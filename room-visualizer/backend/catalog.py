"""Catalog_Loader -- the Asset_Catalog manifest as a live, validated mapping.

Requirement 8.5 is the reason this module exists in the shape it does: dropping a
tile image plus a manifest entry into ``assets/tiles/`` has to make the product
appear with no Python edit and no restart. So the manifest is not read once at
import time and cached forever -- every accessor first compares the manifest's
modification stamp against the one the current mapping was built from, and
re-reads only when it changed. A merchandiser's file save is the trigger; nobody
has to remember to bounce the service.

Three further decisions worth calling out:

**Validation is per entry, never all-or-nothing.** A single malformed tile must
not take the rest of the catalog down with it, so each entry is validated in
isolation; a failure excludes that one entry and logs a WARNING naming the entry
and the specific reason (Requirements 8.3, 8.8). The operator learns *which*
tile is broken and *why* from one log line, which is the whole point of the
requirement.

**Manifest file paths are confined to the tiles directory.** The ``file`` field
is manifest-supplied data, so it is resolved and then checked to sit inside
``assets/tiles/``. An entry naming ``../../etc/passwd`` -- or a symlink pointing
out of the tree, which ``Path.resolve`` collapses before the check -- is rejected
like any other invalid entry. A manifest cannot be used to read arbitrary files.

**Seamless synthesis is memoised per tile.** Building a wrap-continuous pattern
and resampling it to its metric aspect ratio costs 50-200 ms, which would blow
the render budget of Requirement 9.3 on every request. :meth:`seamless`
therefore memoises on ``(tile_id, file_mtime)``: the first render of a tile pays
the cost, later renders do not, and touching the image on disk invalidates just
that one entry. The memo is pruned to the live catalog on every reload, so it
stays bounded by catalog size rather than by history.

The lock is load-bearing rather than defensive, for the same reason it is in
:mod:`backend.cache`: FastAPI dispatches sync route handlers to a threadpool, so
a ``/api/tiles`` reload and a ``/api/render`` memo lookup really do run
concurrently on different threads.

Requirements: 8.2, 8.3, 8.4, 8.5, 8.8.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backend.schemas import TileDefinition
from backend.utils.imageio import DecodeError, decode_image
from backend.utils.texture_helper import SeamlessTexture, make_seamless, to_metric_texture

__all__ = [
    "CatalogLoader",
    "UnknownTile",
    "DEFAULT_MANIFEST_NAME",
    "MANIFEST_VERSION",
]

#: Manifest filename under ``assets/tiles/`` (Requirement 8.2). Mirrors
#: ``Settings.tiles_manifest_name``; passed explicitly by callers that override it.
DEFAULT_MANIFEST_NAME = "manifest.json"

#: Schema version this loader was written against. A higher version is read
#: anyway -- the entry validator is the real gate -- but it is logged, so an
#: operator who ships a newer manifest to an older build finds out from the log
#: rather than from silently missing fields.
MANIFEST_VERSION = 1

_LOGGER = logging.getLogger(__name__)


class UnknownTile(KeyError):
    """Raised by :meth:`CatalogLoader.seamless` for an id not in the catalog.

    Callers that can be handed an arbitrary id -- the render route, resolving a
    client-supplied ``tile_id`` -- should go through :meth:`CatalogLoader.get`
    and turn ``None`` into HTTP 422 ``unknown_tile``. Reaching :meth:`seamless`
    with an unresolved id is a programming error, hence an exception.
    """


class _EntryRejected(Exception):
    """Internal: one manifest entry failed validation, carrying the reason."""


def _reject(reason: str) -> _EntryRejected:
    return _EntryRejected(reason)


def _validate_text(entry: Mapping[str, Any], field: str) -> str:
    if field not in entry:
        raise _reject(f"missing required field {field!r}")
    value = entry[field]
    if not isinstance(value, str):
        raise _reject(f"{field} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise _reject(f"{field} must not be empty")
    return text


def _validate_number(entry: Mapping[str, Any], field: str) -> float:
    if field not in entry:
        raise _reject(f"missing required field {field!r}")
    value = entry[field]
    # ``bool`` is an ``int`` subclass; ``"gloss": true`` is a manifest mistake,
    # not the number 1.0.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _reject(f"{field} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise _reject(f"{field} must be finite, got {value!r}")
    return number


def _validate_positive(entry: Mapping[str, Any], field: str) -> float:
    number = _validate_number(entry, field)
    if number <= 0.0:
        raise _reject(f"{field} must be positive, got {number!r}")
    return number


def _validate_gloss(entry: Mapping[str, Any]) -> float:
    gloss = _validate_number(entry, "gloss")
    if not 0.0 <= gloss <= 1.0:
        raise _reject(f"gloss must be within [0.0, 1.0], got {gloss!r}")
    return gloss


def _validate_optional_grout(entry: Mapping[str, Any]) -> float | None:
    """``grout_mm`` is optional and means "inherit" when absent or null."""
    if entry.get("grout_mm") is None:
        return None
    grout = _validate_number(entry, "grout_mm")
    if grout < 0.0:
        raise _reject(f"grout_mm must not be negative, got {grout!r}")
    return grout


def _file_stamp(path: Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` for ``path``, or ``None`` when it is not a file.

    Size rides along with the modification time because coarse filesystem mtime
    granularity can hide a rewrite that lands within the same tick, and an edit
    that changes a manifest's length is exactly the common case (a merchandiser
    appending a tile).
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


class CatalogLoader:
    """Validated Tile_Definitions from ``assets/tiles/manifest.json``.

    The mapping is rebuilt lazily: every accessor calls
    :meth:`~CatalogLoader._refresh_if_stale` first, so a manifest edit is picked
    up on the next read without a restart (Requirement 8.5). Reads on an
    unchanged manifest cost one ``stat``.
    """

    def __init__(
        self,
        assets_dir: Path,
        logger: logging.Logger | None = None,
        *,
        manifest_name: str = DEFAULT_MANIFEST_NAME,
    ) -> None:
        self._assets_dir = Path(assets_dir)
        self._tiles_dir = self._assets_dir / "tiles"
        self._manifest_path = self._tiles_dir / manifest_name
        self._logger = logger if logger is not None else _LOGGER

        self._lock = threading.RLock()
        self._tiles: dict[str, TileDefinition] = {}
        # ``None`` means "never loaded"; a missing manifest loads to an empty
        # catalog and stamps ``(-1, -1)``, so the reappearance of the file is
        # itself a stamp change and triggers a reload.
        self._manifest_stamp: tuple[int, int] | None = None
        # Memoised seamless textures, keyed by ``(tile_id, mtime_ns, size)`` of
        # the tile image -- the ``(tile_id, file_mtime)`` key of the design, with
        # size added for the same reason as in :func:`_file_stamp`.
        self._seamless: dict[tuple[str, int, int], SeamlessTexture] = {}
        # Decode probes are the expensive half of validation, so their outcome is
        # remembered per image stamp and pruned to the live manifest each load.
        # A hot reload triggered by an appended entry then re-decodes only the
        # new image.
        self._decodable: dict[tuple[str, int, int], bool] = {}

    # ----------------------------------------------------------------- paths --

    @property
    def assets_dir(self) -> Path:
        """Root assets directory this loader was constructed against."""
        return self._assets_dir

    @property
    def tiles_dir(self) -> Path:
        """Directory holding tile images and the manifest (Requirement 8.2)."""
        return self._tiles_dir

    @property
    def manifest_path(self) -> Path:
        """Full path of the Asset_Catalog manifest."""
        return self._manifest_path

    # ----------------------------------------------------------- public API --

    def load(self) -> dict[str, TileDefinition]:
        """Read and validate the manifest now, replacing the current mapping.

        Unconditional: use :meth:`all` or :meth:`get` for the ordinary read path,
        which reloads only when the manifest changed.

        Returns:
            A copy of the freshly built ``tile_id -> TileDefinition`` mapping,
            in manifest order, containing only fully valid entries. Every
            excluded entry has been logged at WARNING level (Requirement 8.8).
        """
        with self._lock:
            stamp = _file_stamp(self._manifest_path)
            tiles = self._read_manifest()

            self._tiles = tiles
            self._manifest_stamp = stamp if stamp is not None else (-1, -1)
            # Keep both memos bounded by catalog size rather than by history:
            # entries for tiles that no longer exist can never be hit again.
            live_ids = set(tiles)
            self._seamless = {
                key: texture for key, texture in self._seamless.items() if key[0] in live_ids
            }
            live_images = {str(tile.image_path) for tile in tiles.values()}
            self._decodable = {
                key: ok for key, ok in self._decodable.items() if key[0] in live_images
            }
            return dict(tiles)

    def get(self, tile_id: str) -> TileDefinition | None:
        """Return the Tile_Definition for ``tile_id``, or ``None`` if unknown.

        ``None`` covers both "never declared" and "declared but invalid", which
        is what the render route turns into HTTP 422 ``unknown_tile``.
        """
        with self._lock:
            self._refresh_if_stale()
            return self._tiles.get(tile_id)

    def all(self) -> list[TileDefinition]:
        """Every valid Tile_Definition, in manifest order (Requirement 8.4)."""
        with self._lock:
            self._refresh_if_stale()
            return list(self._tiles.values())

    def seamless(self, tile_id: str) -> SeamlessTexture:
        """Return the memoised seamless, metrically scaled texture for a tile.

        Built on first use as ``to_metric_texture(make_seamless(image), w, h)``,
        so the returned pattern both wraps without a visible seam (Requirement
        8.1) and carries the declared millimetre dimensions plus the
        ``px_per_mm`` scale the Compositor samples with (Requirements 8.6, 8.7).

        Memoised on ``(tile_id, file_mtime)``: repeat renders of the same tile
        are a dict lookup, and replacing the image on disk rebuilds just that
        entry.

        Raises:
            UnknownTile: ``tile_id`` is not a valid catalog entry.
            OSError or DecodeError: the image was deleted or replaced with
                something unreadable after it was validated and without the
                manifest changing, so no reload had a chance to exclude it.
        """
        with self._lock:
            self._refresh_if_stale()
            tile = self._tiles.get(tile_id)
            if tile is None:
                raise UnknownTile(tile_id)
            stamp = _file_stamp(tile.image_path)
            key = (tile_id, *stamp) if stamp is not None else (tile_id, -1, -1)
            cached = self._seamless.get(key)
            if cached is not None:
                return cached

        # Built outside the lock: synthesis is 50-200 ms and holding the lock
        # would serialise every concurrent render behind the first one. Two
        # threads racing on the same cold tile each build a texture and the last
        # store wins; the results are equivalent, so the duplicated work is a
        # better trade than blocking every other tile.
        texture = self._build_seamless(tile)

        with self._lock:
            return self._seamless.setdefault(key, texture)

    # -------------------------------------------------------------- internals --

    def _refresh_if_stale(self) -> None:
        """Reload when the manifest's modification stamp changed (R8.5)."""
        if self._manifest_stamp != _file_stamp(self._manifest_path):
            self.load()

    def _build_seamless(self, tile: TileDefinition) -> SeamlessTexture:
        image = decode_image(tile.image_path.read_bytes())
        return to_metric_texture(make_seamless(image), tile.width_mm, tile.height_mm)

    def _read_manifest(self) -> dict[str, TileDefinition]:
        """Parse the manifest and validate every entry, logging exclusions."""
        raw = self._read_manifest_document()
        if raw is None:
            return {}

        version = raw.get("version")
        declared = isinstance(version, int) and not isinstance(version, bool)
        if declared and version > MANIFEST_VERSION:
            self._logger.warning(
                "tile manifest declares version=%s, newer than the supported version=%s; "
                "unrecognised fields are ignored (%s)",
                version,
                MANIFEST_VERSION,
                self._manifest_path,
            )

        entries = raw.get("tiles")
        if not isinstance(entries, list):
            self._logger.warning(
                "tile manifest is missing a 'tiles' array; no tiles loaded (%s)",
                self._manifest_path,
            )
            return {}

        tiles: dict[str, TileDefinition] = {}
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                self._log_rejection(
                    self._describe(entry, index),
                    f"entry must be an object, got {type(entry).__name__}",
                )
                continue
            try:
                tile = self._validate_entry(entry)
            except _EntryRejected as exc:
                self._log_rejection(self._describe(entry, index), str(exc))
                continue
            if tile.id in tiles:
                self._log_rejection(tile.id, "duplicate id; the first entry wins")
                continue
            tiles[tile.id] = tile
        return tiles

    def _read_manifest_document(self) -> dict[str, Any] | None:
        """Return the parsed manifest object, or ``None`` if it is unusable.

        A missing manifest is normal before ``setup_assets.py`` has run, so it is
        reported at DEBUG; an unreadable or malformed one is an operator problem
        and is reported at WARNING. Either way the catalog becomes empty rather
        than stale, and because a subsequent save changes the modification stamp,
        the next read recovers on its own -- which matters when the warning was
        caused by reading a manifest halfway through being written.
        """
        try:
            text = self._manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._logger.debug("no tile manifest at %s; catalog is empty", self._manifest_path)
            return None
        except OSError as exc:
            self._logger.warning(
                "tile manifest could not be read (%s): %s; no tiles loaded",
                self._manifest_path,
                exc,
            )
            return None

        try:
            document = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._logger.warning(
                "tile manifest is not valid JSON (%s): %s; no tiles loaded",
                self._manifest_path,
                exc,
            )
            return None

        if not isinstance(document, Mapping):
            self._logger.warning(
                "tile manifest must be a JSON object, got %s (%s); no tiles loaded",
                type(document).__name__,
                self._manifest_path,
            )
            return None
        return dict(document)

    def _validate_entry(self, entry: Mapping[str, Any]) -> TileDefinition:
        """Build a TileDefinition or raise :class:`_EntryRejected` (R8.3).

        Cheap field checks run before the image is touched, so a manifest full of
        malformed entries costs no decodes.
        """
        tile_id = _validate_text(entry, "id")
        name = _validate_text(entry, "name")
        file_field = _validate_text(entry, "file")
        finish = _validate_text(entry, "finish")
        width_mm = _validate_positive(entry, "width_mm")
        height_mm = _validate_positive(entry, "height_mm")
        gloss = _validate_gloss(entry)
        grout_mm = _validate_optional_grout(entry)
        image_path = self._resolve_image(file_field)

        return TileDefinition(
            id=tile_id,
            name=name,
            image_path=image_path,
            width_mm=width_mm,
            height_mm=height_mm,
            finish=finish,
            gloss=gloss,
            grout_mm=grout_mm,
        )

    def _resolve_image(self, file_field: str) -> Path:
        """Confine ``file`` to the tiles directory and prove the image decodes.

        Three separate gates, because they catch different things: an absolute
        path is rejected outright (the field is documented as a name relative to
        ``assets/tiles/``), any ``..`` component is rejected by name, and the
        *resolved* path is then required to sit under the resolved tiles
        directory -- which is what catches a symlink inside the tree pointing out
        of it.

        The returned path is the plain join rather than the resolved form, so it
        stays comparable to the path the caller wrote and reads identically.
        """
        candidate = Path(file_field)
        if candidate.is_absolute():
            raise _reject(f"file must be relative to the tiles directory, got {file_field!r}")
        if any(part == ".." for part in candidate.parts):
            raise _reject(f"file must not traverse out of the tiles directory: {file_field!r}")

        image_path = self._tiles_dir / candidate
        tiles_dir = self._tiles_dir.resolve()
        if not image_path.resolve().is_relative_to(tiles_dir):
            raise _reject(f"file resolves outside the tiles directory: {file_field!r}")
        if not image_path.is_file():
            raise _reject(f"image file not found: {file_field!r}")
        if not self._probe_decodable(image_path):
            raise _reject(f"image file does not decode: {file_field!r}")
        return image_path

    def _probe_decodable(self, path: Path) -> bool:
        """True when ``path`` decodes as a raster image, memoised per stamp."""
        stamp = _file_stamp(path)
        if stamp is None:
            return False
        key = (str(path), *stamp)
        cached = self._decodable.get(key)
        if cached is not None:
            return cached
        try:
            decode_image(path.read_bytes())
        except (DecodeError, OSError):
            decodable = False
        else:
            decodable = True
        self._decodable[key] = decodable
        return decodable

    @staticmethod
    def _describe(entry: Any, index: int) -> str:
        """Best available name for an entry, for the WARNING log (R8.8)."""
        if isinstance(entry, Mapping):
            candidate = entry.get("id")
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return f"<entry {index}>"

    def _log_rejection(self, tile_id: str, reason: str) -> None:
        self._logger.warning(
            "invalid tile manifest entry excluded: id=%s reason=%s (%s)",
            tile_id,
            reason,
            self._manifest_path,
        )
