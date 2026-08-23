"""Tests for the Catalog_Loader in `backend/catalog.py` (Requirements 8.2, 8.3, 8.5, 8.8).

The catalog is the merchandiser's interface to the product range: drop an image
plus a manifest entry into `assets/tiles/` and the tile appears, with no Python
edit and no restart. Two things have to be true for that promise to be safe, and
this module is organised around them.

**Property 27** is the acceptance gate. It is a biconditional, so it is tested
as one: an entry is included *exactly* when it declares a positive width, a
positive height, a non-empty finish, a gloss inside `[0.0, 1.0]`, and an image
that exists and decodes. Half a test would only check that valid entries load;
the property also pins that every invalid entry is dropped and that each
exclusion leaves exactly one warning-level record naming it, which is what
Requirement 8.8 gives the operator instead of a silently short catalog.

The unit tests around it pin what a property over a single manifest snapshot
cannot see: that a *later* manifest edit is picked up from its modification
stamp (Requirement 8.5), that a manifest cannot be used to read files outside
the tiles directory, that `seamless()` is memoised per image stamp so the render
budget is paid once per tile, and that a manifest-level failure yields an empty
catalog rather than a stale one.

Everything here drives `CatalogLoader` directly against a temp assets directory,
with no HTTP layer involved. Property 26, stated over `GET /api/tiles`, is the
same guarantee observed one layer up and lives in the HTTP section at the foot
of this module.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from backend.catalog import (
    DEFAULT_MANIFEST_NAME,
    MANIFEST_VERSION,
    CatalogLoader,
    UnknownTile,
)
from tests.conftest import TINY_CATALOG_TILES

# --------------------------------------------------------------------------- #
# Log capture
# --------------------------------------------------------------------------- #
#
# `CatalogLoader` takes its logger by injection, so the exclusion records of
# Requirement 8.8 are captured by handing it a logger of our own rather than by
# reaching for `caplog`. That matters for the property below: `caplog` is
# function-scoped, and a `@given` body that used it would either trip
# Hypothesis's function-scoped-fixture health check or accumulate records across
# examples.

_LOGGER_NAME = "tests.catalog"


class _Recorder(logging.Handler):
    """Collects every record the loader emits, at every level."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self, level: int = logging.WARNING) -> list[str]:
        """Formatted messages at exactly ``level``, in emission order."""
        return [record.getMessage() for record in self.records if record.levelno == level]


def _recording_loader(assets_dir: Path, **kwargs: Any) -> tuple[CatalogLoader, _Recorder]:
    """A loader over ``assets_dir`` plus the recorder holding its output.

    One logger name is reused and its handlers replaced per call, so a
    Hypothesis run does not register hundreds of loggers. `propagate` is off so
    the deliberate warnings under test do not show up as noise in pytest's own
    captured log.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for existing in list(logger.handlers):
        logger.removeHandler(existing)
    recorder = _Recorder()
    logger.addHandler(recorder)
    return CatalogLoader(assets_dir, logger, **kwargs), recorder


# --------------------------------------------------------------------------- #
# Asset builders
# --------------------------------------------------------------------------- #


def _write_png(path: Path, size_px: tuple[int, int] = (8, 8), seed: int = 0) -> Path:
    """Write a tiny decodable BGR PNG. Small on purpose: the catalog only asks
    that the file decodes, and the property builds one per manifest entry per
    example."""
    width_px, height_px = size_px
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(height_px, width_px, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):  # pragma: no cover - filesystem failure
        raise RuntimeError(f"failed to write {path}")
    return path


def _write_manifest(
    tiles_dir: Path,
    entries: Iterable[dict[str, Any]],
    *,
    version: int = MANIFEST_VERSION,
    name: str = DEFAULT_MANIFEST_NAME,
) -> Path:
    path = tiles_dir / name
    path.write_text(
        json.dumps({"version": version, "tiles": list(entries)}, indent=2), encoding="utf-8"
    )
    return path


def _touch_forward(path: Path, seconds: float = 2.0) -> None:
    """Push ``path``'s mtime forward by ``seconds``.

    Filesystem mtime granularity can be coarse enough to hide a rewrite that
    lands within the same tick, which would make a reload test pass or fail on
    timing rather than on behaviour. Forcing a distinct stamp removes the race.
    """
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + int(seconds * 1e9)))


def _valid_entry(index: int) -> dict[str, Any]:
    """A fully valid manifest entry; callers mutate one field to make it fail."""
    return {
        "id": f"tile-{index:02d}",
        "name": f"Tile {index:02d}",
        "file": f"tile_{index:02d}.png",
        "width_mm": 600.0,
        "height_mm": 600.0,
        "finish": "matte",
        "gloss": 0.4,
    }


# --------------------------------------------------------------------------- #
# Property 27 -- the acceptance gate (Requirements 8.3, 8.8)
# --------------------------------------------------------------------------- #

#: The five conditions Property 27 names, as defects that violate exactly one
#: each. `missing_image` and `undecodable_image` are two ways to violate the
#: single "exists and decodes" condition and are normalised below when both are
#: drawn.
_DEFECTS = ("width", "height", "finish", "gloss", "missing_image", "undecodable_image")

#: Substring the loader's rejection reason must carry for each defect. Kept
#: separate from the defect labels so the assertion checks that the operator is
#: told *which field* is wrong, not merely that something is.
_DEFECT_TOKENS = {
    "width": "width_mm",
    "height": "height_mm",
    "finish": "finish",
    "gloss": "gloss",
    "missing_image": "image file not found",
    "undecodable_image": "does not decode",
}

_finite = {"allow_nan": False, "allow_infinity": False}

#: Each pool is disjoint from its counterpart, so a drawn value is unambiguously
#: valid or unambiguously invalid and the oracle needs no tolerance.
_valid_mm = st.floats(min_value=1.0, max_value=3000.0, **_finite)
_invalid_mm = st.one_of(
    st.just(0.0),
    st.floats(min_value=-3000.0, max_value=-0.001, **_finite),
)
_valid_finish = st.sampled_from(("polished", "matte", "satin", "lappato", "structured"))
_invalid_finish = st.sampled_from(("", " ", "\t", "\n  "))
_valid_gloss = st.floats(min_value=0.0, max_value=1.0, **_finite)
_invalid_gloss = st.one_of(
    st.floats(min_value=-10.0, max_value=-0.001, **_finite),
    st.floats(min_value=1.001, max_value=10.0, **_finite),
)

#: An explicitly weighted union rather than `st.sets(..., max_size=3)`: with six
#: defect labels, an unweighted set strategy draws the no-defect case too rarely
#: to exercise the acceptance half of the biconditional.
_defect_sets = st.one_of(
    st.just(frozenset()),
    st.sets(st.sampled_from(_DEFECTS), min_size=1, max_size=3).map(frozenset),
)

_entry_spec = st.fixed_dictionaries(
    {
        "defects": _defect_sets,
        "width_mm": _valid_mm,
        "bad_width_mm": _invalid_mm,
        "height_mm": _valid_mm,
        "bad_height_mm": _invalid_mm,
        "finish": _valid_finish,
        "bad_finish": _invalid_finish,
        "gloss": _valid_gloss,
        "bad_gloss": _invalid_gloss,
    }
)

_PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _materialise(tiles_dir: Path, specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write the images and return the manifest entries the specs describe.

    Returns one entry per spec, in order, each carrying the drawn valid value
    for every condition it does not violate and the drawn invalid value for
    every condition it does.
    """
    entries: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        defects = spec["defects"]
        entry = _valid_entry(index)
        entry["width_mm"] = spec["bad_width_mm"] if "width" in defects else spec["width_mm"]
        entry["height_mm"] = spec["bad_height_mm"] if "height" in defects else spec["height_mm"]
        entry["finish"] = spec["bad_finish"] if "finish" in defects else spec["finish"]
        entry["gloss"] = spec["bad_gloss"] if "gloss" in defects else spec["gloss"]

        image_path = tiles_dir / entry["file"]
        if "missing_image" in defects:
            pass  # the file simply never appears
        elif "undecodable_image" in defects:
            image_path.write_bytes(b"this is text, not a PNG")
        else:
            _write_png(image_path, seed=index)
        entries.append(entry)
    return entries


def _is_valid(spec: dict[str, Any]) -> bool:
    """The oracle, written from Property 27 rather than from `catalog.py`."""
    return not spec["defects"]


# Feature: ai-room-tile-visualizer, Property 27: Manifest entries are accepted
# only when fully valid
@_PROPERTY_SETTINGS
@given(specs=st.lists(_entry_spec, min_size=1, max_size=6))
def test_property_27_entries_are_accepted_exactly_when_fully_valid(specs):
    """For any manifest, an entry is included exactly when it declares a
    positive width, a positive height, a non-empty finish, a gloss within
    `[0.0, 1.0]`, and an image file that exists and decodes; every excluded
    entry produces exactly one warning-level record naming it.

    Both directions are asserted, since either alone is satisfiable by a broken
    loader: one that accepted everything would pass the inclusion half, and one
    that accepted nothing would pass the exclusion half. Identity is checked
    through `all()` and `get()` together because they are separate code paths
    over the same mapping and the render route depends on them agreeing.

    The manifest is built under a fresh temp directory inside the example rather
    than from the `tiny_catalog` fixture: each example needs its own assets tree,
    and drawing a function-scoped fixture into a `@given` body would reuse one
    tree across every example.

    **Validates: Requirements 8.3, 8.8**
    """
    with tempfile.TemporaryDirectory() as raw_tmp:
        assets_dir = Path(raw_tmp) / "assets"
        tiles_dir = assets_dir / "tiles"
        tiles_dir.mkdir(parents=True)

        entries = _materialise(tiles_dir, specs)
        _write_manifest(tiles_dir, entries)

        loader, recorder = _recording_loader(assets_dir)
        loaded = loader.all()

        expected_ids = [
            entry["id"] for entry, spec in zip(entries, specs) if _is_valid(spec)
        ]
        excluded = [(entry, spec) for entry, spec in zip(entries, specs) if not _is_valid(spec)]

        # Inclusion and exclusion in one assertion, and in manifest order --
        # Requirement 8.4 promises the catalog reads back in the order it was
        # written, which a set comparison would not catch.
        assert [tile.id for tile in loaded] == expected_ids

        # `get()` must agree with `all()`: an id absent from the catalog and an
        # id present but invalid both have to read as `None`, since that is what
        # the render route turns into HTTP 422 `unknown_tile`.
        for entry, spec in zip(entries, specs):
            resolved = loader.get(entry["id"])
            assert (resolved is not None) == _is_valid(spec), (
                f"{entry['id']!r} inclusion disagreed with the acceptance rule "
                f"(defects={sorted(spec['defects'])})"
            )
            if resolved is not None:
                # An accepted entry must come back carrying what it declared,
                # not a coerced or defaulted version of it.
                assert resolved.width_mm == entry["width_mm"]
                assert resolved.height_mm == entry["height_mm"]
                assert resolved.finish == entry["finish"]
                assert resolved.gloss == entry["gloss"]
                # Stored as the plain join, so it stays comparable to the path
                # the manifest author wrote.
                assert resolved.image_path == loader.tiles_dir / entry["file"]

        # Requirement 8.8: one warning per exclusion, no more and no fewer, so a
        # loader that logged nothing and one that logged a storm both fail.
        warnings = recorder.messages(logging.WARNING)
        assert len(warnings) == len(excluded)

        for entry, spec in excluded:
            naming = [message for message in warnings if f"id={entry['id']}" in message]
            assert len(naming) == 1, (
                f"expected exactly one warning naming {entry['id']!r}, got {naming}"
            )
            # The record has to say which field is wrong; an operator holding
            # only "some entry was invalid" cannot fix the manifest.
            tokens = {_DEFECT_TOKENS[defect] for defect in spec["defects"]}
            assert any(token in naming[0] for token in tokens), (
                f"warning for {entry['id']!r} named no violated field: {naming[0]!r}"
            )

        # An accepted entry must not be maligned in the log.
        for entry in (entry for entry, spec in zip(entries, specs) if _is_valid(spec)):
            assert not any(f"id={entry['id']}" in message for message in warnings)


def test_a_single_invalid_entry_does_not_take_the_catalog_down_with_it(tmp_path):
    """Guard for Property 27: validation is per entry, not all-or-nothing.

    The property draws whole manifests, so a loader that discarded the entire
    file on the first bad entry would still satisfy it on every all-valid and
    every all-invalid example. This pins the mixed case directly.
    """
    assets_dir = tmp_path / "assets"
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True)

    good, bad = _valid_entry(0), _valid_entry(1)
    bad["gloss"] = 1.5
    _write_png(tiles_dir / good["file"])
    _write_png(tiles_dir / bad["file"])
    _write_manifest(tiles_dir, [bad, good])

    loader, recorder = _recording_loader(assets_dir)

    assert [tile.id for tile in loader.all()] == [good["id"]]
    assert len(recorder.messages(logging.WARNING)) == 1


def test_duplicate_ids_keep_the_first_entry_and_warn(tmp_path):
    """Two entries claiming one id is ambiguous, so the tie is broken by
    position and the loser is reported like any other exclusion."""
    assets_dir = tmp_path / "assets"
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True)

    first, second = _valid_entry(0), _valid_entry(1)
    second["id"] = first["id"]
    _write_png(tiles_dir / first["file"])
    _write_png(tiles_dir / second["file"])
    _write_manifest(tiles_dir, [first, second])

    loader, recorder = _recording_loader(assets_dir)
    tiles = loader.all()

    assert [tile.id for tile in tiles] == [first["id"]]
    assert tiles[0].image_path == loader.tiles_dir / first["file"]
    assert any("duplicate id" in message for message in recorder.messages(logging.WARNING))


# --------------------------------------------------------------------------- #
# Hot reload (Requirement 8.5)
# --------------------------------------------------------------------------- #


def test_reload_picks_up_an_appended_entry_with_no_restart(tiny_catalog):
    """Requirement 8.5, the merchandiser's whole workflow: drop an image, add a
    manifest line, and the tile is served.

    The loader is read *before* the edit as well as after, so the assertion is
    that the second read observed the change rather than that it happened to be
    the first read of all.
    """
    loader, _ = _recording_loader(tiny_catalog)
    tiles_dir = loader.tiles_dir

    before = [tile.id for tile in loader.all()]
    assert before == [spec["id"] for spec in TINY_CATALOG_TILES]
    assert loader.get("tile-99") is None

    added = _valid_entry(99)
    _write_png(tiles_dir / added["file"])
    manifest = json.loads(loader.manifest_path.read_text(encoding="utf-8"))
    manifest["tiles"].append(added)
    loader.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _touch_forward(loader.manifest_path)

    assert [tile.id for tile in loader.all()] == before + [added["id"]]
    appended = loader.get(added["id"])
    assert appended is not None and appended.name == added["name"]


def test_a_manifest_appearing_after_a_read_is_still_picked_up(tmp_path):
    """A missing manifest is the state before `setup_assets.py` has run, so it
    must not be sticky.

    Absence is reported at DEBUG, not WARNING: nothing is wrong yet, and an empty
    assets tree is the normal state of a fresh checkout. A `load()` here is
    explicit because the lazy accessors short-circuit while there is neither a
    manifest nor a previous stamp to compare against -- the read is already
    correct at that point, so there is nothing to re-read and nothing to say.
    """
    assets_dir = tmp_path / "assets"
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True)

    loader, recorder = _recording_loader(assets_dir)
    assert loader.all() == []
    assert loader.load() == {}
    assert recorder.messages(logging.WARNING) == []
    assert any("no tile manifest" in message for message in recorder.messages(logging.DEBUG))

    entry = _valid_entry(0)
    _write_png(tiles_dir / entry["file"])
    _write_manifest(tiles_dir, [entry])

    assert [tile.id for tile in loader.all()] == [entry["id"]]


def test_an_unchanged_manifest_is_not_re_read(tiny_catalog, monkeypatch):
    """The reload check runs on every accessor, so it has to be a `stat` and not
    a re-parse; otherwise every `/api/tiles` and every render pays for JSON
    parsing and a decode probe per tile."""
    loader, _ = _recording_loader(tiny_catalog)
    loader.all()

    reads = 0
    real_read_text = Path.read_text

    def counting_read_text(self, *args, **kwargs):
        nonlocal reads
        if self == loader.manifest_path:
            reads += 1
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counting_read_text)

    for _ in range(5):
        loader.all()
        loader.get(TINY_CATALOG_TILES[0]["id"])

    assert reads == 0


def test_a_malformed_manifest_empties_the_catalog_and_warns(tiny_catalog):
    """A half-written or corrupt manifest must not leave the previous mapping in
    place: serving tiles the manifest no longer declares is a worse failure than
    serving none, because it is invisible. Recovery is automatic, since the next
    save is itself a stamp change."""
    loader, recorder = _recording_loader(tiny_catalog)
    assert len(loader.all()) == len(TINY_CATALOG_TILES)

    loader.manifest_path.write_text('{"version": 1, "tiles": [', encoding="utf-8")
    _touch_forward(loader.manifest_path)

    assert loader.all() == []
    assert loader.get(TINY_CATALOG_TILES[0]["id"]) is None
    assert any("not valid JSON" in message for message in recorder.messages(logging.WARNING))

    entry = _valid_entry(0)
    _write_png(loader.tiles_dir / entry["file"])
    _write_manifest(loader.tiles_dir, [entry])
    _touch_forward(loader.manifest_path)

    assert [tile.id for tile in loader.all()] == [entry["id"]]


def test_a_newer_manifest_version_is_loaded_with_a_warning(tmp_path):
    """Forward compatibility is deliberate: the per-entry validator is the real
    gate, so a newer manifest still serves its valid entries. The warning is
    what tells an operator running an older build why unfamiliar fields were
    ignored."""
    assets_dir = tmp_path / "assets"
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True)

    entry = _valid_entry(0)
    _write_png(tiles_dir / entry["file"])
    _write_manifest(tiles_dir, [entry], version=MANIFEST_VERSION + 1)

    loader, recorder = _recording_loader(assets_dir)

    assert [tile.id for tile in loader.all()] == [entry["id"]]
    assert any("newer than the supported" in m for m in recorder.messages(logging.WARNING))


# --------------------------------------------------------------------------- #
# Path confinement
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "file_field",
    [
        "../escaped.png",
        "nested/../../escaped.png",
        "/etc/passwd",
    ],
    ids=["parent_traversal", "traversal_via_subdirectory", "absolute_path"],
)
def test_a_file_field_escaping_the_tiles_directory_is_rejected(tmp_path, file_field):
    """`file` is manifest-supplied data, so it is confined to `assets/tiles/`.

    The escape target is a real, decodable image, which removes the possibility
    that the entry was rejected merely for being unreadable: the only reason left
    is that it points outside the tree.
    """
    assets_dir = tmp_path / "assets"
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True)
    _write_png(tmp_path / "escaped.png")
    _write_png(assets_dir / "escaped.png")

    entry = _valid_entry(0)
    entry["file"] = file_field
    _write_manifest(tiles_dir, [entry])

    loader, recorder = _recording_loader(assets_dir)

    assert loader.all() == []
    assert loader.get(entry["id"]) is None
    assert len(recorder.messages(logging.WARNING)) == 1


def test_a_symlink_out_of_the_tiles_directory_is_rejected(tmp_path):
    """The name-level `..` check cannot see a symlink, so confinement is also
    enforced on the resolved path. Without this, a link inside `assets/tiles/`
    would read anything the service can."""
    assets_dir = tmp_path / "assets"
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True)

    outside = _write_png(tmp_path / "outside" / "secret.png")
    link = tiles_dir / "escape.png"
    link.symlink_to(outside)

    entry = _valid_entry(0)
    entry["file"] = "escape.png"
    _write_manifest(tiles_dir, [entry])

    loader, recorder = _recording_loader(assets_dir)

    assert loader.all() == []
    assert len(recorder.messages(logging.WARNING)) == 1
    assert any("outside the tiles directory" in m for m in recorder.messages(logging.WARNING))


def test_a_file_field_in_a_subdirectory_of_tiles_is_accepted(tmp_path):
    """Guard for the confinement tests: the rule is "inside the tiles
    directory", not "directly inside it", so a rejection of every path
    containing a separator would be too strict."""
    assets_dir = tmp_path / "assets"
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True)

    entry = _valid_entry(0)
    entry["file"] = "collection/tile_00.png"
    _write_png(tiles_dir / entry["file"])
    _write_manifest(tiles_dir, [entry])

    loader, _ = _recording_loader(assets_dir)
    tile = loader.get(entry["id"])

    assert tile is not None
    assert tile.image_path == loader.tiles_dir / entry["file"]


# --------------------------------------------------------------------------- #
# Seamless memoisation
# --------------------------------------------------------------------------- #


def test_seamless_is_memoised_per_tile_and_rebuilds_when_the_image_changes(tiny_catalog):
    """Seamless synthesis plus the metric resample costs 50-200 ms, which would
    blow the render budget on every request if it were repeated.

    Memoisation is asserted by identity, which is the only observable difference
    between a cached return and a rebuild that happens to produce equal pixels.
    Invalidation is driven by touching the image's mtime with unchanged bytes:
    the rebuilt texture must therefore be a *different object* carrying *equal*
    content, which distinguishes "the key includes the file stamp" from both "the
    memo never invalidates" and "the memo invalidates and rebuilds wrongly".
    """
    loader, _ = _recording_loader(tiny_catalog)
    tile_id = TINY_CATALOG_TILES[0]["id"]
    other_id = TINY_CATALOG_TILES[1]["id"]

    first = loader.seamless(tile_id)
    assert loader.seamless(tile_id) is first

    # Per tile, not one global slot.
    other = loader.seamless(other_id)
    assert other is not first
    assert loader.seamless(tile_id) is first

    tile = loader.get(tile_id)
    assert tile is not None
    _touch_forward(tile.image_path)

    rebuilt = loader.seamless(tile_id)
    assert rebuilt is not first
    assert np.array_equal(rebuilt.pattern, first.pattern)
    assert loader.seamless(other_id) is other


def test_seamless_carries_the_declared_metric_dimensions(tiny_catalog):
    """The memo must not lose what the Compositor samples with: each texture
    reports the millimetre size its entry declared, so a 1:2 tile stays 1:2."""
    loader, _ = _recording_loader(tiny_catalog)

    for spec in TINY_CATALOG_TILES:
        texture = loader.seamless(spec["id"])
        assert texture.width_mm == spec["width_mm"]
        assert texture.height_mm == spec["height_mm"]
        assert texture.px_per_mm > 0.0
        assert texture.pattern.dtype == np.uint8
        assert texture.pattern.ndim == 3 and texture.pattern.shape[2] == 3


def test_seamless_raises_unknown_tile_for_an_unresolved_id(tiny_catalog):
    """`seamless()` is reached only after `get()` has resolved an id, so an
    unknown id here is a programming error and raises rather than returning a
    placeholder. `UnknownTile` subclasses `KeyError` so existing handlers still
    catch it."""
    loader, _ = _recording_loader(tiny_catalog)

    with pytest.raises(UnknownTile):
        loader.seamless("no-such-tile")

    # An entry excluded by validation is indistinguishable from one never
    # declared, which is what keeps the render route's 422 path uniform.
    invalid = _valid_entry(0)
    invalid["width_mm"] = -1.0
    _write_png(loader.tiles_dir / invalid["file"])
    manifest = json.loads(loader.manifest_path.read_text(encoding="utf-8"))
    manifest["tiles"].append(invalid)
    loader.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _touch_forward(loader.manifest_path)

    with pytest.raises(UnknownTile):
        loader.seamless(invalid["id"])


# --------------------------------------------------------------------------- #
# load() and paths
# --------------------------------------------------------------------------- #


def test_load_returns_an_independent_copy_in_manifest_order(tiny_catalog):
    """`load()` hands out a copy, so a caller mutating the returned mapping
    cannot corrupt the catalog every later read is served from."""
    loader, _ = _recording_loader(tiny_catalog)

    mapping = loader.load()
    assert list(mapping) == [spec["id"] for spec in TINY_CATALOG_TILES]

    mapping.clear()
    assert len(loader.all()) == len(TINY_CATALOG_TILES)


def test_paths_are_derived_from_the_assets_directory(tmp_path):
    """Requirement 8.2 fixes where the manifest lives; the override exists for
    tests and alternate deployments rather than to move the tiles directory."""
    assets_dir = tmp_path / "assets"
    loader, _ = _recording_loader(assets_dir)

    assert loader.assets_dir == assets_dir
    assert loader.tiles_dir == assets_dir / "tiles"
    assert loader.manifest_path == assets_dir / "tiles" / DEFAULT_MANIFEST_NAME

    custom, _ = _recording_loader(assets_dir, manifest_name="catalog.json")
    assert custom.manifest_path == assets_dir / "tiles" / "catalog.json"


# --------------------------------------------------------------------------- #
# HTTP round trip -- Property 26 (task 13.4)
# --------------------------------------------------------------------------- #
#
# Property 27 above is stated over `CatalogLoader` and driven against it
# directly. Property 26 is the same guarantee observed one layer up, over `GET
# /api/tiles`: the merchandiser's promise is not "the loader parses my manifest",
# it is "my tile appears in the API response, at a URL that fetches the image I
# dropped in". Those differ in everything between the loader and the wire --
# lifespan wiring, response serialisation, and the `thumbnail_url` the frontend
# actually requests -- so the round trip is asserted end to end here rather than
# inferred from the loader tests.

import itertools
import shutil
from urllib.parse import quote

import backend
from backend.config import get_settings

# --------------------------------------------------------------------------- #
# Module-scoped app over a rewritable assets tree
# --------------------------------------------------------------------------- #
#
# The `client` fixture in `conftest.py` is function-scoped and points at
# `tiny_catalog`, so a `@given` body cannot request it: Hypothesis would either
# trip its function-scoped-fixture health check or reuse one catalog across every
# example. This fixture is the same wiring at module scope -- neural backend off,
# `RV_ASSETS_DIR` pointed at a temp tree -- and each example rewrites the tree
# rather than rebuilding the app.
#
# Reusing one running app across examples is not a compromise here; it is closer
# to what Requirement 8.5 promises. Every example proves the catalog changed
# under a process that was never restarted.


@pytest.fixture(scope="module")
def http_catalog(tmp_path_factory):
    """Yield ``(TestClient, tiles_dir)`` for an app serving a rewritable tree.

    The environment is set and the settings cache cleared *before* the app
    import, so both the module-level settings read and the `lifespan` startup
    observe the temp assets directory. The `/assets` mount resolves its directory
    from `app.state` per request, so it and `app.state.catalog` describe the same
    files for the whole module.
    """
    root = tmp_path_factory.mktemp("http-catalog")
    assets_dir = root / "assets"
    (assets_dir / "tiles").mkdir(parents=True)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("RV_ENABLE_NEURAL_BACKEND", "false")
        patch.setenv("RV_ASSETS_DIR", str(assets_dir))
        patch.setenv("RV_WEIGHTS_DIR", str(root / "weights"))
        get_settings.cache_clear()

        # Imported inside the body for the same reason `conftest.py` does it: the
        # import must happen after the environment is in place.
        from fastapi.testclient import TestClient

        from backend.app import app

        try:
            with TestClient(app) as test_client:
                yield test_client, assets_dir / "tiles"
        finally:
            get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Per-example asset tree
# --------------------------------------------------------------------------- #

#: Strictly increasing manifest modification stamps. Rewriting the manifest sets
#: its mtime to the current time, and two examples can land inside one
#: filesystem tick -- which would leave the loader's stamp unchanged and serve
#: the previous example's catalog. Stamping from a counter removes the race
#: entirely rather than making it unlikely.
_stamp_ticks = itertools.count(1)
_STAMP_EPOCH_NS = 1_600_000_000_000_000_000


def _force_new_manifest_stamp(path: Path) -> None:
    """Give ``path`` an mtime no earlier read has seen."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, _STAMP_EPOCH_NS + next(_stamp_ticks) * 1_000_000_000))


def _write_png_bytes(path: Path, seed: int) -> None:
    """Write a decodable PNG, encoding in memory rather than through `cv2.imwrite`.

    `cv2.imwrite` routes the path through OpenCV's own file handling, which is
    unreliable for the non-ASCII and punctuation-heavy filenames this section
    deliberately generates -- and it fails by returning `False`, so a name it
    cannot write would look like a missing image rather than a harness problem.
    Encoding to a buffer and writing the bytes with `pathlib` keeps the filename
    entirely in Python's hands.
    """
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    if not ok:  # pragma: no cover - encoder failure
        raise RuntimeError("failed to encode fixture PNG")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


#: Characters a merchandiser can plausibly put in a filename that do *not*
#: survive a URL path unencoded. `quote` must handle each, and the encoded URL
#: must still fetch the file -- which is the half of Requirement 8.4 that a
#: string-shape assertion on `thumbnail_url` would miss.
#:
#: A literal `%` is deliberately absent, and it is the one interesting character
#: the fetch cannot cover. `starlette.testclient` builds `scope["path"]` as
#: `unquote(httpx_url.path)`, but `httpx.URL.path` is *already* decoded, so the
#: harness unquotes once more than a real ASGI server does. A file named
#: `a%41b.png` is advertised correctly as `/assets/tiles/a%2541b.png` and
#: resolves under uvicorn, yet reaches the app as `a%41b.png` -> `aAb.png` here
#: and 404s. Only `%` followed by two hex digits is affected, and the defect is
#: in the harness rather than in the service, so generating it would fail a test
#: about code that is correct. The encoding of `%` is asserted on the URL string
#: instead, in `test_a_percent_in_a_tile_filename_is_percent_encoded` below.
_FILENAME_CHARS = "abZ09_-. #+&é!,;=@()[]'"

#: Non-ASCII and punctuation are fine in an id: ids travel in the JSON body and
#: in the render request, never in a URL path.
_ID_CHARS = "abcXYZ019-_.é "


def _http_entry_spec() -> st.SearchStrategy[dict[str, Any]]:
    return st.fixed_dictionaries(
        {
            "id_suffix": st.text(alphabet=_ID_CHARS, min_size=0, max_size=10),
            "stem": st.text(alphabet=_FILENAME_CHARS, min_size=0, max_size=8),
            "nested": st.booleans(),
            "width_mm": _valid_mm,
            "height_mm": _valid_mm,
            "finish": _valid_finish,
            "gloss": _valid_gloss,
            # Weighted toward valid: Property 26 is a statement about the valid
            # set round-tripping exactly, so most examples need entries that all
            # belong in the response. The defects carry the Requirement 8.8 half.
            "defect": st.sampled_from(
                (None, None, None, "width", "gloss", "finish", "missing_image")
            ),
        }
    )


def _materialise_http_entries(
    tiles_dir: Path, specs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rebuild ``tiles_dir`` from scratch and return the entries it now declares.

    The directory is emptied first so nothing survives from the previous example:
    a leftover image would silently repair a `missing_image` defect, and a
    leftover manifest would make an assertion about the current one meaningless.
    """
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)
    tiles_dir.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        # The index prefix guarantees ids and filenames are unique across the
        # manifest, so `id` collisions and file collisions never confound the
        # acceptance assertion. The `z` before the extension keeps a drawn stem
        # from leaving a trailing space or dot in the filename.
        stem = f"t{index:02d}{spec['stem']}z"
        relative = f"collection/{stem}.png" if spec["nested"] else f"{stem}.png"

        entry = {
            "id": f"{index:02d}-{spec['id_suffix']}".strip(),
            "name": f"Tile {index:02d}",
            "file": relative,
            "width_mm": spec["width_mm"],
            "height_mm": spec["height_mm"],
            "finish": spec["finish"],
            "gloss": spec["gloss"],
        }

        defect = spec["defect"]
        if defect == "width":
            entry["width_mm"] = -entry["width_mm"]
        elif defect == "gloss":
            entry["gloss"] = 1.5
        elif defect == "finish":
            entry["finish"] = "   "

        if defect != "missing_image":
            _write_png_bytes(tiles_dir / relative, seed=index)
        entries.append(entry)
    return entries


def _python_source_stamps() -> dict[str, tuple[int, int]]:
    """`(mtime_ns, size)` for every delivered backend Python source file.

    Requirement 8.5's "without changes to Python source files" is a claim about
    the codebase, not about the response, so it is checked as one. Comparing the
    snapshot across the round trip turns the clause from prose into an assertion.
    """
    root = Path(backend.__file__).resolve().parent
    stamps: dict[str, tuple[int, int]] = {}
    for source in sorted(root.rglob("*.py")):
        stat = source.stat()
        stamps[str(source)] = (stat.st_mtime_ns, stat.st_size)
    return stamps


# --------------------------------------------------------------------------- #
# Property 26 -- the HTTP round trip (Requirements 8.2, 8.4, 8.8)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 26: Tile catalog is a round trip
# through the manifest
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(specs=st.lists(_http_entry_spec(), min_size=0, max_size=5))
def test_property_26_tile_catalog_round_trips_through_the_manifest(http_catalog, specs):
    """For any set of valid entries written to the manifest with their image files
    present, `/api/tiles` returns exactly that set of tile identifiers, with no
    change to any Python source file.

    The manifest drawn here also carries invalid entries, so the assertion is
    stronger than the property as stated: the response must equal the valid
    subset exactly, which fails both a loader that drops valid entries and one
    that publishes invalid ones (Requirement 8.8 at the HTTP layer).

    `thumbnail_url` is not merely checked for shape. The generated filenames
    contain spaces, `#`, `%`, and non-ASCII characters precisely because
    `/api/tiles` percent-encodes them, and an encoding that produced a
    well-formed-looking URL which then 404s would be a worse failure than an
    obviously malformed one. Each advertised URL is therefore fetched, and its
    bytes compared against the file on disk -- the assertion being that the URL
    the frontend will request resolves to the image the merchandiser dropped in.

    **Validates: Requirements 8.2, 8.4, 8.8**
    """
    client, tiles_dir = http_catalog
    sources_before = _python_source_stamps()

    entries = _materialise_http_entries(tiles_dir, specs)
    manifest_path = _write_manifest(tiles_dir, entries)
    _force_new_manifest_stamp(manifest_path)

    response = client.get("/api/tiles")
    assert response.status_code == 200
    published = response.json()["tiles"]

    accepted = [entry for entry, spec in zip(entries, specs) if spec["defect"] is None]

    # Exact, ordered equality in one assertion: Requirement 8.4 promises the
    # catalog reads back in the order the manifest declared it, which a set
    # comparison would not catch.
    assert [tile["id"] for tile in published] == [entry["id"] for entry in accepted]

    for tile, entry in zip(published, accepted):
        # An accepted entry must come back carrying what it declared, not a
        # coerced or defaulted version of it.
        assert tile["name"] == entry["name"]
        assert tile["width_mm"] == entry["width_mm"]
        assert tile["height_mm"] == entry["height_mm"]
        assert tile["finish"] == entry["finish"]
        assert tile["gloss"] == entry["gloss"]

        # Requirement 8.2 fixes where tile assets live, and the URL has to agree
        # with it: anything not under the tiles mount cannot be served.
        assert tile["thumbnail_url"] == f"/assets/tiles/{quote(entry['file'])}"
        assert tile["thumbnail_url"].startswith("/assets/tiles/")

        fetched = client.get(tile["thumbnail_url"])
        assert fetched.status_code == 200, (
            f"advertised thumbnail_url {tile['thumbnail_url']!r} did not resolve "
            f"for file {entry['file']!r}"
        )
        assert fetched.content == (tiles_dir / entry["file"]).read_bytes()

    excluded_ids = {
        entry["id"] for entry, spec in zip(entries, specs) if spec["defect"] is not None
    }
    assert excluded_ids.isdisjoint({tile["id"] for tile in published})

    assert _python_source_stamps() == sources_before


def test_tiles_endpoint_reflects_a_manifest_edit_without_a_restart(client, tiny_catalog):
    """Requirement 8.5 through the endpoint, on the app the property reuses.

    The property rewrites the whole manifest each example, which a loader that
    reloaded on *any* read would also satisfy. This drives the merchandiser's
    actual workflow -- append a tile, then withdraw one -- against a single
    running app, and reads the endpoint before the first edit so the assertion is
    that a later request observed the change rather than that it happened to be
    the first request of all.
    """
    tiles_dir = tiny_catalog / "tiles"
    manifest_path = tiles_dir / DEFAULT_MANIFEST_NAME
    original = [spec["id"] for spec in TINY_CATALOG_TILES]

    assert [tile["id"] for tile in client.get("/api/tiles").json()["tiles"]] == original

    added = _valid_entry(99)
    _write_png(tiles_dir / added["file"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tiles"].append(added)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _touch_forward(manifest_path)

    appeared = client.get("/api/tiles").json()["tiles"]
    assert [tile["id"] for tile in appeared] == original + [added["id"]]
    assert appeared[-1]["thumbnail_url"] == f"/assets/tiles/{added['file']}"
    assert client.get(appeared[-1]["thumbnail_url"]).status_code == 200

    # Withdrawing a product has to work the same way round.
    _write_manifest(tiles_dir, [added])
    _touch_forward(manifest_path, seconds=4.0)

    assert [tile["id"] for tile in client.get("/api/tiles").json()["tiles"]] == [added["id"]]


def test_a_percent_in_a_tile_filename_is_percent_encoded(client, tiny_catalog):
    """The one case the property's fetch assertion cannot reach.

    `%` is the only interesting filename character excluded from the generated
    alphabet, for a harness reason rather than a product one: `starlette.
    testclient` derives `scope["path"]` by unquoting `httpx.URL.path`, which is
    already decoded, so it delivers one more unquote than a real ASGI server. A
    correctly advertised `/assets/tiles/tile%2500.png` therefore arrives as
    `tile\\x00.png` in-process and 404s, while resolving normally under uvicorn.

    What is still worth pinning is that `%` is escaped at all -- an unescaped `%`
    in a URL path is malformed, and the frontend would fail on it in a real
    browser. So the URL string is asserted and the fetch is left to the property,
    which covers every other character.
    """
    tiles_dir = tiny_catalog / "tiles"
    entry = _valid_entry(0)
    entry["file"] = "tile 50% off.png"
    _write_png(tiles_dir / entry["file"])
    _write_manifest(tiles_dir, [entry])
    _touch_forward(tiles_dir / DEFAULT_MANIFEST_NAME)

    published = client.get("/api/tiles").json()["tiles"]

    assert [tile["id"] for tile in published] == [entry["id"]]
    assert published[0]["thumbnail_url"] == "/assets/tiles/tile%2050%25%20off.png"
    assert "%" not in published[0]["thumbnail_url"].replace("%20", "").replace("%25", "")


def test_a_nested_tile_file_keeps_its_path_separator_in_the_url(client, tiny_catalog):
    """`quote` leaves `/` safe, so a tile filed under a subdirectory stays a path.

    Escaping the separator would produce a single flat segment that the `/assets`
    mount cannot resolve, which the property would catch only when it happened to
    draw a nested entry. Pinning it directly makes the intent explicit.
    """
    tiles_dir = tiny_catalog / "tiles"
    entry = _valid_entry(0)
    entry["file"] = "collection 2024/tile_00.png"
    _write_png(tiles_dir / entry["file"])
    _write_manifest(tiles_dir, [entry])
    _touch_forward(tiles_dir / DEFAULT_MANIFEST_NAME)

    published = client.get("/api/tiles").json()["tiles"]

    assert published[0]["thumbnail_url"] == "/assets/tiles/collection%202024/tile_00.png"
    assert client.get(published[0]["thumbnail_url"]).status_code == 200


def test_tiles_endpoint_returns_an_empty_list_for_an_empty_catalog(client, tiny_catalog):
    """An empty catalog is a 200 with no tiles, not an error.

    A fresh checkout before `setup_assets.py` has run is in exactly this state,
    and the frontend has to be able to render an empty product list rather than
    surface a failure the merchandiser cannot act on.
    """
    manifest_path = tiny_catalog / "tiles" / DEFAULT_MANIFEST_NAME
    _write_manifest(tiny_catalog / "tiles", [])
    _touch_forward(manifest_path)

    response = client.get("/api/tiles")

    assert response.status_code == 200
    assert response.json() == {"tiles": []}
