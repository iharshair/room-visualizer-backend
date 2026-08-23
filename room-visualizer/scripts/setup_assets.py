#!/usr/bin/env python3
"""Setup_Tool -- generate the starter Asset_Catalog and the sample room.

Run once after cloning::

    python scripts/setup_assets.py             # fill in anything missing
    python scripts/setup_assets.py --force     # regenerate everything

It produces three things under ``assets/``:

1. **Eight starter tiles** (Requirement 11.2) -- marble, wood, concrete, and
   terrazzo, each in a 600x600 and a 600x1200 millimetre format, written as
   ``assets/tiles/<finish>_<w>x<h>.png``. Every raster is synthesized by the
   Texture_Helper's procedural generators, passed through
   :func:`~backend.utils.texture_helper.make_seamless`, and then *measured*: a
   pattern whose wrapped edge continuity exceeds
   :data:`~backend.utils.texture_helper.SEAMLESS_TOLERANCE` aborts setup rather
   than shipping a visibly seamed asset (Requirement 8.1).
2. **The Asset_Catalog manifest** (Requirement 11.3), ``assets/tiles/manifest.json``,
   carrying a schema ``version`` plus each tile's ``id``, ``name``, ``file``,
   ``width_mm``, ``height_mm``, ``finish``, and ``gloss`` -- exactly the fields
   the Catalog_Loader validates.
3. **The sample room** (Requirement 11.4), ``assets/samples/synthetic_room.png``
   with its ``synthetic_room.truth.json`` sidecar, rendered through the *same*
   ``make_synthetic_room`` the Test_Suite uses, at the *same* pinned seed
   (:data:`SAMPLE_ROOM_SEED`), so the shipped sample and the anchored
   ``synthetic_room`` fixture are pixel-for-pixel the same scene and cannot
   drift apart.

Three properties of this script are load-bearing rather than incidental:

**It is offline and imagery-free** (Requirement 11.5). The only imports are the
standard library, ``numpy``, ``opencv-python``, ``backend/utils/texture_helper.py``,
and ``tests/fixtures/synthetic.py``. Nothing here opens a socket and no
third-party picture is embedded or fetched -- every pixel is computed.

**Re-running it is safe.** Without ``--force`` an existing file is left exactly
as it is, and its generation is skipped entirely. That is what lets a
merchandiser drop their own product images and manifest entries into
``assets/tiles/`` and still re-run setup without losing them. ``--force`` is the
explicit opt-in to overwrite.

**Seeding is split.** The eight tile rasters are a deterministic function of
``--seed`` (default :data:`DEFAULT_SEED`), so the same seed reproduces
byte-identical tiles on any platform. The sample room is *not* reseedable: it is
pinned to :data:`SAMPLE_ROOM_SEED`, which is what makes the "cannot drift apart"
claim above hold for any invocation rather than only the default one. Reseeding
the sample would both break that anchor and risk an occluder pose the
Classical_Backend cannot recover; see :data:`SAMPLE_ROOM_SEED` for the specific
failure that motivated pinning it.

Requirements: 11.2, 11.3, 11.4, 11.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

# Running this file directly puts ``scripts/`` on ``sys.path``, not the project
# root, so ``backend`` and ``tests`` would not import. Prepending the project
# root keeps ``python scripts/setup_assets.py`` working from any directory
# without requiring an installed package or a PYTHONPATH incantation.
_PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from backend.utils.texture_helper import (  # noqa: E402  (path setup must precede)
    SEAMLESS_TOLERANCE,
    edge_continuity,
    generate_concrete,
    generate_marble,
    generate_terrazzo,
    generate_wood_plank,
    make_seamless,
)
from tests.fixtures.synthetic import make_synthetic_room, write_room  # noqa: E402

__all__ = [
    "FinishSpec",
    "TileSpec",
    "SetupError",
    "FINISHES",
    "FORMATS_MM",
    "MANIFEST_VERSION",
    "DEFAULT_SEED",
    "SAMPLE_ROOM_SEED",
    "TILE_SHORT_EDGE_PX",
    "tile_specs",
    "build_tile",
    "manifest_document",
    "generate_tiles",
    "write_manifest",
    "generate_sample_room",
    "run_setup",
    "main",
]

#: Manifest schema version, matching ``backend.catalog.MANIFEST_VERSION``.
MANIFEST_VERSION = 1

#: Default ``--seed``, governing the *tile rasters only*. Any integer works;
#: this one is simply what the README documents so a fresh clone and the docs
#: agree. It deliberately does not reach the sample room -- see
#: :data:`SAMPLE_ROOM_SEED`.
DEFAULT_SEED = 7

#: Seed for the sample room, pinned rather than configurable.
#:
#: This is ``make_synthetic_room``'s own default, which is the pose the
#: Test_Suite's ``synthetic_room`` fixture is anchored to, so the shipped sample
#: *is* the scene every geometry, lighting, and compositing assertion is written
#: against. ``--seed`` cannot move it: an operator who reseeds the tile rasters
#: would otherwise silently ship a sample room the suite has never checked, and
#: the seeds that put the fixture's two occluders in an awkward pose are not
#: hypothetical. At seed 7 they cover a quarter of the frame and run off the left
#: edge, outside the envelope the Classical_Backend documents for its occluder
#: heuristics, so foreground recall on the shipped sample collapses to zero and
#: the demo composite tiles straight over the furniture. At this seed recall is
#: about 0.92 and the demo shows the occlusion behaviour Requirement 7.6
#: promises.
SAMPLE_ROOM_SEED = 0

#: Authoring resolution for the *short* metric edge of each tile, in pixels. The
#: long edge follows from the declared millimetre ratio, so a 600x1200 plank is
#: authored at 512x1024 and never at a square resolution that would have to be
#: stretched later. 512 keeps a 600 mm tile at roughly 0.85 px/mm -- fine enough
#: that the Compositor's ``INTER_LINEAR`` sampling has detail to work with, and
#: inside the Texture_Helper's 1024 px sampling budget without a rescale.
TILE_SHORT_EDGE_PX = 512

#: The two formats every finish ships in, as ``(width_mm, height_mm)``
#: (Requirement 11.2).
FORMATS_MM: tuple[tuple[int, int], ...] = ((600, 600), (600, 1200))


class SetupError(RuntimeError):
    """A generated asset failed its quality gate, or an output could not be written.

    Raised rather than warned: shipping a seamed tile or a half-written manifest
    would surface as a rendering defect much later and much less legibly than a
    failed setup run.
    """


@dataclass(frozen=True, slots=True)
class FinishSpec:
    """One tile finish: how to synthesize it and how to describe it.

    ``key`` is the filename and identifier stem (``marble``), while ``finish`` is
    the shopper-facing surface label the manifest carries (``polished``) and the
    Compositor pairs with ``gloss``. They are separate because the material and
    its surface treatment are separate facts -- honed marble and polished marble
    are the same stone with very different specular behaviour.
    """

    key: str
    finish: str
    display_name: str
    gloss: float
    generator: Callable[[tuple[int, int], int], np.ndarray]


#: The four starter finishes (Requirement 11.2) with the gloss values the design
#: fixes: polished marble is near-mirror, matte concrete is almost purely
#: diffuse, satin wood and honed terrazzo sit between them (Requirement 11.3).
FINISHES: tuple[FinishSpec, ...] = (
    FinishSpec("marble", "polished", "Polished Marble", 0.85, generate_marble),
    FinishSpec("wood", "satin", "Satin Wood Plank", 0.35, generate_wood_plank),
    FinishSpec("concrete", "matte", "Matte Concrete", 0.10, generate_concrete),
    FinishSpec("terrazzo", "honed", "Honed Terrazzo", 0.45, generate_terrazzo),
)


@dataclass(frozen=True, slots=True)
class TileSpec:
    """One concrete tile: a finish in one metric format, with its own seed."""

    tile_id: str
    name: str
    file_name: str
    width_mm: int
    height_mm: int
    finish: str
    gloss: float
    size_px: tuple[int, int]  # (width, height), matching the generator argument
    seed: int
    generator: Callable[[tuple[int, int], int], np.ndarray]

    def manifest_entry(self) -> dict[str, object]:
        """The manifest object for this tile (Requirement 11.3).

        Field names and types are exactly what ``CatalogLoader._validate_entry``
        requires, so a manifest written here validates without exclusions.
        """
        return {
            "id": self.tile_id,
            "name": self.name,
            "file": self.file_name,
            "width_mm": float(self.width_mm),
            "height_mm": float(self.height_mm),
            "finish": self.finish,
            "gloss": self.gloss,
        }


def _pixel_size(width_mm: int, height_mm: int, short_edge_px: int) -> tuple[int, int]:
    """Authoring pixel size honouring the declared millimetre ratio.

    Derived from a single pixels-per-millimetre scale pinned to the *short* metric
    edge, so both formats of a finish are authored at the same physical detail
    density and neither is anisotropically stretched. Returned width-first, which
    is the ``size_px`` convention the generators and ``cv2`` share.
    """
    scale = short_edge_px / float(min(width_mm, height_mm))
    return (max(2, round(width_mm * scale)), max(2, round(height_mm * scale)))


def tile_specs(
    seed: int = DEFAULT_SEED,
    *,
    finishes: Sequence[FinishSpec] = FINISHES,
    formats_mm: Sequence[tuple[int, int]] = FORMATS_MM,
    short_edge_px: int = TILE_SHORT_EDGE_PX,
) -> tuple[TileSpec, ...]:
    """Enumerate the eight starter tiles for a given base ``seed``.

    Each tile gets its own derived seed, so the eight rasters differ from one
    another while the whole set stays a pure function of ``seed``. The derivation
    is positional arithmetic rather than a hash of the id, because Python's
    string hashing is salted per process and would make the assets
    irreproducible across runs.
    """
    specs: list[TileSpec] = []
    for index, finish in enumerate(finishes):
        for format_index, (width_mm, height_mm) in enumerate(formats_mm):
            format_label = f"{width_mm}x{height_mm}"
            specs.append(
                TileSpec(
                    tile_id=f"{finish.key}-{finish.finish}-{format_label}",
                    name=f"{finish.display_name} {format_label}",
                    file_name=f"{finish.key}_{format_label}.png",
                    width_mm=width_mm,
                    height_mm=height_mm,
                    finish=finish.finish,
                    gloss=finish.gloss,
                    size_px=_pixel_size(width_mm, height_mm, short_edge_px),
                    seed=int(seed) + 1_000 * index + 17 * format_index,
                    generator=finish.generator,
                )
            )
    return tuple(specs)


def build_tile(spec: TileSpec) -> np.ndarray:
    """Synthesize one tile raster and prove it tiles without a seam.

    ``generate_x`` then :func:`make_seamless` is the composition the
    Texture_Helper documents; :func:`edge_continuity` is the measurable form of
    Requirement 8.1 and the gate is the design's ``<= 0.02``. Measuring here --
    at generation, on the exact bytes that get written -- is what stops a
    regression in a generator from reaching a shopper's screen as a visible grid
    of repeat lines.

    Raises:
        SetupError: the synthesized pattern is not seamless within tolerance.
    """
    pattern = make_seamless(spec.generator(spec.size_px, spec.seed))
    continuity = edge_continuity(pattern)
    if continuity > SEAMLESS_TOLERANCE:
        raise SetupError(
            f"{spec.file_name}: wrapped edge continuity {continuity:.4f} exceeds the "
            f"seamless tolerance {SEAMLESS_TOLERANCE:.2f}; the generator for finish "
            f"{spec.finish!r} regressed"
        )
    return pattern


def _write_png(path: Path, image: np.ndarray, *, force: bool) -> bool:
    """Write ``image`` unless the file exists and ``force`` is false.

    Returns True when the file was written, False when it was left untouched.
    """
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise SetupError(f"failed to write {path}")
    return True


def generate_tiles(
    tiles_dir: Path,
    specs: Iterable[TileSpec],
    *,
    force: bool,
    log: Callable[[str], None],
) -> tuple[list[Path], list[Path]]:
    """Write every tile raster that is missing (or all of them under ``force``).

    Generation is skipped outright for a tile whose file already exists and is
    not being forced -- both to keep a re-run fast and, more importantly, so a
    merchandiser's own image at that filename is never even recomputed, let alone
    overwritten.

    Returns:
        ``(written, skipped)`` paths.
    """
    tiles_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    skipped: list[Path] = []

    for spec in specs:
        path = tiles_dir / spec.file_name
        if path.exists() and not force:
            skipped.append(path)
            log(f"  kept     {path.name} (exists; use --force to regenerate)")
            continue
        pattern = build_tile(spec)
        _write_png(path, pattern, force=True)
        written.append(path)
        log(
            f"  wrote    {path.name} "
            f"({pattern.shape[1]}x{pattern.shape[0]} px, "
            f"{spec.width_mm}x{spec.height_mm} mm, {spec.finish}, gloss {spec.gloss:.2f})"
        )

    return written, skipped


def manifest_document(specs: Iterable[TileSpec]) -> dict[str, object]:
    """The full Asset_Catalog manifest document (Requirement 11.3)."""
    return {
        "version": MANIFEST_VERSION,
        "tiles": [spec.manifest_entry() for spec in specs],
    }


def write_manifest(
    tiles_dir: Path,
    specs: Iterable[TileSpec],
    *,
    force: bool,
    log: Callable[[str], None],
) -> Path | None:
    """Write ``manifest.json``, or leave an existing one alone without ``force``.

    An existing manifest is the merchandiser's file: they may have appended their
    own products to it, and rewriting it would silently delete that work. So
    without ``--force`` it is preserved untouched, exactly like the tile images.

    Returns:
        The manifest path when written, ``None`` when it was left untouched.
    """
    manifest_path = tiles_dir / "manifest.json"
    if manifest_path.exists() and not force:
        log(f"  kept     {manifest_path.name} (exists; use --force to regenerate)")
        return None

    tiles_dir.mkdir(parents=True, exist_ok=True)
    document = manifest_document(specs)
    manifest_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    log(f"  wrote    {manifest_path.name} ({len(document['tiles'])} tiles)")  # type: ignore[arg-type]
    return manifest_path


def generate_sample_room(
    samples_dir: Path,
    *,
    seed: int = SAMPLE_ROOM_SEED,
    force: bool,
    log: Callable[[str], None],
    stem: str = "synthetic_room",
) -> dict[str, Path]:
    """Render the sample perspective room and its ground-truth sidecar (R11.4).

    Delegates to ``tests.fixtures.synthetic.make_synthetic_room`` at *its* own
    defaults, the generator and the pose the Test_Suite's ``synthetic_room``
    fixture uses, so the shipped sample is pixel-for-pixel the scene the geometry
    assertions are written against. That pose yields a floor plus all three walls
    -- comfortably the "at least two walls" Requirement 11.4 asks for -- with two
    solid boxes standing on the floor as occluders.

    ``seed`` is a parameter so a caller can render an alternative pose
    deliberately, but :func:`run_setup` never varies it: ``--seed`` reseeds the
    tile rasters and leaves the sample anchored. See :data:`SAMPLE_ROOM_SEED`.

    Rendering is skipped when every output already exists and ``force`` is false.

    Returns:
        The written paths keyed ``image``, ``truth``, and ``occluders``; empty
        when nothing needed rendering.
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "image": samples_dir / f"{stem}.png",
        "truth": samples_dir / f"{stem}.truth.json",
        "occluders": samples_dir / f"{stem}.occluders.png",
    }
    if not force and all(path.exists() for path in expected.values()):
        for path in expected.values():
            log(f"  kept     {path.name} (exists; use --force to regenerate)")
        return {}

    room = make_synthetic_room(seed=seed)
    paths = write_room(room, samples_dir, stem=stem, force=force)
    height, width = room.shape
    planes = ", ".join(room.plane_names())
    log(f"  wrote    {paths['image'].name} ({width}x{height} px, planes: {planes})")
    log(f"  wrote    {paths['truth'].name} (camera, VPs, horizon, homographies)")
    log(f"  wrote    {paths['occluders'].name} (foreground occluder mask)")
    return paths


def run_setup(
    out_dir: Path,
    *,
    seed: int = DEFAULT_SEED,
    force: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, object]:
    """Generate every starter asset under ``out_dir``.

    Args:
        out_dir: assets root. ``tiles/`` and ``samples/`` are created inside it.
        seed: base seed for the tile rasters; the eight of them are a
            deterministic function of it. The sample room ignores it and stays
            pinned to :data:`SAMPLE_ROOM_SEED`.
        force: overwrite existing files instead of preserving them.
        log: line sink, so a caller can silence or capture the progress report.

    Returns:
        A summary mapping with the resolved directories and the tile paths
        written and kept, the manifest path, and the sample room paths.

    Raises:
        SetupError: a generated tile failed its seamlessness gate, or an output
            could not be written.
    """
    assets_dir = Path(out_dir)
    tiles_dir = assets_dir / "tiles"
    samples_dir = assets_dir / "samples"
    specs = tile_specs(seed)

    log(
        f"Asset setup -> {assets_dir}  (tile seed {seed}, "
        f"sample room seed {SAMPLE_ROOM_SEED} (pinned)"
        f"{', force' if force else ''})"
    )

    log(f"Tiles ({len(specs)}):")
    written, skipped = generate_tiles(tiles_dir, specs, force=force, log=log)

    log("Manifest:")
    manifest_path = write_manifest(tiles_dir, specs, force=force, log=log)

    log("Sample room:")
    # Deliberately not ``seed=seed``: the sample room is the Test_Suite's anchor
    # and must not move when the tile rasters are reseeded (SAMPLE_ROOM_SEED).
    sample_paths = generate_sample_room(samples_dir, force=force, log=log)

    log(
        f"Done. {len(written)} tile image(s) written, {len(skipped)} kept; "
        f"manifest {'written' if manifest_path is not None else 'kept'}; "
        f"sample room {'written' if sample_paths else 'kept'}."
    )

    return {
        "assets_dir": assets_dir,
        "tiles_dir": tiles_dir,
        "samples_dir": samples_dir,
        "tiles_written": written,
        "tiles_kept": skipped,
        "manifest": manifest_path,
        "sample_room": sample_paths,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="setup_assets.py",
        description=(
            "Generate the starter tile catalog and the sample perspective room. "
            "Fully offline and procedural: no network access and no third-party imagery."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Existing files are preserved unless --force is given, so tile images "
            "and manifest entries added by hand survive a re-run."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "assets root to write into; 'tiles/' and 'samples/' are created inside it "
            f"(default: {_PROJECT_DIR / 'assets'})"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            f"base seed for the generated tile rasters (default: {DEFAULT_SEED}). "
            f"The sample room is not affected: it stays pinned to seed "
            f"{SAMPLE_ROOM_SEED}, the pose the test suite is anchored to"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing assets instead of leaving them untouched",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the progress report; errors are still reported",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point. Returns a process exit status."""
    args = _build_parser().parse_args(argv)
    out_dir = args.out if args.out is not None else _PROJECT_DIR / "assets"
    log: Callable[[str], None] = (lambda _message: None) if args.quiet else print

    try:
        run_setup(out_dir, seed=args.seed, force=args.force, log=log)
    except (SetupError, OSError, ValueError) as exc:
        print(f"setup_assets: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
