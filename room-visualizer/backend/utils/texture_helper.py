"""Texture_Helper -- seamless tile synthesis and metric scaling.

Two jobs live here, both consumed by the Catalog_Loader and the Setup_Tool:

* **Seamless synthesis.** :func:`make_seamless` turns an arbitrary tile
  photograph into a pattern whose wrapped opposite edges match, so the
  Compositor can tile it with ``cv2.BORDER_WRAP`` and show no seam
  (Requirement 8.1). :func:`edge_continuity` is the measurable form of that
  requirement; the design target is ``<= 0.02``.
* **Metric scaling.** :func:`to_metric_texture` resamples a pattern so its pixel
  aspect ratio matches the tile's declared millimetre aspect ratio and records
  ``px_per_mm``, which is what stops a 600x1200 plank from being rendered as a
  square (Requirements 8.6, 8.7). The Compositor's metric-to-texture conversion
  is then a single multiply.

* **Procedural synthesis.** :func:`generate_marble`, :func:`generate_wood_plank`,
  :func:`generate_concrete`, and :func:`generate_terrazzo` build the starter
  tile finishes the Setup_Tool ships. Every one is seeded and deterministic and
  draws only on ``numpy``/``cv2``, so ``setup_assets.py`` needs no network and no
  third-party imagery (Requirements 11.2, 11.5). Each returns a raw ``(h,w,3)``
  ``uint8`` BGR pattern; the Setup_Tool composes them as
  ``to_metric_texture(make_seamless(generate_x(...)), w_mm, h_mm)``, which is
  what makes the shipped assets satisfy Requirement 8.1 by construction.

Requirements: 8.1, 8.6, 8.7, 11.2, 11.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = [
    "SeamlessTexture",
    "make_seamless",
    "edge_continuity",
    "to_metric_texture",
    "generate_marble",
    "generate_wood_plank",
    "generate_concrete",
    "generate_terrazzo",
    "SEAMLESS_TOLERANCE",
    "METRIC_RATIO_TOLERANCE",
    "MIN_TEXTURE_EDGE",
    "MAX_TEXTURE_EDGE",
    "MAX_GENERATED_EDGE",
]

#: Requirement 8.1: mean absolute wrapped-edge luminance difference, as a
#: fraction of the full 8-bit luminance range, must not exceed this.
SEAMLESS_TOLERANCE = 0.02

#: Requirements 8.6/8.7: the resampled pixel ratio must match the declared
#: millimetre ratio to within 0.1 percent.
METRIC_RATIO_TOLERANCE = 0.001

#: Resolution budget for :func:`to_metric_texture`. The long edge follows the
#: source pattern -- neither inventing detail nor discarding it -- clamped into
#: this band so a stray 8 px thumbnail or a 6000 px product shot both yield a
#: texture the Compositor can sample cheaply.
MIN_TEXTURE_EDGE = 16
MAX_TEXTURE_EDGE = 1024

#: Sanity ceiling on a requested generator edge. The generators are asked for
#: authoring resolutions, which may exceed the Compositor's sampling budget
#: (:data:`MAX_TEXTURE_EDGE`) before :func:`to_metric_texture` scales them down,
#: but a request past this is a caller mistake rather than a tile.
MAX_GENERATED_EDGE = 4096

#: How far either side of the target long edge :func:`to_metric_texture` will
#: look for a pixel pair that honours the declared ratio.
_RATIO_SEARCH_SPAN = 512

#: Narrow band used to repair wrapped edges after resampling. Resize kernels
#: clamp at the border instead of wrapping, which can reopen a seam the source
#: pattern had closed; the repair is a border-scale artifact, so the correcting
#: cross-fade stays narrow and leaves the tile's interior alone.
_CORRECTION_BLEND_FRAC = 0.02

#: Repair fires at half the published tolerance, so a resampled pattern that
#: merely drifts close to the limit is still brought back with headroom.
_CORRECTION_TRIGGER = SEAMLESS_TOLERANCE / 2.0


@dataclass(frozen=True, slots=True)
class SeamlessTexture:
    """A wrap-continuous pattern plus the metric scale it was built at."""

    pattern: np.ndarray  # (h,w,3) uint8, opposite edges continuous
    width_mm: float
    height_mm: float
    px_per_mm: float

    @property
    def width_px(self) -> int:
        """Pattern width in pixels -- ``round(width_mm * px_per_mm)``."""
        return int(self.pattern.shape[1])

    @property
    def height_px(self) -> int:
        """Pattern height in pixels -- ``round(height_mm * px_per_mm)``."""
        return int(self.pattern.shape[0])


# --------------------------------------------------------------------------- #
# Seamless synthesis
# --------------------------------------------------------------------------- #


def _validate_image(image: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a numpy array, got {type(image)!r}")
    if image.ndim not in (2, 3):
        raise ValueError(f"{name} must be (h,w) or (h,w,c), got shape {image.shape}")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise ValueError(f"{name} must have 1, 3, or 4 channels, got {image.shape[2]}")
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError(f"{name} must be at least 2x2, got shape {image.shape}")
    return image


def _crossfade_alpha(length: int, seam: int, blend_frac: float) -> np.ndarray:
    """Linear cross-fade ramp along one axis, centred on ``seam``.

    ``seam`` is the index of the first sample *after* the discontinuity, so the
    seam itself sits between ``seam - 1`` and ``seam``. The ramp peaks at 0.5 on
    both of those samples and falls linearly to 0 at half the band width either
    side. Peaking at exactly 0.5 is what makes the blend a true cross-fade: the
    two samples straddling the seam both become the same 50/50 mix of the two
    sides, so they are bitwise equal and the wrapped edge is continuous by
    construction rather than merely by attenuation.
    """
    half_band = max(1.0, blend_frac * length / 2.0)
    index = np.arange(length, dtype=np.float32)
    # Distance from the seam line, measured so both straddling samples read 0.
    distance = np.abs(index - seam + 0.5) - 0.5
    return 0.5 * np.clip(1.0 - distance / half_band, 0.0, 1.0)


def _blend_across_seam(
    source: np.ndarray, mirrored: np.ndarray, alpha: np.ndarray, axis: int
) -> np.ndarray:
    """Cross-fade ``source`` towards ``mirrored`` with a per-row/column alpha."""
    shape = [1] * source.ndim
    shape[axis] = alpha.size
    weight = alpha.reshape(shape)
    return source * (1.0 - weight) + mirrored * weight


def make_seamless(image_bgr: np.ndarray, blend_frac: float = 0.15) -> np.ndarray:
    """Return a wrap-continuous version of ``image_bgr`` (Requirement 8.1).

    Mirror-offset blending, chosen because it needs no frequency-domain
    assumptions and so works on arbitrary product photography:

    1. Offset the image by half its width and half its height with wraparound,
       which moves the original outer edges into the middle as two seams.
    2. Build a linear cross-fade ramp of width ``blend_frac * dimension``
       centred on each seam.
    3. Blend the offset image against its horizontally mirrored copy across the
       vertical seam, then the result against its vertically mirrored copy
       across the horizontal seam. Each copy is mirrored about its seam, so it
       supplies statistically matching content on both sides and the blend
       reads as texture rather than smear.
    4. Roll back by the same offset, which returns the cross-faded seams to the
       pattern's outer edges.

    The horizontal pass leaves the two columns straddling the vertical seam
    bitwise equal, and the vertical pass mixes rows only -- so it preserves that
    equality while establishing the same for rows. Both wrapped edges therefore
    match exactly and :func:`edge_continuity` reports ~0.

    Args:
        image_bgr: ``(h,w)`` or ``(h,w,c)`` image, at least 2x2. ``uint8`` input
            returns ``uint8``; any other dtype returns ``float32``.
        blend_frac: Cross-fade band width as a fraction of each dimension.

    Returns:
        A pattern of the same shape as the input whose opposite edges wrap
        continuously.
    """
    image = _validate_image(image_bgr, "image_bgr")
    if not 0.0 < blend_frac <= 1.0:
        raise ValueError(f"blend_frac must be in (0, 1], got {blend_frac}")

    height, width = image.shape[:2]
    offset_y, offset_x = height // 2, width // 2

    work = np.asarray(image, dtype=np.float32)
    # Step 1: the original edges now meet at columns offset_x-1|offset_x and
    # rows offset_y-1|offset_y.
    rolled = np.roll(np.roll(work, offset_x, axis=1), offset_y, axis=0)

    # Step 3a: mirror about the vertical seam, then cross-fade across it. The
    # index form is parity-agnostic; a plain ``[:, ::-1]`` only lands on the
    # seam for even widths.
    columns = np.arange(width)
    mirror_columns = (2 * offset_x - 1 - columns) % width
    blended = _blend_across_seam(
        rolled,
        rolled[:, mirror_columns],
        _crossfade_alpha(width, offset_x, blend_frac),
        axis=1,
    )

    # Step 3b: same across the horizontal seam.
    rows = np.arange(height)
    mirror_rows = (2 * offset_y - 1 - rows) % height
    blended = _blend_across_seam(
        blended,
        blended[mirror_rows, :],
        _crossfade_alpha(height, offset_y, blend_frac),
        axis=0,
    )

    # Step 4.
    pattern = np.roll(np.roll(blended, -offset_y, axis=0), -offset_x, axis=1)

    if image.dtype == np.uint8:
        return np.clip(np.rint(pattern), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(pattern)


def edge_continuity(pattern: np.ndarray) -> float:
    """Mean wrapped-edge luminance mismatch as a fraction of full range.

    The measurable form of Requirement 8.1: convert to luminance, average the
    mean absolute difference between the first and last columns with that
    between the first and last rows, and divide by 255. Lower is better; the
    design target is ``<= 0.02`` (:data:`SEAMLESS_TOLERANCE`).
    """
    image = _validate_image(pattern, "pattern")

    array = image if image.dtype == np.uint8 else np.clip(np.rint(image), 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[2] == 3:
        luminance = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 3 and array.shape[2] == 4:
        luminance = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    elif array.ndim == 3:
        luminance = array[:, :, 0]
    else:
        luminance = array

    luminance = luminance.astype(np.float32)
    column_gap = float(np.mean(np.abs(luminance[:, 0] - luminance[:, -1])))
    row_gap = float(np.mean(np.abs(luminance[0, :] - luminance[-1, :])))
    return (column_gap + row_gap) / 2.0 / 255.0


# --------------------------------------------------------------------------- #
# Metric scaling
# --------------------------------------------------------------------------- #


def _px_per_mm(width_px: int, height_px: int, width_mm: float, height_mm: float) -> float:
    """The single scale both axes share, averaged over the two measurements."""
    return (width_px / width_mm + height_px / height_mm) / 2.0


def _resolve_pixel_dimensions(
    width_mm: float, height_mm: float, target_long_px: int
) -> tuple[int, int, float]:
    """Pick ``(width_px, height_px, px_per_mm)`` honouring the declared ratio.

    Rounding a fixed long edge to an integer short edge can distort the ratio by
    far more than the 0.1 percent Requirement 8.6 allows, so the long edge is
    not treated as fixed. Candidates are scanned outward from ``target_long_px``
    and the first is accepted whose pixel ratio matches the millimetre ratio
    within :data:`METRIC_RATIO_TOLERANCE` *and* whose single ``px_per_mm``
    reproduces both pixel dimensions exactly -- so the Compositor can convert in
    either direction with one multiply and no drift.
    """
    ratio = width_mm / height_mm
    width_is_long = width_mm >= height_mm

    for delta in range(_RATIO_SEARCH_SPAN + 1):
        for long_px in {target_long_px + delta, target_long_px - delta}:
            if long_px < 2:
                continue
            if width_is_long:
                width_px, height_px = long_px, max(1, round(long_px / ratio))
            else:
                height_px, width_px = long_px, max(1, round(long_px * ratio))

            scale = _px_per_mm(width_px, height_px, width_mm, height_mm)
            error = abs((width_px / height_px) / ratio - 1.0)
            if (
                error <= METRIC_RATIO_TOLERANCE
                and round(width_mm * scale) == width_px
                and round(height_mm * scale) == height_px
            ):
                return width_px, height_px, scale

    raise ValueError(
        f"cannot represent a {width_mm}x{height_mm} mm tile near {target_long_px} px "
        f"within {METRIC_RATIO_TOLERANCE:.1%} of its declared aspect ratio"
    )


def to_metric_texture(
    image_bgr: np.ndarray,
    width_mm: float,
    height_mm: float,
    *,
    max_edge_px: int = MAX_TEXTURE_EDGE,
    ensure_seamless: bool = True,
) -> SeamlessTexture:
    """Resample a pattern to its declared metric aspect ratio.

    Requirements 8.6 and 8.7: the output pixel ratio matches
    ``width_mm / height_mm`` to within 0.1 percent, and ``px_per_mm`` reproduces
    both pixel dimensions from the millimetre dimensions, so no isotropic or
    anisotropic stretch can creep into the render. This is the single place a
    600x1200 plank is prevented from being squashed to a square.

    Scaling is the only content change: pass a :func:`make_seamless` result, so
    the composition order is ``to_metric_texture(make_seamless(img), w, h)``.
    Because resize kernels clamp at the border rather than wrapping, a strong
    anisotropic rescale can reopen a seam the source had closed; when that
    happens the wrapped edges are repaired with a narrow corrective cross-fade
    so the returned pattern always satisfies :data:`SEAMLESS_TOLERANCE`.

    Args:
        image_bgr: Source pattern, ``(h,w)`` or ``(h,w,c)``, at least 2x2.
        width_mm: Declared real-world tile width in millimetres, positive.
        height_mm: Declared real-world tile height in millimetres, positive.
        max_edge_px: Upper bound on the output long edge.
        ensure_seamless: Repair wrapped edges if resampling perturbed them past
            :data:`SEAMLESS_TOLERANCE`. Set ``False`` for a pure resample.

    Returns:
        A :class:`SeamlessTexture` carrying the resampled pattern, the declared
        millimetre dimensions, and the shared pixel-per-millimetre scale.
    """
    image = _validate_image(image_bgr, "image_bgr")
    if not np.isfinite(width_mm) or width_mm <= 0.0:
        raise ValueError(f"width_mm must be a positive finite number, got {width_mm}")
    if not np.isfinite(height_mm) or height_mm <= 0.0:
        raise ValueError(f"height_mm must be a positive finite number, got {height_mm}")
    if max_edge_px < MIN_TEXTURE_EDGE:
        raise ValueError(f"max_edge_px must be at least {MIN_TEXTURE_EDGE}, got {max_edge_px}")

    source_long = max(image.shape[0], image.shape[1])
    target_long = int(min(max(source_long, MIN_TEXTURE_EDGE), max_edge_px))
    width_px, height_px, px_per_mm = _resolve_pixel_dimensions(
        float(width_mm), float(height_mm), target_long
    )

    if (width_px, height_px) == (image.shape[1], image.shape[0]):
        pattern = np.ascontiguousarray(image)
    else:
        shrinking = width_px * height_px < image.shape[1] * image.shape[0]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
        pattern = cv2.resize(image, (width_px, height_px), interpolation=interpolation)
        if ensure_seamless and edge_continuity(pattern) > _CORRECTION_TRIGGER:
            pattern = make_seamless(pattern, blend_frac=_CORRECTION_BLEND_FRAC)

    return SeamlessTexture(
        pattern=pattern,
        width_mm=float(width_mm),
        height_mm=float(height_mm),
        px_per_mm=float(px_per_mm),
    )

# --------------------------------------------------------------------------- #
# Procedural generators -- shared noise machinery
# --------------------------------------------------------------------------- #
#
# Requirements 11.2 and 11.5: the Setup_Tool ships marble, wood, concrete, and
# terrazzo starter tiles that it synthesizes itself, so a clone needs no network
# and carries no third-party imagery. Everything below is driven by a single
# ``numpy.random.Generator`` seeded from the caller's ``seed``, and PCG64 plus
# integer-sized ``cv2`` kernels are reproducible across platforms, so a given
# seed yields byte-identical output anywhere.
#
# The noise primitives keep their features periodic where it is cheap to do so
# (wrap-padded upsampling grids, wrap-bordered filters, chips drawn across the
# border). That is a quality choice, not the seamlessness guarantee: callers
# still pass the result through :func:`make_seamless`, which is what Requirement
# 8.1 is measured against. Periodic primitives just mean that pass has almost
# nothing left to hide.

#: Colour anchors, all BGR because that is what ``cv2`` writes and what the
#: Compositor samples. Named in RGB terms in the comments for readability.
_MARBLE_BASE_BGR = (238.0, 241.0, 246.0)  # warm white
_MARBLE_VEIN_BGR = (150.0, 154.0, 163.0)  # grey-taupe vein
_WOOD_LIGHT_BGR = (122.0, 162.0, 198.0)  # light oak
_WOOD_DARK_BGR = (52.0, 84.0, 122.0)  # dark grain
_CONCRETE_BASE_BGR = (172.0, 171.0, 168.0)  # neutral-cool grey
_TERRAZZO_MATRIX_BGR = (233.0, 234.0, 232.0)  # light cement matrix

#: Aggregate palette for :func:`generate_terrazzo`, BGR.
_TERRAZZO_PALETTE = (
    (64.0, 62.0, 60.0),  # charcoal
    (72.0, 96.0, 178.0),  # terracotta
    (84.0, 168.0, 214.0),  # ochre
    (124.0, 150.0, 126.0),  # sage
    (150.0, 118.0, 96.0),  # slate blue
    (238.0, 244.0, 246.0),  # off-white
    (150.0, 150.0, 206.0),  # rose
)

#: Marble: octaves of the turbulence field, cells across the short edge at
#: octave 0, and warp displacement as a fraction of the short edge.
_MARBLE_OCTAVES = 6
_MARBLE_BASE_CELLS = 3.0
_MARBLE_WARP_FRAC = 0.18

#: Marble veining. Veins are the zero crossings of a directional phase ramp that
#: turbulence displaces by up to ``_MARBLE_VEIN_DISTORTION`` cycles; the ramp is
#: what makes them long and roughly parallel instead of closed loops. Wave counts
#: are integers per axis so the ramp is periodic and the veins survive tiling.
_MARBLE_WAVES_X = (1, 3)
_MARBLE_WAVES_Y = (2, 5)
_MARBLE_VEIN_DISTORTION = 0.9
_MARBLE_VEIN_WIDTH = 0.24
_MARBLE_VEIN_GAMMA = 1.7
_MARBLE_VEIN_STRENGTH = 0.9
_MARBLE_HAIRLINE_MULTIPLE = 3
_MARBLE_HAIRLINE_WIDTH = 0.09
_MARBLE_HAIRLINE_STRENGTH = 0.38
#: Soft grey shoulder either side of a vein -- a wider, much weaker ridge off the
#: same phase field, so stone reads cloudy rather than as ink on paper.
_MARBLE_SHOULDER_MULTIPLE = 4.0
_MARBLE_SHOULDER_STRENGTH = 0.16
_MARBLE_GRAIN_SIGMA = 1.4

#: Wood: plank length-to-width ratio the plank count is derived from, grain and
#: ring frequencies, and the per-plank jitter bands.
_WOOD_PLANK_ASPECT = 6.0
_WOOD_GRAIN_CELLS_ALONG = 3
_WOOD_RING_BANDS = 3.5
_WOOD_TONE_JITTER = 0.07
_WOOD_HUE_JITTER = 0.05
_WOOD_SEAM_DARKEN = 0.72
_WOOD_GRAIN_SIGMA = 2.0

#: Concrete: contrast of the fine and broad noise terms about mid grey, speckle
#: density per pixel, and how dark a fully-weighted speckle goes.
_CONCRETE_FINE_CONTRAST = 0.22
_CONCRETE_MOTTLE_CONTRAST = 0.30
_CONCRETE_SPECKLE_DENSITY = 6.0e-4
_CONCRETE_SPECKLE_DEPTH = 0.42
_CONCRETE_GRAIN_SIGMA = 2.4

#: Terrazzo: chip centre spacing as a fraction of the short edge, chip radius as
#: a fraction of that spacing, and the polygon vertex range.
_TERRAZZO_SPACING_FRAC = 0.075
_TERRAZZO_CHIP_RADIUS_RANGE = (0.34, 0.48)
_TERRAZZO_CHIP_VERTICES = (5, 9)
_TERRAZZO_CHIP_ANGLE_JITTER = 0.34
_TERRAZZO_CHIP_RADIUS_JITTER = (0.72, 1.14)
_TERRAZZO_MATRIX_CONTRAST = 0.10
_TERRAZZO_GRAIN_SIGMA = 2.2

#: Neighbourhood used to spread single-pixel speckle seeds into small blobs.
#: Wrap-bordered, so a speckle on the edge blooms across it.
_SPECKLE_KERNEL = np.array(
    [[0.35, 0.70, 0.35], [0.70, 1.00, 0.70], [0.35, 0.70, 0.35]], dtype=np.float32
)


def _resolve_size(size_px: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a generator size request to ``(height, width)`` in pixels.

    ``size_px`` is either a single edge for a square tile or a
    ``(width, height)`` pair -- width first, matching both ``cv2``'s size
    convention and this module's ``width_mm, height_mm`` argument order -- so the
    600x1200 formats are requested as ``(512, 1024)``.
    """
    if isinstance(size_px, (str, bytes)):
        # Iterable, so it would otherwise be read digit by digit.
        raise TypeError(f"size_px must be an int or a (width, height) pair, got {size_px!r}")
    if isinstance(size_px, (int, np.integer)):
        width = height = int(size_px)
    else:
        try:
            values = tuple(int(value) for value in size_px)
        except TypeError as exc:
            raise TypeError(
                f"size_px must be an int or a (width, height) pair, got {type(size_px)!r}"
            ) from exc
        if len(values) != 2:
            raise ValueError(f"size_px must hold exactly two edges, got {len(values)}")
        width, height = values

    for name, value in (("width", width), ("height", height)):
        if not MIN_TEXTURE_EDGE <= value <= MAX_GENERATED_EDGE:
            raise ValueError(
                f"size_px {name} must be within "
                f"[{MIN_TEXTURE_EDGE}, {MAX_GENERATED_EDGE}] px, got {value}"
            )
    return height, width


def _rng(seed: int) -> np.random.Generator:
    """Seeded generator for one texture. Rejects a non-integral ``seed``."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed)!r}")
    return np.random.default_rng(int(seed))


def _isotropic_cells(height: int, width: int, cells_short_edge: float) -> tuple[int, int]:
    """Cell counts that give both axes the same cell size in pixels.

    Scaling the cell count with each edge is what keeps a 1:2 plank's noise
    features round instead of stretched, so a non-square ``size_px`` changes how
    much of the pattern you see rather than the scale of what you see.
    """
    short_edge = float(min(height, width))
    cells_y = max(1, int(round(cells_short_edge * height / short_edge)))
    cells_x = max(1, int(round(cells_short_edge * width / short_edge)))
    return cells_y, cells_x


def _value_noise(
    rng: np.random.Generator, height: int, width: int, cells_y: int, cells_x: int
) -> np.ndarray:
    """One octave: a random cell grid, bilinearly upsampled to ``(height, width)``.

    The grid is wrap-padded by one cell before upsampling and the oversized
    result cropped, so the interpolated field is periodic in both axes and does
    not have to be repaired at the border.
    """
    cells_y = int(min(max(1, cells_y), height))
    cells_x = int(min(max(1, cells_x), width))

    grid = rng.random((cells_y, cells_x), dtype=np.float32)
    wrapped = np.pad(grid, ((0, 1), (0, 1)), mode="wrap")

    pad_y = max(1, int(round(height / cells_y)))
    pad_x = max(1, int(round(width / cells_x)))
    upsampled = cv2.resize(
        wrapped, (width + pad_x, height + pad_y), interpolation=cv2.INTER_LINEAR
    )
    return np.ascontiguousarray(upsampled[:height, :width])


def _fbm(
    rng: np.random.Generator,
    height: int,
    width: int,
    cells_y: int,
    cells_x: int,
    *,
    octaves: int = 5,
    gain: float = 0.5,
    lacunarity: float = 2.0,
) -> np.ndarray:
    """Fractal Brownian motion: amplitude-weighted octaves normalized to [0, 1].

    Cell counts are per-axis so callers can ask for anisotropic noise -- few
    cells along one axis and many across it is exactly the stretched field wood
    grain needs.
    """
    total = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    weight_sum = 0.0
    frequency_y = float(cells_y)
    frequency_x = float(cells_x)

    for _ in range(max(1, int(octaves))):
        total += amplitude * _value_noise(
            rng, height, width, int(round(frequency_y)), int(round(frequency_x))
        )
        weight_sum += amplitude
        amplitude *= gain
        frequency_y *= lacunarity
        frequency_x *= lacunarity

    return total / weight_sum


def _isotropic_fbm(
    rng: np.random.Generator,
    height: int,
    width: int,
    cells_short_edge: float,
    *,
    octaves: int = 5,
    gain: float = 0.5,
) -> np.ndarray:
    """:func:`_fbm` with square cells, sized from the short edge."""
    cells_y, cells_x = _isotropic_cells(height, width, cells_short_edge)
    return _fbm(rng, height, width, cells_y, cells_x, octaves=octaves, gain=gain)


def _normalize01(field: np.ndarray) -> np.ndarray:
    """Rescale to full [0, 1] range; a flat field becomes a constant 0.5."""
    low = float(field.min())
    high = float(field.max())
    if high - low < 1e-6:
        return np.full_like(field, 0.5)
    return (field - low) / (high - low)


def _warp(field: np.ndarray, shift_x: np.ndarray, shift_y: np.ndarray) -> np.ndarray:
    """Advect ``field`` by a per-pixel displacement, wrapping at the border.

    ``cv2.remap`` has no wrap border mode, so the sample coordinates are taken
    modulo the field size instead -- which is the same thing for a field that is
    already periodic.
    """
    height, width = field.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = np.mod(grid_x + shift_x, float(width)).astype(np.float32)
    map_y = np.mod(grid_y + shift_y, float(height)).astype(np.float32)
    return cv2.remap(
        field, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def _vein_mask(phase: np.ndarray, width: float, gamma: float) -> np.ndarray:
    """Thin ridges along the zero crossings of a phase field, in [0, 1].

    ``phase`` counts cycles, so ``|sin(pi * phase)|`` is the distance to the
    nearest crossing; anything inside ``width`` of one becomes vein, falling off
    with ``gamma``. Reading the crossings instead of the crests is what yields a
    line rather than a band.
    """
    distance = np.abs(np.sin(np.pi * phase)).astype(np.float32)
    ridge = np.clip(1.0 - distance / max(1e-6, width), 0.0, 1.0)
    return np.power(ridge, gamma, dtype=np.float32)


def _speckle_mask(
    rng: np.random.Generator, height: int, width: int, density: float
) -> np.ndarray:
    """Sparse soft blobs in [0, 1] -- ``density`` seeds per pixel, wrap-aware."""
    mask = np.zeros((height, width), dtype=np.float32)
    count = max(1, int(round(density * height * width)))
    rows = rng.integers(0, height, count)
    columns = rng.integers(0, width, count)
    weights = rng.uniform(0.35, 0.95, count).astype(np.float32)
    np.add.at(mask, (rows, columns), weights)
    # ``filter2D`` rejects BORDER_WRAP outright, so the wrap is done by padding a
    # one-pixel ring by hand -- the kernel is 3x3, so that ring is all the
    # neighbourhood an edge pixel needs -- and cropping the result back.
    padded = np.pad(mask, 1, mode="wrap")
    spread = cv2.filter2D(padded, -1, _SPECKLE_KERNEL)[1:-1, 1:-1]
    return np.clip(spread, 0.0, 1.0)


def _tint(colour_bgr: tuple[float, float, float], field: np.ndarray) -> np.ndarray:
    """Broadcast a BGR anchor over a ``(h,w)`` multiplier field."""
    anchor = np.asarray(colour_bgr, dtype=np.float32).reshape(1, 1, 3)
    return anchor * field[:, :, np.newaxis]


def _mix(
    dark_bgr: tuple[float, float, float], light_bgr: tuple[float, float, float], field: np.ndarray
) -> np.ndarray:
    """Per-pixel interpolation between two BGR anchors over a [0, 1] field."""
    dark = np.asarray(dark_bgr, dtype=np.float32).reshape(1, 1, 3)
    light = np.asarray(light_bgr, dtype=np.float32).reshape(1, 1, 3)
    return dark + (light - dark) * field[:, :, np.newaxis]


def _quantize(
    colour: np.ndarray, rng: np.random.Generator, grain_sigma: float
) -> np.ndarray:
    """Add achromatic grain and land on ``(h,w,3)`` ``uint8`` BGR.

    The grain is a single channel broadcast across BGR, so it reads as sensor
    noise on a photographed tile rather than as colour fringing.
    """
    if grain_sigma > 0.0:
        grain = rng.normal(0.0, grain_sigma, (colour.shape[0], colour.shape[1], 1))
        colour = colour + grain.astype(np.float32)
    return np.clip(np.rint(colour), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Procedural generators -- finishes
# --------------------------------------------------------------------------- #


def generate_marble(size_px: int | tuple[int, int], seed: int = 0) -> np.ndarray:
    """Synthesize a marble pattern (Requirement 11.2).

    fBm value noise warped by a second, lower-frequency noise field, then used to
    displace the phase of a directional ramp. The veins are the ramp's zero
    crossings, so they run long and roughly parallel the way bedding planes in
    stone do, while the turbulence bends and splits them. Thresholding near the
    crossing rather than at the crest is what keeps a vein a line: a band count on
    its own gives closed blobs, which read as camouflage. A finer parallel
    hairline set and a slow tonal mottle sit under it on a warm-white base.

    Args:
        size_px: A single edge for a square tile, or a ``(width, height)`` pair.
        seed: Any integer; identical seeds give byte-identical output.

    Returns:
        ``(height, width, 3)`` ``uint8`` BGR. Pass through :func:`make_seamless`
        before use as a tiling texture.
    """
    height, width = _resolve_size(size_px)
    rng = _rng(seed)

    cells_y, cells_x = _isotropic_cells(height, width, _MARBLE_BASE_CELLS)
    veining = _fbm(rng, height, width, cells_y, cells_x, octaves=_MARBLE_OCTAVES)

    # A second field displaces the first. Two independent components, so the
    # warp swirls rather than sliding everything one way.
    warp_cells_y, warp_cells_x = _isotropic_cells(height, width, 2.0)
    strength = _MARBLE_WARP_FRAC * min(height, width)
    shift_x = (_fbm(rng, height, width, warp_cells_y, warp_cells_x, octaves=3) - 0.5) * strength
    shift_y = (_fbm(rng, height, width, warp_cells_y, warp_cells_x, octaves=3) - 0.5) * strength
    turbulence = _normalize01(_warp(veining, shift_x, shift_y))

    # Integer wave counts per axis, so the ramp closes on itself at the border and
    # the vein direction is whatever diagonal those two counts describe.
    waves_x = int(rng.integers(*_MARBLE_WAVES_X))
    waves_y = int(rng.integers(*_MARBLE_WAVES_Y))
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32) / width,
        np.arange(height, dtype=np.float32) / height,
    )
    phase = (
        waves_x * grid_x
        + waves_y * grid_y
        + (turbulence - 0.5) * 2.0 * _MARBLE_VEIN_DISTORTION
    )

    veins = _vein_mask(phase, _MARBLE_VEIN_WIDTH, _MARBLE_VEIN_GAMMA) * _MARBLE_VEIN_STRENGTH
    hairlines = (
        _vein_mask(phase * _MARBLE_HAIRLINE_MULTIPLE, _MARBLE_HAIRLINE_WIDTH, 1.0)
        * _MARBLE_HAIRLINE_STRENGTH
    )
    shoulder = (
        _vein_mask(phase, _MARBLE_VEIN_WIDTH * _MARBLE_SHOULDER_MULTIPLE, 1.0)
        * _MARBLE_SHOULDER_STRENGTH
    )
    veins = np.maximum(np.maximum(veins, hairlines), shoulder)

    mottle = _isotropic_fbm(rng, height, width, 1.5, octaves=3, gain=0.6)
    base = _tint(_MARBLE_BASE_BGR, 0.965 + 0.07 * mottle)
    vein_colour = np.asarray(_MARBLE_VEIN_BGR, dtype=np.float32).reshape(1, 1, 3)
    colour = base * (1.0 - veins[:, :, np.newaxis]) + vein_colour * veins[:, :, np.newaxis]

    return _quantize(colour, rng, _MARBLE_GRAIN_SIGMA)


def generate_wood_plank(size_px: int | tuple[int, int], seed: int = 0) -> np.ndarray:
    """Synthesize a wood-plank pattern (Requirement 11.2).

    Grain is anisotropic noise -- a handful of cells along the plank's long axis
    against many across it -- so the figure stretches lengthwise the way sawn
    timber does. A low-frequency sine of a second stretched field adds the ring
    figure. The tile is then divided into planks across the short axis, each one
    rolled along its length and given its own tone and warm/cool jitter, so no
    two planks repeat, with a darkened line at every plank seam.

    Plank count follows a fixed physical plank proportion
    (:data:`_WOOD_PLANK_ASPECT`), so a square tile shows several planks and a 1:2
    tile shows proportionally fewer at the same plank width.

    Args:
        size_px: A single edge for a square tile, or a ``(width, height)`` pair.
        seed: Any integer; identical seeds give byte-identical output.

    Returns:
        ``(height, width, 3)`` ``uint8`` BGR. Pass through :func:`make_seamless`
        before use as a tiling texture.
    """
    height, width = _resolve_size(size_px)
    rng = _rng(seed)

    # Work planks-horizontal, then transpose back if the long axis was vertical.
    transposed = height > width
    work_h, work_w = (width, height) if transposed else (height, width)

    plank_count = max(1, int(round(work_h * _WOOD_PLANK_ASPECT / work_w)))
    plank_count = min(plank_count, max(1, work_h // 8))
    edges = np.linspace(0, work_h, plank_count + 1).round().astype(int)

    grain = _fbm(
        rng,
        work_h,
        work_w,
        max(2, work_h // 10),
        _WOOD_GRAIN_CELLS_ALONG,
        octaves=3,
        gain=0.55,
    )
    ring_field = _fbm(rng, work_h, work_w, max(2, work_h // 24), 2, octaves=2, gain=0.6)
    rings = 0.5 + 0.5 * np.sin(ring_field * _WOOD_RING_BANDS * 2.0 * np.pi)
    figure = _normalize01(0.65 * grain + 0.35 * rings.astype(np.float32))

    colour = np.zeros((work_h, work_w, 3), dtype=np.float32)
    for index in range(plank_count):
        top, bottom = int(edges[index]), int(edges[index + 1])
        if bottom <= top:
            continue

        band = np.roll(figure[top:bottom], int(rng.integers(0, work_w)), axis=1)
        plank = _mix(_WOOD_DARK_BGR, _WOOD_LIGHT_BGR, band)

        tone = 1.0 + float(rng.uniform(-_WOOD_TONE_JITTER, _WOOD_TONE_JITTER))
        hue = float(rng.uniform(-_WOOD_HUE_JITTER, _WOOD_HUE_JITTER))
        # Warm/cool jitter pushes blue and red apart around a fixed green.
        channel_gain = np.asarray([1.0 - hue, 1.0, 1.0 + hue], dtype=np.float32)
        plank *= tone * channel_gain.reshape(1, 1, 3)

        # Seam at the leading edge of every plank, including plank 0, so the
        # wrapped top and bottom edges stay statistically alike.
        seam_rows = min(2, bottom - top)
        plank[:seam_rows] *= _WOOD_SEAM_DARKEN

        colour[top:bottom] = plank

    pattern = _quantize(colour, rng, _WOOD_GRAIN_SIGMA)
    if transposed:
        pattern = np.ascontiguousarray(np.transpose(pattern, (1, 0, 2)))
    return pattern


def generate_concrete(size_px: int | tuple[int, int], seed: int = 0) -> np.ndarray:
    """Synthesize a concrete pattern (Requirement 11.2).

    Multi-octave noise held at deliberately low contrast about mid grey, plus a
    broad two-octave mottle for the uneven pour, plus sparse dark speckles for
    aggregate pinholes. Concrete's whole character is that nothing is high
    contrast, so both noise terms are scaled well below full range -- the finish
    should read as texture under lighting, not as pattern.

    Args:
        size_px: A single edge for a square tile, or a ``(width, height)`` pair.
        seed: Any integer; identical seeds give byte-identical output.

    Returns:
        ``(height, width, 3)`` ``uint8`` BGR. Pass through :func:`make_seamless`
        before use as a tiling texture.
    """
    height, width = _resolve_size(size_px)
    rng = _rng(seed)

    fine = _isotropic_fbm(rng, height, width, 8.0, octaves=5, gain=0.5)
    mottle = _isotropic_fbm(rng, height, width, 1.5, octaves=2, gain=0.6)
    value = np.clip(
        0.5
        + (fine - 0.5) * _CONCRETE_FINE_CONTRAST
        + (mottle - 0.5) * _CONCRETE_MOTTLE_CONTRAST,
        0.0,
        1.0,
    )

    # Mid value lands on the base colour unchanged; the band either side is what
    # the shading pass then has something to grip.
    colour = _tint(_CONCRETE_BASE_BGR, 0.86 + 0.28 * value)

    speckles = _speckle_mask(rng, height, width, _CONCRETE_SPECKLE_DENSITY)
    colour *= (1.0 - _CONCRETE_SPECKLE_DEPTH * speckles)[:, :, np.newaxis]

    return _quantize(colour, rng, _CONCRETE_GRAIN_SIGMA)


def _poisson_disk(
    rng: np.random.Generator, height: int, width: int, radius: float, attempts: int = 24
) -> np.ndarray:
    """Poisson-disk sample ``(x, y)`` centres at least ``radius`` apart.

    Bridson's algorithm with two adjustments for texture work: distances are
    measured on the torus, and candidates wrap, so chip placement is periodic and
    survives tiling. A background grid of cell size ``radius / sqrt(2)`` holds at
    most one sample, which keeps the rejection test to a fixed 5x5 lookup instead
    of a scan over every placed point.

    Returns:
        ``(n, 2)`` ``float32`` of ``(x, y)`` centres, never empty.
    """
    cell = max(1e-3, radius / np.sqrt(2.0))
    grid_h = max(1, int(np.ceil(height / cell)))
    grid_w = max(1, int(np.ceil(width / cell)))
    grid = np.full((grid_h, grid_w), -1, dtype=np.int32)
    points: list[tuple[float, float]] = []
    radius_sq = radius * radius

    def insert(x: float, y: float) -> int:
        index = len(points)
        points.append((x, y))
        grid[int(y / cell) % grid_h, int(x / cell) % grid_w] = index
        return index

    def far_enough(x: float, y: float) -> bool:
        centre_y, centre_x = int(y / cell), int(x / cell)
        for offset_y in range(-2, 3):
            for offset_x in range(-2, 3):
                index = grid[(centre_y + offset_y) % grid_h, (centre_x + offset_x) % grid_w]
                if index < 0:
                    continue
                other_x, other_y = points[index]
                delta_x = abs(other_x - x)
                delta_y = abs(other_y - y)
                delta_x = min(delta_x, width - delta_x)
                delta_y = min(delta_y, height - delta_y)
                if delta_x * delta_x + delta_y * delta_y < radius_sq:
                    return False
        return True

    active = [insert(float(rng.uniform(0.0, width)), float(rng.uniform(0.0, height)))]
    while active:
        slot = int(rng.integers(0, len(active)))
        origin_x, origin_y = points[active[slot]]
        placed = False
        for _ in range(attempts):
            angle = float(rng.uniform(0.0, 2.0 * np.pi))
            distance = float(rng.uniform(radius, 2.0 * radius))
            x = (origin_x + np.cos(angle) * distance) % width
            y = (origin_y + np.sin(angle) * distance) % height
            if far_enough(x, y):
                active.append(insert(x, y))
                placed = True
                break
        if not placed:
            # Exhausted this point's neighbourhood; drop it by swapping in the
            # tail so the active list stays O(1) to shrink.
            active[slot] = active[-1]
            active.pop()

    return np.asarray(points, dtype=np.float32)


def _chip_polygon(
    rng: np.random.Generator, centre_x: float, centre_y: float, chip_radius: float
) -> np.ndarray:
    """A jittered convex polygon around a chip centre, as ``int32`` vertices.

    Vertices are evenly spaced in angle before jitter, which bounds the angular
    step away from zero and so keeps the polygon convex and free of the slivers
    that sorting uniform random angles produces.
    """
    vertex_count = int(rng.integers(*_TERRAZZO_CHIP_VERTICES))
    angles = np.arange(vertex_count, dtype=np.float32) / vertex_count * 2.0 * np.pi
    angles += rng.uniform(
        -_TERRAZZO_CHIP_ANGLE_JITTER, _TERRAZZO_CHIP_ANGLE_JITTER, vertex_count
    ).astype(np.float32)
    radii = chip_radius * rng.uniform(*_TERRAZZO_CHIP_RADIUS_JITTER, vertex_count).astype(
        np.float32
    )
    points = np.stack(
        (centre_x + radii * np.cos(angles), centre_y + radii * np.sin(angles)), axis=1
    )
    return np.rint(points).astype(np.int32)


def _fill_wrapped(canvas: np.ndarray, polygon: np.ndarray, colour: tuple[float, ...]) -> None:
    """Fill ``polygon`` on ``canvas``, repeating it across any border it crosses."""
    height, width = canvas.shape[:2]
    bgr = tuple(float(channel) for channel in colour)

    min_x, min_y = polygon.min(axis=0)
    max_x, max_y = polygon.max(axis=0)
    shifts_x = [0]
    shifts_y = [0]
    if min_x < 0:
        shifts_x.append(width)
    if max_x >= width:
        shifts_x.append(-width)
    if min_y < 0:
        shifts_y.append(height)
    if max_y >= height:
        shifts_y.append(-height)

    for shift_y in shifts_y:
        for shift_x in shifts_x:
            offset = polygon + np.asarray((shift_x, shift_y), dtype=np.int32)
            cv2.fillPoly(canvas, [offset], bgr, lineType=cv2.LINE_AA)


def generate_terrazzo(size_px: int | tuple[int, int], seed: int = 0) -> np.ndarray:
    """Synthesize a terrazzo pattern (Requirement 11.2).

    Chip centres come from a Poisson-disk sample, which is what gives terrazzo
    its characteristic even-but-unstructured scatter -- uniform random points
    clump and read as noise, a lattice reads as tile. Each centre is drawn as a
    jittered convex polygon in one of a small aggregate palette over a light
    cement matrix, and chips crossing a border are repeated on the far side so
    the scatter survives tiling.

    Args:
        size_px: A single edge for a square tile, or a ``(width, height)`` pair.
        seed: Any integer; identical seeds give byte-identical output.

    Returns:
        ``(height, width, 3)`` ``uint8`` BGR. Pass through :func:`make_seamless`
        before use as a tiling texture.
    """
    height, width = _resolve_size(size_px)
    rng = _rng(seed)

    matrix_value = _isotropic_fbm(rng, height, width, 3.0, octaves=4, gain=0.5)
    canvas = _tint(
        _TERRAZZO_MATRIX_BGR,
        1.0 + _TERRAZZO_MATRIX_CONTRAST * (matrix_value - 0.5),
    )

    spacing = max(4.0, _TERRAZZO_SPACING_FRAC * min(height, width))
    centres = _poisson_disk(rng, height, width, spacing)
    low, high = _TERRAZZO_CHIP_RADIUS_RANGE
    palette_size = len(_TERRAZZO_PALETTE)

    for centre_x, centre_y in centres:
        colour = _TERRAZZO_PALETTE[int(rng.integers(0, palette_size))]
        # Per-chip brightness jitter so repeats of one palette entry still differ.
        gain = 1.0 + float(rng.uniform(-0.10, 0.10))
        chip_radius = spacing * float(rng.uniform(low, high))
        polygon = _chip_polygon(rng, float(centre_x), float(centre_y), chip_radius)
        _fill_wrapped(canvas, polygon, tuple(np.clip(np.asarray(colour) * gain, 0.0, 255.0)))

    return _quantize(canvas, rng, _TERRAZZO_GRAIN_SIGMA)
