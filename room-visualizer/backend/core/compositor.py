"""Compositor -- metric tiling by inverse warp.

This module turns a Tile_Definition's seamless pattern into an image-aligned
tiling of one Structural_Plane. The whole job is done as an *inverse* warp: for
every destination pixel the plane's ``H^-1`` gives the millimetre position on
the plane, the millimetre position is reduced modulo the tile pitch, and the
in-cell millimetre position converts to a texture pixel through the texture's
own ``px_per_mm``. A single :func:`cv2.remap` then samples the entire plane in
one pass.

Doing it this way is not just faster than warping tile-by-tile -- it is the
reason two requirements hold by construction rather than by tuning:

* Steps 1-3 happen entirely in millimetres, before any pixel conversion, so the
  ratio of the ``u`` pitch to the ``v`` pitch is exactly
  ``width_mm : height_mm`` on every plane and in either geometry mode
  (Requirements 8.6, 8.7). No stretch can creep in, because no stretch is ever
  applied.
* ``H^-1`` compresses distant regions, so the modulo cells shrink in image space
  with depth without a depth term appearing anywhere in this file
  (Requirement 5.7).

Grout is part of the same metric pass rather than a post-process: the tile pitch
is ``dimension + grout_mm``, and the in-cell position past the tile dimension is
grout. That is a comparison on arrays the sample map already computed, so it
costs almost nothing, and grout lines foreshorten with the tiles instead of
staying a constant pixel width.

No Python loop runs per tile or per pixel anywhere here, which is what keeps a
render inside the budget of Requirement 9.3. That budget is per tiled plane --
70 ms fixed plus 26 ms per plane -- because this module's work is per plane:
one inverse warp and one blend over each plane's mask bounding box. Measured at
1600x1200: 27 ms for one plane, 82 ms for four, against 41 and 123 ms before the
optimisations recorded through this file.

Three of those are worth finding from here. :func:`_plane_bbox` explains why the
tiled box comes from the plane *mask* and not from its contour -- a correctness
point as much as a speed one. :func:`_highlight_luts` explains how the gloss term
becomes two saturating 8-bit passes with no loss of a single output byte.
:func:`build_sample_maps` explains where the ``float64``/``float32`` boundary sits
and what was measured to put it there rather than one step earlier.

The second half of the module turns that flat tiling into something that looks
photographed. :func:`blend_lighting` re-lights the tile with the photograph's own
illumination -- multiply on the shadowed side of the plane median, soft-light on
the lit side, plus a gloss-scaled specular term (Requirements 7.4, 7.5).
:func:`feather_alpha` softens the plane boundary over ``feather_width_px``
(Requirement 7.7). :func:`compose` walks the planes in a fixed order, alpha
composites each one, and finally redraws the Foreground_Mask from the original
photograph so occluders stay in front (Requirement 7.6). :func:`encode_render`
puts the bytes on the wire in the configured format.

One detail of the blend is worth recording here rather than leaving it to be
rediscovered in :func:`_blend_lut`. The soft-light branch is evaluated on shading
**re-normalised about the plane median**, not on absolute normalised shading:
``S == M`` maps to 0.5, which is soft-light's own fixed point, so the lit branch
starts from the unmodified tile value and can only lighten from there. That is
what makes the two branches meet at the tile value for *every* plane median
instead of only for a plane whose median happens to sit at mid-grey -- an
absolute-shading soft-light darkens the lit side by as much as 49 levels at a
median of 30, which reads as the whole surface dimming on the wrong side of the
threshold. At a median of exactly 127.5 the re-normalisation is the identity, so
a mid-grey plane blends exactly as an absolute-shading formulation would.

The branch predicate is ``S < M``, so a pixel sitting exactly on the median takes
the soft-light branch -- where it is the identity, and where the multiply branch's
own limit as ``S`` rises to ``M`` also lands. Which side of the comparison owns
the median is therefore a labelling choice with no observable consequence, which
is the point of the re-normalisation.

Requirements: 5.7, 7.4, 7.5, 7.6, 7.7, 8.6, 8.7, 9.3.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Final, Mapping, MutableMapping, NamedTuple, Sequence

import cv2
import numpy as np

from backend.config import Settings, get_settings
from backend.core.geometry import invert_homography
from backend.core.lighting import NEUTRAL_DETAIL
from backend.schemas import (
    PlaneMetadata,
    PlaneName,
    PlaneRenderSpec,
    SceneState,
    TileDefinition,
)
from backend.utils.imageio import encode_image
from backend.utils.texture_helper import SeamlessTexture

__all__ = [
    "SampleMaps",
    "COMPOSITE_ORDER",
    "HIGHLIGHT_GAIN",
    "NEUTRAL_DETAIL",
    "resolve_grout",
    "build_sample_maps",
    "tile_field_for_plane",
    "blend_lighting",
    "feather_alpha",
    "compose",
    "encode_render",
]

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

#: Floor on the homogeneous divisor ``w`` of the inverse warp. The Geometry
#: Engine pins the sign of every homography so ``w`` is positive over the
#: plane's interior, which lets the divide below ignore the sign -- but ``w``
#: still crosses zero on the plane's vanishing line, and a pixel there has no
#: finite metric position at all. Those pixels are marked invalid and sampled at
#: the texture origin; they sit outside the plane mask, so the alpha composite in
#: :func:`compose` never shows them.
_MIN_DIVISOR: Final[float] = 1e-9

#: Slack added around the plane's own extent when deriving the sample-map
#: bounding box, in pixels. The contour is a *simplified* polygon and the mask it
#: was traced from is rasterised, so a mask pixel can sit a hair outside the
#: contour's own bounds; one pixel of margin costs nothing and keeps the tiling
#: from stopping short of the plane edge the feathered alpha will ask for.
_BBOX_MARGIN_PX: Final[int] = 1

#: Whether this OpenCV build honours ``BORDER_WRAP`` in :func:`cv2.remap`.
#: Sample coordinates are reduced modulo the pattern size before remapping, so
#: the border mode only decides how the final sub-pixel interpolation at the
#: wrap seam behaves. Builds that reject the flag fall back to sampling a
#: one-pixel wrap-padded copy of the pattern, which is exact for the same
#: reason. Probed once, on first failure, rather than per call.
_WRAP_BORDER_SUPPORTED: bool = True

#: Order planes are drawn in by :func:`compose`, fixed by the design. Plane masks
#: are a partition (Requirement 3.3) so the order between them cannot change a
#: pixel, but pinning it keeps the output byte-identical across runs. Note this
#: is deliberately *not* the Segmenter's ``PLANE_PRIORITY``: that order resolves
#: which plane owns a contested mask pixel, this one only decides paint order.
COMPOSITE_ORDER: Final[tuple[PlaneName, ...]] = (
    "wall_back",
    "wall_left",
    "wall_right",
    "floor",
)

#: Scale on the gloss-weighted specular term of :func:`blend_lighting`. Not a
#: :class:`Settings` field: it calibrates the meaning of the catalog's ``gloss``
#: column against the detail map, so an operator retuning it per deployment would
#: silently redefine what "gloss 0.85" means for every tile.
#:
#: 1.0 is the identity choice and the reason it is the default: at ``gloss = 1.0``
#: the photograph's entire high-frequency residual is transferred onto the tile,
#: so a mirror-finish tile shows exactly the local luminance structure the camera
#: saw, and ``gloss`` reads directly as the fraction of that residual kept.
HIGHLIGHT_GAIN: Final[float] = 1.0

#: Floor on the normalised plane median used as the multiply divisor. A plane
#: median of 0 means the plane is pure black in ``L*``, where the multiply branch
#: has no defined slope; this keeps the divide finite instead of producing inf.
_MIN_MEDIAN_NORM: Final[float] = 1.0 / 255.0

#: Floor on either half-span of the soft-light re-normalisation, in the shading
#: map's 8-bit units. The lit branch divides by ``255 - M`` and the shadowed one
#: by ``M``, and a plane median pinned at 255 or at 0 collapses the corresponding
#: span to zero. Both collapses are degenerate rather than meaningful -- the span
#: that vanishes holds at most the single shading value ``S == M`` -- so flooring
#: the divisor keeps the map finite and leaves that lone value at 0.5, which is
#: exactly where the median belongs.
_MIN_MEDIAN_SPAN: Final[float] = 1.0 / 255.0

#: Fallback plane median, in the shading map's 8-bit units, for a plane whose
#: cached median is missing or unusable *and* whose mask selects no pixels to
#: re-derive one from. Mid-grey is the only neutral choice: it puts the plane
#: exactly on the branch threshold, so the tile passes through close to unchanged
#: rather than being uniformly darkened or lightened by a guess.
_FALLBACK_MEDIAN: Final[float] = 127.5

#: Feather width applied to the Foreground_Mask edge, in pixels. The design asks
#: for a 1-pixel soft edge on the final occluder redraw; see
#: :func:`_foreground_alpha` for why that is implemented as an outward ramp.
_FOREGROUND_EDGE_PX: Final[int] = 1


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


class SampleMaps(NamedTuple):
    """Where each destination pixel of one plane samples the tile pattern.

    The arrays cover the plane's bounding box only, not the whole frame: a
    floor filling a fifth of a 1600x1200 photograph is a fifth of the sample-map
    arithmetic, and the maps are the largest transient allocation a render
    makes.

    Attributes:
        map_x: ``(h, w)`` ``float32`` texture column per destination pixel,
            already reduced into ``[0, pattern_width)``.
        map_y: ``(h, w)`` ``float32`` texture row, reduced into
            ``[0, pattern_height)``.
        grout: ``(h, w)`` ``uint8``, 255 where the pixel's in-cell metric
            position falls in the grout band between tiles and 0 elsewhere.
            An 8-bit 0/255 mask rather than a ``bool`` one so the grout colour
            can be written with :func:`cv2.copyTo`, which measures around
            nineteen times faster than the equivalent boolean fancy-index
            assignment (0.16 ms against 3.07 ms on a full-frame plane) and is
            worth a documented dtype for at that margin.
        bbox: ``(x0, y0, x1, y1)`` half-open destination rectangle the arrays
            cover, in image pixels.
    """

    map_x: np.ndarray
    map_y: np.ndarray
    grout: np.ndarray
    bbox: tuple[int, int, int, int]

    @property
    def shape(self) -> tuple[int, int]:
        """The ``(height, width)`` of the destination rectangle covered."""
        return self.map_x.shape[0], self.map_x.shape[1]

    @property
    def is_empty(self) -> bool:
        """Whether the plane's bounding box selects no destination pixels."""
        return self.map_x.size == 0

    @property
    def region(self) -> tuple[slice, slice]:
        """The ``bbox`` as ``(rows, cols)`` slices for indexing a full frame.

        Kept for callers that hold a :class:`SampleMaps` and a frame; the module's
        own paths carry the box as four integers because they also need it to size
        a patch.
        """
        x0, y0, x1, y1 = self.bbox
        return slice(y0, y1), slice(x0, x1)


# --------------------------------------------------------------------------- #
# Grout resolution
# --------------------------------------------------------------------------- #


def resolve_grout(
    spec: PlaneRenderSpec | None = None,
    tile: TileDefinition | None = None,
    settings: Settings | None = None,
) -> tuple[float, tuple[int, int, int]]:
    """Resolve the effective grout width and colour for one plane.

    Implements the inheritance chain a ``None`` in :class:`PlaneRenderSpec`
    stands for: the render request wins, then the Tile_Definition's own
    ``grout_mm``, then the configured defaults. A tile carries no colour of its
    own, so ``grout_rgb`` inherits straight from settings.

    Args:
        spec: the plane's render spec; ``None`` means "inherit everything".
        tile: the selected Tile_Definition, consulted for ``grout_mm`` only.
        settings: source of ``default_grout_mm`` / ``default_grout_rgb``;
            defaults to :func:`get_settings`.

    Returns:
        ``(grout_mm, grout_rgb)`` with ``grout_mm >= 0`` and each channel of
        ``grout_rgb`` in ``[0, 255]``. The colour stays in **RGB** order, as its
        name says; :func:`tile_field_for_plane` reverses it when writing into a
        BGR frame.

    Raises:
        ValueError: a resolved grout width is negative or not finite, or a
            resolved colour is not three channels in ``[0, 255]``.
    """
    cfg = settings or get_settings()

    if spec is not None and spec.grout_mm is not None:
        width_mm = float(spec.grout_mm)
    elif tile is not None and tile.grout_mm is not None:
        width_mm = float(tile.grout_mm)
    else:
        width_mm = float(cfg.default_grout_mm)
    if not math.isfinite(width_mm) or width_mm < 0.0:
        raise ValueError(f"grout_mm must be a non-negative finite number, got {width_mm}")

    raw = cfg.default_grout_rgb
    if spec is not None and spec.grout_rgb is not None:
        raw = spec.grout_rgb
    channels = tuple(int(channel) for channel in raw)
    if len(channels) != 3 or any(channel < 0 or channel > 255 for channel in channels):
        raise ValueError(f"grout_rgb must be three channels in [0, 255], got {raw!r}")

    return width_mm, (channels[0], channels[1], channels[2])


# --------------------------------------------------------------------------- #
# Shared validation
# --------------------------------------------------------------------------- #


def _frame_shape(image_shape: Sequence[int]) -> tuple[int, int]:
    """Coerce ``(H, W)`` or ``(H, W, C)`` to a validated ``(height, width)``."""
    if len(image_shape) < 2:
        raise ValueError(f"image_shape must be (H, W[, C]), got {tuple(image_shape)!r}")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"image_shape must be positive, got {(height, width)!r}")
    return height, width


def _plane_homography_inv(plane: PlaneMetadata) -> np.ndarray:
    """The plane's cached ``H^-1``, recomputed from ``H`` only if it is absent.

    Requirement 9.3 is the reason the inverse is cached on the Plane_Metadata at
    analysis time: a render must not pay for a matrix inversion per plane per
    request. This falls back to inverting ``H`` so a hand-built
    :class:`PlaneMetadata` still works.
    """
    if plane.homography_inv is not None:
        matrix = np.asarray(plane.homography_inv, dtype=np.float64)
        if matrix.shape == (3, 3) and np.isfinite(matrix).all():
            return matrix

    inverse = invert_homography(plane.homography)
    if inverse is None:
        raise ValueError(f"plane {plane.name!r} has no usable inverse homography")
    return inverse


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Half-open ``(x0, y0, x1, y1)`` bounding box of a mask's set pixels.

    Empty for an all-zero mask. Derived from the mask rather than the plane
    contour because this box bounds the pixels that will actually be *written*,
    and the mask is the thing the alpha is built from.
    """
    binary = np.asarray(mask)
    if binary.dtype != np.uint8 or not binary.flags["C_CONTIGUOUS"]:
        binary = np.ascontiguousarray((binary != 0).view(np.uint8) * 255)
    x, y, w, h = cv2.boundingRect(binary)
    if w <= 0 or h <= 0:
        return 0, 0, 0, 0
    return x, y, x + w, y + h


def _plane_bbox(
    plane: PlaneMetadata,
    shape: tuple[int, int],
    mask: np.ndarray | None = None,
) -> tuple[int, int, int, int]:
    """Half-open ``(x0, y0, x1, y1)`` bounding box of a plane, clipped to frame.

    Preference order is **mask, then contour, then whole frame**, and the order
    matters for correctness rather than only for speed.

    The mask is the authoritative extent, because the plane alpha
    :func:`compose` composites through is built from the mask and is zero
    everywhere outside it. So the mask's bounding box provably contains every
    pixel a render can write, and tiling it can never clip the rendered region.
    The contour cannot make that promise: :func:`simplify_contour` traces only
    the *largest* external contour, so a mask with more than one connected
    component -- which is the norm, not the exception, once the Segmenter has
    subtracted the Foreground_Mask out of a plane -- has set pixels outside the
    contour's own bounds. Those pixels are opaque in the alpha and were being
    composited from the zero-filled region of the tile field, which is to say
    painted black. On the 1600x1200 four-plane fixture that is 6 800 to 14 100
    pixels per plane.

    ``mask`` is therefore threaded in from :func:`compose`, which has already
    computed this box for the blend, so the tighter box costs no extra work. A
    caller with no mask to hand -- :func:`tile_field_for_plane`, and the metric
    probes in the test suite that read "outside the tiled box the field is
    black" -- still gets the contour box.

    The whole-frame fallback for a plane with no usable geometry at all is
    unchanged, and for the same reason: tiling too much is corrected by the plane
    alpha, whereas tiling too little leaves a visible hole.
    """
    height, width = shape

    if mask is not None:
        array = np.asarray(mask)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim == 2 and array.shape == (height, width):
            mx0, my0, mx1, my1 = _mask_bbox(array)
            if mx1 > mx0 and my1 > my0:
                return _grow_bbox((mx0, my0, mx1, my1), shape)

    candidates = []
    for source in (plane.contour, plane.bounding_points):
        if source is None:
            continue
        points = np.asarray(source, dtype=np.float64).reshape(-1, 2)
        if points.size and np.isfinite(points).all():
            candidates.append(points)
    if not candidates:
        return 0, 0, width, height

    points = np.concatenate(candidates, axis=0)
    return _grow_bbox(
        (
            int(math.floor(float(points[:, 0].min()))),
            int(math.floor(float(points[:, 1].min()))),
            int(math.ceil(float(points[:, 0].max()))) + 1,
            int(math.ceil(float(points[:, 1].max()))) + 1,
        ),
        shape,
    )


def _grow_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Pad a box by :data:`_BBOX_MARGIN_PX` on every side and clip it to frame."""
    height, width = shape
    x0, y0, x1, y1 = bbox
    x0 = max(x0 - _BBOX_MARGIN_PX, 0)
    y0 = max(y0 - _BBOX_MARGIN_PX, 0)
    x1 = min(x1 + _BBOX_MARGIN_PX, width)
    y1 = min(y1 + _BBOX_MARGIN_PX, height)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, 0, 0
    return x0, y0, x1, y1


def _wrap_in_place(values: np.ndarray, pitch: float, scratch: np.ndarray) -> None:
    """Reduce ``values`` modulo ``pitch`` in place, over a reusable ``scratch``.

    ``values - floor(values / pitch) * pitch`` rather than :func:`numpy.mod`,
    which is not a micro-optimisation: ``np.mod`` on float64 measures roughly
    twenty times slower than this on a full-frame plane, and this function runs
    four times per plane per render. Both spellings agree bit for bit on the
    ranges involved here, and both give a non-negative result for a positive
    pitch, which is what makes the grout comparison in
    :func:`build_sample_maps` a plain ``>=``.
    """
    np.multiply(values, 1.0 / pitch, out=scratch)
    np.floor(scratch, out=scratch)
    scratch *= pitch
    values -= scratch


def _as_bgr(pattern: np.ndarray) -> np.ndarray:
    """Coerce a tile pattern to a contiguous ``(h, w, 3)`` ``uint8`` BGR array."""
    array = np.asarray(pattern)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 1:
        array = cv2.cvtColor(array[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    elif array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"tile pattern must be (h,w[,1|3|4]), got shape {array.shape!r}")
    if array.dtype != np.uint8:
        raise ValueError(f"tile pattern must be uint8, got dtype {array.dtype!r}")
    return np.ascontiguousarray(array)


# --------------------------------------------------------------------------- #
# Sample-map construction
# --------------------------------------------------------------------------- #


def build_sample_maps(
    plane: PlaneMetadata,
    texture: SeamlessTexture,
    spec: PlaneRenderSpec,
    image_shape: Sequence[int],
    *,
    grout_mm: float | None = None,
    settings: Settings | None = None,
    mask: np.ndarray | None = None,
) -> SampleMaps:
    """Build the inverse-warp sample maps and grout mask for one plane.

    Fully vectorised over the plane's bounding box, in this order:

    1. ``[u, v, w] = H^-1 @ [x, y, 1]``, then ``u_mm = u/w``, ``v_mm = v/w``.
    2. ``[u', v'] = R(rotation_deg) @ [u_mm, v_mm] + offset_mm``, still in
       millimetres.
    3. Reduce modulo the tile pitch -- ``width_mm + grout_mm`` across, and
       ``height_mm + grout_mm`` down -- giving the position inside one cell.
       Because the pitch is built from the declared millimetre dimensions, the
       rendered footprint ratio is the declared ratio by construction
       (Requirements 8.6, 8.7), and because ``H^-1`` compresses with depth the
       cells shrink in image space as they recede (Requirement 5.7).
    4. Multiply by ``texture.px_per_mm`` to land on texture pixels, reduced
       modulo the pattern size so every coordinate is in range and the tiling
       wraps continuously.

    The grout mask falls out of step 3 for free: a cell spans ``[0, pitch)`` with
    the tile occupying ``[0, dimension)``, so everything past the dimension is
    the grout band between this tile and the next.

    **Where the precision boundary sits, and why there.** Steps 1 to 3 and the
    grout comparison run in ``float64``; only step 4 runs in ``float32``. That
    line is drawn from measurement, not preference. Requirement 8.6's rendered
    aspect ratio is verified by bracketing the rendered footprint between the
    last tile pixel and the first grout pixel in metric coordinates, with one
    micron of slack for the renderer and the check evaluating ``H^-1`` in a
    different order. Everything before the reduction therefore has to be exact to
    well under a micron, and ``float32`` is not: over the plane extents of the
    reference room it costs up to **1.9 mm** through the perspective divide,
    and still **3.3e-3 mm** when the divide is kept in ``float64`` and only the
    rotation and reduction are narrowed -- three times the slack, because the
    pre-reduction metric coordinate runs to millions of millimetres near the
    plane's vanishing line and ``float32`` carries only about seven significant
    digits.

    After the reduction the same coordinate lies in ``[0, pitch)``, a few hundred
    millimetres, where ``float32`` costs at most **1.6e-4 mm** -- six times inside
    the slack, and in any case invisible to that check, since the grout band it
    reads is classified in ``float64`` before the narrowing. What ``float32``
    perturbs from there on is the sub-pixel texture sample position, by under
    1.4e-4 of a texture pixel. So the narrowing is free of consequence exactly
    where it is free of consequence, and step 4 is a third of the cost of the
    ``float64`` spelling.

    Args:
        plane: the Structural_Plane, supplying ``homography_inv`` and the
            contour the bounding box is derived from.
        texture: the tile's :class:`SeamlessTexture` -- its ``width_mm``,
            ``height_mm``, and ``px_per_mm`` set the metric pitch and scale.
        spec: the plane's render spec, supplying ``rotation_deg`` and
            ``offset_mm``.
        image_shape: destination frame shape, ``(H, W)`` or ``(H, W, C)``.
        grout_mm: pre-resolved grout width; ``None`` resolves it through
            :func:`resolve_grout`.
        settings: settings used only when ``grout_mm`` needs resolving.
        mask: the plane's pixel mask, when the caller has one. The bounding box
            is then taken from the mask rather than from the plane contour, which
            is both the correct extent and the tightest one that is correct --
            see :func:`_plane_bbox`.

    Returns:
        A :class:`SampleMaps` over the plane's bounding box.

    Raises:
        ValueError: the frame shape, the plane's homography, the texture's
            metric dimensions, or the grout width is unusable.
    """
    shape = _frame_shape(image_shape)
    inverse = _plane_homography_inv(plane)
    bbox = _plane_bbox(plane, shape, mask)

    width_mm = float(texture.width_mm)
    height_mm = float(texture.height_mm)
    px_per_mm = float(texture.px_per_mm)
    if not (width_mm > 0.0 and height_mm > 0.0 and px_per_mm > 0.0):
        raise ValueError(
            "texture must declare positive width_mm, height_mm, and px_per_mm, got "
            f"{width_mm}, {height_mm}, {px_per_mm}"
        )
    pattern_w = float(texture.width_px)
    pattern_h = float(texture.height_px)

    if grout_mm is None:
        grout_mm, _ = resolve_grout(spec, None, settings)
    grout_mm = float(grout_mm)
    if not math.isfinite(grout_mm) or grout_mm < 0.0:
        raise ValueError(f"grout_mm must be a non-negative finite number, got {grout_mm}")

    x0, y0, x1, y1 = bbox
    box_h, box_w = y1 - y0, x1 - x0
    if box_h <= 0 or box_w <= 0:
        empty_f32 = np.zeros((0, 0), dtype=np.float32)
        return SampleMaps(
            empty_f32, empty_f32.copy(), np.zeros((0, 0), dtype=np.uint8), bbox
        )

    # --- step 1: destination pixels to metric plane coordinates ------------ #
    # `np.add.outer` builds each (h, w) plane in one allocation from the row and
    # column terms, so the pixel grid itself is never materialised.
    xs = np.arange(x0, x1, dtype=np.float64)
    ys = np.arange(y0, y1, dtype=np.float64)
    u = np.add.outer(inverse[0, 1] * ys + inverse[0, 2], inverse[0, 0] * xs)
    v = np.add.outer(inverse[1, 1] * ys + inverse[1, 2], inverse[1, 0] * xs)
    divisor = np.add.outer(inverse[2, 1] * ys + inverse[2, 2], inverse[2, 0] * xs)

    # The sign of `divisor` is already pinned positive over the plane interior,
    # so only its magnitude needs guarding: a pixel on the plane's vanishing line
    # has no finite metric position at all. That is rare enough to be worth
    # testing for rather than masking unconditionally.
    degenerate = np.abs(divisor) < _MIN_DIVISOR
    if degenerate.any():
        np.copyto(divisor, 1.0, where=degenerate)
    else:
        degenerate = None
    u /= divisor
    v /= divisor
    del divisor

    # --- step 2: rotation and offset, in millimetres ----------------------- #
    rotation = math.radians(float(spec.rotation_deg))
    if rotation:
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        u, v = cos_r * u - sin_r * v, sin_r * u + cos_r * v
    offset_u, offset_v = (float(spec.offset_mm[0]), float(spec.offset_mm[1]))
    if offset_u:
        u += offset_u
    if offset_v:
        v += offset_v

    # --- step 3: reduce into one tile cell, and read off the grout band ---- #
    scratch = np.empty_like(u)
    _wrap_in_place(u, width_mm + grout_mm, scratch)
    _wrap_in_place(v, height_mm + grout_mm, scratch)

    if grout_mm > 0.0:
        # Built with `cv2.compare` rather than numpy comparisons so it comes out
        # as the contiguous 0/255 mask `cv2.copyTo` wants when the grout colour
        # is written, without a second pass to convert it.
        grout = cv2.bitwise_or(
            cv2.compare(u, width_mm, cv2.CMP_GE),
            cv2.compare(v, height_mm, cv2.CMP_GE),
        )
        if degenerate is not None:
            # A pixel with no finite metric position is not grout; it is nowhere.
            grout[degenerate] = 0
    else:
        grout = np.zeros((box_h, box_w), dtype=np.uint8)

    # --- step 4: in-cell millimetres to texture pixels, in float32 --------- #
    # The narrowing happens here rather than at the end, so the scale and the
    # pattern reduction both run at half the memory traffic. See the docstring
    # for the measurement that puts this boundary here and not one step earlier.
    del scratch
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    del u, v
    scratch32 = np.empty_like(map_x)

    map_x *= np.float32(px_per_mm)
    map_y *= np.float32(px_per_mm)
    # Grout-band positions run past the pattern's own width; wrapping keeps every
    # coordinate samplable, and the grout colour overwrites them regardless.
    _wrap_in_place(map_x, pattern_w, scratch32)
    _wrap_in_place(map_y, pattern_h, scratch32)
    del scratch32

    if degenerate is not None:
        np.copyto(map_x, 0.0, where=degenerate)
        np.copyto(map_y, 0.0, where=degenerate)

    return SampleMaps(map_x=map_x, map_y=map_y, grout=grout, bbox=bbox)


# --------------------------------------------------------------------------- #
# Warped tiling
# --------------------------------------------------------------------------- #


def _remap_wrapped(pattern: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Sample ``pattern`` at the given coordinates with wrap semantics.

    One :func:`cv2.remap` over the whole plane, which is the design's main
    performance lever for Requirement 9.3.
    """
    global _WRAP_BORDER_SUPPORTED

    if _WRAP_BORDER_SUPPORTED:
        try:
            return cv2.remap(
                pattern, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP
            )
        except cv2.error:  # pragma: no cover - build-dependent
            _WRAP_BORDER_SUPPORTED = False

    # Coordinates are already inside the pattern, so the only sample that reaches
    # past the far edge is the interpolation partner at the wrap seam. Appending
    # the first row and column supplies exactly that partner.
    padded = np.pad(pattern, ((0, 1), (0, 1), (0, 0)), mode="wrap")
    return cv2.remap(padded, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _tile_patch_for_plane(
    plane: PlaneMetadata,
    texture: SeamlessTexture,
    spec: PlaneRenderSpec,
    image_shape: tuple[int, int],
    *,
    tile: TileDefinition | None,
    settings: Settings,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """The warped, grouted tiling of one plane as a **contiguous bbox patch**.

    The whole of :func:`tile_field_for_plane` except the frame-sized wrapper.
    :func:`compose` wants the patch, not the frame: it composites over the same
    box the patch covers, so a frame-sized field would mean zeroing 5.8 MB per
    plane, copying the patch into it, and then handing the blend a *strided* view
    of the result. Returning the patch skips all three.

    Returns:
        ``(patch, bbox)``. The patch is ``(y1 - y0, x1 - x0, 3)`` ``uint8`` BGR
        and contiguous; an empty box gives a zero-sized patch.
    """
    grout_mm, grout_rgb = resolve_grout(spec, tile, settings)
    maps = build_sample_maps(
        plane, texture, spec, image_shape, grout_mm=grout_mm, settings=settings, mask=mask
    )
    if maps.is_empty:
        return np.zeros((0, 0, 3), dtype=np.uint8), maps.bbox

    patch = _remap_wrapped(_as_bgr(texture.pattern), maps.map_x, maps.map_y)
    if grout_mm > 0.0:
        # `grout_rgb` is RGB by contract; the frame is BGR.
        bgr = (grout_rgb[2], grout_rgb[1], grout_rgb[0])
        solid = _grout_image(bgr, maps.shape, image_shape[0] * image_shape[1])
        # A masked `cv2.copyTo` rather than `patch[mask_bool] = colour`: same
        # result, about nineteen times faster (0.14 ms against 2.7 ms on a
        # full-frame plane), and it is why `SampleMaps.grout` is an 8-bit mask.
        written = cv2.copyTo(solid, maps.grout, patch)
        if written is not None and not np.may_share_memory(written, patch):
            patch = written  # pragma: no cover - build-dependent fallback
    return patch, maps.bbox


def tile_field_for_plane(
    plane: PlaneMetadata,
    texture: SeamlessTexture,
    spec: PlaneRenderSpec,
    image_shape: Sequence[int],
    *,
    tile: TileDefinition | None = None,
    settings: Settings | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render one plane's tiling, grout included, aligned to the full frame.

    The field is frame-sized so the caller can composite it with a feathered
    plane alpha without tracking offsets, but only the plane's bounding box is
    ever computed or written -- everything outside stays zero, and the plane
    alpha is zero there too.

    Grout is written after remapping, using the mask
    :func:`build_sample_maps` derived in metric space, so grout lines
    foreshorten with the tiles rather than holding a constant pixel width.

    Args:
        plane: the Structural_Plane to tile.
        texture: the tile's :class:`SeamlessTexture`.
        spec: the plane's render spec.
        image_shape: destination frame shape, ``(H, W)`` or ``(H, W, C)``.
        tile: the selected Tile_Definition, consulted for its ``grout_mm``.
        settings: settings for grout defaults; defaults to
            :func:`get_settings`.
        mask: the plane's pixel mask, when the caller has one. Supplying it
            takes the tiled box from the mask instead of the plane contour,
            which is the only box guaranteed to cover every pixel the plane
            alpha will ask for -- see :func:`_plane_bbox`. Left out, the box
            comes from the contour, so a caller reading "outside the tiled box
            the field is zero" as a probe keeps that reading.

    Returns:
        A ``(H, W, 3)`` ``uint8`` BGR field: the warped tiling inside the
        plane's bounding box, zero elsewhere.
    """
    shape = _frame_shape(image_shape)
    cfg = settings or get_settings()
    patch, bbox = _tile_patch_for_plane(
        plane, texture, spec, shape, tile=tile, settings=cfg, mask=mask
    )

    field = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if patch.size:
        x0, y0, x1, y1 = bbox
        field[y0:y1, x0:x1] = patch
    return field


# --------------------------------------------------------------------------- #
# Scratch buffers
# --------------------------------------------------------------------------- #


class _Workspace:
    """Reusable scratch planes for the blend, allocated once per request.

    Requirement 9.3 gives a render 26 ms of server time per tiled plane, and at
    1600x1200 the ``(H, W, 3)`` ``uint16`` index plane alone is 11.5 MB.
    Allocating a fresh set per plane would put four rounds of multi-megabyte
    allocation and first-touch page faults inside that budget, so :func:`compose`
    builds one workspace and every plane borrows a view of it.

    The backing buffers are deliberately **flat** and reshaped per plane rather
    than frame-shaped and sliced. Both give a right-sized view, but a
    ``buffer[:h, :w]`` slice of a frame-shaped buffer is not contiguous, and
    numpy's strided store path for a non-contiguous ``out=`` measures about two
    and a half times slower than the contiguous one on a full-frame plane -- 10 ms
    against 4 ms just to build the blend's index array. Reshaping a flat prefix
    keeps every scratch plane contiguous no matter what shape is asked for.

    Capacity grows on demand and never shrinks, so a workspace sized for the
    frame serves every plane inside it without another allocation.
    """

    __slots__ = ("_capacity", "_index", "_wide", "_highlight")

    def __init__(self, height: int, width: int) -> None:
        self._capacity = 0
        self._reserve(int(height) * int(width))

    def _reserve(self, pixels: int) -> None:
        """(Re)allocate so every buffer holds at least ``pixels`` pixels."""
        if pixels <= self._capacity:
            return
        self._capacity = pixels
        # Packed LUT index, `shading * 256 + tile`, one entry per colour channel.
        self._index = np.empty(pixels * 3, dtype=np.uint16)
        # A three-channel 8-bit plane, borrowed twice per glossy plane: once for
        # the broadcast shading that feeds the index, once for the broadcast
        # detail map the highlight LUTs read.
        self._wide = np.empty(pixels * 3, dtype=np.uint8)
        # The looked-up highlight magnitude, three-channel so `cv2.add` and
        # `cv2.subtract` can take it against the blended patch directly.
        self._highlight = np.empty(pixels * 3, dtype=np.uint8)

    def index(self, shape: tuple[int, int]) -> np.ndarray:
        """A contiguous ``(h, w, 3)`` ``uint16`` view for the packed LUT index."""
        pixels = shape[0] * shape[1]
        self._reserve(pixels)
        return self._index[: pixels * 3].reshape(shape[0], shape[1], 3)

    def wide(self, shape: tuple[int, int]) -> np.ndarray:
        """A contiguous ``(h, w, 3)`` ``uint8`` view for a broadcast 8-bit map."""
        pixels = shape[0] * shape[1]
        self._reserve(pixels)
        return self._wide[: pixels * 3].reshape(shape[0], shape[1], 3)

    def highlight(self, shape: tuple[int, int]) -> np.ndarray:
        """A contiguous ``(h, w, 3)`` ``uint8`` view for the highlight term."""
        pixels = shape[0] * shape[1]
        self._reserve(pixels)
        return self._highlight[: pixels * 3].reshape(shape[0], shape[1], 3)


@lru_cache(maxsize=8)
def _grout_source(bgr: tuple[int, int, int], pixels: int) -> np.ndarray:
    """A flat read-only ``uint8`` buffer of ``pixels`` repetitions of ``bgr``.

    The fast grout write needs a source image of the flat grout colour matching
    the destination's shape, and every plane in a render asks for a differently
    shaped one. Filling a fresh image per plane would put back most of the cost
    the fast write just removed.

    A flat buffer of the repeating ``B, G, R`` triple answers every shape at once:
    reshaping any prefix whose length is a multiple of three to ``(h, w, 3)``
    yields a contiguous image of that colour for *any* ``h`` and ``w``, because
    the triple's phase is preserved by construction. So one fill serves every
    plane -- and, since it is only ever read, one fill serves every render and
    every thread too, which is why it is cached here rather than rebuilt per
    request. Returned read-only so that stays true.

    ``pixels`` is the frame's pixel count rather than the plane's, so a render
    reuses one entry across all four planes instead of interning a buffer per
    plane size. At 1600x1200 an entry is 5.8 MB and the cache holds at most eight,
    which is bounded by the number of distinct grout colours in play.
    """
    flat = np.empty(int(pixels) * 3, dtype=np.uint8)
    flat.reshape(-1, 3)[:] = bgr
    flat.flags.writeable = False
    return flat


def _grout_image(
    bgr: tuple[int, int, int], shape: tuple[int, int], capacity_px: int
) -> np.ndarray:
    """A contiguous read-only ``(h, w, 3)`` ``uint8`` image of the grout colour."""
    pixels = shape[0] * shape[1]
    return _grout_source(bgr, max(capacity_px, pixels))[: pixels * 3].reshape(
        shape[0], shape[1], 3
    )


# --------------------------------------------------------------------------- #
# Lighting blend
# --------------------------------------------------------------------------- #


def _as_single_channel(array: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a ``(h, w)`` ``uint8`` map and return it as an ndarray."""
    plane = np.asarray(array)
    if plane.ndim == 3 and plane.shape[2] == 1:
        plane = plane[:, :, 0]
    if plane.ndim != 2:
        raise ValueError(f"expected a 2-D {name}, got shape {plane.shape!r}")
    if plane.dtype != np.uint8:
        raise ValueError(f"expected a uint8 {name}, got dtype {plane.dtype!r}")
    return plane


@lru_cache(maxsize=32)
def _highlight_luts(gloss: float) -> tuple[np.ndarray, np.ndarray]:
    """The gloss-scaled specular term as a split pair of 256-entry 8-bit tables.

    The highlight of Requirement 7.5 is ``(D - 128) * gloss * HIGHLIGHT_GAIN``,
    a signed quantity added to an already-clipped blend. Split into its positive
    and negative magnitudes it becomes two unsigned 8-bit tables, and the add
    becomes a *saturating* ``cv2.add`` followed by a saturating ``cv2.subtract``.

    That identity is exact, not an approximation. The blend the tables are
    applied to is already in ``[0, 255]``, and at most one of the two magnitudes
    is non-zero for any given ``D``, so ``sat_sub(sat_add(B, pos), neg)`` equals
    ``clip(B + highlight, 0, 255)`` term for term -- which is precisely what the
    int16-accumulator formulation computed, using two 6.9 MB int16 passes and a
    clip where this uses two 8-bit ones. Measured on the 1600x1200 four-plane
    fixture, the whole blend goes from 76 ms to 37 ms at catalog gloss values,
    with output bytes unchanged.

    The rounding is done in ``float32`` deliberately: that is the width the
    superseded implementation rounded at, so keeping it here makes the two
    bit-identical rather than merely close. Verified exhaustively over the entire
    ``(S, T)`` domain across a sweep of medians, gloss values, and detail maps.

    Args:
        gloss: the Tile_Definition's gloss, already clamped to ``[0, 1]``.

    Returns:
        ``(positive, negative)``, both ``(256,)`` ``uint8`` and read-only:
        callers share them.
    """
    detail = np.arange(256, dtype=np.float32) - np.float32(NEUTRAL_DETAIL)
    detail *= np.float32(gloss * HIGHLIGHT_GAIN)
    np.rint(detail, out=detail)

    positive = np.clip(detail, 0.0, 255.0).astype(np.uint8)
    negative = np.clip(-detail, 0.0, 255.0).astype(np.uint8)
    positive.flags.writeable = False
    negative.flags.writeable = False
    return positive, negative


@lru_cache(maxsize=16)
def _blend_lut(median: float) -> np.ndarray:
    """The design's blend as a flat ``256 * 256`` table, indexed ``S * 256 + T``.

    This is the module's second performance lever, after the single-pass remap.
    Strip the gloss highlight away and the blend is a function of exactly two
    8-bit quantities -- the tile value ``T`` and the shading ``S`` -- plus one
    per-plane constant, the median ``M``. So it has only 65 536 distinct answers,
    and evaluating it as written (two branches, a square root, a select, all over
    three colour channels of every pixel in the plane) recomputes each of those
    answers thousands of times.

    Tabulating it instead turns the whole blend into one gather, and measures
    roughly three times faster on a full-frame plane -- the difference between
    fitting Requirement 9.3's per-plane budget and missing it. The table is built in
    float64 and rounded once, so the arithmetic is *more* faithful to the design
    than a float32 evaluation would be; the only cost is that the highlight is
    added to an already-rounded blend, which moves the result by at most one
    8-bit step.

    Cached across calls because a median repeats for every render of the same
    scene. Both returned arrays are read-only: callers share them.

    The median enters twice, not once: it is the branch threshold *and* the point
    the soft-light branch's input is re-normalised about. Tabulating per median
    is what makes that affordable -- the re-normalisation is two divisions and a
    select over a 256x256 grid, paid once per plane rather than per pixel.

    Args:
        median: the plane's shading median in 8-bit units.

    Returns:
        A flat read-only ``(65536,)`` ``uint8`` table holding values in
        ``[0, 255]``. One table serves matte and glossy tiles alike: the
        highlight is applied on top of it by the saturating 8-bit pair of
        :func:`_highlight_luts`, so no wider accumulator is needed.
    """
    tile = np.arange(256, dtype=np.float64).reshape(1, 256)  # T, the column axis
    shade = np.arange(256, dtype=np.float64).reshape(256, 1)  # S, the row axis
    t = tile / 255.0

    # Shadowed side: multiply on absolute normalised shading, so full shadow goes
    # to black and the branch reaches the tile value exactly as S rises to M.
    # Clamped by max(median_norm, eps) to keep the divide finite at median 0.
    multiply = tile * ((shade / 255.0) / max(median / 255.0, _MIN_MEDIAN_NORM))

    # Lit side: soft-light on shading re-normalised about the plane median. The
    # two half-ranges [0, M] and [M, 255] are each mapped linearly onto [0, 0.5]
    # and [0.5, 1], which places S == M at 0.5 -- soft-light's fixed point, where
    # the curve returns the tile unchanged. Both halves have positive slope and
    # agree at the join, so the map is monotone in S over the whole range and
    # lands inside [0, 1] by construction; the clip below is belt and braces
    # against float error at the endpoints, not a correction.
    #
    # At median 127.5 the two half-slopes are equal and the map collapses to
    # S/255, so a plane whose median sits at mid-grey blends identically to an
    # absolute-shading soft-light.
    s = np.where(
        shade < median,
        0.5 * (shade / max(median, _MIN_MEDIAN_SPAN)),
        0.5 + 0.5 * ((shade - median) / max(255.0 - median, _MIN_MEDIAN_SPAN)),
    )
    np.clip(s, 0.0, 1.0, out=s)

    # Soft-light lightens without blowing out the way a screen blend would. Only
    # the s >= 0.5 half is ever selected, since the branch below hands this one
    # every pixel with S >= M; the other half is written out because the curve is
    # the standard two-sided one and truncating it here would obscure that.
    soft_light = (
        np.where(
            s <= 0.5,
            2.0 * t * s + t * t * (1.0 - 2.0 * s),
            2.0 * t * (1.0 - s) + np.sqrt(t) * (2.0 * s - 1.0),
        )
        * 255.0
    )

    # `S < M` is multiply, `S >= M` is soft-light. The median itself therefore
    # goes to soft-light, where s == 0.5 makes it the identity -- the same value
    # the multiply branch tends to from below, so the choice is documentation
    # rather than a discontinuity.
    blended = np.where(shade < median, multiply, soft_light)
    table = np.clip(np.rint(blended), 0.0, 255.0).reshape(-1).astype(np.uint8)
    table.flags.writeable = False
    return table


def blend_lighting(
    tile_bgr: np.ndarray,
    shading: np.ndarray,
    detail: np.ndarray,
    median: float,
    gloss: float = 0.0,
    settings: Settings | None = None,
    *,
    out: np.ndarray | None = None,
    workspace: "_Workspace | None" = None,
) -> np.ndarray:
    """Re-light a flat tile field with the photograph's illumination. R7.4, R7.5

    Per pixel, with tile value ``T``, shading ``S``, plane median ``M``, detail
    ``D``, and gloss ``g``, writing ``t = T/255``::

        multiply   = T * (S/255) / max(M/255, eps)

        # shading re-normalised about the plane median: S == M maps to 0.5
        s          = where(S < M, 0.5 * S / max(M, eps),
                                  0.5 + 0.5 * (S - M) / max(255 - M, eps))
        soft_light = where(s <= 0.5, 2ts + t^2 (1 - 2s),
                                    2t(1 - s) + sqrt(t)(2s - 1)) * 255

        blended    = where(S < M, multiply, soft_light)
        highlight  = (D - 128) * g * HIGHLIGHT_GAIN
        out        = clip(blended + highlight, 0, 255)

    ``where(S < M, ...)`` is the literal statement of Requirement 7.4: below the
    plane's own median the shading *darkens* the tile multiplicatively, so a cast
    shadow keeps falling across the new surface; at or above it the soft-light
    curve lightens without driving highlights to pure white the way a plain screen
    blend would. The median is per plane rather than global because a dim floor
    and a sunlit back wall have different neutral points -- a global threshold
    would push a whole surface onto one branch.

    The soft-light input is re-normalised about ``M`` rather than taken as
    ``S/255``, which is what makes "at or above the median the tile does not
    darken" true for every plane instead of only for one whose median sits at
    mid-grey. Both halves of that map have positive slope and meet at 0.5, so it
    is monotone in ``S`` and confined to ``[0, 1]``; at ``M = 127.5`` it reduces
    exactly to ``S/255``. A pixel with ``S == M`` takes the soft-light branch,
    where ``s == 0.5`` makes the curve the identity -- the same value the multiply
    branch tends to from below, so both branches agree on the median contour and
    which one formally owns it does not affect a single output byte.

    ``highlight`` is exactly zero at ``gloss = 0`` and scales linearly with
    gloss, which is Requirement 7.5: a matte concrete tile ignores the
    photograph's specular structure, a polished marble reproduces it.

    Everything except the highlight is read out of :func:`_blend_lut`, which
    tabulates the expression exactly for all 65 536 ``(S, T)`` pairs at this
    plane's median. The highlight is then read out of :func:`_highlight_luts` and
    applied as a saturating 8-bit add and subtract, which is that clip term for
    term rather than an approximation of it. See both functions for why.

    ``out`` may alias ``tile_bgr``, which is how :func:`compose` avoids a second
    per-plane buffer: the ``(S, T)`` index is fully materialised before the
    first byte of the result is written, so overwriting the tile patch in place
    cannot disturb the values still to be read.

    Args:
        tile_bgr: ``(h, w, 3)`` ``uint8`` flat tile field.
        shading: ``(h, w)`` ``uint8`` shading map over the same pixels.
        detail: ``(h, w)`` ``uint8`` detail map, neutral at
            :data:`~backend.core.lighting.NEUTRAL_DETAIL`.
        median: the plane's shading median in the same 8-bit units as
            ``shading``.
        gloss: the Tile_Definition's gloss, clamped to ``[0, 1]``.
        settings: accepted for interface symmetry with the rest of the module;
            the blend itself has no configurable term.
        out: optional ``(h, w, 3)`` ``uint8`` destination, written in place.
        workspace: scratch to borrow instead of allocating; supplied by
            :func:`compose` so the buffers are reused across planes.

    Returns:
        A ``(h, w, 3)`` ``uint8`` BGR array -- ``out`` when it was given.

    Raises:
        ValueError: shapes or dtypes disagree, or ``median`` is not finite.
    """
    del settings  # no configurable term; kept for the documented signature

    tile = np.asarray(tile_bgr)
    if tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) tile field, got shape {tile.shape!r}")
    if tile.dtype != np.uint8:
        raise ValueError(f"expected a uint8 tile field, got dtype {tile.dtype!r}")

    shade = _as_single_channel(shading, name="shading map")
    fine = _as_single_channel(detail, name="detail map")
    shape = (tile.shape[0], tile.shape[1])
    if shade.shape != shape or fine.shape != shape:
        raise ValueError(
            f"shading {shade.shape!r} and detail {fine.shape!r} must both match the "
            f"tile field's {shape!r}"
        )

    median = float(median)
    if not math.isfinite(median):
        raise ValueError(f"plane median must be finite, got {median}")
    gloss = float(np.clip(float(gloss), 0.0, 1.0))

    if out is None:
        result = np.empty((shape[0], shape[1], 3), dtype=np.uint8)
    else:
        result = out
        if result.shape != (shape[0], shape[1], 3) or result.dtype != np.uint8:
            raise ValueError(
                f"out must be a {(shape[0], shape[1], 3)!r} uint8 array, got "
                f"shape {result.shape!r} dtype {result.dtype!r}"
            )
    if tile.size == 0:
        return result

    ws = workspace or _Workspace(shape[0], shape[1])

    # `where(S < M, multiply, soft_light)` comes out of the table, which already
    # encodes the branch selection of Requirement 7.4. Pack (S, T) into one
    # uint16 -- 255*256 + 255 is exactly 65535, so the index cannot overflow --
    # and the whole blend becomes a single gather.
    #
    # The shading map is broadcast to three channels with `cv2.cvtColor` before
    # the multiply rather than by a numpy `[:, :, None]` view. Same values, but
    # the multiply then reads a contiguous three-channel plane instead of
    # broadcasting a stride-0 axis, which measures about a third faster on a
    # full-frame plane and is the cheapest 1.3x available in this function.
    spread = ws.wide(shape)
    cv2.cvtColor(shade, cv2.COLOR_GRAY2BGR, dst=spread)
    index = ws.index(shape)
    np.multiply(spread, np.uint16(256), out=index, dtype=np.uint16)
    np.add(index, tile, out=index)
    np.take(_blend_lut(median), index, out=result, mode="clip")

    if gloss <= 0.0 or HIGHLIGHT_GAIN == 0.0:
        # A matte tile takes no highlight; Requirement 7.5's term is exactly zero
        # at gloss 0, so there is nothing left to do.
        return result

    # Gloss-scaled specular term, applied as a saturating 8-bit add of its
    # positive magnitude and a saturating subtract of its negative one. See
    # `_highlight_luts` for why that is exactly `clip(blended + highlight, 0,
    # 255)` rather than an approximation of it. `spread` is free to be reused
    # here: the index it fed has already been gathered.
    positive, negative = _highlight_luts(gloss)
    cv2.cvtColor(fine, cv2.COLOR_GRAY2BGR, dst=spread)
    highlight = ws.highlight(shape)
    cv2.LUT(spread, positive, dst=highlight)
    cv2.add(result, highlight, dst=result)
    cv2.LUT(spread, negative, dst=highlight)
    cv2.subtract(result, highlight, dst=result)
    return result


# --------------------------------------------------------------------------- #
# Mask feathering
# --------------------------------------------------------------------------- #


def feather_alpha(mask: np.ndarray, width_px: int) -> np.ndarray:
    """Linear-ramp alpha over the inner ``width_px`` of a mask edge. R7.7

    ``cv2.distanceTransform`` gives every set pixel its Euclidean distance to
    the nearest unset pixel; alpha is that distance divided by ``width_px`` and
    clipped, so it is 1 at distance ``>= width_px``, 0 everywhere outside the
    mask, and linear in the band between. Feathering rather than hard-edged
    pasting is what removes the stair-step aliasing along a plane boundary that
    a rasterised contour otherwise leaves.

    ``cv2.DIST_MASK_PRECISE`` is used rather than the default 3x3 chamfer
    approximation: the approximation is off by a few percent, which is enough to
    leave a pixel at distance exactly ``width_px`` at alpha 0.98 instead of 1.0
    and turn a documented "fully opaque" guarantee into a near miss.

    Args:
        mask: ``(H, W)`` mask; any non-zero pixel counts as set.
        width_px: feather width in pixels. ``0`` gives a hard binary alpha,
            which is the honest reading of "no feathering" and the escape hatch
            for a caller that wants the raw mask.

    Returns:
        A ``(H, W)`` ``float32`` array in ``[0, 1]``.

    Raises:
        ValueError: the mask is not 2-D, or ``width_px`` is negative.
    """
    binary = np.asarray(mask)
    if binary.ndim == 3 and binary.shape[2] == 1:
        binary = binary[:, :, 0]
    if binary.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {binary.shape!r}")
    width_px = int(width_px)
    if width_px < 0:
        raise ValueError(f"width_px must be non-negative, got {width_px}")

    if binary.dtype != np.uint8 or not binary.flags["C_CONTIGUOUS"]:
        binary = np.ascontiguousarray((binary != 0).view(np.uint8) * 255)
    if binary.size == 0:
        return np.zeros(binary.shape, dtype=np.float32)

    if width_px == 0:
        return (binary != 0).astype(np.float32)

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    alpha = distance.astype(np.float32, copy=False)
    alpha *= np.float32(1.0 / width_px)
    np.clip(alpha, 0.0, 1.0, out=alpha)
    return alpha


def _foreground_alpha(mask: np.ndarray) -> np.ndarray:
    """Alpha for the final occluder redraw: 1 inside, one soft pixel outside.

    Requirement 7.6 is absolute -- a foreground pixel must come from the
    original photograph, byte for byte -- so the ramp cannot eat into the mask
    the way :func:`feather_alpha` does. It is placed *outside* instead: full
    alpha on every mask pixel, half alpha on the one-pixel ring around it, zero
    beyond. That is the design's "1-pixel feathered foreground edge" without
    trading away the guarantee it exists to protect, and a one-pixel ramp has
    exactly one intermediate step to offer regardless of how it is computed.

    Built from a dilation and two writes rather than by running
    :func:`feather_alpha` over a dilated mask. The two agree on any ordinary
    silhouette, but the explicit form is roughly an order of magnitude cheaper --
    no distance transform -- and it does not quietly promote a ring pixel in a
    concave notch to full alpha the way a distance-based ramp would.
    """
    binary = np.asarray(mask)
    if binary.ndim == 3 and binary.shape[2] == 1:
        binary = binary[:, :, 0]
    if binary.ndim != 2:
        raise ValueError(f"expected a 2-D foreground mask, got shape {binary.shape!r}")
    if binary.dtype != np.uint8 or not binary.flags["C_CONTIGUOUS"]:
        binary = np.ascontiguousarray((binary != 0).view(np.uint8) * 255)

    kernel = np.ones((3, 3), dtype=np.uint8)
    grown = cv2.dilate(binary, kernel, iterations=_FOREGROUND_EDGE_PX)
    alpha = np.zeros(binary.shape, dtype=np.float32)
    alpha[grown != 0] = 1.0 / (_FOREGROUND_EDGE_PX + 1)
    alpha[binary != 0] = 1.0
    return alpha


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #


def _composite_over(base: np.ndarray, top: np.ndarray, alpha: np.ndarray) -> None:
    """``base = base + alpha * (top - base)``, in place on a ``uint8`` view.

    Split by alpha rather than blended uniformly, because a feathered mask is
    almost entirely saturated: alpha is exactly 1 across the plane's interior and
    fractional only in a band ``feather_width_px`` wide along its edge. The
    interior is therefore a straight masked copy, and the float arithmetic runs
    over a gathered list of edge pixels whose length is proportional to the
    plane's *perimeter*, not its area.

    That matters twice over. It is an order of magnitude faster on a full-frame
    plane, and it makes the interior copy exact -- an opaque pixel is the tile
    value byte for byte, with no chance of a rounding step showing up as a
    one-level difference across the surface.

    The interior copy goes through :func:`cv2.copyTo` rather than
    :func:`numpy.copyto`. Same result, but OpenCV's masked copy is vectorised and
    measures around fifteen times faster here (0.4 ms against 6 ms on a
    1512x546 plane), which is a meaningful slice of Requirement 9.3's budget for
    a one-line change. OpenCV writes through the strided ``base`` view in place;
    the ``may_share_memory`` guard catches the case where a build declines to and
    hands back a fresh buffer instead.
    """
    # Both selections are built with OpenCV rather than numpy comparisons: they
    # come out as contiguous 0/255 masks in one vectorised pass each, which is
    # what `cv2.copyTo` wants anyway. The saturating subtract is exact here --
    # 255 - 255 == 0 and 255 - 0 == 255 -- so `band` is precisely
    # "alpha > 0 and alpha < 1" with no epsilon fudge.
    opaque = cv2.compare(alpha, 1.0, cv2.CMP_GE)
    written = cv2.copyTo(top, opaque, base)
    if written is not None and not np.may_share_memory(written, base):
        np.copyto(base, written)  # pragma: no cover - build-dependent fallback

    band = cv2.subtract(cv2.compare(alpha, 0.0, cv2.CMP_GT), opaque)
    if not cv2.countNonZero(band):
        return

    # Gather the ramp pixels by coordinate, so the float arithmetic runs over a
    # perimeter-sized list. Done in float32 so a tile darker than the photograph
    # under it does not wrap the way a uint8 subtract would.
    #
    # `cv2.findNonZero` rather than `np.nonzero`, for the same reason `cv2.copyTo`
    # replaced `np.copyto` above: both scan the whole band mask, but OpenCV's scan
    # is vectorised and measures around six times faster (0.46 ms against 2.7 ms
    # on a full-frame plane). Since this scan is over the *frame*, not over the
    # perimeter it finds, it was the dominant cost in this function -- larger than
    # the interior copy and the ramp arithmetic put together.
    found = cv2.findNonZero(band)
    if found is None:  # pragma: no cover - countNonZero already proved otherwise
        return
    coords = found.reshape(-1, 2)
    cols, rows = coords[:, 0], coords[:, 1]
    under = base[rows, cols].astype(np.float32)
    over = top[rows, cols].astype(np.float32)
    weight = alpha[rows, cols].astype(np.float32)[:, None]
    mixed = under + weight * (over - under)
    np.clip(mixed, 0.0, 255.0, out=mixed)
    np.rint(mixed, out=mixed)
    base[rows, cols] = mixed.astype(np.uint8)


def _resolve_median(
    plane: PlaneMetadata,
    mask: np.ndarray,
    shading: np.ndarray,
    override: float | None,
) -> float:
    """The plane median that selects the blend branch, in shading units.

    Preference order: an explicit override from a fresh
    :class:`~backend.core.lighting.LightingMaps`, then the value cached on the
    Plane_Metadata at analysis time (which is what a render normally uses --
    Requirement 9.2 forbids redoing analysis work), then a median re-derived
    from the mask, then mid-grey. A plane whose mask selected nothing is absent
    from ``plane_medians`` by the Lighting_Engine's own contract, so the missing
    case is real and has to be handled rather than asserted away.
    """
    for candidate in (override, plane.luminance_median):
        if candidate is None:
            continue
        value = float(candidate)
        if math.isfinite(value) and value > 0.0:
            return value

    selected = shading[np.asarray(mask) > 0]
    if selected.size:
        return float(np.median(selected))
    return _FALLBACK_MEDIAN


def compose(
    scene: SceneState,
    specs: Mapping[PlaneName, PlaneRenderSpec],
    textures: Mapping[PlaneName, SeamlessTexture],
    settings: Settings | None = None,
    *,
    tiles: Mapping[PlaneName, TileDefinition] | None = None,
    plane_medians: Mapping[PlaneName, float] | None = None,
    alpha_cache: MutableMapping[PlaneName, np.ndarray] | None = None,
    warnings: list[str] | None = None,
) -> np.ndarray:
    """Composite the requested tile selections onto a cached scene.

    Draw order is fixed by the design (Requirement 7.6):

    1. Start from a copy of ``scene.image``, so a plane nobody selected keeps its
       photographic appearance.
    2. Draw each requested plane in :data:`COMPOSITE_ORDER` --
       ``wall_back``, ``wall_left``, ``wall_right``, ``floor`` -- alpha
       compositing the re-lit tile field through the feathered plane alpha.
    3. Redraw ``scene.image`` wherever ``foreground_mask`` is set. Plane masks
       already exclude foreground pixels (Requirement 3.4), so this last pass is
       a guarantee of last resort rather than the primary mechanism -- but a
       visible sticker-over-furniture artifact is the most damaging failure this
       product has, so it is worth paying for twice.

    Only each plane's mask bounding box is blended and composited; the rest of
    the frame is already correct from step 1. Scratch buffers are allocated once
    here and reused across every plane (Requirement 9.3).

    Unusable requests degrade rather than raise: a plane the scene never
    detected, a plane with no texture supplied, or a plane whose mask is empty is
    skipped with a message appended to ``warnings``, because a render that drops
    one surface is far more useful to the caller than a 500.

    Args:
        scene: the cached Scene_State; its masks, homographies, and lighting
            maps are reused as-is, with no analysis repeated (Requirement 9.2).
        specs: per-plane render specs. Planes absent here are left untouched.
        textures: the seamless, metrically scaled texture per plane.
        settings: feather width, grout defaults; defaults to
            :func:`get_settings`.
        tiles: the resolved Tile_Definition per plane, consulted for ``gloss``
            and ``grout_mm``. A plane with no entry renders at gloss 0.
        plane_medians: optional override of the blend threshold per plane, for a
            caller holding a fresh :class:`LightingMaps`.
        alpha_cache: optional mapping the feathered plane alphas are read from
            and written to, so repeated renders of one scene pay the distance
            transform once (Requirement 9.2).
        warnings: optional list that skipped-plane messages are appended to.

    Returns:
        A new ``(H, W, 3)`` ``uint8`` BGR image. ``scene.image`` is never
        mutated.

    Raises:
        ValueError: the scene has been released, or its image and lighting maps
            do not agree on a shape.
    """
    cfg = settings or get_settings()
    notes = warnings if warnings is not None else []

    image = scene.image
    if image is None:
        raise ValueError("scene state has been released; it cannot be rendered from")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"scene image must be an (H, W, 3) uint8 BGR array, got shape "
            f"{image.shape!r} dtype {image.dtype!r}"
        )
    height, width = image.shape[0], image.shape[1]

    shading = _as_single_channel(scene.shading_map, name="shading map")
    detail = _as_single_channel(scene.detail_map, name="detail map")
    if shading.shape != (height, width) or detail.shape != (height, width):
        raise ValueError(
            f"lighting maps {shading.shape!r}/{detail.shape!r} do not match the "
            f"scene image {(height, width)!r}"
        )

    out = image.copy()
    if not specs:
        return out

    workspace = _Workspace(height, width)

    for name in COMPOSITE_ORDER:
        spec = specs.get(name)
        if spec is None:
            continue

        plane = scene.planes.get(name)
        if plane is None:
            notes.append(f"plane {name!r} was not detected in this scene; skipped")
            continue
        texture = textures.get(name)
        if texture is None:
            notes.append(f"no texture supplied for plane {name!r}; skipped")
            continue
        mask = scene.plane_masks.get(name)
        if mask is None:
            notes.append(f"plane {name!r} has no cached mask; skipped")
            continue
        mask = np.asarray(mask)
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = mask[:, :, 0]
        if mask.shape != (height, width):
            notes.append(
                f"plane {name!r} mask shape {mask.shape!r} does not match the "
                f"scene; skipped"
            )
            continue

        if not cv2.countNonZero(
            mask if mask.dtype == np.uint8 and mask.flags["C_CONTIGUOUS"]
            else np.ascontiguousarray((mask != 0).view(np.uint8) * 255)
        ):
            notes.append(f"plane {name!r} mask is empty; skipped")
            continue

        tile = tiles.get(name) if tiles else None
        # The patch, not a frame-sized field: the box it covers is derived from
        # this same mask, so it is exactly the box composited below. That saves
        # zeroing and writing 5.8 MB per plane, and it keeps the blend's inputs
        # contiguous. It also closes a hole -- see `_plane_bbox` on why a
        # contour-derived box can leave masked pixels untiled and therefore
        # painted black.
        patch, (x0, y0, x1, y1) = _tile_patch_for_plane(
            plane,
            texture,
            spec,
            (height, width),
            tile=tile,
            settings=cfg,
            mask=mask,
        )
        if not patch.size:
            notes.append(f"plane {name!r} covers no pixels to tile; skipped")
            continue
        region = (slice(y0, y1), slice(x0, x1))

        alpha = alpha_cache.get(name) if alpha_cache is not None else None
        if (
            alpha is None
            or alpha.shape != (height, width)
            or alpha.dtype != np.float32
        ):
            alpha = feather_alpha(mask, cfg.feather_width_px)
            if alpha_cache is not None:
                alpha_cache[name] = alpha

        median = _resolve_median(
            plane, mask, shading, (plane_medians or {}).get(name)
        )
        gloss = float(tile.gloss) if tile is not None else 0.0

        # Blend in place over the tile patch: it is a per-plane temporary, so
        # nothing else can observe the overwrite.
        blend_lighting(
            patch,
            shading[region],
            detail[region],
            median,
            gloss,
            cfg,
            out=patch,
            workspace=workspace,
        )
        _composite_over(out[region], patch, alpha[region])

    foreground = scene.foreground_mask
    if foreground is None:
        return out
    foreground = np.asarray(foreground)
    if foreground.ndim == 3 and foreground.shape[2] == 1:
        foreground = foreground[:, :, 0]
    if foreground.shape != (height, width):
        notes.append(
            f"foreground mask shape {foreground.shape!r} does not match the scene; "
            "occlusion pass skipped"
        )
        return out

    fx0, fy0, fx1, fy1 = _mask_bbox(foreground)
    if fx1 <= fx0 or fy1 <= fy0:
        return out

    # Grow the box by the ramp width so the soft ring just outside the mask is
    # included in the redraw.
    pad = _FOREGROUND_EDGE_PX + 1
    fx0, fy0 = max(fx0 - pad, 0), max(fy0 - pad, 0)
    fx1, fy1 = min(fx1 + pad, width), min(fy1 + pad, height)
    fg_region = (slice(fy0, fy1), slice(fx0, fx1))
    _composite_over(
        out[fg_region], image[fg_region], _foreground_alpha(foreground[fg_region])
    )
    return out


# --------------------------------------------------------------------------- #
# Output encoding
# --------------------------------------------------------------------------- #


def encode_render(
    image: np.ndarray,
    settings: Settings | None = None,
    *,
    fmt: str | None = None,
) -> tuple[bytes, str]:
    """Encode a composited image for the wire. R9.3

    PNG by default, because it is lossless: JPEG ringing lands hardest on the
    flat grout lines and hard plane edges this renderer spends its whole budget
    getting right. ``RV_RENDER_FORMAT=jpeg`` trades that for size at
    ``render_jpeg_quality``.

    Args:
        image: the composited ``(H, W, 3)`` ``uint8`` BGR image.
        settings: source of ``render_format`` and ``render_jpeg_quality``;
            defaults to :func:`get_settings`.
        fmt: per-request override of the configured format, as a Render_Request
            may carry.

    Returns:
        ``(encoded_bytes, mime_type)``.

    Raises:
        EncodeError: the format is unsupported or OpenCV refused the array.
    """
    cfg = settings or get_settings()
    return encode_image(
        image,
        fmt=fmt or cfg.render_format,
        jpeg_quality=cfg.render_jpeg_quality,
    )
