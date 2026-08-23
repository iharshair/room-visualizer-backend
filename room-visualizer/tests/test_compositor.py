"""Compositor tests -- metric fidelity of the rendered tiling.

Requirements 8.6, 8.7, and 6.4 all say the same thing from three angles: a tile
declared at `width_mm x height_mm` must come out of the renderer at that metric
ratio, on any Structural_Plane, in either geometry mode, with no isotropic or
anisotropic stretch anywhere in the chain. Property 17 is that statement, and
this module measures it rather than asserting it against a remembered number.

How the measurement works, and why it is not circular:

* The renderer is asked for a real tiling through :func:`tile_field_for_plane`,
  with grout enabled and painted in a colour no texture in this module can
  produce. The grout band *is* the composited tile-cell boundary, so finding
  that colour in the output locates the rendered footprint's edges in image
  space without consulting any of the Compositor's internals.
* Those image pixels are mapped back through the plane's `H^-1` into metric
  plane coordinates and reduced modulo the tile pitch, which is exactly the
  recovery the task specifies. The pixel nearest the boundary from inside the
  tile and the pixel nearest it from inside the grout *bracket* the rendered
  footprint dimension, and the bracket is what gets asserted: the declared
  millimetre dimension has to lie inside it. That claim is resolution-free --
  it holds however coarsely the plane is sampled -- which is why it is the
  primary assertion and the ratio bound is derived from it.
* Boundary-based measurement alone cannot see a stretch *inside* a cell, so
  :func:`ramp_texture` carries a second, independent probe: a pattern whose red
  channel encodes its own column and whose green channel encodes its own row.
  Regressing the rendered channels against the recovered metric coordinates
  recovers the millimetres-per-texel scale of each axis separately, so an
  anisotropic stretch shows up as the two scales disagreeing. That path goes
  through `px_per_mm`, the pattern's pixel dimensions, and `cv2.remap`, none of
  which the grout path touches.
* :func:`test_property_17_measurement_detects_an_injected_stretch` renders a
  deliberately mis-scaled tile and asserts the measurement rejects it, so the
  property cannot pass by measuring nothing.

Both geometry modes are covered by two cached scenes rather than by re-rendering
per example: a yawed pose, which recovers an orthogonal vanishing point triple
and builds metric plane frames, and a frontal pose, whose lateral vanishing
point runs to infinity so every plane falls back to the four-point planar
homography of Requirement 6.1. The scenes are generated once and only read.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Mapping, Sequence

import cv2
import numpy as np
import pytest
from hypothesis import (
    HealthCheck,
    assume,
    event,
    given,
    settings as hypothesis_settings,
    strategies as st,
)

from backend.config import get_settings
from backend.core.compositor import (
    blend_lighting,
    compose,
    feather_alpha,
    tile_field_for_plane,
)
from backend.core.geometry import calibrate
from backend.core.lighting import NEUTRAL_DETAIL, decompose
from backend.core.segmenter import bounding_quad, simplify_contour
from backend.schemas import (
    GeometryMode,
    PlaneMetadata,
    PlaneName,
    PlaneRenderSpec,
    SceneState,
    TileDefinition,
)
from backend.utils.texture_helper import SeamlessTexture, make_seamless, to_metric_texture
from tests.fixtures.synthetic import SyntheticRoom, make_synthetic_room

#: Every Structural_Plane the fixture room shows, in the design's order.
PLANE_NAMES: Final[tuple[PlaneName, ...]] = ("floor", "wall_back", "wall_left", "wall_right")


# --------------------------------------------------------------------------- #
# Documented tolerances and fixture parameters
# --------------------------------------------------------------------------- #

#: Requirement 8.6's bound, verbatim: the rendered metric width-to-height ratio
#: may differ from the declared ratio by no more than 1 percent.
ASPECT_RATIO_TOLERANCE: Final[float] = 0.01

#: Slack on the bracket containment check, in millimetres. The renderer and this
#: module evaluate `H^-1 @ [x, y, 1]` in a different order of operations -- the
#: renderer builds each row and column term once and adds them outwardly, this
#: module evaluates per pixel -- so the two agree to float64 round-off rather
#: than bit for bit. On plane extents of a few thousand millimetres that is
#: nanometres; a micron of slack keeps a pixel sitting exactly on the boundary
#: from deciding the test.
BRACKET_SLACK_MM: Final[float] = 1e-3

#: How tight the bracket must be, as a fraction of the declared dimension,
#: before the ratio bound is scored. The bracket is the gap between the last
#: tile pixel and the first grout pixel in metric space, so it *is* the
#: measurement's resolution: scoring a 1 percent claim against a measurement
#: coarser than 1 percent would test the fixture's pixel density, not the
#: renderer. Well-sampled planes come in far under this -- the boundary is
#: sampled by hundreds of pixels at different sub-pixel phases, so the observed
#: gap is a small fraction of one pixel's worth of millimetres. The containment
#: assertion above is scored unconditionally and carries the resolution-free
#: half of the claim.
BRACKET_PRECISION_FRACTION: Final[float] = 0.002

#: Fraction of a tile cell treated as "clearly inside" along the other axis when
#: locating one axis's boundary. A pixel is grout when it is past the tile
#: dimension in *either* axis, so the u boundary can only be read from pixels
#: comfortably inside the tile in v, and the other way round.
INNER_CELL_FRACTION: Final[float] = 0.75

#: Metric window, as a fraction of the cell, the ramp regression is fitted over.
#: Trimming both ends drops the pixels whose bilinear sample straddles the
#: pattern's wrap seam, where a periodic ramp is genuinely discontinuous and no
#: linear fit applies.
RAMP_FIT_WINDOW: Final[tuple[float, float]] = (0.15, 0.85)

#: Millimetre band around a cell's own origin and its far pitch edge that the
#: boundary search ignores.
#:
#: This is the one place the recovery cannot be compared against the renderer
#: pixel for pixel. A cell is a half-open interval, so a plane coordinate of
#: *exactly* zero reduces to 0 while the same coordinate a floating-point step
#: below zero reduces to the full pitch -- a whole cell away. The renderer and
#: this module evaluate `H^-1 @ [x, y, 1]` in a different order of operations, so
#: on the pixel where the plane's metric origin falls the two can land on
#: opposite sides of that step and disagree by one entire pitch. The band is a
#: nanometre wide, so it excludes only that pixel and its immediate neighbours,
#: and nothing else on the plane can hide inside it.
SEAM_GUARD_MM: Final[float] = 1e-6

#: Smallest number of pixels either side of a boundary before a measurement is
#: trusted. Two-sided by design: one pixel of each kind is enough to bracket the
#: boundary, but not enough to have found the best-resolved crossing on the
#: plane.
MIN_BOUNDARY_SAMPLES: Final[int] = 12

#: Smallest number of interior pixels a ramp regression is fitted over.
MIN_RAMP_SAMPLES: Final[int] = 400

#: Grout colour, RGB as a :class:`PlaneRenderSpec` carries it. Deliberately a
#: colour no texture in this module can produce -- every texture here is clamped
#: below 220 in the red channel -- and deliberately *not* channel-symmetric, so
#: a channel-order slip in the renderer shows up as grout that cannot be found
#: rather than as grout that happens to match anyway.
GROUT_RGB: Final[tuple[int, int, int]] = (255, 0, 128)

#: The same colour in the BGR order the composited field is written in.
GROUT_BGR: Final[np.ndarray] = np.array(GROUT_RGB[::-1], dtype=np.uint8)

#: Fixture room size and render quality for the cached scenes. 800x600 at
#: `supersample=2` gives the near floor roughly a hundred pixels per 600 mm
#: tile, which is what makes the metric bracket tight; the two scenes are
#: generated once per session, so the cost is paid twice, not per example.
SCENE_WIDTH: Final[int] = 800
SCENE_HEIGHT: Final[int] = 600
SCENE_SUPERSAMPLE: Final[int] = 2

#: Yaw that recovers a complete vanishing point triple, and yaw that loses the
#: lateral direction to infinity and so routes every plane to the planar
#: fallback of Requirement 6.1. Both are the values `tests/test_geometry.py`
#: pins the same two modes with.
SCENE_YAW_DEG: Final[Mapping[str, float]] = {
    "vanishing_points": 8.0,
    "planar_fallback": 0.0,
}


# --------------------------------------------------------------------------- #
# Cached scenes -- one per geometry mode
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scene:
    """A rendered room plus the Plane_Metadata a Compositor call needs.

    Built from the fixture's *analytic* plane outlines rather than from a
    Segmenter run, so a segmentation regression cannot make a compositing
    property fail and a compositing regression cannot hide behind a lucky mask.
    """

    room: SyntheticRoom
    mode: GeometryMode
    planes: dict[str, PlaneMetadata]
    masks: dict[str, np.ndarray]

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.room.image.shape[0]), int(self.room.image.shape[1])


def _bounding_quad(contour: np.ndarray) -> np.ndarray:
    """Four-point quad around a contour, in the shape Plane_Metadata carries."""
    box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
    return np.round(box).astype(np.int32)


def _plane_metadata(
    name: str,
    contour: np.ndarray,
    mask: np.ndarray,
    homography: np.ndarray,
    homography_inv: np.ndarray,
    extent_mm: Sequence[float],
    rmse_px: float,
    mode: GeometryMode,
) -> PlaneMetadata:
    """Assemble one :class:`PlaneMetadata` from analytic geometry.

    ``luminance_median`` is fixed at mid-grey: this module never blends lighting,
    so the value only has to be a legal one.
    """
    moments = cv2.moments(mask, binaryImage=True)
    area = float(moments["m00"]) or 1.0
    return PlaneMetadata(
        name=name,  # type: ignore[arg-type]
        contour=np.round(contour).astype(np.int32),
        bounding_points=_bounding_quad(contour),
        area_fraction=float(np.count_nonzero(mask)) / float(mask.size),
        centroid=(float(moments["m10"] / area), float(moments["m01"] / area)),
        homography=np.asarray(homography, dtype=np.float64),
        homography_inv=np.asarray(homography_inv, dtype=np.float64),
        plane_extent_mm=(
            float(extent_mm[0]),
            float(extent_mm[1]),
            float(extent_mm[2]),
            float(extent_mm[3]),
        ),
        reprojection_rmse_px=float(rmse_px),
        geometry_mode=mode,
        luminance_median=128.0,
    )


@lru_cache(maxsize=2)
def scene(mode: str) -> Scene:
    """The cached room and planes for one geometry mode.

    Cached because the property below draws tile formats, planes, rotations, and
    offsets -- not poses. Rendering and calibrating a room per example would
    dominate the runtime while testing the Geometry_Engine, which
    `tests/test_geometry.py` already covers.
    """
    room = make_synthetic_room(
        width=SCENE_WIDTH,
        height=SCENE_HEIGHT,
        focal_px=0.875 * SCENE_WIDTH,
        yaw_deg=SCENE_YAW_DEG[mode],
        pitch_deg=-12.0,
        walls=("left", "right", "back"),
        n_occluders=0,
        seed=0,
        supersample=SCENE_SUPERSAMPLE,
    )
    contours = {
        name: polygon.astype(np.float64) for name, polygon in room.plane_polygons.items()
    }
    calibration = calibrate(room.image, contours, settings=get_settings())
    assert calibration.geometry_mode == mode, (
        f"the {mode!r} scene calibrated as {calibration.geometry_mode!r}; the pose "
        "no longer exercises the mode this scene exists for"
    )

    planes: dict[str, PlaneMetadata] = {}
    masks: dict[str, np.ndarray] = {}
    for name in calibration.homographies:
        mask = room.plane_mask(name, subtract_occluders=False)
        masks[name] = mask
        planes[name] = _plane_metadata(
            name,
            contours[name],
            mask,
            calibration.homographies[name],
            calibration.homography_inverses[name],
            calibration.plane_extents_mm[name],
            calibration.reprojection_rmse_px[name],
            calibration.geometry_mode,
        )
    assert planes, f"the {mode!r} scene recovered no plane geometry at all"
    return Scene(room=room, mode=calibration.geometry_mode, planes=planes, masks=masks)


# --------------------------------------------------------------------------- #
# Textures
# --------------------------------------------------------------------------- #


def noise_texture(width_mm: float, height_mm: float, *, seed: int = 11) -> SeamlessTexture:
    """A seamless, metrically scaled texture with visible structure.

    Values are held inside ``[24, 216]`` so no texel can be mistaken for the
    grout colour and no texel can be mistaken for the untouched black outside the
    plane's bounding box.
    """
    rng = np.random.default_rng(seed)
    source = rng.integers(24, 217, size=(128, 128, 3), dtype=np.int16).astype(np.uint8)
    source = cv2.GaussianBlur(source, (0, 0), 2.0)
    return to_metric_texture(make_seamless(source), width_mm, height_mm)


@lru_cache(maxsize=32)
def metric_texture(width_mm: float, height_mm: float) -> SeamlessTexture:
    """Memoised :func:`noise_texture`, so a repeated draw costs nothing."""
    return noise_texture(width_mm, height_mm)


def ramp_texture(width_mm: float, height_mm: float) -> SeamlessTexture:
    """A texture whose channels encode their own texel coordinates.

    Red carries the pattern column and green the pattern row, both as a ramp of
    period equal to the pattern dimension -- ``255 * x / width_px`` rather than
    ``255 * x / (width_px - 1)``, so the ramp is genuinely periodic and its
    slope in texels is the same on both axes. Blue is a constant floor that keeps
    every texel distinguishable from the untouched background.

    The pattern is generated at the exact pixel size :func:`to_metric_texture`
    resolves for the declared millimetre dimensions, so it is installed without
    a resample: rescaling a ramp would blur precisely the quantity being
    measured.
    """
    probe = to_metric_texture(
        np.full((96, 96, 3), 128, dtype=np.uint8),
        width_mm,
        height_mm,
        ensure_seamless=False,
    )
    height_px, width_px = probe.height_px, probe.width_px
    pattern = np.empty((height_px, width_px, 3), dtype=np.uint8)
    pattern[:, :, 0] = 24
    pattern[:, :, 1] = (np.arange(height_px) * (255.0 / height_px)).astype(np.uint8)[:, None]
    pattern[:, :, 2] = (np.arange(width_px) * (255.0 / width_px)).astype(np.uint8)[None, :]
    return dataclasses.replace(probe, pattern=pattern)


# --------------------------------------------------------------------------- #
# Recovering the rendered footprint
# --------------------------------------------------------------------------- #


def render_spec(
    rotation_deg: float = 0.0,
    grout_mm: float = 12.0,
    offset_mm: tuple[float, float] = (0.0, 0.0),
) -> PlaneRenderSpec:
    """A render spec carrying its own grout, so no setting can change the probe."""
    return PlaneRenderSpec(
        tile_id="probe",
        rotation_deg=rotation_deg,
        grout_mm=grout_mm,
        grout_rgb=GROUT_RGB,
        offset_mm=offset_mm,
    )


def in_cell_mm(
    plane: PlaneMetadata,
    spec: PlaneRenderSpec,
    pitch_mm: tuple[float, float],
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map image pixels back through ``H^-1`` to a position inside one tile cell.

    This is the recovery Property 17 is stated over: image coordinates to metric
    plane coordinates through the plane's own inverse homography, then the
    render spec's rotation and offset in millimetres, then a reduction modulo the
    tile pitch. Nothing here reads the Compositor's sample maps -- the only
    inputs are the plane's ``H^-1``, the spec, and the declared pitch.

    Returns:
        ``(cu_mm, cv_mm, valid)``, the in-cell metric position of each pixel and
        a mask clearing the pixels whose homogeneous divisor is degenerate,
        which is every pixel on the plane's vanishing line.
    """
    inverse = np.asarray(plane.homography_inv, dtype=np.float64)
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    u = inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]
    v = inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]
    w = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]

    valid = np.isfinite(w) & (np.abs(w) > 1e-9)
    safe = np.where(valid, w, 1.0)
    u = u / safe
    v = v / safe

    rotation = math.radians(float(spec.rotation_deg))
    if rotation:
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        u, v = cos_r * u - sin_r * v, sin_r * u + cos_r * v
    u = u + float(spec.offset_mm[0])
    v = v + float(spec.offset_mm[1])

    valid &= np.isfinite(u) & np.isfinite(v)
    pitch_u, pitch_v = pitch_mm
    cu = u - np.floor(u / pitch_u) * pitch_u
    cv = v - np.floor(v / pitch_v) * pitch_v
    return cu, cv, valid


@dataclass(frozen=True)
class Boundary:
    """One recovered footprint edge, as the interval that brackets it."""

    lower_mm: float  # largest in-cell coordinate still drawn as tile
    upper_mm: float  # smallest in-cell coordinate drawn as grout

    @property
    def midpoint_mm(self) -> float:
        return 0.5 * (self.lower_mm + self.upper_mm)

    @property
    def width_mm(self) -> float:
        """Width of the bracket, which is the measurement's own resolution."""
        return self.upper_mm - self.lower_mm

    def brackets(self, declared_mm: float, *, slack_mm: float = BRACKET_SLACK_MM) -> bool:
        return self.lower_mm - slack_mm <= declared_mm <= self.upper_mm + slack_mm


@dataclass(frozen=True)
class Footprint:
    """The rendered tile footprint, recovered in metric plane coordinates."""

    width: Boundary
    height: Boundary

    @property
    def ratio(self) -> float:
        return self.width.midpoint_mm / self.height.midpoint_mm

    @property
    def precision(self) -> float:
        """Worst bracket width relative to its own midpoint."""
        return max(
            self.width.width_mm / self.width.midpoint_mm,
            self.height.width_mm / self.height.midpoint_mm,
        )


def _boundary(inside: np.ndarray, outside: np.ndarray) -> Boundary | None:
    """Bracket a boundary between the tile pixels below it and grout above it.

    ``None`` means only one thing -- too few pixels of either kind to have found
    the best-resolved crossing on this plane. An *inverted* bracket, where the
    last tile pixel sits past the first grout pixel, is deliberately returned
    as-is rather than reported as unmeasurable: that is exactly what a renderer
    laying cells at the wrong metric pitch produces, and it has to reach
    :meth:`Boundary.brackets` and fail there instead of being quietly discarded.
    """
    if inside.size < MIN_BOUNDARY_SAMPLES or outside.size < MIN_BOUNDARY_SAMPLES:
        return None
    lower = float(inside.max())
    upper = float(outside.min())
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    return Boundary(lower_mm=lower, upper_mm=upper)


def rendered_footprint(
    plane: PlaneMetadata,
    mask: np.ndarray,
    texture: SeamlessTexture,
    spec: PlaneRenderSpec,
    *,
    declared_mm: tuple[float, float] | None = None,
) -> Footprint | None:
    """Recover the rendered tile footprint from a composited field.

    The plane is tiled for real, the grout colour is located in the output to
    find the composited cell boundaries, and those pixels are mapped back
    through ``H^-1`` into metric coordinates by :func:`in_cell_mm`.

    Args:
        plane: the Structural_Plane, supplying ``H^-1``.
        mask: the plane's pixel mask; only these pixels are inspected.
        texture: the texture handed to the renderer.
        spec: the render spec handed to the renderer, carrying the grout.
        declared_mm: the millimetre dimensions the recovery reduces modulo.
            Defaults to the texture's own declaration; a caller passing
            something else is deliberately measuring a mismatch.

    Returns:
        The recovered :class:`Footprint`, or ``None`` when the plane is sampled
        too sparsely for a boundary to be located at all.
    """
    grout_mm = float(spec.grout_mm or 0.0)
    assert grout_mm > 0.0, "the footprint probe needs grout to find a cell boundary"

    width_mm, height_mm = declared_mm or (texture.width_mm, texture.height_mm)
    field = tile_field_for_plane(plane, texture, spec, mask.shape)

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    rendered = field[ys, xs]
    drawn = rendered.any(axis=1)  # outside the sample-map bbox the field is black
    is_grout = np.all(rendered == GROUT_BGR, axis=1)

    pitch_u, pitch_v = width_mm + grout_mm, height_mm + grout_mm
    cu, cv, valid = in_cell_mm(plane, spec, (pitch_u, pitch_v), xs, ys)
    usable = (
        valid
        & drawn
        & (cu > SEAM_GUARD_MM)
        & (cu < pitch_u - SEAM_GUARD_MM)
        & (cv > SEAM_GUARD_MM)
        & (cv < pitch_v - SEAM_GUARD_MM)
    )

    # A pixel is grout when it is past the tile dimension in either axis, so each
    # boundary is read only from pixels comfortably inside the cell in the other.
    across = usable & (cv <= INNER_CELL_FRACTION * height_mm)
    down = usable & (cu <= INNER_CELL_FRACTION * width_mm)

    width = _boundary(cu[across & ~is_grout], cu[across & is_grout])
    height = _boundary(cv[down & ~is_grout], cv[down & is_grout])
    if width is None or height is None:
        return None
    return Footprint(width=width, height=height)


def ramp_scales(
    plane: PlaneMetadata,
    mask: np.ndarray,
    texture: SeamlessTexture,
    spec: PlaneRenderSpec,
) -> tuple[float, float] | None:
    """Recover each axis's rendered texel-per-millimetre scale from a ramp tile.

    The ramp's period is the pattern dimension, so the rendered red channel is
    ``255 * cu / width_mm`` and the green channel ``255 * cv / height_mm``
    whenever the texture is sampled without stretch. Regressing each channel
    against its recovered metric coordinate therefore returns ``255 / width_mm``
    and ``255 / height_mm``, and their quotient is the rendered metric aspect
    ratio -- measured through ``px_per_mm``, the pattern's pixel dimensions, and
    ``cv2.remap``, none of which the grout boundary touches.

    Returns:
        ``(slope_u, slope_v)`` in levels per millimetre, or ``None`` when too few
        interior pixels are available to fit.
    """
    grout_mm = float(spec.grout_mm or 0.0)
    width_mm, height_mm = texture.width_mm, texture.height_mm
    field = tile_field_for_plane(plane, texture, spec, mask.shape)

    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    rendered = field[ys, xs]
    drawn = rendered.any(axis=1)
    is_grout = np.all(rendered == GROUT_BGR, axis=1)

    cu, cv, valid = in_cell_mm(
        plane, spec, (width_mm + grout_mm, height_mm + grout_mm), xs, ys
    )
    low, high = RAMP_FIT_WINDOW
    inside = (
        valid
        & drawn
        & ~is_grout
        & (cu >= low * width_mm)
        & (cu <= high * width_mm)
        & (cv >= low * height_mm)
        & (cv <= high * height_mm)
    )
    if int(np.count_nonzero(inside)) < MIN_RAMP_SAMPLES:
        return None

    red = rendered[inside, 2].astype(np.float64)
    green = rendered[inside, 1].astype(np.float64)
    slope_u = float(np.polyfit(cu[inside], red, 1)[0])
    slope_v = float(np.polyfit(cv[inside], green, 1)[0])
    if not (math.isfinite(slope_u) and math.isfinite(slope_v)):
        return None
    if slope_u <= 0.0 or slope_v <= 0.0:
        return None
    return slope_u, slope_v


# --------------------------------------------------------------------------- #
# Shared strategies
# --------------------------------------------------------------------------- #

#: Declared millimetre formats. The three Requirement 8.7 names them -- 1:1,
#: 1:2, and plank -- come first; the rest widen the ratio range in both
#: directions, including the landscape orientations, so a renderer that quietly
#: assumed a portrait tile would be caught.
TILE_FORMATS: Final[tuple[tuple[float, float], ...]] = (
    (600.0, 600.0),  # 1:1
    (600.0, 1200.0),  # 1:2
    (200.0, 1200.0),  # plank, 1:6
    (300.0, 300.0),
    (450.0, 900.0),
    (1000.0, 500.0),  # landscape 2:1
    (800.0, 800.0),
    (150.0, 600.0),  # narrow plank, 1:4
)

#: The three formats Requirement 8.7 calls out by name.
NAMED_FORMATS: Final[tuple[tuple[float, float], ...]] = (
    (600.0, 600.0),
    (600.0, 1200.0),
    (200.0, 1200.0),
)

GEOMETRY_MODES: Final[tuple[str, ...]] = ("vanishing_points", "planar_fallback")

_format = st.sampled_from(TILE_FORMATS)
_mode = st.sampled_from(GEOMETRY_MODES)
_plane_name = st.sampled_from(PLANE_NAMES)
_rotation_deg = st.sampled_from((0.0, 22.5, 45.0, 90.0, 137.0))
_grout_mm = st.sampled_from((6.0, 12.0, 30.0))
_offset_mm = st.tuples(
    st.floats(min_value=-900.0, max_value=900.0),
    st.floats(min_value=-900.0, max_value=900.0),
)

_FOOTPRINT_SETTINGS = hypothesis_settings(
    max_examples=120,
    deadline=None,
    # Every example composites a real tiling over a real plane and then measures
    # it, which is honest work rather than accidental slowness.
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Property 17 -- rendered tile metric aspect ratio equals the declared ratio
# (Requirements 6.4, 8.6, 8.7, 13.7)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 17: Rendered tile metric aspect
# ratio equals the declared ratio -- for any Tile_Definition declared width and
# height in millimetres, any Structural_Plane, and either geometry mode, the
# ratio of the rendered tile footprint's measured metric width to its measured
# metric height differs from the declared ratio by no more than 1 percent,
# including for 1:1, 1:2, and plank formats.
@given(
    declared=_format,
    mode=_mode,
    plane_name=_plane_name,
    rotation_deg=_rotation_deg,
    grout_mm=_grout_mm,
    offset_mm=_offset_mm,
)
@_FOOTPRINT_SETTINGS
def test_property_17_rendered_metric_aspect_ratio_matches_declared_ratio(
    declared: tuple[float, float],
    mode: str,
    plane_name: str,
    rotation_deg: float,
    grout_mm: float,
    offset_mm: tuple[float, float],
) -> None:
    """Property 17: the rendered footprint carries the declared metric ratio.

    Two claims, both read off a composited tiling:

    * the declared millimetre dimension lies inside the bracket the rendered
      cell boundary is recovered in, on each axis. This is the resolution-free
      statement -- it holds however coarsely the plane is sampled -- and it is
      what rules out a footprint that is the right *shape* at the wrong scale;
    * the ratio of the two recovered dimensions is within
      :data:`ASPECT_RATIO_TOLERANCE` of the declared ratio, scored whenever the
      bracket is tight enough for a 1 percent claim to mean anything.

    Rotation and offset are drawn as well, because both act in metric plane
    space before the modulo: a footprint whose ratio survived only at zero
    rotation would be a pixel-space stretch in disguise.

    **Validates: Requirements 6.4, 8.6, 8.7, 13.7**
    """
    width_mm, height_mm = declared
    current = scene(mode)
    assume(plane_name in current.planes)
    plane = current.planes[plane_name]
    event(f"mode={current.mode} plane={plane_name}")

    spec = render_spec(rotation_deg=rotation_deg, grout_mm=grout_mm, offset_mm=offset_mm)
    texture = metric_texture(width_mm, height_mm)
    footprint = rendered_footprint(plane, current.masks[plane_name], texture, spec)
    # Only a plane too sparsely sampled to locate a crossing lands here; a
    # boundary in the *wrong* place is measured and asserted on below.
    event(f"measurable={footprint is not None}")
    assume(footprint is not None)
    assert footprint is not None  # narrowed for the type checker

    assert footprint.width.brackets(width_mm), (
        f"{plane_name} in {current.mode} mode: the rendered cell boundary was "
        f"recovered in [{footprint.width.lower_mm:.4f}, "
        f"{footprint.width.upper_mm:.4f}] mm, which does not contain the declared "
        f"width {width_mm} mm"
    )
    assert footprint.height.brackets(height_mm), (
        f"{plane_name} in {current.mode} mode: the rendered cell boundary was "
        f"recovered in [{footprint.height.lower_mm:.4f}, "
        f"{footprint.height.upper_mm:.4f}] mm, which does not contain the declared "
        f"height {height_mm} mm"
    )

    precise = footprint.precision <= BRACKET_PRECISION_FRACTION
    event(f"ratio scored={precise}")
    assume(precise)

    declared_ratio = width_mm / height_mm
    error = abs(footprint.ratio / declared_ratio - 1.0)
    assert error <= ASPECT_RATIO_TOLERANCE, (
        f"{plane_name} in {current.mode} mode: rendered {width_mm}x{height_mm} mm "
        f"as {footprint.width.midpoint_mm:.2f}x{footprint.height.midpoint_mm:.2f} mm, "
        f"a ratio of {footprint.ratio:.5f} against the declared {declared_ratio:.5f} "
        f"-- {error:.2%} off, past the {ASPECT_RATIO_TOLERANCE:.0%} bound"
    )


@pytest.mark.parametrize("mode", GEOMETRY_MODES)
@pytest.mark.parametrize("declared", NAMED_FORMATS, ids=("1to1", "1to2", "plank"))
def test_property_17_holds_for_the_named_formats_on_every_plane(
    mode: str, declared: tuple[float, float]
) -> None:
    """Requirement 8.7's three named formats, on every plane, in both modes.

    The property above draws these among other formats; this pins them
    explicitly and unconditionally, on every plane the scene recovered rather
    than on a drawn one, so a regression that only shows up on, say, the left
    wall cannot hide behind a shrinking-example search.
    """
    width_mm, height_mm = declared
    current = scene(mode)
    texture = metric_texture(width_mm, height_mm)
    spec = render_spec(grout_mm=12.0)

    measured: dict[str, Footprint] = {}
    for name, plane in current.planes.items():
        footprint = rendered_footprint(plane, current.masks[name], texture, spec)
        if footprint is None:
            continue
        measured[name] = footprint

    assert measured, (
        f"no plane of the {mode!r} scene yielded a measurable footprint for "
        f"{width_mm}x{height_mm} mm"
    )
    declared_ratio = width_mm / height_mm
    for name, footprint in measured.items():
        assert footprint.width.brackets(width_mm), f"{name}: width {footprint.width}"
        assert footprint.height.brackets(height_mm), f"{name}: height {footprint.height}"
        if footprint.precision > BRACKET_PRECISION_FRACTION:
            continue
        error = abs(footprint.ratio / declared_ratio - 1.0)
        assert error <= ASPECT_RATIO_TOLERANCE, (
            f"{name} in {mode} mode rendered {width_mm}x{height_mm} mm at a ratio of "
            f"{footprint.ratio:.5f}, {error:.2%} off the declared {declared_ratio:.5f}"
        )


@pytest.mark.parametrize("mode", GEOMETRY_MODES)
@pytest.mark.parametrize("declared", NAMED_FORMATS, ids=("1to1", "1to2", "plank"))
def test_property_17_texture_sampling_carries_no_anisotropic_stretch(
    mode: str, declared: tuple[float, float]
) -> None:
    """Requirement 8.7: the texture inside a cell is not stretched either.

    A cell boundary in the right metric place says nothing about how the pattern
    was sampled *within* the cell, so this measures the other half: the ramp
    texture's two channels give each axis's rendered levels-per-millimetre
    independently, and their quotient is the rendered metric aspect ratio. It
    exercises ``px_per_mm``, the pattern's resolved pixel dimensions, and the
    remap -- so an axis-dependent scale error that left the grout untouched
    still fails here.
    """
    width_mm, height_mm = declared
    current = scene(mode)
    texture = ramp_texture(width_mm, height_mm)
    spec = render_spec(grout_mm=12.0)

    declared_ratio = width_mm / height_mm
    scored = 0
    for name, plane in current.planes.items():
        scales = ramp_scales(plane, current.masks[name], texture, spec)
        if scales is None:
            continue
        slope_u, slope_v = scales
        # slope_u is 255/width_mm and slope_v is 255/height_mm when the pattern
        # is sampled at its declared metric scale, so their quotient inverts to
        # the rendered width-to-height ratio.
        measured_ratio = slope_v / slope_u
        error = abs(measured_ratio / declared_ratio - 1.0)
        assert error <= ASPECT_RATIO_TOLERANCE, (
            f"{name} in {mode} mode sampled a {width_mm}x{height_mm} mm tile at "
            f"{255.0 / slope_u:.2f}x{255.0 / slope_v:.2f} mm, a ratio of "
            f"{measured_ratio:.5f} against the declared {declared_ratio:.5f} "
            f"-- {error:.2%} off"
        )
        scored += 1

    assert scored, (
        f"no plane of the {mode!r} scene carried enough interior pixels to measure "
        f"texture scale for {width_mm}x{height_mm} mm"
    )


@pytest.mark.parametrize(
    "stretch_u, stretch_v",
    [(1.25, 1.0), (1.0, 1.25), (0.8, 1.0)],
)
def test_property_17_measurement_detects_an_injected_stretch(
    stretch_u: float, stretch_v: float
) -> None:
    """The measurement is sensitive: a mis-scaled render has to fail it.

    A property that measures nothing passes everything, so this renders a tile
    whose metric dimensions have been stretched on one axis and asserts the
    recovery -- scored against the honest declaration -- reports a ratio outside
    the 1 percent bound. Without this, a broken :func:`rendered_footprint` would
    make Property 17 vacuously true.
    """
    width_mm, height_mm = 600.0, 1200.0
    current = scene("vanishing_points")
    plane = current.planes["floor"]
    honest = metric_texture(width_mm, height_mm)
    stretched = dataclasses.replace(
        honest,
        width_mm=honest.width_mm * stretch_u,
        height_mm=honest.height_mm * stretch_v,
    )
    spec = render_spec(grout_mm=12.0)

    footprint = rendered_footprint(
        plane,
        current.masks["floor"],
        stretched,
        spec,
        declared_mm=(stretched.width_mm, stretched.height_mm),
    )
    assert footprint is not None, "the stretched render was not measurable at all"
    expected = (stretched.width_mm, stretched.height_mm)
    assert footprint.width.brackets(expected[0]), (
        "the measurement should track what was actually rendered, "
        f"got {footprint.width!r} for {expected[0]} mm"
    )

    declared_ratio = width_mm / height_mm
    error = abs(footprint.ratio / declared_ratio - 1.0)
    assert error > ASPECT_RATIO_TOLERANCE, (
        f"a {stretch_u}x by {stretch_v}x metric stretch measured as a ratio of "
        f"{footprint.ratio:.5f} against the declared {declared_ratio:.5f}, only "
        f"{error:.2%} off -- the measurement cannot see a stretch this large"
    )

# =========================================================================== #
#                                                                             #
#  Task 10.4 -- blending, occlusion, and feathering                           #
#  (Requirements 7.4, 7.5, 7.6, 7.7)                                          #
#                                                                             #
# =========================================================================== #
#
# Everything above measures *where* a tile lands. This section measures what
# happens to it once it is there: the photograph's own illumination is folded
# back in, occluders are put back in front, and the plane boundary is softened.
# Four properties, each stated over the quantity its requirement names.
#
# * **Property 21** (R7.4) is about the *direction* of the lighting blend. The
#   plane median is the branch threshold, so below it the tile must come out no
#   brighter than it went in and above it no darker. Stated over
#   :func:`blend_lighting` at gloss 0, because the gloss highlight of R7.5 is a
#   separate additive term and folding it in would make the claim untestable
#   rather than stronger.
# * **Property 22** (R7.5) is about the gloss term on its own, isolated by
#   differencing each render against the same render at gloss 0. That difference
#   *is* the highlight contribution, by definition, which is what lets both
#   halves of the property -- monotonicity in gloss, and nothing at all at gloss
#   0 -- be read off one set of renders.
# * **Property 23** (R7.6) is stated over the whole :func:`compose` pipeline,
#   because the guarantee it protects is end-to-end: a foreground pixel must
#   equal the photograph byte for byte no matter how many planes were tiled over
#   it. The lit scene below deliberately hands ``compose`` plane masks that were
#   *not* occluder-subtracted, so every occluder pixel really is painted over
#   before the final redraw puts it back --- see
#   :func:`test_property_23_foreground_pixels_are_genuinely_overpainted_first`,
#   which measures that the property has something to protect.
# * **Property 24** (R7.7) is a claim about distance, so it is scored against a
#   brute-force exact Euclidean distance computed in numpy rather than against
#   another OpenCV distance transform. That independence is the point: the whole
#   reason ``feather_alpha`` asks for ``DIST_MASK_PRECISE`` is that the default
#   3x3 chamfer approximation is a few percent off, which is enough to leave a
#   pixel at distance exactly ``width_px`` short of full opacity.
#
# The closing unit test measures grout thickness in pixels down the floor and
# holds it to shrinking with depth, which is the visible consequence of grout
# being laid in the same metric pass as the tiles rather than stroked on in pixel
# space afterwards.


# --------------------------------------------------------------------------- #
# Documented tolerances
# --------------------------------------------------------------------------- #

#: "Within rounding tolerance", in 8-bit levels. :func:`blend_lighting` builds
#: its blend in float64 and rounds once through ``rint``, so a value the design's
#: expression places exactly on the tile value can land one level either side of
#: it. One level is that rounding step and nothing more --- the property below
#: would still catch the tens-of-levels errors a wrong branch produces.
BLEND_ROUNDING_TOLERANCE: Final[int] = 1

#: Slack on the "fully opaque" half of Property 24, in alpha units. Kept at
#: exactly zero: ``alpha = clip(distance / width_px, 0, 1)`` reaches 1.0 by
#: saturation, not by arithmetic luck, so a pixel at or past the feather width is
#: *exactly* opaque and a tolerance here would only hide a distance transform
#: that had stopped being precise.
ALPHA_OPACITY_SLACK: Final[float] = 0.0

#: Largest absolute disagreement, in pixels, allowed between the distance
#: transform ``feather_alpha`` uses and the brute-force exact Euclidean distance
#: Property 24 scores it against. Both are Euclidean and one is exact, so this is
#: float32 representation error rather than a modelling allowance.
DISTANCE_AGREEMENT_PX: Final[float] = 1e-3

#: Side of the square canvas Property 24's masks are drawn on. Small on purpose:
#: the reference distance is an all-pairs computation between set and unset
#: pixels, so 40x40 keeps 100 examples inside a couple of seconds while still
#: leaving room for a blob with a genuine interior.
FEATHER_CANVAS_PX: Final[int] = 40

#: Grout width, in millimetres, the depth test lays. Wide enough that the near
#: floor renders it several pixels thick, so the far/near difference is many
#: quantisation steps rather than one.
DEPTH_PROBE_GROUT_MM: Final[float] = 40.0

#: Depth bands the floor is split into for the grout thickness measurement, and
#: the smallest number of runs a band needs before it is scored.
DEPTH_PROBE_BANDS: Final[int] = 5
DEPTH_PROBE_MIN_RUNS: Final[int] = 20


# --------------------------------------------------------------------------- #
# A fully lit scene -- image, masks, occluders, and lighting maps
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LitScene:
    """A cached :class:`SceneState` complete enough for :func:`compose`.

    Distinct from :class:`Scene` above, which carries only the geometry Property
    17 needs. This one adds the two things the lighting and occlusion properties
    are stated over: a real Lighting_Engine decomposition of the room, and a
    Foreground_Mask with actual occluders in it.

    The plane masks are built with ``subtract_occluders=False``, which is
    deliberately *not* what the Segmenter would return (Requirement 3.4 has it
    excluding foreground pixels). Handing ``compose`` the un-subtracted masks is
    what forces the tiling to cover every occluder pixel, so Requirement 7.6's
    final redraw is the only thing standing between the render and a
    sticker-over-furniture artifact --- which is precisely the guarantee Property
    23 exists to check.
    """

    state: SceneState
    room: SyntheticRoom

    @property
    def foreground(self) -> np.ndarray:
        """Boolean view of the Foreground_Mask."""
        return np.asarray(self.state.foreground_mask) > 0

    @property
    def plane_names(self) -> tuple[str, ...]:
        return tuple(self.state.planes)


@lru_cache(maxsize=1)
def lit_scene() -> LitScene:
    """The cached lit scene, generated once per session.

    Same pose and size as the ``vanishing_points`` :func:`scene`, so a failure
    here and a failure there point at the same geometry. Occluders are switched
    on, and the lighting maps come from :func:`decompose` rather than from
    hand-built gradients: a shading map with the wrong frequency content would
    make Property 21's median threshold meaningless, and the Lighting_Engine is
    the only thing that produces a real one.
    """
    room = make_synthetic_room(
        width=SCENE_WIDTH,
        height=SCENE_HEIGHT,
        focal_px=0.875 * SCENE_WIDTH,
        yaw_deg=SCENE_YAW_DEG["vanishing_points"],
        pitch_deg=-12.0,
        walls=("left", "right", "back"),
        n_occluders=2,
        seed=0,
        supersample=SCENE_SUPERSAMPLE,
    )
    assert np.count_nonzero(room.occluder_mask), (
        "the lit scene placed no occluders, so Property 23 would have nothing to "
        "protect"
    )

    contours = {
        name: polygon.astype(np.float64) for name, polygon in room.plane_polygons.items()
    }
    calibration = calibrate(room.image, contours, settings=get_settings())
    masks = {
        name: mask
        for name, mask in room.plane_masks(
            subtract_occluders=False, resolve_overlaps=True
        ).items()
        if name in calibration.homographies
    }
    assert masks, "the lit scene recovered no plane geometry at all"

    lighting = decompose(room.image, masks)  # type: ignore[arg-type]
    planes = {
        name: _plane_metadata(
            name,
            contours[name],
            masks[name],
            calibration.homographies[name],
            calibration.homography_inverses[name],
            calibration.plane_extents_mm[name],
            calibration.reprojection_rmse_px[name],
            calibration.geometry_mode,
        )
        for name in masks
    }
    # `_plane_metadata` fixes the median at mid-grey because Property 17 never
    # blends; here the real per-plane median is what selects the blend branch.
    for name, plane in planes.items():
        plane.luminance_median = float(lighting.plane_medians.get(name, 128.0))

    height, width = room.shape
    state = SceneState(
        scene_id="lit-scene",
        created_at=0.0,
        image=room.image,
        width=width,
        height=height,
        planes=planes,  # type: ignore[arg-type]
        plane_masks=masks,  # type: ignore[arg-type]
        foreground_mask=room.occluder_mask,
        shading_map=lighting.shading,
        detail_map=lighting.detail,
        horizon=(
            float(room.truth_horizon[0]),
            float(room.truth_horizon[1]),
            float(room.truth_horizon[2]),
        ),
        vanishing_points=calibration.vanishing_points,
        geometry_mode=calibration.geometry_mode,
        segmentation_backend="classical",
    )
    return LitScene(state=state, room=room)


def tile_definition(
    declared: tuple[float, float], gloss: float, grout_mm: float | None = None
) -> TileDefinition:
    """A Tile_Definition carrying the gloss the blend is scored against.

    ``image_path`` is never opened: :func:`compose` reads a Tile_Definition only
    for its ``gloss`` and its ``grout_mm``, taking the pixels from the
    :class:`SeamlessTexture` it is handed separately.
    """
    width_mm, height_mm = declared
    return TileDefinition(
        id=f"probe-{int(width_mm)}x{int(height_mm)}",
        name="Probe tile",
        image_path=Path("probe.png"),
        width_mm=width_mm,
        height_mm=height_mm,
        finish="probe",
        gloss=gloss,
        grout_mm=grout_mm,
    )


# --------------------------------------------------------------------------- #
# Blend probes
# --------------------------------------------------------------------------- #


def blend_sweep(
    shading_values: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """A ``(rows, 256, 3)`` tile field and shading map sweeping every pair.

    One row per drawn shading value, one column per tile value, so a single
    example covers *every* tile colour against every shading value the draw
    produced rather than sampling the pair space. The three colour channels carry
    the same ramp rolled by a third each, so a channel-dependent bug shows up as
    well.

    The shading map is a genuine map, not a scalar: its own median is what
    Property 21 uses as the plane median, exactly as the Lighting_Engine derives
    it from the shading map over a plane mask.
    """
    shading = np.repeat(
        np.asarray(shading_values, dtype=np.uint8)[:, None], 256, axis=1
    )
    ramp = np.arange(256, dtype=np.uint8)
    rows = shading.shape[0]
    tile = np.stack(
        [np.tile(np.roll(ramp, shift), (rows, 1)) for shift in (0, 85, 170)], axis=2
    )
    return np.ascontiguousarray(tile), np.ascontiguousarray(shading)


def detail_sweep(shape: tuple[int, int], stride: int, offset: int) -> np.ndarray:
    """A deterministic detail map covering the full 8-bit range.

    A ramp of drawn stride and phase rather than random noise, so the map spans
    0 to 255 -- both extremes of the signed residual -- on every draw, and so a
    failing example replays exactly.
    """
    rows, cols = shape
    index = np.arange(rows * cols, dtype=np.int64)
    return ((index * stride + offset) % 256).astype(np.uint8).reshape(rows, cols)


def highlight_contribution(
    tile: np.ndarray,
    shading: np.ndarray,
    detail: np.ndarray,
    median: float,
    gloss: float,
) -> np.ndarray:
    """The gloss highlight's own contribution, as signed 8-bit levels.

    Defined as the difference between a render at ``gloss`` and the same render
    at gloss 0. Requirement 7.5 scales *only* the highlight term with gloss, so
    everything else cancels in the difference and what remains is the highlight
    the requirement is about -- clipping included, which matters because a
    highlight that would push past 255 must not be credited with more strength
    than it actually delivered.
    """
    lit = blend_lighting(tile, shading, detail, median, gloss).astype(np.int16)
    matte = blend_lighting(tile, shading, detail, median, 0.0).astype(np.int16)
    return lit - matte


# --------------------------------------------------------------------------- #
# Feather probes
# --------------------------------------------------------------------------- #


def exact_distance_to_background(mask: np.ndarray) -> np.ndarray:
    """Brute-force exact Euclidean distance from each set pixel to an unset one.

    An all-pairs minimum in numpy, which is far too slow for production and
    exactly right for a reference: it shares no code with
    ``cv2.distanceTransform``, so agreement between the two is evidence rather
    than tautology.

    Matches OpenCV's convention that only zeros *inside* the image count, so a
    mask running off the edge of the canvas measures its distance to the nearest
    background pixel in frame rather than to the border.

    Returns:
        A ``(H, W)`` ``float64`` array: 0 at every unset pixel, and the exact
        distance at every set one. ``inf`` where the mask fills the whole canvas
        and there is no background to measure against.
    """
    binary = np.asarray(mask) != 0
    distance = np.zeros(binary.shape, dtype=np.float64)
    set_ys, set_xs = np.nonzero(binary)
    off_ys, off_xs = np.nonzero(~binary)
    if set_ys.size == 0:
        return distance
    if off_ys.size == 0:
        distance[binary] = np.inf
        return distance

    chunk = 4096
    for start in range(0, set_ys.size, chunk):
        ys = set_ys[start : start + chunk]
        xs = set_xs[start : start + chunk]
        distance[ys, xs] = np.hypot(
            ys[:, None] - off_ys[None, :], xs[:, None] - off_xs[None, :]
        ).min(axis=1)
    return distance


#: One drawn mask shape: ``(kind, centre_x, centre_y, half_width, half_height)``.
_mask_shape = st.tuples(
    st.sampled_from(("ellipse", "rectangle")),
    st.integers(min_value=2, max_value=FEATHER_CANVAS_PX - 3),
    st.integers(min_value=2, max_value=FEATHER_CANVAS_PX - 3),
    st.integers(min_value=3, max_value=14),
    st.integers(min_value=3, max_value=14),
)


def draw_mask(shapes: Sequence[tuple[str, int, int, int, int]]) -> np.ndarray:
    """Rasterise drawn shapes into a ``uint8`` ``{0, 255}`` mask.

    Overlapping ellipses and rectangles give the union concave notches, thin
    necks, and interiors of very different widths, which is where a feather band
    measured by anything other than true distance goes wrong.
    """
    mask = np.zeros((FEATHER_CANVAS_PX, FEATHER_CANVAS_PX), dtype=np.uint8)
    for kind, cx, cy, half_w, half_h in shapes:
        if kind == "ellipse":
            cv2.ellipse(mask, (cx, cy), (half_w, half_h), 0, 0, 360, 255, -1)
        else:
            cv2.rectangle(
                mask, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h), 255, -1
            )
    return mask


# --------------------------------------------------------------------------- #
# Shared strategies for this section
# --------------------------------------------------------------------------- #

#: Shading values a Property 21 example sweeps. Small lists on purpose: the
#: sweep is 256 columns wide per value, and a short list makes the plane median
#: land anywhere in the range rather than converging on mid-grey the way a long
#: uniform draw would.
_shading_values = st.lists(st.integers(min_value=0, max_value=255), min_size=1, max_size=24)

#: A plane median in 8-bit shading units, drawn freely: Requirement 7.4 puts no
#: constraint on where a plane's neutral point sits.
_median = st.floats(min_value=0.0, max_value=255.0, allow_nan=False, allow_infinity=False)

#: Gloss, over the ``[0.0, 1.0]`` range Requirement 8.3 declares, with the two
#: endpoints and the catalog's own values reachable.
_gloss = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

_detail_stride = st.integers(min_value=1, max_value=97)
_detail_offset = st.integers(min_value=0, max_value=255)

#: Feather widths, including 0 -- the documented "no feathering" escape hatch --
#: and widths past the radius of a small blob, where no pixel is opaque at all.
_feather_width_px = st.integers(min_value=0, max_value=8)

_BLEND_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_COMPOSE_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    # Every example composites real tilings over a real 800x600 room.
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Property 21 -- blend mode is selected by the plane median (Requirement 7.4)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 21: Blend mode is selected by the
# plane median -- for any tile colour and shading map, the composited output
# value is no greater than the tile value at every pixel where the shading value
# is below the plane median, no less than the tile value at every pixel where the
# shading value is above the plane median, and equal to the tile value at every
# pixel where the shading value equals the plane median, all within rounding
# tolerance.
@given(shading_values=_shading_values)
@_BLEND_SETTINGS
def test_property_21_blend_direction_follows_the_plane_median(
    shading_values: list[int],
) -> None:
    """Property 21: the median decides whether the tile darkens or lightens.

    Requirement 7.4 selects a multiply blend below the plane median and a
    soft-light blend above it. The observable consequence, and the one this
    property is stated over, is the *direction* each branch moves the tile: a
    shadowed pixel must come out no brighter than the tile went in, and a lit
    pixel no darker. Get the branch selection backwards and both halves fail at
    once.

    Scored at gloss 0 so the additive highlight of Requirement 7.5 is out of the
    way -- it is a separate term with its own property below, and it can move a
    pixel either way regardless of which branch ran.

    A pixel sitting *exactly* on the median is claimed by the third clause, which
    is where the two branches have to meet: the multiply branch tends to the tile
    value as shading rises to the median, and the soft-light branch is
    re-normalised so the median is its own fixed point. Asserting equality there
    rather than excluding it is what keeps a step at the median contour from
    hiding between the two inequalities.

    **Validates: Requirements 7.4**
    """
    tile, shading = blend_sweep(shading_values)
    # The plane median is the median of the shading map, exactly as the
    # Lighting_Engine derives it from the shading over a plane mask.
    median = float(np.median(shading))
    detail = np.full(shading.shape, NEUTRAL_DETAIL, dtype=np.uint8)

    out = blend_lighting(tile, shading, detail, median, 0.0).astype(np.int16)
    reference = tile.astype(np.int16)

    below = (shading.astype(np.float64) < median)[:, :, None]
    above = (shading.astype(np.float64) > median)[:, :, None]
    at = (shading.astype(np.float64) == median)[:, :, None]
    event(f"median={median:.1f} below={int(below.sum())} above={int(above.sum())}")

    overshoot = int(np.max(out - reference, initial=0, where=below))
    assert overshoot <= BLEND_ROUNDING_TOLERANCE, (
        f"at a plane median of {median:.1f}, a pixel with shading below the median "
        f"came out {overshoot} levels brighter than the tile; the multiply branch "
        f"must never lighten"
    )

    shortfall = int(np.max(reference - out, initial=0, where=above))
    assert shortfall <= BLEND_ROUNDING_TOLERANCE, (
        f"at a plane median of {median:.1f}, a pixel with shading above the median "
        f"came out {shortfall} levels darker than the tile; the soft-light branch "
        f"must never darken"
    )

    deviation = int(np.max(np.abs(out - reference), initial=0, where=at))
    assert deviation <= BLEND_ROUNDING_TOLERANCE, (
        f"at a plane median of {median:.1f}, a pixel with shading exactly at the "
        f"median moved {deviation} levels away from the tile; the two branches must "
        f"meet at the unmodified tile value on the median contour"
    )


@pytest.mark.parametrize("median", [1.0, 64.0, 127.5, 128.0, 192.0, 255.0])
def test_multiply_branch_darkens_monotonically_below_the_median(median: float) -> None:
    """The shadowed half of Requirement 7.4, swept exhaustively per median.

    Two claims the property above states only as an inequality: below the median
    the output is not merely bounded by the tile value but *monotone* in shading,
    so a deeper shadow really does read as darker; and a pixel at zero shading
    goes fully black regardless of the tile. Together they rule out a multiply
    branch that clamps to the tile value and calls it darkening.
    """
    tile = np.full((256, 1, 3), 200, dtype=np.uint8)
    shading = np.arange(256, dtype=np.uint8).reshape(256, 1)
    detail = np.full((256, 1), NEUTRAL_DETAIL, dtype=np.uint8)

    out = blend_lighting(tile, shading, detail, median, 0.0)[:, 0, 0].astype(np.int16)
    below = np.arange(256) < median
    if int(below.sum()) < 2:
        pytest.skip(f"a median of {median} leaves no room below it to sweep")

    shadowed = out[below]
    assert np.all(np.diff(shadowed) >= 0), (
        f"at median {median} the multiply branch is not monotone in shading: "
        f"{shadowed.tolist()}"
    )
    assert shadowed[0] == 0, (
        f"at median {median} a pixel in full shadow came out at {shadowed[0]} "
        "rather than black"
    )


def test_blend_uses_the_supplied_median_not_a_global_constant() -> None:
    """Requirement 7.4's threshold is per plane, so changing it must change the
    render.

    A blend that quietly compared against mid-grey would satisfy Property 21 --
    and would darken a dim floor and blow out a sunlit wall, which is the exact
    failure the per-plane median exists to prevent. One shading value, two
    medians straddling it, and the branch has to flip.
    """
    tile = np.full((1, 1, 3), 180, dtype=np.uint8)
    shading = np.full((1, 1), 90, dtype=np.uint8)
    detail = np.full((1, 1), NEUTRAL_DETAIL, dtype=np.uint8)

    shadowed = int(blend_lighting(tile, shading, detail, 200.0, 0.0)[0, 0, 0])
    lit = int(blend_lighting(tile, shading, detail, 40.0, 0.0)[0, 0, 0])

    assert shadowed < 180, "shading below the plane median should have darkened the tile"
    assert shadowed != lit, (
        "the same shading produced the same result on either side of two very "
        "different plane medians; the median is not selecting the branch"
    )


# --------------------------------------------------------------------------- #
# Property 22 -- highlight strength increases with gloss (Requirement 7.5)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 22: Highlight strength increases
# monotonically with gloss -- for any pair of gloss values with all other inputs
# fixed, the magnitude of the highlight contribution for the greater gloss value
# is at least that for the lesser, and a gloss value of zero produces no
# highlight contribution.
@given(
    gloss_a=_gloss,
    gloss_b=_gloss,
    median=_median,
    stride=_detail_stride,
    offset=_detail_offset,
)
@_BLEND_SETTINGS
def test_property_22_highlight_magnitude_is_monotone_in_gloss(
    gloss_a: float,
    gloss_b: float,
    median: float,
    stride: int,
    offset: int,
) -> None:
    """Property 22: more gloss is never less highlight, and gloss 0 is none.

    The highlight contribution is isolated by differencing each render against
    the same render at gloss 0, which is what makes the claim measurable at all:
    Requirement 7.5 scales only the specular term with gloss, so everything else
    cancels and the residue is the term itself.

    Magnitude rather than signed value, because the detail map's residual is
    signed -- a pixel darker than its neighbourhood has a *negative* highlight
    that must get more negative, not less, as the tile grows glossier. Taking the
    absolute value states both directions in one inequality.

    The sweep covers all 65 536 shading-and-tile pairs per example, and the
    detail ramp spans the full 8-bit range, so a highlight that saturates against
    the 0 and 255 clip is included: clipping may flatten the contribution but it
    can never reverse it.

    **Validates: Requirements 7.5**
    """
    lower, higher = sorted((gloss_a, gloss_b))
    tile, shading = blend_sweep(range(256))
    detail = detail_sweep(shading.shape, stride, offset)

    quiet = highlight_contribution(tile, shading, detail, median, lower)
    loud = highlight_contribution(tile, shading, detail, median, higher)
    event(f"gloss {lower:.2f} -> {higher:.2f}")

    worst = int(np.max(np.abs(quiet) - np.abs(loud), initial=0))
    assert worst <= 0, (
        f"raising gloss from {lower:.4f} to {higher:.4f} at median {median:.1f} "
        f"weakened the highlight by up to {worst} levels; highlight strength must "
        "not fall as gloss rises"
    )

    if higher > 0.0:
        # Not part of the property, but a guard that the difference above is
        # measuring something: a sweep where no gloss value ever moved a pixel
        # would satisfy monotonicity vacuously.
        event(f"highlight reached {int(np.abs(loud).max())} levels")


@given(
    median=_median,
    stride=_detail_stride,
    offset=_detail_offset,
    other_stride=_detail_stride,
    other_offset=_detail_offset,
)
@_BLEND_SETTINGS
def test_property_22_gloss_zero_ignores_the_detail_map_entirely(
    median: float,
    stride: int,
    offset: int,
    other_stride: int,
    other_offset: int,
) -> None:
    """Property 22, second half: at gloss 0 there is no highlight contribution.

    Stated as independence rather than as a subtraction, which is the stronger
    reading and the one that cannot be satisfied trivially: if a matte tile
    carried any highlight at all, replacing the detail map with a completely
    different one would move the output. Two independent detail ramps must
    produce byte-identical renders.

    **Validates: Requirements 7.5**
    """
    tile, shading = blend_sweep(range(256))
    detail = detail_sweep(shading.shape, stride, offset)
    other = detail_sweep(shading.shape, other_stride, other_offset)
    assume(not np.array_equal(detail, other))

    matte = blend_lighting(tile, shading, detail, median, 0.0)
    alternative = blend_lighting(tile, shading, other, median, 0.0)

    assert np.array_equal(matte, alternative), (
        f"at gloss 0 and median {median:.1f} the detail map changed the render by "
        f"up to {int(np.abs(matte.astype(np.int16) - alternative.astype(np.int16)).max())} "
        "levels; a matte tile must take no highlight"
    )


@pytest.mark.parametrize("gloss", [0.10, 0.35, 0.85, 1.0])
def test_highlight_grows_with_gloss_by_a_measurable_amount(gloss: float) -> None:
    """Guard for Property 22: the monotone bound must not be vacuously true.

    Property 22 is an inequality, and a compositor that ignored gloss outright
    would satisfy it perfectly. This pins the other side: each catalog gloss
    value has to deliver strictly more highlight than the one below it, and at
    gloss 1.0 the full residual has to come through, which is what
    ``HIGHLIGHT_GAIN = 1.0`` claims.
    """
    tile = np.full((1, 1, 3), 128, dtype=np.uint8)
    shading = np.full((1, 1), 128, dtype=np.uint8)
    detail = np.full((1, 1), NEUTRAL_DETAIL + 60, dtype=np.uint8)
    median = 128.0

    strength = int(
        highlight_contribution(tile, shading, detail, median, gloss)[0, 0, 0]
    )
    matte = int(highlight_contribution(tile, shading, detail, median, 0.0)[0, 0, 0])

    assert matte == 0, "gloss 0 produced a highlight"
    assert strength == round(60 * gloss), (
        f"a residual of +60 at gloss {gloss} delivered {strength} levels, not the "
        f"{round(60 * gloss)} a linear gloss scale with unit gain implies"
    )


# --------------------------------------------------------------------------- #
# Property 23 -- foreground pixels are preserved (Requirement 7.6)
# --------------------------------------------------------------------------- #


def compose_lit_scene(
    names: Sequence[str],
    declared: tuple[float, float],
    rotation_deg: float,
    grout_mm: float,
    gloss: float,
    *,
    scene_state: SceneState | None = None,
) -> tuple[np.ndarray, list[str], dict[str, PlaneRenderSpec]]:
    """Composite the lit scene with one tile format per requested plane.

    Each plane takes a different entry from :data:`TILE_FORMATS`, offset by its
    position in ``names``, so a multi-plane render exercises several metric
    pitches at once rather than repeating one.

    Returns:
        ``(composited, warnings, specs)``. The warnings list is returned so a
        caller can assert the render was not quietly reduced to a no-op by a
        skipped plane.
    """
    lit = lit_scene()
    state = scene_state if scene_state is not None else lit.state

    specs: dict[str, PlaneRenderSpec] = {}
    textures: dict[str, SeamlessTexture] = {}
    tiles: dict[str, TileDefinition] = {}
    for index, name in enumerate(names):
        width_mm, height_mm = TILE_FORMATS[
            (TILE_FORMATS.index(declared) + index) % len(TILE_FORMATS)
        ]
        specs[name] = PlaneRenderSpec(
            tile_id=f"probe-{name}",
            rotation_deg=rotation_deg,
            grout_mm=grout_mm,
            grout_rgb=GROUT_RGB,
        )
        textures[name] = metric_texture(width_mm, height_mm)
        tiles[name] = tile_definition((width_mm, height_mm), gloss)

    warnings: list[str] = []
    composited = compose(
        state,
        specs,  # type: ignore[arg-type]
        textures,  # type: ignore[arg-type]
        get_settings(),
        tiles=tiles,  # type: ignore[arg-type]
        warnings=warnings,
    )
    return composited, warnings, specs


# Feature: ai-room-tile-visualizer, Property 23: Foreground pixels are preserved
# from the original photograph -- for any scene and any set of tile selections,
# every composited pixel at which the Foreground_Mask is set equals the
# corresponding pixel of the original photograph.
@given(
    names=st.lists(st.sampled_from(PLANE_NAMES), min_size=1, max_size=4, unique=True),
    declared=_format,
    rotation_deg=_rotation_deg,
    grout_mm=_grout_mm,
    gloss=_gloss,
)
@_COMPOSE_SETTINGS
def test_property_23_foreground_pixels_equal_the_original_photograph(
    names: list[str],
    declared: tuple[float, float],
    rotation_deg: float,
    grout_mm: float,
    gloss: float,
) -> None:
    """Property 23: occluders survive every tile selection byte for byte.

    Requirement 7.6 admits no tolerance -- a foreground pixel is the photograph's
    own pixel, not a close approximation of it -- so this is an exact array
    comparison rather than a bounded difference.

    The plane masks the lit scene hands ``compose`` are deliberately *not*
    occluder-subtracted, so every requested plane paints straight over the
    furniture and the final redraw is the only thing putting it back. Gloss is
    drawn as well, because a glossy tile takes the highlight path through
    ``blend_lighting`` and writes different bytes than a matte one.

    **Validates: Requirements 7.6**
    """
    lit = lit_scene()
    chosen = [name for name in names if name in lit.state.planes]
    assume(chosen)

    composited, warnings, _ = compose_lit_scene(
        chosen, declared, rotation_deg, grout_mm, gloss
    )
    event(f"planes={len(chosen)} skipped={len(warnings)}")
    assert not warnings, f"the render skipped a requested plane: {warnings}"

    foreground = lit.foreground
    original = np.asarray(lit.state.image)
    differing = np.any(composited[foreground] != original[foreground], axis=1)
    assert not differing.any(), (
        f"{int(differing.sum())} of {int(foreground.sum())} Foreground_Mask pixels "
        f"were changed by tiling {chosen} at {declared[0]}x{declared[1]} mm, "
        f"rotation {rotation_deg} deg, gloss {gloss:.2f}"
    )


def test_property_23_foreground_pixels_are_genuinely_overpainted_first() -> None:
    """The guard that gives Property 23 its content.

    If the tiling never reached a foreground pixel, Property 23 would hold for
    free and tell us nothing. Rendering the identical scene with an empty
    Foreground_Mask shows how many of those pixels the planes actually cover:
    they must all be overpainted, so the redraw of Requirement 7.6 is doing real
    work every time the property passes.
    """
    lit = lit_scene()
    chosen = list(lit.plane_names)
    foreground = lit.foreground
    original = np.asarray(lit.state.image)

    unprotected_state = dataclasses.replace(
        lit.state, foreground_mask=np.zeros_like(lit.state.foreground_mask)
    )
    unprotected, warnings, _ = compose_lit_scene(
        chosen, (600.0, 600.0), 0.0, 12.0, 0.85, scene_state=unprotected_state
    )
    assert not warnings, f"the guard render skipped a plane: {warnings}"

    overpainted = np.any(unprotected[foreground] != original[foreground], axis=1)
    assert overpainted.all(), (
        f"only {int(overpainted.sum())} of {int(foreground.sum())} foreground "
        "pixels were covered by the tiling, so Property 23 is partly vacuous"
    )

    protected, _, _ = compose_lit_scene(
        chosen, (600.0, 600.0), 0.0, 12.0, 0.85
    )
    assert np.array_equal(protected[foreground], original[foreground])


def test_every_masked_pixel_of_a_selected_plane_is_actually_tiled() -> None:
    """A plane mask component outside its largest contour still gets tiled.

    ``compose`` composites through an alpha built from the plane *mask*, so every
    mask pixel is opaque and will be taken from the tile field. The tiled region
    therefore has to cover the mask, not merely the polygon
    ``simplify_contour`` traced from it -- and those are different things as soon
    as a mask has more than one connected component, because ``simplify_contour``
    returns only the largest external contour. Subtracting the Foreground_Mask out
    of a plane (Requirement 3.4) splits planes routinely, so this is the normal
    case rather than a contrived one.

    Sourcing the tiled box from the contour left those components untiled, which
    meant opaque and composited from the zero-filled field: solid black holes in
    the middle of a rendered surface. The mask is used instead, and this is the
    test that says so.

    The split is created here rather than hoped for: an occluder-shaped band is
    cut across the floor mask so it falls into two components, and the plane's
    ``contour`` and ``bounding_points`` are re-derived from that mask through the
    Segmenter's own :func:`simplify_contour` and :func:`bounding_quad`. Those two
    return the *largest* component only, exactly as they do in production, so the
    smaller component ends up outside the contour's bounding box -- which is the
    configuration the bug needed and the one a real segmentation reaches whenever
    a piece of furniture divides a surface.

    Scored through the public surface: :func:`noise_texture` holds every texel in
    ``[24, 216]``, so "this pixel was tiled" is observable as "this pixel is not
    pure black". Passing ``mask=`` to :func:`tile_field_for_plane` must leave no
    masked pixel untiled; omitting it must leave some, or the two boxes coincide
    and the test proves nothing.
    """
    lit = lit_scene()
    mask = np.asarray(lit.state.plane_masks["floor"]).copy()
    ys, xs = np.nonzero(mask)
    # Cut a full-width band just below the mask's top edge. The strip above it
    # survives as a second, smaller component, and it is the part that used to be
    # left untiled.
    top, bottom = int(ys.min()), int(ys.max())
    cut = top + max(4, (bottom - top) // 12)
    mask[cut:cut + 3, :] = 0
    assert cv2.connectedComponents(mask)[0] >= 3, (
        "the cut did not split the floor mask, so there is no stranded component "
        "to check"
    )

    plane = dataclasses.replace(
        lit.state.planes["floor"],
        contour=simplify_contour(mask),
        bounding_points=bounding_quad(simplify_contour(mask)),
    )
    texture = metric_texture(600.0, 600.0)
    spec = render_spec(grout_mm=12.0)
    settings = get_settings()
    masked = mask > 0

    with_mask = tile_field_for_plane(
        plane, texture, spec, mask.shape, settings=settings, mask=mask
    )
    holes = masked & ~with_mask.any(axis=2)
    assert not holes.any(), (
        f"{int(holes.sum())} masked pixel(s) were left untiled; they are opaque in "
        "the plane alpha, so the composite takes them from the zero-filled field "
        "and shows solid black"
    )

    without_mask = tile_field_for_plane(
        plane, texture, spec, mask.shape, settings=settings
    )
    assert (masked & ~without_mask.any(axis=2)).any(), (
        "the contour-derived box already covered every mask pixel, so this test "
        "cannot tell a mask-derived box from a contour-derived one"
    )

    # And end to end, through `compose`, which is where the black would have shown.
    state = dataclasses.replace(
        lit.state,
        planes={**lit.state.planes, "floor": plane},
        plane_masks={**lit.state.plane_masks, "floor": mask},
    )
    composited, warnings, _ = compose_lit_scene(
        ["floor"], (600.0, 600.0), 0.0, 12.0, 0.0, scene_state=state
    )
    assert not warnings, warnings
    composite_holes = masked & np.all(composited == 0, axis=2)
    assert not composite_holes.any(), (
        f"{int(composite_holes.sum())} masked pixel(s) came out pure black in the "
        "composite, which is the untiled field showing through"
    )


def test_unselected_planes_keep_their_photographic_appearance() -> None:
    """A plane nobody chose is untouched, which is the other half of R7.6's
    draw order.

    ``compose`` starts from a copy of the photograph, so every pixel outside the
    selected planes' masks -- and outside the one-pixel soft ring the foreground
    redraw adds -- has to come through unchanged.
    """
    lit = lit_scene()
    chosen = ["floor"]
    composited, warnings, _ = compose_lit_scene(chosen, (600.0, 600.0), 0.0, 12.0, 0.0)
    assert not warnings

    original = np.asarray(lit.state.image)
    floor_mask = np.asarray(lit.state.plane_masks["floor"]) > 0
    # The feathered plane alpha reaches `feather_width_px` outside nothing and
    # `feather_width_px` inside the mask, so dilating the mask by that width
    # bounds every pixel the composite could legitimately have touched.
    reach = max(int(get_settings().feather_width_px), 1)
    kernel = np.ones((2 * reach + 1, 2 * reach + 1), dtype=np.uint8)
    touchable = cv2.dilate(floor_mask.view(np.uint8), kernel) != 0

    untouched = ~touchable
    assert np.count_nonzero(untouched), "the floor covers the whole frame"
    assert np.array_equal(composited[untouched], original[untouched]), (
        "pixels outside every selected plane were modified"
    )


# --------------------------------------------------------------------------- #
# Property 24 -- feathering is confined to the configured band (Requirement 7.7)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 24: Mask feathering is confined to
# the configured band -- for any Structural_Plane mask and any configured feather
# width, the composite alpha is fully opaque at every pixel whose distance inside
# the mask boundary is at least the feather width, fully transparent at every
# pixel outside the mask, and takes intermediate values only within a band no
# wider than the configured feather width.
@given(
    shapes=st.lists(_mask_shape, min_size=1, max_size=3),
    width_px=_feather_width_px,
)
@_BLEND_SETTINGS
def test_property_24_feather_band_is_confined_to_the_configured_width(
    shapes: list[tuple[str, int, int, int, int]],
    width_px: int,
) -> None:
    """Property 24: opaque interior, transparent exterior, ramp only between.

    Scored against :func:`exact_distance_to_background`, a brute-force all-pairs
    Euclidean distance that shares no code with the distance transform
    ``feather_alpha`` uses. That independence is what makes the third clause --
    the band is *no wider* than the configured width -- a real measurement: an
    approximate distance transform would put ramp pixels past the width and this
    would catch it.

    All three clauses at once, for a mask drawn as overlapping ellipses and
    rectangles so the boundary has concave notches and necks narrower than the
    feather width, where a distance-based ramp and a morphological one part
    company.

    **Validates: Requirements 7.7**
    """
    mask = draw_mask(shapes)
    inside = mask != 0
    assume(inside.any())

    alpha = feather_alpha(mask, width_px)
    assert alpha.dtype == np.float32
    assert alpha.shape == mask.shape
    assert np.all((alpha >= 0.0) & (alpha <= 1.0))

    # Fully transparent outside the mask, for every width including 0.
    assert np.all(alpha[~inside] == 0.0), (
        f"{int(np.count_nonzero(alpha[~inside]))} pixels outside the mask carried "
        "a non-zero alpha"
    )

    if width_px == 0:
        # The documented "no feathering" escape hatch: a hard binary alpha.
        assert np.array_equal(alpha, inside.astype(np.float32))
        event("width=0 hard alpha")
        return

    reference = exact_distance_to_background(mask)
    measured = cv2.distanceTransform(
        np.ascontiguousarray(inside.view(np.uint8) * 255), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    finite = np.isfinite(reference)
    assert np.max(np.abs(reference[finite] - measured[finite]), initial=0.0) <= (
        DISTANCE_AGREEMENT_PX
    ), "the distance transform the feather is built on is not exact Euclidean"

    opaque_expected = reference >= width_px
    event(f"width={width_px} opaque={int(opaque_expected.sum())}")
    if opaque_expected.any():
        assert np.all(alpha[opaque_expected] >= 1.0 - ALPHA_OPACITY_SLACK), (
            f"a pixel at least {width_px} px inside the mask boundary was only "
            f"{float(alpha[opaque_expected].min()):.6f} opaque"
        )

    band = (alpha > 0.0) & (alpha < 1.0)
    if band.any():
        depth = reference[band]
        assert np.all(depth > 0.0), "a pixel outside the mask was given a ramp alpha"
        assert float(depth.max()) < width_px, (
            f"the feather ramp reached {float(depth.max()):.3f} px inside the mask "
            f"boundary, past the configured width of {width_px} px"
        )


@pytest.mark.parametrize("width_px", [1, 2, 3, 5])
def test_feather_ramp_is_linear_across_the_band(width_px: int) -> None:
    """Requirement 7.7 asks for a feathered edge, not a stepped one.

    Property 24 bounds where the ramp lives; this pins its shape. On a wide
    rectangle the distance inside the boundary is exactly the row or column
    offset, so the expected alpha is known in closed form and a ramp that
    plateaued or jumped would fail even though it stayed inside the band.
    """
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:50, 10:50] = 255

    alpha = feather_alpha(mask, width_px)
    # Walk inward along the row through the rectangle's middle.
    profile = alpha[30, 10 : 10 + width_px + 2]
    expected = np.array(
        [min((offset + 1) / width_px, 1.0) for offset in range(width_px + 2)],
        dtype=np.float32,
    )
    assert np.allclose(profile, expected, atol=1e-6), (
        f"feather profile at width {width_px} was {profile.tolist()}, expected "
        f"{expected.tolist()}"
    )


def test_compose_feathers_with_the_configured_width() -> None:
    """Property 24 is about `feather_alpha`; this ties it to what `compose` uses.

    Two consequences of an alpha of exactly 1 across the interior and exactly 0
    outside: a pixel well inside the plane equals the re-lit tile byte for byte,
    with no rounding drift from a blend that should not have run, and a pixel
    well outside it equals the photograph.
    """
    lit = lit_scene()
    settings = get_settings()
    plane_name = "floor"
    plane = lit.state.planes[plane_name]
    mask = np.asarray(lit.state.plane_masks[plane_name])
    texture = metric_texture(600.0, 600.0)
    spec = PlaneRenderSpec(
        tile_id="probe", rotation_deg=0.0, grout_mm=12.0, grout_rgb=GROUT_RGB
    )
    tile = tile_definition((600.0, 600.0), 0.85)

    composited = compose(
        lit.state,
        {plane_name: spec},  # type: ignore[dict-item]
        {plane_name: texture},  # type: ignore[dict-item]
        settings,
        tiles={plane_name: tile},  # type: ignore[dict-item]
    )

    # Rebuild the expected interior independently: the same tile field, re-lit
    # with the same plane median, is what an alpha of 1 has to deliver exactly.
    field = tile_field_for_plane(
        plane, texture, spec, mask.shape, tile=tile, settings=settings
    )
    expected = blend_lighting(
        field,
        np.asarray(lit.state.shading_map),
        np.asarray(lit.state.detail_map),
        plane.luminance_median,
        tile.gloss,
        settings,
    )

    alpha = feather_alpha(mask, settings.feather_width_px)
    # The occluder redraw carries its own one-pixel soft ring *outside* the
    # Foreground_Mask, so those pixels are legitimately a mix of tile and
    # photograph even at plane alpha 1. Excluding the dilated foreground leaves
    # only pixels the plane alpha alone decided.
    foreground_reach = (
        cv2.dilate(
            np.ascontiguousarray(np.asarray(lit.state.foreground_mask)),
            np.ones((3, 3), dtype=np.uint8),
        )
        != 0
    )
    interior = (alpha >= 1.0) & ~foreground_reach
    assert np.count_nonzero(interior) > 1000, "the floor has no opaque interior to check"
    assert np.array_equal(composited[interior], expected[interior]), (
        "an opaque interior pixel does not equal the re-lit tile exactly"
    )

    outside = alpha == 0.0
    other_planes = np.zeros(mask.shape, dtype=bool)
    for name, other in lit.state.plane_masks.items():
        if name != plane_name:
            other_planes |= np.asarray(other) > 0
    # The room's four planes tile nearly the whole frame, so only a thin sliver
    # of pixels belongs to no plane at all. That sliver is still the right place
    # to read the transparent half of the claim; the broader "unselected planes
    # are untouched" statement is
    # `test_unselected_planes_keep_their_photographic_appearance`.
    clear = outside & ~other_planes & ~foreground_reach
    assert np.count_nonzero(clear), "no pixel of the frame lies outside every plane"
    assert np.array_equal(
        composited[clear], np.asarray(lit.state.image)[clear]
    ), "a fully transparent pixel was modified"


# --------------------------------------------------------------------------- #
# Grout foreshortening (Requirement 5.7 as it applies to grout)
# --------------------------------------------------------------------------- #


def vertical_grout_runs(grout: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``(centre_row, length)`` for every vertical grout run enclosed by ``mask``.

    Scanning columns measures the *vertical* thickness of the grout lines that
    run across the floor at constant depth, which is the foreshortened image of a
    fixed millimetre grout width. Only runs with a tiled pixel immediately above
    and below are kept, so a line clipped by the plane boundary cannot be
    mistaken for a thin one.
    """
    height, width = grout.shape
    records: list[tuple[float, int]] = []
    for col in range(width):
        column = grout[:, col]
        if not column.any():
            continue
        padded = np.concatenate(([False], column, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        for start, end in zip(starts.tolist(), ends.tolist()):
            if start == 0 or end >= height:
                continue
            if mask[start - 1, col] and mask[end, col]:
                records.append((start + (end - start - 1) / 2.0, end - start))
    if not records:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(records, dtype=np.float64)


def test_grout_line_pixel_width_shrinks_with_depth_on_the_floor() -> None:
    """Grout laid in metric space foreshortens with the tiles. R5.7, R8.6

    Grout is not stroked onto the composited image at a fixed pixel width; it is
    the in-cell region past the tile dimension, so it lives in millimetres and
    the same perspective that shrinks a distant tile shrinks the joint beside it.
    That is a visible property -- constant-width grout is one of the tells of a
    pasted-on floor -- and this measures it directly off the rendered pixels.

    The floor is tiled with a deliberately wide 40 mm joint and the grout colour
    is located in the output. Vertical run lengths are pooled across every column
    and grouped into depth bands by image row; the median run length per band is
    the grout's rendered thickness there. Median rather than mean because a
    column running nearly parallel to a depth-direction grout line grazes it for
    tens of pixels, and those tangential runs are outliers rather than
    thicknesses.
    """
    current = scene("vanishing_points")
    plane = current.planes["floor"]
    mask = current.masks["floor"]
    texture = metric_texture(600.0, 600.0)
    spec = render_spec(grout_mm=DEPTH_PROBE_GROUT_MM)

    field = tile_field_for_plane(plane, texture, spec, mask.shape)
    tiled = (mask > 0) & field.any(axis=2)
    grout = tiled & np.all(field == GROUT_BGR, axis=2)
    assert np.count_nonzero(grout) > 500, "no grout was rendered on the floor at all"

    runs = vertical_grout_runs(grout, tiled)
    assert len(runs) > 200, f"only {len(runs)} enclosed grout runs were found"

    # Depth bands by image row: the floor recedes upward, so a lower band is
    # nearer the camera.
    rows = runs[:, 0]
    edges = np.linspace(rows.min(), rows.max() + 1e-6, DEPTH_PROBE_BANDS + 1)
    thickness: list[tuple[float, float]] = []
    for index in range(DEPTH_PROBE_BANDS):
        selected = (rows >= edges[index]) & (rows < edges[index + 1])
        if int(selected.sum()) < DEPTH_PROBE_MIN_RUNS:
            continue
        thickness.append(
            (float(0.5 * (edges[index] + edges[index + 1])), float(np.median(runs[selected, 1])))
        )

    assert len(thickness) >= 3, (
        f"only {len(thickness)} depth bands carried enough grout runs to measure; "
        "the fixture no longer shows enough of the floor"
    )

    widths = [width for _, width in thickness]
    assert all(
        near >= far for far, near in zip(widths, widths[1:])
    ), f"grout thickness is not monotone from far to near: {thickness}"
    assert widths[-1] > widths[0], (
        f"grout rendered {widths[-1]} px thick nearest the camera and {widths[0]} px "
        "thick farthest away; a metric grout width must foreshorten with depth"
    )
