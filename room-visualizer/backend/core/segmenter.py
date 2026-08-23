"""Segmenter interface and the shared mask post-processing pipeline.

Two backends produce Structural_Plane proposals -- MobileSAM through onnxruntime
and a pure-OpenCV classical path -- but only one of them may ever be active, and
Requirements 3.3 and 3.4 have to hold either way. So the invariants do not live
in the backends: every backend emits raw :class:`Region` proposals and then hands
them to the module-level functions here.

The pipeline is:

1. :func:`assign_structural_planes` -- score each region against the four plane
   names and award each name to at most one region (Requirements 3.3, 3.5).
2. :func:`enforce_plane_invariants` -- subtract the Foreground_Mask, resolve
   residual overlaps by priority, drop undersized planes (Requirements 3.3, 3.4).
3. :func:`simplify_contour` / :func:`bounding_quad` / :func:`plane_area_fraction`
   -- describe each surviving mask for transport (Requirements 3.6, 1.3).

Steps 1 and 2 are pure subtraction over binary masks, so disjointness and
foreground exclusion hold *by construction* rather than by assertion. That is the
whole reason the enforcement pass is shared code and not a backend concern.

Requirements: 3.3, 3.4, 3.5, 3.6, 4.1.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Final, Mapping, Protocol, Sequence

import cv2
import numpy as np

from backend.config import Settings, get_settings
from backend.core.geometry import detect_line_segments, estimate_horizon_hint
from backend.schemas import PLANE_NAMES, PlaneName, SegmentationBackend

__all__ = [
    "Region",
    "SegmentationResult",
    "Segmenter",
    "ClassicalSegmenter",
    "NeuralSegmenter",
    "InferenceSessionLike",
    "UnsupportedModelSignature",
    "PROMPT_GRID_SIDE",
    "PLANE_PRIORITY",
    "SCORE_FLOOR",
    "FOREGROUND_MIN_COMPONENT_FRACTION",
    "assign_structural_planes",
    "score_regions",
    "enforce_plane_invariants",
    "simplify_contour",
    "bounding_quad",
    "plane_area_fraction",
    "describe_planes",
    "finalize_segmentation",
    "clean_mask",
    "drop_small_components",
    "fill_holes",
    "binarize",
]

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

#: Overlap priority for the invariant pass. A pixel on a shared room edge is
#: awarded to the earliest name in this tuple. The floor leads because a
#: mis-assigned floor pixel is the most visible compositing error, and the back
#: wall precedes the side walls because its silhouette is the one the side walls
#: are cut against. ``tests/fixtures/synthetic.py`` resolves its ground-truth
#: masks with this exact order, so fixture and implementation cannot drift.
PLANE_PRIORITY: Final[tuple[PlaneName, ...]] = (
    "floor",
    "wall_back",
    "wall_left",
    "wall_right",
)

#: Minimum combined score a region must reach before a plane name is awarded to
#: it. Below this the name is omitted from the result entirely rather than
#: returned as an empty mask (Requirement 3.5).
SCORE_FLOOR: Final[float] = 0.45

# Cue weights per plane name. Each cue is normalised to [0, 1] so a score is
# also in [0, 1] and the floor threshold above is comparable across planes.
_W_FLOOR_BELOW: Final[float] = 0.45
_W_FLOOR_LOW: Final[float] = 0.35
_W_FLOOR_CONVERGENCE: Final[float] = 0.20

# Frontality carries the most weight of the back wall's three cues because it is
# the only one that stays sharp under camera yaw: a yawed back wall slides well
# off centre, gutting the `central` cue, but its horizontal edges stay nearly
# level while a side wall's converge hard.
_W_BACK_ABOVE: Final[float] = 0.35
_W_BACK_CENTRAL: Final[float] = 0.20
_W_BACK_FRONTAL: Final[float] = 0.45

# Lateral position leads for the side walls for the mirror-image reason: a side
# wall frequently contains no detectable line segment at all (its texture runs
# almost edge-on to the camera), so its centroid has to be able to carry the
# decision unaided.
_W_SIDE_ABOVE: Final[float] = 0.25
_W_SIDE_LATERAL: Final[float] = 0.35
_W_SIDE_ORIENTATION: Final[float] = 0.40

# Normalised horizontal distance over which the lateral cues saturate. A left
# wall's centroid rarely reaches the image edge, so 0.35 image widths from centre
# is treated as "fully lateral".
_LATERAL_SPAN: Final[float] = 0.35

# The centre span is deliberately wider than the lateral one. Camera yaw
# translates the back wall's centroid sideways without making it any less the
# back wall, and at 20 degrees of yaw that offset reaches ~0.3 image widths; a
# tighter span would zero the cue on a perfectly ordinary photograph.
_CENTRE_SPAN: Final[float] = 0.45

# Image-space slope at which a region's contained segments count as fully
# converging. A frontal back wall's horizontal lines sit near 0; a receding floor
# or side wall easily exceeds this.
_SLOPE_REF: Final[float] = 0.25

# Segments flatter than this cannot be extrapolated to a horizon crossing with
# useful conditioning, so they inform the convergence *magnitude* but not its
# sign.
_MIN_SIGNED_SLOPE: Final[float] = 0.05

# Neutral orientation score used when a region carries no usable segments.
_ORIENTATION_UNKNOWN: Final[float] = 0.5

# Default kernel size for morphological cleanup, as a fraction of the image's
# shorter edge.
_CLEAN_KERNEL_FRAC: Final[float] = 0.01


# --------------------------------------------------------------------------- #
# Mask primitives
# --------------------------------------------------------------------------- #


def binarize(mask: np.ndarray) -> np.ndarray:
    """Return ``mask`` as a fresh contiguous ``uint8`` array of ``{0, 255}``.

    Backends variously produce boolean arrays, ``{0, 1}`` labels, and float
    probabilities already thresholded. Normalising once here means every function
    below can assume one representation, and the ``uint8`` choice satisfies
    Requirement 12.4's 8-bit storage bound.

    Raises:
        ValueError: the array is not 2-D or is empty.
    """
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {arr.shape!r}")
    if arr.size == 0:
        raise ValueError("expected a non-empty mask")
    return np.where(arr > 0, np.uint8(255), np.uint8(0))


def _odd_size(shape: tuple[int, int], fraction: float, minimum: int = 3) -> int:
    """Odd window size that is ``fraction`` of the image's shorter edge.

    Every window this module opens -- morphological, blurring, or box-filtering
    -- is expressed as a fraction of the frame rather than in absolute pixels, so
    a 2048 px upload and its 400 px thumbnail are processed at the same relative
    scale (Requirement 2.7's downscale must not change what the segmenter sees).
    """
    short_edge = min(shape[0], shape[1])
    size = max(minimum, int(round(fraction * short_edge)))
    return size + 1 if size % 2 == 0 else size


def _kernel_for(shape: tuple[int, int], kernel_frac: float) -> np.ndarray:
    """Odd-sized elliptical structuring element scaled to the image."""
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_odd_size(shape, kernel_frac), _odd_size(shape, kernel_frac))
    )


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill every interior hole of ``mask``, keeping its outer silhouette.

    Implemented by refilling the external contours rather than by a border flood
    fill, so a component touching the image edge is filled just like an interior
    one -- which matters because a plane region almost always runs off frame.
    """
    binary = binarize(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return binary
    filled = np.zeros_like(binary)
    cv2.drawContours(filled, list(contours), -1, 255, thickness=cv2.FILLED)
    return filled


def drop_small_components(mask: np.ndarray, min_fraction: float) -> np.ndarray:
    """Remove connected components smaller than ``min_fraction`` of the frame.

    Args:
        mask: 2-D mask; any non-zero pixel counts as set.
        min_fraction: area threshold as a fraction of total image pixels. Values
            at or below zero return the mask unchanged apart from binarisation.
    """
    binary = binarize(mask)
    if min_fraction <= 0.0:
        return binary
    total = float(binary.shape[0] * binary.shape[1])
    min_pixels = min_fraction * total
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for label in range(1, count):
        if float(stats[label, cv2.CC_STAT_AREA]) >= min_pixels:
            out[labels == label] = 255
    return out


def clean_mask(
    mask: np.ndarray,
    *,
    kernel_frac: float = _CLEAN_KERNEL_FRAC,
    min_component_fraction: float = 0.0,
    close_holes: bool = False,
) -> np.ndarray:
    """Morphologically tidy a raw proposal mask.

    Opening first removes the speckle that k-means and point-prompt proposals
    both leave behind; closing afterwards welds the resulting gaps back up. Doing
    it in that order matters -- closing first would grow the speckle into blobs
    large enough to survive the component filter.

    Args:
        mask: raw 2-D proposal mask.
        kernel_frac: structuring-element size as a fraction of the shorter edge.
        min_component_fraction: drop components below this fraction of the frame.
        close_holes: also fill interior holes, for foreground silhouettes where a
            gap in the middle of a sofa is always an artifact.
    """
    binary = binarize(mask)
    if not binary.any():
        return binary
    kernel = _kernel_for(binary.shape, kernel_frac)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if close_holes:
        binary = fill_holes(binary)
    if min_component_fraction > 0.0:
        binary = drop_small_components(binary, min_component_fraction)
    return binary


def plane_area_fraction(mask: np.ndarray) -> float:
    """Set pixels of ``mask`` over its total pixels, in ``[0, 1]``. R3.6"""
    binary = binarize(mask)
    total = binary.shape[0] * binary.shape[1]
    return float(int(np.count_nonzero(binary)) / float(total))


# --------------------------------------------------------------------------- #
# Region proposals
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Region:
    """One backend-agnostic region proposal fed to the shared pipeline.

    ``segments`` carries the line segments the Geometry_Engine already detected
    for this photograph, as an ``(N,4)`` array of ``(x1, y1, x2, y2)`` endpoints.
    Passing them in rather than re-detecting per region is what lets line
    detection run exactly once per upload while still giving
    :func:`assign_structural_planes` its orientation cue. ``None`` and an empty
    array both mean "no orientation evidence", which leaves the orientation cues
    *neutral* rather than asserting frontality -- a region with no contained
    segments is unmeasured, not flat, and scoring it as flat is what would let a
    segment-free side wall outscore a genuine back wall.
    """

    mask: np.ndarray
    segments: np.ndarray | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        self.mask = binarize(self.mask)
        if self.segments is not None:
            segments = np.asarray(self.segments, dtype=np.float32).reshape(-1, 4)
            self.segments = segments if len(segments) else None


@dataclass(slots=True, frozen=True)
class _RegionFeatures:
    """Cues extracted once per region and reused across all four plane scores."""

    area: int
    centroid_x: float
    centroid_y: float
    below_fraction: float
    convergence: float
    converges_right: float  # 1.0 right, 0.0 left, 0.5 undetermined
    has_orientation: bool  # False when no contained segment could be measured
    below_cue: float  # "is a floor" vertical evidence, in [0, 1]
    above_cue: float  # "is a wall" vertical evidence, in [0, 1]


def _segment_cues(
    region: Region,
    centroid_x: float,
    horizon_y: float,
) -> tuple[float, float]:
    """Convergence magnitude and direction from the segments inside ``region``.

    A segment belongs to the region when its midpoint falls on the mask. Only
    segments flatter than 45 degrees are considered: steeper ones are the room's
    vertical structure, which converges toward the *vertical* vanishing point and
    says nothing about whether a surface is a floor, a side wall, or frontal.

    Returns:
        ``(convergence, converges_right, has_orientation)`` where convergence is
        the length-weighted mean absolute image slope normalised to ``[0, 1]``,
        ``converges_right`` is 1.0 when the contained segments meet the horizon
        to the right of the region centroid, 0.0 to the left, and 0.5 when no
        segment is steep enough to extrapolate, and ``has_orientation`` is False
        when the region contained no measurable segment at all.

        The last flag matters: a convergence of 0.0 means "measured, and flat"
        when ``has_orientation`` is True but "unknown" when it is False, and the
        two must not score alike.
    """
    segments = region.segments
    if segments is None or len(segments) == 0:
        return 0.0, _ORIENTATION_UNKNOWN, False

    height, width = region.mask.shape
    x1, y1, x2, y2 = (segments[:, i] for i in range(4))
    mid_x = 0.5 * (x1 + x2)
    mid_y = 0.5 * (y1 + y2)
    col = np.clip(np.rint(mid_x), 0, width - 1).astype(np.intp)
    row = np.clip(np.rint(mid_y), 0, height - 1).astype(np.intp)
    inside = region.mask[row, col] > 0
    if not inside.any():
        return 0.0, _ORIENTATION_UNKNOWN, False

    dx = (x2 - x1)[inside]
    dy = (y2 - y1)[inside]
    mid_x = mid_x[inside]
    mid_y = mid_y[inside]
    length = np.hypot(dx, dy)

    # Flatter than 45 degrees, and not exactly vertical (which would divide by 0).
    flat = (np.abs(dy) < np.abs(dx)) & (length > 0.0)
    if not flat.any():
        return 0.0, _ORIENTATION_UNKNOWN, False

    slope = dy[flat] / dx[flat]
    weight = length[flat]
    total_weight = float(weight.sum())
    if total_weight <= 0.0:  # pragma: no cover - guarded by `length > 0` above
        return 0.0, _ORIENTATION_UNKNOWN, False

    mean_abs_slope = float(np.abs(slope) @ weight / total_weight)
    convergence = float(np.clip(mean_abs_slope / _SLOPE_REF, 0.0, 1.0))

    steep = np.abs(slope) >= _MIN_SIGNED_SLOPE
    if not steep.any():
        # Every contained segment is near-horizontal: the surface reads as
        # frontal, so there is no convergence *side* to report -- but the
        # measurement itself succeeded, which is what the True below records.
        return convergence, _ORIENTATION_UNKNOWN, True

    # Extrapolate each segment to the horizon row. The median crossing is the
    # region's apparent horizontal vanishing point; which side of the centroid it
    # falls on is the sign the design uses to separate wall_left from wall_right.
    crossing = mid_x[flat][steep] + (horizon_y - mid_y[flat][steep]) / slope[steep]
    crossing = crossing[np.isfinite(crossing)]
    if len(crossing) == 0:  # pragma: no cover - finite by construction
        return convergence, _ORIENTATION_UNKNOWN, True
    vp_x = float(np.median(crossing))
    if abs(vp_x - centroid_x) < 1.0:
        return convergence, _ORIENTATION_UNKNOWN, True
    return convergence, (1.0 if vp_x > centroid_x else 0.0), True


def _features(
    region: Region,
    horizon_y: float,
    height: int,
    horizon_in_frame: bool,
    raw_horizon_y: float,
) -> _RegionFeatures | None:
    """Extract every scoring cue for one region, or ``None`` if it is empty.

    ``horizon_in_frame`` selects how the vertical cue is measured, implementing
    the design's first cue -- centroid position relative to the horizon hint,
    combined with where the region's mass is concentrated.

    When the horizon row lies inside the image, the fraction of the region's mass
    below it is the sharpest discriminator available: a floor lies wholly below
    the horizon while a wall straddles it.

    When the horizon falls outside the frame -- a steeply pitched camera -- that
    fraction saturates to 1.0 for *every* region and separates nothing. The cue
    then degrades to a soft signed offset of the centroid from the unclamped
    horizon row. Using the unclamped value matters: a horizon off the *top* of
    the frame means every visible surface lies below it, which is real evidence
    that the frame is floor-dominated, and clamping to row 0 would throw exactly
    that away.
    """
    mask = region.mask
    rows, cols = np.nonzero(mask)
    area = int(rows.size)
    if area == 0:
        return None

    centroid_x = float(cols.mean())
    centroid_y = float(rows.mean())
    below_fraction = float(np.count_nonzero(rows > horizon_y) / area)
    convergence, converges_right, has_orientation = _segment_cues(
        region, centroid_x, horizon_y
    )

    if horizon_in_frame:
        below_cue = below_fraction
    else:
        below_cue = 0.5 + (centroid_y - raw_horizon_y) / max(height, 1)
    below_cue = float(np.clip(below_cue, 0.0, 1.0))
    above_cue = 1.0 - below_cue

    return _RegionFeatures(
        area=area,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        below_fraction=below_fraction,
        convergence=convergence,
        converges_right=converges_right,
        has_orientation=has_orientation,
        below_cue=below_cue,
        above_cue=above_cue,
    )


def _plane_scores(features: _RegionFeatures, shape: tuple[int, int]) -> dict[PlaneName, float]:
    """Score one region against all four plane names, each in ``[0, 1]``.

    The three cues are the design's: vertical position relative to the horizon
    hint, dominant contained-segment orientation with its convergence sign, and
    horizontal centroid position.
    """
    height, width = shape
    cx = features.centroid_x / max(width - 1, 1)
    cy = features.centroid_y / max(height - 1, 1)
    below = features.below_cue
    above = features.above_cue
    convergence = features.convergence

    lateral_left = float(np.clip((0.5 - cx) / _LATERAL_SPAN, 0.0, 1.0))
    lateral_right = float(np.clip((cx - 0.5) / _LATERAL_SPAN, 0.0, 1.0))
    central = float(np.clip(1.0 - abs(cx - 0.5) / _CENTRE_SPAN, 0.0, 1.0))

    # Orientation cues are *evidence-gated*. Without a measurable contained
    # segment there is no convergence sign and no frontality reading, so both
    # side-wall and back-wall orientation terms fall back to neutral and the
    # decision rests on position alone. Asserting frontality here instead -- the
    # obvious `1 - convergence` with convergence defaulting to 0 -- is what makes
    # a segment-free side wall impersonate the back wall.
    right_bias = features.converges_right
    if features.has_orientation:
        # Both the sign *and* the strength of convergence have to be present for
        # a side-wall claim: a region whose contained edges are level is not a
        # side wall no matter which way its centroid leans. Multiplying the side
        # bias by the convergence magnitude is what encodes that, and it is the
        # single cue that most reliably separates a yawed back wall -- whose
        # centroid can sit far off centre -- from a genuine side wall.
        orient_left = right_bias * convergence
        orient_right = (1.0 - right_bias) * convergence
        frontality = 1.0 - convergence
    else:
        orient_left = _ORIENTATION_UNKNOWN
        orient_right = _ORIENTATION_UNKNOWN
        frontality = _ORIENTATION_UNKNOWN

    return {
        "floor": _W_FLOOR_BELOW * below + _W_FLOOR_LOW * cy + _W_FLOOR_CONVERGENCE * convergence,
        "wall_back": (
            _W_BACK_ABOVE * above + _W_BACK_CENTRAL * central + _W_BACK_FRONTAL * frontality
        ),
        "wall_left": (
            _W_SIDE_ABOVE * above
            + _W_SIDE_LATERAL * lateral_left
            + _W_SIDE_ORIENTATION * orient_left
        ),
        "wall_right": (
            _W_SIDE_ABOVE * above
            + _W_SIDE_LATERAL * lateral_right
            + _W_SIDE_ORIENTATION * orient_right
        ),
    }


def _normalised_horizon(horizon_y_hint: float, height: int) -> float:
    """Clamp the horizon hint into the frame, defaulting to mid-height."""
    try:
        value = float(horizon_y_hint)
    except (TypeError, ValueError):
        value = float("nan")
    if not np.isfinite(value):
        value = (height - 1) / 2.0
    return float(np.clip(value, 0.0, max(height - 1, 0)))


def score_regions(
    regions: Sequence[Region],
    horizon_y_hint: float,
    shape: tuple[int, int],
) -> list[dict[PlaneName, float]]:
    """Per-region plane scores, aligned index-for-index with ``regions``.

    Exposed separately from :func:`assign_structural_planes` because the neural
    backend needs to know which proposals failed *all four* structural scores in
    order to union them into the Foreground_Mask, and recomputing the scores
    there would let the two paths disagree.

    Empty regions yield an all-zero score map rather than being dropped, so the
    returned list always matches the input length.
    """
    height, width = int(shape[0]), int(shape[1])
    horizon_y = _normalised_horizon(horizon_y_hint, height)

    # `_normalised_horizon` clamps into the frame, so compare against the raw
    # hint to learn whether the true horizon was ever inside it. A clamped
    # horizon sitting exactly on the first or last row is degenerate in the same
    # way as one off frame entirely: every region lands on one side of it.
    try:
        raw_horizon = float(horizon_y_hint)
    except (TypeError, ValueError):
        raw_horizon = float("nan")
    horizon_in_frame = bool(
        np.isfinite(raw_horizon) and 0.0 < raw_horizon < float(height - 1)
    )
    if not np.isfinite(raw_horizon):
        raw_horizon = horizon_y

    zero: dict[PlaneName, float] = {name: 0.0 for name in PLANE_NAMES}
    out: list[dict[PlaneName, float]] = []
    for region in regions:
        features = _features(region, horizon_y, height, horizon_in_frame, raw_horizon)
        out.append(dict(zero) if features is None else _plane_scores(features, (height, width)))
    return out


def assign_structural_planes(
    regions: Sequence[Region],
    horizon_y_hint: float,
    shape: tuple[int, int],
) -> dict[PlaneName, np.ndarray]:
    """Award each plane name to at most one region by greedy assignment.

    Every ``(region, plane)`` pair is scored, then the pairs are consumed in
    descending score order; a pair is accepted only when neither its plane name
    nor its region has already been claimed. One-to-one in both directions is
    what makes the returned masks disjoint before enforcement ever runs, which is
    the primary mechanism for Requirement 3.3 --
    :func:`enforce_plane_invariants` is the guarantee of last resort.

    A plane name whose best candidate scores below :data:`SCORE_FLOOR` is omitted
    from the result rather than mapped to an empty mask (Requirement 3.5).

    Args:
        regions: candidate proposals, each carrying a mask of shape ``shape``.
        horizon_y_hint: cheap horizon row from ``estimate_horizon_hint``;
            non-finite values fall back to mid-height.
        shape: ``(height, width)`` of the photograph.

    Returns:
        Mapping from plane name to a ``uint8`` ``{0, 255}`` mask. Keys are
        ordered by :data:`PLANE_PRIORITY` so downstream iteration is stable.
    """
    height, width = int(shape[0]), int(shape[1])
    scores = score_regions(regions, horizon_y_hint, (height, width))

    # (-score, plane index, region index) sorts by descending score with a fully
    # deterministic tie-break, so the same photograph always yields the same
    # assignment regardless of dict iteration details.
    candidates = [
        (-scores[r_idx][plane], p_idx, r_idx, plane)
        for r_idx in range(len(regions))
        for p_idx, plane in enumerate(PLANE_PRIORITY)
        if scores[r_idx][plane] >= SCORE_FLOOR
    ]
    candidates.sort()

    claimed_planes: dict[PlaneName, int] = {}
    claimed_regions: set[int] = set()
    for _, _, r_idx, plane in candidates:
        if plane in claimed_planes or r_idx in claimed_regions:
            continue
        claimed_planes[plane] = r_idx
        claimed_regions.add(r_idx)

    assigned: dict[PlaneName, np.ndarray] = {}
    for plane in PLANE_PRIORITY:
        r_idx = claimed_planes.get(plane)
        if r_idx is None:
            continue
        mask = regions[r_idx].mask
        if mask.shape != (height, width):
            raise ValueError(
                f"region mask shape {mask.shape!r} does not match image shape {(height, width)!r}"
            )
        assigned[plane] = mask.copy()
    return assigned


# --------------------------------------------------------------------------- #
# Invariant enforcement
# --------------------------------------------------------------------------- #


def enforce_plane_invariants(
    plane_masks: dict[PlaneName, np.ndarray],
    foreground: np.ndarray,
    *,
    min_area_fraction: float | None = None,
    settings: Settings | None = None,
    kernel_frac: float = _CLEAN_KERNEL_FRAC,
) -> dict[PlaneName, np.ndarray]:
    """Make the plane masks a genuine partition of the non-foreground pixels.

    Four deterministic passes, in this order:

    1. Morphological cleanup of each mask -- open then close, so proposal
       speckle and pinholes do not survive as one-pixel plane fragments.
    2. Subtract the Foreground_Mask from every plane mask (Requirement 3.4).
    3. Resolve residual overlaps in :data:`PLANE_PRIORITY` order, subtracting
       each higher-priority mask from every lower-priority one (Requirement 3.3).
    4. Drop planes whose remaining area falls under ``min_area_fraction``.

    Cleanup runs *first* on purpose: passes 2 and 3 are pure subtraction, so
    nothing after cleanup can reintroduce an overlap or a foreground pixel. Were
    the order reversed, a closing operation could dilate one plane back across a
    boundary it had just been cut from.

    Args:
        plane_masks: assigned masks, keyed by plane name. Not mutated.
        foreground: ``(H, W)`` Foreground_Mask; any non-zero pixel is excluded.
        min_area_fraction: area floor below which a plane is dropped. Defaults to
            ``settings.min_plane_area_fraction``.
        settings: settings to read the default area floor from; defaults to
            :func:`get_settings`.
        kernel_frac: structuring-element size for pass 1, as a fraction of the
            image's shorter edge. Zero disables cleanup.

    Returns:
        A new mapping containing only the surviving planes, ordered by
        :data:`PLANE_PRIORITY`. Masks are pairwise disjoint and intersect the
        foreground nowhere.

    Raises:
        ValueError: a plane mask's shape does not match ``foreground``.
    """
    fg = binarize(foreground)
    shape = fg.shape

    if min_area_fraction is None:
        min_area_fraction = (settings or get_settings()).min_plane_area_fraction

    cleaned: dict[PlaneName, np.ndarray] = {}
    for plane in PLANE_PRIORITY:
        raw = plane_masks.get(plane)
        if raw is None:
            continue
        mask = binarize(raw)
        if mask.shape != shape:
            raise ValueError(
                f"plane {plane!r} mask shape {mask.shape!r} does not match "
                f"foreground shape {shape!r}"
            )
        if kernel_frac > 0.0:
            mask = clean_mask(mask, kernel_frac=kernel_frac)
        cleaned[plane] = mask

    # Pass 2: foreground exclusion. R3.4
    for mask in cleaned.values():
        mask[fg > 0] = 0

    # Pass 3: priority-ordered overlap resolution. R3.3
    claimed = np.zeros(shape, dtype=bool)
    for plane in PLANE_PRIORITY:
        mask = cleaned.get(plane)
        if mask is None:
            continue
        mask[claimed] = 0
        claimed |= mask > 0

    # Pass 4: area floor. A plane that survives assignment but ends up as a
    # sliver behind furniture is not usable for compositing, so it is omitted
    # rather than reported with a near-zero area_fraction.
    total = float(shape[0] * shape[1])
    return {
        plane: mask
        for plane, mask in cleaned.items()
        if int(np.count_nonzero(mask)) / total >= min_area_fraction
    }


# --------------------------------------------------------------------------- #
# Contours and bounding quads
# --------------------------------------------------------------------------- #


def _clip_points(points: np.ndarray, shape: tuple[int, int] | None) -> np.ndarray:
    """Clamp integer image points into ``[0, W-1] x [0, H-1]``."""
    if shape is None:
        return points
    height, width = int(shape[0]), int(shape[1])
    out = points.copy()
    np.clip(out[:, 0], 0, max(width - 1, 0), out=out[:, 0])
    np.clip(out[:, 1], 0, max(height - 1, 0), out=out[:, 1])
    return out


def simplify_contour(mask: np.ndarray, epsilon_frac: float = 0.01) -> np.ndarray:
    """Largest external contour of ``mask``, simplified for JSON transport.

    Epsilon is proportional to the contour's arc length rather than absolute, so
    the same fraction gives a comparable vertex count on a 400 px and a 2048 px
    image (Requirement 3.6).

    The polygon is guaranteed to have at least three points: if the requested
    epsilon collapses the shape, the epsilon is halved until three points
    survive, and a genuinely degenerate contour falls back to its convex hull and
    finally to the corners of its bounding rectangle.

    Args:
        mask: ``(H, W)`` mask with at least one set pixel.
        epsilon_frac: approximation tolerance as a fraction of the arc length.
            Must be positive.

    Returns:
        ``(N, 2) int32`` image points with ``N >= 3``, all inside the mask bounds.

    Raises:
        ValueError: the mask is empty or ``epsilon_frac`` is not positive.
    """
    if epsilon_frac <= 0.0:
        raise ValueError(f"epsilon_frac must be positive, got {epsilon_frac}")
    binary = binarize(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("cannot simplify an empty mask")

    largest = max(contours, key=cv2.contourArea)
    perimeter = float(cv2.arcLength(largest, True))

    approx = largest
    if perimeter > 0.0:
        epsilon = epsilon_frac * perimeter
        while epsilon > 1e-3:
            candidate = cv2.approxPolyDP(largest, epsilon, True)
            if len(candidate) >= 3:
                approx = candidate
                break
            epsilon *= 0.5
        else:  # pragma: no cover - a 3-point contour always survives some epsilon
            approx = largest

    points = np.asarray(approx, dtype=np.int32).reshape(-1, 2)
    if len(points) < 3:
        hull = cv2.convexHull(largest)
        points = np.asarray(hull, dtype=np.int32).reshape(-1, 2)
    if len(points) < 3:
        x, y, w, h = cv2.boundingRect(largest)
        points = np.array(
            [[x, y], [x + w - 1, y], [x + w - 1, y + h - 1], [x, y + h - 1]],
            dtype=np.int32,
        )
    return _clip_points(points, binary.shape)


def _order_quad(points: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left.

    Uses the rotated coordinates ``x + y`` and ``x - y``: their extremes pick out
    the four corners for any convex quad, including the rotated ones
    ``cv2.minAreaRect`` returns in arbitrary winding.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    total = pts[:, 0] + pts[:, 1]
    diff = pts[:, 0] - pts[:, 1]
    order = [int(np.argmin(total)), int(np.argmax(diff)), int(np.argmax(total)), int(np.argmin(diff))]
    if len(set(order)) != 4:
        # Degenerate quad (collinear or duplicated corners): fall back to a
        # simple angular sort about the centroid, which is still deterministic.
        centre = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
        order = list(np.argsort(angles))
    return pts[order]


def bounding_quad(
    contour: np.ndarray,
    shape: tuple[int, int] | None = None,
    *,
    rectangularity: float = 0.85,
) -> np.ndarray:
    """Reduce a contour to exactly four points for UI hit-testing. R1.3, R10.3

    Near-rectangular contours -- which is what a wall or a frontal floor patch
    usually is -- take the ``cv2.minAreaRect`` path, whose corners hug the shape
    tightly. Everything else takes the four extreme convex-hull points, because a
    minimum-area rectangle around, say, an L-shaped floor would bound large
    regions that are not floor at all.

    Args:
        contour: ``(N, 2)`` or ``(N, 1, 2)`` image points, ``N >= 3``.
        shape: optional ``(height, width)`` used to clamp the result in bounds.
            ``minAreaRect`` can place a corner a pixel outside a mask that
            touches the frame edge, and the API contract requires in-bounds
            points.
        rectangularity: hull-area to rect-area ratio at or above which the
            contour counts as near-rectangular.

    Returns:
        ``(4, 2) int32`` points ordered top-left, top-right, bottom-right,
        bottom-left.

    Raises:
        ValueError: fewer than three input points.
    """
    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError(f"bounding_quad needs at least 3 points, got {len(pts)}")

    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    hull_pts = np.asarray(hull, dtype=np.float32).reshape(-1, 2)
    hull_area = float(cv2.contourArea(hull))

    rect = cv2.minAreaRect(hull)
    rect_area = float(rect[1][0] * rect[1][1])

    use_rect = rect_area <= 0.0 or (hull_area / rect_area) >= rectangularity
    if use_rect:
        quad = np.asarray(cv2.boxPoints(rect), dtype=np.float64)
    else:
        quad = _extreme_hull_quad(hull_pts)

    ordered = _order_quad(quad)
    out = np.rint(ordered).astype(np.int32)
    return _clip_points(out, shape)


def _extreme_hull_quad(hull_pts: np.ndarray) -> np.ndarray:
    """Four extreme convex-hull points by rotated coordinates.

    Falls back to the minimum-area rectangle when the extremes do not resolve to
    four distinct points, which happens for hulls that are nearly a triangle.
    """
    total = hull_pts[:, 0] + hull_pts[:, 1]
    diff = hull_pts[:, 0] - hull_pts[:, 1]
    indices = [int(np.argmin(total)), int(np.argmax(diff)), int(np.argmax(total)), int(np.argmin(diff))]
    if len(set(indices)) == 4:
        return hull_pts[indices].astype(np.float64)
    rect = cv2.minAreaRect(hull_pts.reshape(-1, 1, 2))
    return np.asarray(cv2.boxPoints(rect), dtype=np.float64)


def describe_planes(
    plane_masks: dict[PlaneName, np.ndarray],
    *,
    epsilon_frac: float = 0.01,
) -> tuple[
    dict[PlaneName, np.ndarray],
    dict[PlaneName, np.ndarray],
    dict[PlaneName, float],
]:
    """Contours, bounding quads, and area fractions for each plane mask. R3.6

    Returns three mappings whose key sets are identical to ``plane_masks``'
    surviving planes, ordered by :data:`PLANE_PRIORITY`.
    """
    contours: dict[PlaneName, np.ndarray] = {}
    quads: dict[PlaneName, np.ndarray] = {}
    fractions: dict[PlaneName, float] = {}
    for plane in PLANE_PRIORITY:
        mask = plane_masks.get(plane)
        if mask is None:
            continue
        binary = binarize(mask)
        contour = simplify_contour(binary, epsilon_frac)
        contours[plane] = contour
        quads[plane] = bounding_quad(contour, binary.shape)
        fractions[plane] = plane_area_fraction(binary)
    return contours, quads, fractions


# --------------------------------------------------------------------------- #
# Result and interface
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SegmentationResult:
    """Everything one Segmenter run produces for one photograph.

    ``plane_masks`` holds only the planes actually detected; an absent plane is
    absent from every mapping here rather than present with an empty mask
    (Requirement 3.5). The masks are pairwise disjoint and share no pixel with
    ``foreground_mask``, because they come out of
    :func:`enforce_plane_invariants` (Requirements 3.3, 3.4).
    """

    plane_masks: dict[PlaneName, np.ndarray]  # uint8 {0,255}, disjoint   R3.1
    foreground_mask: np.ndarray  # uint8 {0,255}                          R3.2
    contours: dict[PlaneName, np.ndarray]  # (N,2) int32, N >= 3          R3.6
    bounding_points: dict[PlaneName, np.ndarray]  # (4,2) int32           R1.3
    area_fractions: dict[PlaneName, float]  # mask pixels / (H*W)         R3.6
    backend_name: SegmentationBackend  # R4.6

    @property
    def plane_names(self) -> tuple[PlaneName, ...]:
        """Detected plane names in :data:`PLANE_PRIORITY` order."""
        return tuple(plane for plane in PLANE_PRIORITY if plane in self.plane_masks)


def finalize_segmentation(
    plane_masks: dict[PlaneName, np.ndarray],
    foreground: np.ndarray,
    backend_name: SegmentationBackend,
    *,
    min_area_fraction: float | None = None,
    settings: Settings | None = None,
    epsilon_frac: float = 0.01,
    kernel_frac: float = _CLEAN_KERNEL_FRAC,
) -> SegmentationResult:
    """Run enforcement plus description and package a :class:`SegmentationResult`.

    Both backends end their ``segment`` call here, which is what guarantees the
    partition invariants and the contour contract hold identically whichever one
    ran (Requirement 4.1).
    """
    enforced = enforce_plane_invariants(
        plane_masks,
        foreground,
        min_area_fraction=min_area_fraction,
        settings=settings,
        kernel_frac=kernel_frac,
    )
    contours, quads, fractions = describe_planes(enforced, epsilon_frac=epsilon_frac)
    return SegmentationResult(
        plane_masks=enforced,
        foreground_mask=binarize(foreground),
        contours=contours,
        bounding_points=quads,
        area_fractions=fractions,
        backend_name=backend_name,
    )


class Segmenter(ABC):
    """The one segmentation interface both backends implement. R4.1

    Implementations receive nothing but the photograph -- no user annotations, no
    hints from the caller (Requirement 1.5) -- and must return plane masks that
    already satisfy the partition invariants, which they get for free by ending
    in :func:`finalize_segmentation`.
    """

    @property
    @abstractmethod
    def backend_name(self) -> SegmentationBackend:
        """Value reported as ``segmentation_backend`` by the API. R4.6"""
        ...

    @abstractmethod
    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Segment one ``(H, W, 3)`` ``uint8`` BGR photograph. R3.1, R3.2"""
        ...

# --------------------------------------------------------------------------- #
# Classical backend -- OpenCV only, no weights, no network
# --------------------------------------------------------------------------- #

#: Connected-component area floor for the Foreground_Mask, as a fraction of the
#: frame. 0.2 percent of a 2048x1536 upload is about 6 300 px: below that a blob
#: is sensor noise or a colour-cluster fringe, not an occluder worth cutting a
#: tile around. The same floor gates region proposals, so a speck can never
#: become a plane candidate either.
FOREGROUND_MIN_COMPONENT_FRACTION: Final[float] = 0.002

#: Longest edge of the downscaled copy the colour clustering runs on. k-means
#: over a full 2048 px frame costs seconds and buys nothing: the clusters are
#: region-scale, and the label map is upsampled with nearest-neighbour and then
#: morphologically cleaned anyway.
_ANALYSIS_EDGE_PX: Final[int] = 320

#: k for the Lab colour clustering. Three is the design's number and it matches
#: what a room photograph contains at region scale -- a floor, a lit wall and a
#: shaded wall -- which is also exactly why the furniture is held out of the
#: second clustering pass: three clusters have no spare one to spend on a sofa.
#: Where two surfaces still share a cluster, the horizon cut and the subset unions
#: in :func:`_classical_proposals` are what separate them again.
_KMEANS_K: Final[int] = 3
_KMEANS_ATTEMPTS: Final[int] = 3
_KMEANS_MAX_ITER: Final[int] = 20
_KMEANS_EPSILON: Final[float] = 1.0

#: OpenCV's k-means seeds its centres from the global cv2 RNG, so the seed is
#: pinned before every call. Without it two runs over the same photograph can
#: return different plane masks, which would make the whole pipeline
#: irreproducible and Requirement 9.x's cached scene meaningless.
_KMEANS_SEED: Final[int] = 0x5EED

#: Colour-smoothing window on the analysis image, as a fraction of its shorter
#: edge. A tiled floor is two alternating colours; clustering it raw splits one
#: surface into two clusters of checkers. Smoothing at surface scale first is
#: what makes a cluster mean a *surface* rather than a tile.
_ANALYSIS_SMOOTH_FRAC: Final[float] = 0.06

#: Weight applied to Lab's ``L`` channel before clustering. Lightness is the
#: channel that varies *within* a surface -- shading falloff, tile tone, a
#: sunlit patch -- while chroma is closer to the surface's signature, so
#: clustering on raw Lab groups a floor's light tiles with a lit wall instead of
#: with its own dark tiles. Down-weighting L by this factor is what makes one
#: cluster mean one surface. Measured over the synthetic room, floor recovery
#: peaks broadly around 0.3 and stays flat across smoothing scales there, while
#: weights below about 0.2 start losing the occluders into surface clusters.
#: Only clustering is affected: the Foreground_Mask's colour test below measures
#: deviation in unweighted Lab, where an occluder's lightness is its loudest cue.
_CLUSTER_LIGHTNESS_WEIGHT: Final[float] = 0.3

#: Colour-smoothing window for the full-resolution Lab image the residual
#: foreground is measured in. Much tighter than the analysis window: here the
#: goal is to suppress sensor noise while keeping an occluder's silhouette sharp.
_RESIDUAL_SMOOTH_FRAC: Final[float] = 0.008

#: Upper bound on region proposals handed to the shared pipeline. Scoring is
#: O(regions), and past the dozen or so genuine surface fragments a room
#: photograph yields the rest are fringes that lose every greedy contest.
_MAX_PROPOSALS: Final[int] = 20

#: Most components a proposal may be made of. Beyond this it is texture rather
#: than a surface and is dropped, union included.
_MAX_PROPOSAL_COMPONENTS: Final[int] = 4

#: Filled area over convex-hull area a proposal must reach. The other half of the
#: texture rejection: a checkerboard's light squares are diagonally connected, so
#: they pass the component count as a single blob, but they fill only about half
#: their own hull. A real surface, even one with occluders punched out of it, fills
#: most of its hull.
_MIN_PROPOSAL_SOLIDITY: Final[float] = 0.6

#: A plane region smaller than this fraction of the frame carries too few pixels
#: for the robust colour and texture statistics below to mean anything.
_MIN_RESIDUAL_AREA_FRACTION: Final[float] = 0.005

#: Largest share of the frame a candidate occluder may occupy. A fifth of the
#: picture is a generous wardrobe; past that the region is far more likely to be a
#: surface the plane assignment simply failed to name -- a wall whose colour it
#: shared with another, or one seen through a doorway. Erring low here is the
#: right way round: a missed occluder costs one badly composited object, while a
#: wall mistaken for furniture costs the whole surface.
_MAX_FLOATING_AREA_FRACTION: Final[float] = 0.2

#: Share of its own colour cluster a component must hold to read as an object.
#: Furniture is its own colour: a sofa forms a cluster and then fills it. A tile
#: is one of dozens of blobs sharing a cluster, so it holds a small share of it --
#: which separates the two even when a patterned surface slips past the
#: component-count guard because most of its cluster merged into one blob.
_MIN_FLOATING_CLUSTER_SHARE: Final[float] = 0.3

#: Share of a region's outline that may run along the frame edge before it counts
#: as reaching out of the picture rather than floating inside it. Measured over
#: the synthetic room's ground truth, structural planes sit at 0.24 and above --
#: a wall gives at least one whole side to the frame -- while free-standing
#: occluders sit at 0.18 and below, including ones whose base is clipped by the
#: bottom of the frame. A plain "touches the edge at all" test would classify
#: every one of those clipped occluders as a surface, which is why the contact is
#: measured as a proportion of the outline instead.
_MAX_FLOATING_BORDER_CONTACT: Final[float] = 0.2

#: Share of a proposal that may be floating region before the proposal is dropped
#: as furniture rather than surface. Only whole candidates are judged, so a large
#: union that happens to contain a sofa is unaffected -- it is the sofa's *own*
#: component that has to be kept out of the plane contest.
_MAX_FLOATING_OVERLAP: Final[float] = 0.5

#: Pixels sampled when locating a plane's dominant Lab cluster. The centre of a
#: cluster is a mean; a few tens of thousands of pixels fix it as precisely as
#: three million do.
_RESIDUAL_SAMPLE_CAP: Final[int] = 20_000

#: Share of a plane's pixels a colour cluster must hold to count as one of that
#: plane's dominant colours. A tiled floor is genuinely two-toned, and measuring
#: deviation from a single centre would report half of every tiled surface as an
#: occluder. The largest cluster always qualifies, so a plain wall still has
#: exactly one dominant colour.
_DOMINANT_CLUSTER_SHARE: Final[float] = 0.2

#: Colour deviation is flagged past ``median + sigmas * MAD`` of the in-plane
#: deviation distribution -- adaptive, as the design requires, so a busy patterned
#: wall raises its own bar instead of being reported as one large occluder.
_COLOUR_DEVIATION_SIGMAS: Final[float] = 3.0

#: Floor on the MAD spread, in Lab units. A perfectly flat synthetic wall has a
#: near-zero spread, and without this floor its own quantisation noise would clear
#: an adaptive threshold of ``median + 0``.
_MIN_LAB_DEVIATION: Final[float] = 8.0

#: How far a local edge direction may sit from every one of the plane's texture
#: directions before it counts as inconsistent. Generous, because a plane's
#: texture converges: a floor's tile edges sweep through a range of image angles,
#: and the mode set below is what captures that spread.
_TEXTURE_TOLERANCE_RAD: Final[float] = np.pi / 5.0

#: Angle histogram used to find a plane's texture directions, over ``[0, pi)``.
_TEXTURE_ANGLE_BINS: Final[int] = 18
_TEXTURE_MODE_FRACTION: Final[float] = 0.2
_TEXTURE_MAX_MODES: Final[int] = 5

#: Neighbourhood the inconsistent-energy fraction is pooled over.
_TEXTURE_WINDOW_FRAC: Final[float] = 0.02

#: A neighbourhood must be *mostly* inconsistent before it is called foreground,
#: whatever the adaptive threshold says. Texture is the weaker of the two cues --
#: it fires on legitimate perspective convergence too -- so it gets a hard floor.
_TEXTURE_FRACTION_FLOOR: Final[float] = 0.6
_TEXTURE_SIGMAS: Final[float] = 3.0
_TEXTURE_MIN_SPREAD: Final[float] = 0.05

_EPS: Final[float] = 1e-6


def _as_bgr_u8(image: np.ndarray) -> np.ndarray:
    """Coerce an input photograph to a contiguous ``(H, W, 3) uint8`` BGR image.

    The API only ever hands ``segment`` a decoded BGR upload, but tests and the
    Setup_Tool pass grayscale and BGRA arrays, and normalising here keeps every
    function below free of channel checks.

    Raises:
        TypeError: the input is not a numpy array.
        ValueError: the array is empty, not 2-D or 3-D, or has an unusable
            channel count.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected a numpy array, got {type(image)!r}")
    if image.size == 0:
        raise ValueError("expected a non-empty image")
    arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.shape[2] == 1:
            arr = cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] != 3:
            raise ValueError(f"unsupported channel count {arr.shape[2]}")
    else:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {image.shape!r}")
    return np.ascontiguousarray(arr)


def _kmeans_lab(samples: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic Lab k-means over ``(N, 3)`` samples.

    Degenerate inputs -- a flat colour, or fewer distinct colours than ``k`` --
    are the common case on synthetic and low-contrast frames, so they return a
    smaller centre set rather than raising: the caller then simply has fewer
    clusters to propose from.

    Returns:
        ``(labels, centres)`` with ``labels`` of shape ``(N,)`` indexing
        ``centres`` of shape ``(k', 3) float32``, ``1 <= k' <= k``.
    """
    data = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1, 3)
    distinct = np.unique(data, axis=0)
    k_eff = int(min(max(k, 1), len(distinct)))
    if k_eff <= 1:
        centre = distinct[0] if len(distinct) else data.mean(axis=0)
        return (
            np.zeros(len(data), dtype=np.int32),
            np.asarray(centre, dtype=np.float32).reshape(1, 3),
        )

    cv2.setRNGSeed(_KMEANS_SEED)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        _KMEANS_MAX_ITER,
        _KMEANS_EPSILON,
    )
    _, labels, centres = cv2.kmeans(
        data, k_eff, None, criteria, _KMEANS_ATTEMPTS, cv2.KMEANS_PP_CENTERS
    )
    return (
        np.asarray(labels, dtype=np.int32).reshape(-1),
        np.asarray(centres, dtype=np.float32).reshape(-1, 3),
    )


def _cluster_label_map(
    image_bgr: np.ndarray, exclude: np.ndarray | None = None
) -> np.ndarray:
    """Per-pixel dominant-colour cluster index, as an ``(H, W) uint8`` label map.

    Clustering runs on a downscaled, surface-scale-smoothed Lab copy and the
    labels are upsampled with nearest-neighbour interpolation. Both choices say
    the same thing: the label map is a coarse *where is each surface* prior, and
    its boundaries are refined afterwards by morphology and by the shared
    invariant pass -- not by clustering precision.

    Args:
        image_bgr: ``(H, W, 3) uint8`` photograph.
        exclude: optional mask of pixels left out of the *fit*. Every pixel is
            still labelled, by nearest surviving centre. This is how the second
            clustering pass keeps furniture from consuming one of only three
            clusters -- see :meth:`ClassicalSegmenter.segment`.
    """
    height, width = image_bgr.shape[:2]
    scale = min(1.0, _ANALYSIS_EDGE_PX / float(max(height, width)))
    if scale < 1.0:
        small = cv2.resize(
            image_bgr,
            (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image_bgr

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    window = _odd_size(small.shape[:2], _ANALYSIS_SMOOTH_FRAC)
    lab = cv2.GaussianBlur(lab, (window, window), 0)
    lab[:, :, 0] *= _CLUSTER_LIGHTNESS_WEIGHT
    samples = lab.reshape(-1, 3)

    fit = samples
    if exclude is not None and exclude.any():
        mask = cv2.resize(
            np.where(exclude, np.uint8(255), np.uint8(0)),
            (small.shape[1], small.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        keep = mask.reshape(-1) == 0
        if int(np.count_nonzero(keep)) >= _KMEANS_K:
            fit = samples[keep]

    _labels, centres = _kmeans_lab(fit, _KMEANS_K)
    distance = np.linalg.norm(samples[:, None, :] - centres[None, :, :], axis=2)
    label_small = np.argmin(distance, axis=1).reshape(small.shape[:2]).astype(np.uint8)
    if label_small.shape == (height, width):
        return label_small
    return cv2.resize(label_small, (width, height), interpolation=cv2.INTER_NEAREST)


def _residual_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Full-resolution Lab image, lightly smoothed, for colour-deviation tests."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    window = _odd_size(image_bgr.shape[:2], _RESIDUAL_SMOOTH_FRAC)
    return cv2.GaussianBlur(lab, (window, window), 0)


def _gradient_fields(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient magnitude and local *edge* direction of a grayscale frame.

    The returned angle is the gradient direction rotated by ``pi/2`` and reduced
    modulo ``pi``, so it is directly comparable with the line-segment angles the
    Geometry_Engine reports -- a segment's angle *is* an edge direction, while a
    raw image gradient points across the edge.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    edge_angle = np.mod(np.arctan2(gy, gx) + 0.5 * np.pi, np.pi).astype(np.float32)
    return magnitude, edge_angle


def _segments_inside(mask: np.ndarray, segments: np.ndarray | None) -> np.ndarray | None:
    """The subset of ``segments`` whose midpoint falls on ``mask``.

    Same containment test :func:`_segment_cues` uses for scoring, so a plane's
    texture directions are derived from exactly the segments that earned it its
    orientation score.
    """
    if segments is None or len(segments) == 0:
        return None
    height, width = mask.shape
    mid_x = 0.5 * (segments[:, 0] + segments[:, 2])
    mid_y = 0.5 * (segments[:, 1] + segments[:, 3])
    col = np.clip(np.rint(mid_x), 0, width - 1).astype(np.intp)
    row = np.clip(np.rint(mid_y), 0, height - 1).astype(np.intp)
    inside = mask[row, col] > 0
    return segments[inside] if inside.any() else None


def _angle_modes(angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Dominant directions of a weighted ``[0, pi)`` angle population.

    Every bin carrying at least :data:`_TEXTURE_MODE_FRACTION` of the peak weight
    is kept, up to :data:`_TEXTURE_MAX_MODES`. Keeping a *set* rather than a
    single mode is what makes the cue survive perspective: a tiled floor's edges
    fan across several bins, and calling all of them the plane's texture is the
    difference between flagging an occluder and flagging the far half of the
    floor.
    """
    if angles.size == 0:
        return np.empty(0, dtype=np.float32)
    bins = _TEXTURE_ANGLE_BINS
    index = np.minimum((angles / np.pi * bins).astype(np.intp), bins - 1)
    histogram = np.zeros(bins, dtype=np.float64)
    np.add.at(histogram, index, weights)
    peak = float(histogram.max())
    if peak <= 0.0:
        return np.empty(0, dtype=np.float32)
    keep = np.flatnonzero(histogram >= _TEXTURE_MODE_FRACTION * peak)
    keep = keep[np.argsort(histogram[keep])[::-1][:_TEXTURE_MAX_MODES]]
    centres = (keep.astype(np.float64) + 0.5) * (np.pi / bins)
    return centres.astype(np.float32)


def _segment_angle_modes(segments: np.ndarray | None) -> np.ndarray:
    """Texture directions from a plane's contained line segments, length-weighted."""
    if segments is None or len(segments) == 0:
        return np.empty(0, dtype=np.float32)
    dx = segments[:, 2] - segments[:, 0]
    dy = segments[:, 3] - segments[:, 1]
    length = np.hypot(dx, dy)
    usable = length > 0.0
    if not usable.any():
        return np.empty(0, dtype=np.float32)
    angles = np.mod(np.arctan2(dy[usable], dx[usable]), np.pi)
    return _angle_modes(angles, length[usable].astype(np.float64))


def _gradient_angle_modes(
    magnitude: np.ndarray, edge_angle: np.ndarray, inside: np.ndarray
) -> np.ndarray:
    """Texture directions from the plane's own gradients, the segment-free path.

    A side wall seen almost edge-on frequently yields no detectable segment at
    all, yet still has a coherent texture direction. Only above-median-magnitude
    pixels vote, so flat regions do not dilute the histogram with noise angles.
    """
    mags = magnitude[inside]
    if mags.size == 0:
        return np.empty(0, dtype=np.float32)
    strong = mags > float(np.median(mags))
    if not strong.any():
        return np.empty(0, dtype=np.float32)
    angles = edge_angle[inside][strong].astype(np.float64)
    return _angle_modes(angles, mags[strong].astype(np.float64))


def _min_angle_distance(angles: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """Distance from each angle to its nearest mode, respecting the wrap at ``pi``.

    Accumulated mode by mode rather than over an ``(H, W, K)`` broadcast, which at
    2048 px and five modes would allocate hundreds of megabytes.
    """
    best = np.full(angles.shape, np.pi, dtype=np.float32)
    for mode in modes:
        diff = np.abs(angles - float(mode))
        np.minimum(diff, np.pi - diff, out=diff)
        np.minimum(best, diff, out=best)
    return best


def _robust_threshold(values: np.ndarray, sigmas: float, min_spread: float) -> float:
    """``median + sigmas * MAD`` with a floor on the spread.

    The median absolute deviation is used rather than the standard deviation
    because the population being described *contains* the outliers being looked
    for: a sofa covering a fifth of the floor would drag a mean-and-sigma
    threshold up past itself.
    """
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * 1.4826
    return median + sigmas * max(mad, min_spread)


def _dominant_lab_centres(lab: np.ndarray, inside: np.ndarray) -> np.ndarray:
    """The Lab colours that describe a plane, from k-means over its own pixels.

    Three clusters are fitted and those holding at least
    :data:`_DOMINANT_CLUSTER_SHARE` of the plane are kept, the largest always
    among them. On a plain wall that is one colour; on a two-tone tiled floor it
    is two, and the third cluster -- which is where an occluder's pixels land --
    is exactly what gets left out.

    Returns:
        ``(K, 3) float32`` centres, ``1 <= K <= 3``.
    """
    pixels = lab[inside]
    step = max(1, len(pixels) // _RESIDUAL_SAMPLE_CAP)
    labels, centres = _kmeans_lab(pixels[::step], _KMEANS_K)
    counts = np.bincount(labels, minlength=len(centres)).astype(np.float64)
    share = counts / max(counts.sum(), 1.0)
    keep = np.flatnonzero(share >= _DOMINANT_CLUSTER_SHARE)
    if len(keep) == 0:  # pragma: no cover - the largest share always clears 1/3
        keep = np.array([int(np.argmax(counts))], dtype=np.intp)
    return centres[keep]


def _colour_residual(lab: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pixels inside ``mask`` whose Lab colour departs from its dominant colours.

    Deviation is the distance to the *nearest* dominant centre, thresholded
    adaptively against the in-plane deviation distribution, so a busy patterned
    surface raises its own bar rather than being reported as one large occluder.
    The threshold uses a median and a MAD because the population being described
    contains the outliers being looked for.

    Returns:
        Boolean array the shape of ``mask``.
    """
    inside = mask > 0
    if not inside.any():
        return np.zeros(mask.shape, dtype=bool)

    centres = _dominant_lab_centres(lab, inside)
    deviation = np.full(mask.shape, np.inf, dtype=np.float32)
    for centre in centres:
        distance = np.linalg.norm(lab - centre.reshape(1, 1, 3), axis=2)
        np.minimum(deviation, distance, out=deviation)

    threshold = _robust_threshold(
        deviation[inside], _COLOUR_DEVIATION_SIGMAS, _MIN_LAB_DEVIATION
    )
    return inside & (deviation > threshold)


def _texture_residual(
    magnitude: np.ndarray,
    edge_angle: np.ndarray,
    mask: np.ndarray,
    segments: np.ndarray | None,
) -> np.ndarray:
    """Pixels inside ``mask`` whose local edge energy contradicts its texture.

    The plane's projected texture directions come from its contained line
    segments, falling back to its own gradient histogram. Each pixel's edge
    direction is compared against that set, and the *fraction* of local gradient
    energy that is inconsistent -- rather than its absolute amount -- is
    thresholded, so the cue is invariant to how strongly textured the surface is.

    Two gates keep it conservative, because unlike colour this cue also fires on
    legitimate perspective convergence: a neighbourhood must carry above-median
    edge energy to have an opinion at all, and it must be inconsistent past both
    the adaptive threshold and the hard :data:`_TEXTURE_FRACTION_FLOOR`.

    Returns:
        Boolean array the shape of ``mask``.
    """
    inside = mask > 0
    if not inside.any():
        return np.zeros(mask.shape, dtype=bool)

    modes = _segment_angle_modes(_segments_inside(mask, segments))
    if modes.size == 0:
        modes = _gradient_angle_modes(magnitude, edge_angle, inside)
    if modes.size == 0:
        # No texture evidence in either source: the surface is featureless, so
        # this cue abstains and colour deviation decides alone.
        return np.zeros(mask.shape, dtype=bool)

    deviation = _min_angle_distance(edge_angle, modes)
    inconsistent = np.where(deviation > _TEXTURE_TOLERANCE_RAD, magnitude, np.float32(0.0))

    window = _odd_size(mask.shape, _TEXTURE_WINDOW_FRAC, minimum=5)
    local_bad = cv2.boxFilter(inconsistent, -1, (window, window), normalize=True)
    local_all = cv2.boxFilter(magnitude, -1, (window, window), normalize=True)
    fraction = local_bad / np.maximum(local_all, _EPS)

    energy_floor = float(np.median(local_all[inside]))
    threshold = max(
        _robust_threshold(fraction[inside], _TEXTURE_SIGMAS, _TEXTURE_MIN_SPREAD),
        _TEXTURE_FRACTION_FLOOR,
    )
    return inside & (local_all > energy_floor) & (fraction > threshold)


def _horizon_parts(mask: np.ndarray, below: np.ndarray) -> list[np.ndarray]:
    """``mask`` plus the part of it that lies above the horizon hint.

    This is where the design's "colour clusters combined with the line-derived
    horizon hint" happens: a floor and a lit back wall routinely land in one
    colour cluster, and the horizon is what tells them apart.

    The cut is deliberately one-sided. The horizon is the floor plane's line at
    infinity, so *no* floor pixel can image above it -- everything above is wall
    or ceiling, which makes the upper part a sound wall proposal. Below the line
    the two are genuinely mixed, because a wall runs from the horizon all the way
    down to where it meets the floor. Proposing that lower part as well produces a
    band of wall whose mass is entirely below the horizon and whose tile edges
    converge hard, and such a band outscores the real floor as ``floor``: it wins
    the vertical cue outright and the convergence cue too. Keeping only the upper
    cut is what stops a side wall being tiled as a floor.
    """
    above = np.where(below, np.uint8(0), mask)
    count = int(np.count_nonzero(above))
    if 0 < count < int(np.count_nonzero(mask)):
        return [mask, above]
    return [mask]


def _solidity(mask: np.ndarray) -> float:
    """Filled area over the area of the convex hulls of its components, in ``[0, 1]``.

    Summed per component rather than over one global hull, so a plane genuinely
    split in two by an occluder is not punished for the empty space between the
    halves.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = sum(float(cv2.contourArea(cv2.convexHull(c))) for c in contours)
    if hull_area <= 0.0:
        return 0.0
    return float(np.count_nonzero(mask)) / hull_area


def _surface_candidates(mask: np.ndarray, min_fraction: float) -> list[np.ndarray]:
    """``mask`` restricted to its usable components, plus those components alone.

    A mask in more than :data:`_MAX_PROPOSAL_COMPONENTS` pieces, or filling less
    than :data:`_MIN_PROPOSAL_SOLIDITY` of its own hull, is rejected outright
    rather than proposed as a union. Those two rules are what keep *texture* out
    of the proposal set: the light squares of a tiled floor form a perfectly good
    colour cluster spanning the whole floor, and being concentrated in the near
    field its mask outscores the contiguous floor it is only half of. A surface
    fragmented by a few occluders still passes, which is the case worth keeping.

    Returns:
        Zero to ``_MAX_PROPOSAL_COMPONENTS + 1`` masks: the cleaned union first,
        then each component when there is more than one.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_pixels = min_fraction * float(mask.shape[0] * mask.shape[1])
    keep = [
        label
        for label in range(1, count)
        if float(stats[label, cv2.CC_STAT_AREA]) >= min_pixels
    ]
    if not keep or len(keep) > _MAX_PROPOSAL_COMPONENTS:
        return []
    components = [np.where(labels == label, np.uint8(255), np.uint8(0)) for label in keep]
    out = components if len(components) == 1 else None
    if out is None:
        union = np.zeros_like(mask)
        for component in components:
            union |= component
        out = [union, *components]
    return [candidate for candidate in out if _solidity(candidate) >= _MIN_PROPOSAL_SOLIDITY]


def _cluster_subsets(label_map: np.ndarray) -> list[np.ndarray]:
    """Every non-empty union of the colour clusters, as boolean masks.

    Single clusters alone are not enough. A tiled floor's light squares can be a
    closer colour match to a lit wall than to their own dark squares, so the
    floor arrives split across two clusters, one of which it shares with a wall.
    Proposing the *unions* as well gives the horizon split below something
    contiguous to cut, and cutting "floor plus wall" at the horizon is what
    recovers each of them.

    With :data:`_KMEANS_K` at three this is at most seven masks, and
    :func:`_kmeans_lab` never returns more centres than that, so the exponential
    is bounded by construction.
    """
    count = int(label_map.max()) + 1
    clusters = [label_map == index for index in range(count)]
    out: list[np.ndarray] = []
    for bits in range(1, 1 << count):
        member = np.zeros(label_map.shape, dtype=bool)
        for index in range(count):
            if bits >> index & 1:
                member |= clusters[index]
        if member.any():
            out.append(member)
    return out


def _floating_regions(
    label_map: np.ndarray, horizon_y: float, min_fraction: float
) -> np.ndarray:
    """Cluster components that float inside the frame rather than reaching its edge.

    This is the classical stand-in for the neural backend's enclosure test, and it
    rests on one observation about photographs of rooms: a Structural_Plane always
    runs off the edge of the picture, because a floor or a wall continues past
    whatever the lens caught. Furniture does not -- it sits somewhere in the middle
    with room surface all around it.

    Without this test a large piece of furniture wins its plane outright: a dark
    sofa filling a quarter of the lower frame scores higher as ``floor`` than the
    floor does, because it is lower, and nothing in the structural scoring knows
    what is in front of what.

    Three further conditions bound the damage the other way. The region must be
    small enough to be furniture rather than a wall seen through a doorway; it
    must hold most of its own colour cluster, so a tile is not mistaken for an
    object; and its base must lie below the horizon hint, which is not a heuristic
    but a projective fact -- every point of the ground plane images below the
    horizon, so anything standing on the floor has its lowest pixel there. Reading
    it off the horizon rather than off a fixed share of the frame is what keeps the
    test working when the camera is pitched steeply down and the furniture appears
    near the top of the picture.

    Returns:
        Boolean union of the floating components, an empty mask when there are
        none.
    """
    height, width = label_map.shape
    total = float(height * width)
    out = np.zeros(label_map.shape, dtype=bool)
    for index in range(int(label_map.max()) + 1):
        cluster = np.where(label_map == index, np.uint8(255), np.uint8(0))
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            cluster, connectivity=8
        )
        usable = [
            label
            for label in range(1, count)
            if float(stats[label, cv2.CC_STAT_AREA]) >= min_fraction * total
        ]
        if len(usable) > _MAX_PROPOSAL_COMPONENTS:
            # A colour that recurs as many separate blobs is a pattern, not an
            # object: the light squares of a tiled floor are individually compact,
            # solid and clear of the frame edge, so each one would otherwise read
            # as a small piece of furniture. Furniture comes in ones and twos;
            # tiling comes in dozens.
            continue
        share_floor = _MIN_FLOATING_CLUSTER_SHARE * float(np.count_nonzero(cluster))
        for label in usable:
            if float(stats[label, cv2.CC_STAT_AREA]) < share_floor:
                continue
            component = np.where(labels == label, np.uint8(255), np.uint8(0))
            if _is_object_shaped(component, horizon_y, total):
                out |= component > 0
    return out


def _is_object_shaped(component: np.ndarray, horizon_y: float, total: float) -> bool:
    """Whether one component could be a thing in the room rather than a surface.

    The three tests every occluder candidate has to pass, wherever it came from:
    small enough to be furniture, standing on the floor -- so its lowest pixel is
    below the horizon, which every point of the ground plane must be -- and not
    giving much of its outline to the frame edge, because a surface runs out of
    the picture and an object does not.
    """
    if float(np.count_nonzero(component)) > _MAX_FLOATING_AREA_FRACTION * total:
        return False
    rows = np.flatnonzero(component.any(axis=1))
    if len(rows) == 0 or int(rows[-1]) <= horizon_y:
        return False
    return _border_contact_ratio(component) <= _MAX_FLOATING_BORDER_CONTACT


def _border_contact_ratio(mask: np.ndarray) -> float:
    """Share of ``mask``'s outline that lies along the edge of the frame."""
    contact = int(
        mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum()
    ) // 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
    if perimeter <= 0.0:
        return 1.0
    return contact / perimeter


def _classical_proposals(
    label_map: np.ndarray,
    horizon_y: float,
    segments: np.ndarray,
    min_fraction: float,
    kernel_frac: float,
    floating: np.ndarray | None = None,
) -> list[Region]:
    """Region proposals from the colour clusters and the horizon hint.

    Every candidate is morphologically cleaned and area-filtered before it is
    proposed, and near-duplicates -- a union whose horizon split changed nothing,
    a component identical to its parent, two subsets that clean to the same
    silhouette -- are collapsed by ``(area, bounding box)`` so they do not crowd
    out genuine surfaces under the proposal cap.

    Candidates that are mostly ``floating`` region are dropped, which is what
    keeps a large piece of furniture from being awarded a plane name. The floating
    pixels are *not* subtracted from the surviving candidates: leaving them in
    keeps each proposal's silhouette intact for the solidity test, and they are
    removed for good when the Foreground_Mask is subtracted during enforcement.
    """
    height, width = label_map.shape
    below = np.arange(height, dtype=np.intp)[:, None] > horizon_y
    below = np.broadcast_to(below, label_map.shape)
    min_pixels = min_fraction * float(height * width)

    seen: set[tuple[int, int, int, int, int]] = set()
    scored: list[tuple[int, np.ndarray]] = []
    for member in _cluster_subsets(label_map):
        cleaned = clean_mask(
            np.where(member, np.uint8(255), np.uint8(0)),
            kernel_frac=kernel_frac,
            min_component_fraction=min_fraction,
        )
        if not cleaned.any():
            continue
        for part in _horizon_parts(cleaned, below):
            for candidate in _surface_candidates(part, min_fraction):
                area = int(np.count_nonzero(candidate))
                if area < min_pixels:
                    continue
                if floating is not None and floating.any():
                    overlap = int(np.count_nonzero(floating & (candidate > 0)))
                    if overlap > _MAX_FLOATING_OVERLAP * area:
                        continue
                x, y, w, h = cv2.boundingRect(candidate)
                key = (area, x, y, w, h)
                if key in seen:
                    continue
                seen.add(key)
                scored.append((area, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        Region(mask, segments=segments, source="kmeans_lab")
        for _area, mask in scored[:_MAX_PROPOSALS]
    ]


def _grow_assigned_planes(
    assigned: dict[PlaneName, np.ndarray], label_map: np.ndarray, horizon_y: float
) -> dict[PlaneName, np.ndarray]:
    """Regrow each assigned plane over the rest of its own colour cluster.

    The horizon hint is a *scoring* cue, not a surface boundary, but the two are
    hard to keep apart: a wall genuinely straddles the horizon, so its unsplit
    proposal has only half its mass above the hint and is always outscored by its
    own above-the-horizon fragment. Left alone, every wall would be reported
    truncated at eye level.

    So after assignment each plane reclaims the pixels that (a) belong to its own
    dominant colour cluster, (b) no other plane was awarded, and (c) are connected
    to the plane it grew from. Condition (c) is what keeps the growth honest: it
    can only extend a surface the assignment already found, never invent a second
    disjoint patch of it. Planes are grown in :data:`PLANE_PRIORITY` order and
    each one's result is claimed before the next runs, so the operation cannot
    create an overlap.

    The floor additionally may not grow above the horizon hint. When a floor and a
    wall do end up sharing a colour cluster, this is what stops the floor
    swallowing the room: no point of the ground plane images above its own line at
    infinity. The constraint is applied to the growth only, never to the assigned
    seed, because the hint is an estimate and evidence outranks inference.
    """
    if not assigned:
        return {}
    original = {plane: mask > 0 for plane, mask in assigned.items()}
    claimed = np.zeros(label_map.shape, dtype=bool)
    for seed in original.values():
        claimed |= seed
    below_horizon = np.arange(label_map.shape[0], dtype=np.intp)[:, None] > horizon_y

    grown: dict[PlaneName, np.ndarray] = {}
    for plane in PLANE_PRIORITY:
        seed = original.get(plane)
        if seed is None:
            continue
        blocked = claimed & ~seed
        counts = np.bincount(label_map[seed], minlength=int(label_map.max()) + 1)
        cluster = label_map == int(np.argmax(counts))
        if plane == "floor":
            cluster = cluster & below_horizon
        merged = seed | (cluster & ~blocked)

        # Keep only the components the seed actually reaches.
        count, components = cv2.connectedComponents(
            np.where(merged, np.uint8(255), np.uint8(0)), connectivity=8
        )
        reached = np.unique(components[seed])
        reached = reached[reached > 0]
        result = np.isin(components, reached) if count > 1 else seed

        grown[plane] = np.where(result, np.uint8(255), np.uint8(0))
        claimed |= result
    return grown


def _fitted_plane_masks(
    assigned: dict[PlaneName, np.ndarray],
) -> dict[PlaneName, np.ndarray]:
    """Hole-fill each assigned mask into the plane's *fitted* silhouette.

    A sofa standing on the floor punches a hole in the floor's colour cluster, so
    the raw proposal describes the visible floor rather than the floor surface.
    Filling that hole is what turns the mask into a fitted plane and puts the
    occluder's pixels *inside* it, which is the precondition for deriving the
    Foreground_Mask as the residual of structural fitting at all. The foreground
    is subtracted again by :func:`enforce_plane_invariants`, so the fill never
    survives into the result.

    A fill may not annex pixels another plane's proposal already claims -- without
    that guard a floor whose contour wraps a doorway could swallow the back wall
    outright and then win it on priority.
    """
    if not assigned:
        return {}
    raw = {plane: mask > 0 for plane, mask in assigned.items()}
    fitted: dict[PlaneName, np.ndarray] = {}
    for plane in PLANE_PRIORITY:
        mask = assigned.get(plane)
        if mask is None:
            continue
        filled = fill_holes(mask)
        for other, other_mask in raw.items():
            if other != plane:
                filled[other_mask & ~raw[plane]] = 0
        fitted[plane] = filled
    return fitted


def _structural_pockets(
    fitted: dict[PlaneName, np.ndarray], horizon_y: float
) -> np.ndarray:
    """Pixels the fitted planes enclose but none of them claims.

    The per-plane colour and texture residuals only see *inside* a plane, which
    misses the most common occluder of all: anything tall enough to break a
    plane's silhouette rather than punch a hole in it. A wardrobe standing against
    the back wall interrupts both the floor and the wall, so it is a hole in
    neither -- but it is unmistakably a hole in their *union*.

    Sealing that union and taking what the seal added is therefore the other half
    of "the residual of structural fitting".

    Each pocket still has to look like an object, because not every gap in the
    union is furniture: when a plane is missed altogether -- three walls that
    share a colour and only one of them wins a name -- the surface left over is a
    gap too, and by far the biggest one. :func:`_is_object_shaped` is what tells
    the two apart, and a missed back wall gives itself away by handing a long
    stretch of its outline to the top of the frame.
    """
    if not fitted:
        return np.zeros((1, 1), dtype=bool)
    masks = list(fitted.values())
    union = np.zeros(masks[0].shape, dtype=np.uint8)
    for mask in masks:
        union |= binarize(mask)

    gaps = np.where((fill_holes(union) > 0) & (union == 0), np.uint8(255), np.uint8(0))
    total = float(union.shape[0] * union.shape[1])
    out = np.zeros(union.shape, dtype=bool)
    count, labels = cv2.connectedComponents(gaps, connectivity=8)
    for label in range(1, count):
        component = np.where(labels == label, np.uint8(255), np.uint8(0))
        if _is_object_shaped(component, horizon_y, total):
            out |= component > 0
    return out


class ClassicalSegmenter(Segmenter):
    """Weightless OpenCV segmentation backend. R4.1, R4.5

    This is the backend the service falls back to whenever MobileSAM weights are
    unavailable, so it may use nothing but OpenCV and numpy: no network at import
    or call time, no model files, no optional dependency (Requirement 4.5, and
    Requirement 13.1's no-weights test run).

    Region proposals come from Lab-space colour clustering combined with the
    Geometry_Engine's horizon hint (:func:`_classical_proposals`), and the
    Foreground_Mask is the residual of structural fitting, in four parts:

    * regions that float inside the frame instead of running off its edge, which
      are furniture and are barred from the plane contest (:func:`_floating_regions`);
    * pockets the fitted planes enclose but none of them claims
      (:func:`_structural_pockets`);
    * pixels whose colour departs from their plane's dominant colours
      (:func:`_colour_residual`);
    * pixels whose local edge energy contradicts their plane's texture direction
      (:func:`_texture_residual`).

    Line detection runs exactly once per photograph, feeding the horizon hint,
    the per-region orientation cue, and the per-plane texture directions from one
    call (Requirement 5.1).

    Known limits, all of which degrade the labelling without ever breaking the
    partition invariants: three walls painted the same colour cluster together and
    are reported as one plane; an occluder covering more than
    :data:`_MAX_FLOATING_AREA_FRACTION` of the frame *and* running off its edge is
    indistinguishable from a surface by every cue here; and a camera pitched
    steeply enough to push the horizon out of frame leaves the vertical cue with
    nothing to separate a floor from a wall. The neural backend is the accurate
    path -- this one only has to be correct, weightless and never raise
    (Requirements 4.5, 4.7).
    """

    __slots__ = ("_settings",)

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def backend_name(self) -> SegmentationBackend:
        """Always ``"classical"``, which is what the API reports. R4.6"""
        return "classical"

    @property
    def settings(self) -> Settings:
        """The settings this backend reads its area thresholds from."""
        return self._settings

    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Segment one photograph into Structural_Planes and a Foreground_Mask.

        Never raises for a decodable image: a frame with no recoverable surface
        returns a result with no planes and an empty foreground, leaving the
        "no usable plane" decision to the API layer (Requirement 6.5).

        Args:
            image_bgr: ``(H, W, 3) uint8`` BGR photograph. Grayscale and BGRA
                input is converted.

        Returns:
            A :class:`SegmentationResult` whose masks satisfy the partition
            invariants, because it comes from :func:`finalize_segmentation`.

        Raises:
            TypeError: the input is not a numpy array.
            ValueError: the input is empty or not an image-shaped array.
        """
        bgr = _as_bgr_u8(image_bgr)
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # One detection, three consumers: the horizon hint, the structural
        # orientation cue, and each plane's texture direction. R5.1
        segments = detect_line_segments(gray)
        horizon_y = estimate_horizon_hint(gray, segments, self._settings)

        # Two clustering passes. The first is only there to find the furniture:
        # with three clusters to spend and a room containing a floor, walls and a
        # sofa, spending one on the sofa leaves the floor and the walls sharing a
        # cluster, and a single region covering the whole room then wins `floor`
        # and is tiled over the walls. So the objects found by the first pass are
        # held out of the second pass's fit, and the three clusters it settles on
        # describe surfaces alone. Every pixel is still labelled either way.
        floating = _floating_regions(
            _cluster_label_map(bgr), horizon_y, FOREGROUND_MIN_COMPONENT_FRACTION
        )
        label_map = _cluster_label_map(bgr, exclude=floating)
        regions = _classical_proposals(
            label_map,
            horizon_y,
            segments,
            FOREGROUND_MIN_COMPONENT_FRACTION,
            _CLEAN_KERNEL_FRAC,
            floating,
        )
        assigned = assign_structural_planes(regions, horizon_y, (height, width))
        fitted = _fitted_plane_masks(_grow_assigned_planes(assigned, label_map, horizon_y))

        foreground = self._foreground(
            bgr, gray, fitted, segments, floating, horizon_y
        )
        return finalize_segmentation(
            fitted, foreground, self.backend_name, settings=self._settings
        )

    def _foreground(
        self,
        bgr: np.ndarray,
        gray: np.ndarray,
        fitted: dict[PlaneName, np.ndarray],
        segments: np.ndarray | None,
        floating: np.ndarray,
        horizon_y: float,
    ) -> np.ndarray:
        """Foreground_Mask as the residual of structural fitting. R3.2

        Four residuals are unioned: the floating regions barred from the plane
        contest, the pockets the fitted planes enclose without claiming, and
        per-plane colour deviation and texture inconsistency. A plane too small
        for its own robust statistics to mean anything is skipped by the last two
        but still contributes to the second.

        Returns:
            ``(H, W) uint8`` ``{0, 255}`` mask, morphologically closed,
            hole-filled, and stripped of components under
            :data:`FOREGROUND_MIN_COMPONENT_FRACTION`.
        """
        shape = gray.shape[:2]
        raw = floating.copy()
        if not fitted:
            return np.zeros(shape, dtype=np.uint8)

        lab = _residual_lab(bgr)
        magnitude, edge_angle = _gradient_fields(gray)
        min_area = _MIN_RESIDUAL_AREA_FRACTION * float(shape[0] * shape[1])

        for mask in fitted.values():
            if int(np.count_nonzero(mask)) < min_area:
                continue
            raw |= _colour_residual(lab, mask)
            raw |= _texture_residual(magnitude, edge_angle, mask, segments)
        raw |= _structural_pockets(fitted, horizon_y)

        return clean_mask(
            np.where(raw, np.uint8(255), np.uint8(0)),
            kernel_frac=_CLEAN_KERNEL_FRAC,
            min_component_fraction=FOREGROUND_MIN_COMPONENT_FRACTION,
            close_holes=True,
        )

# --------------------------------------------------------------------------- #
# Neural backend -- MobileSAM through onnxruntime
# --------------------------------------------------------------------------- #

_logger = logging.getLogger(__name__)

#: Square side MobileSAM's image encoder expects. The photograph is resized so
#: its *longest* edge reaches this and the remainder is zero-padded, which is
#: SAM's own ``ResizeLongestSide`` preprocessing -- the prompt coordinates have to
#: be scaled by the same factor or every mask comes back describing the wrong
#: part of the picture. Only used as a default: the real side is read off the
#: session's input metadata when it is static.
_SAM_INPUT_SIZE: Final[int] = 1024

#: SAM's channel statistics, in RGB order (the checkpoint's own values). Applied
#: only when the encoder declares a floating-point input; an encoder exported with
#: normalisation folded into the graph declares ``uint8`` and is fed raw pixels.
_SAM_PIXEL_MEAN_RGB: Final[tuple[float, float, float]] = (123.675, 116.28, 103.53)
_SAM_PIXEL_STD_RGB: Final[tuple[float, float, float]] = (58.395, 57.12, 57.375)

#: Side of the low-resolution mask the decoder accepts as its optional
#: ``mask_input``. Always fed as zeros with ``has_mask_input=0``: the design
#: prompts each point once rather than iteratively refining, so there is never a
#: previous mask to pass.
_SAM_LOW_RES_SIZE: Final[int] = 256

#: Points per axis in the prompt grid, so ``PROMPT_GRID_SIDE ** 2`` decoder calls
#: per photograph. The grid is what makes the proposals *class-agnostic*: nothing
#: here knows what a sofa is, it just asks "what is the object at this pixel?"
#: across the frame. Six per axis puts a prompt inside every surface and every
#: substantial occluder of a room photograph while keeping the decoder budget at
#: 36 calls, which is the term that dominates neural analysis time.
PROMPT_GRID_SIDE: Final[int] = 6

#: SAM's decoder emits mask *logits*, so zero is the probability-0.5 contour.
_MASK_LOGIT_THRESHOLD: Final[float] = 0.0

#: IoU at or above which a new proposal is considered a duplicate of one already
#: kept. Neighbouring prompts that land on the same sofa return near-identical
#: masks, and without this collapse those duplicates would crowd the scoring.
_PROPOSAL_DEDUP_IOU: Final[float] = 0.8

#: Share of a candidate's visible outline that must abut structural surface before
#: it counts as *enclosed by* a structural region. Below 1.0 because a proposal's
#: boundary and the plane's rarely agree to the pixel. Measured over the synthetic
#: room across several occluder placements, things standing in the room sit at 0.98
#: and above while unclaimed *surfaces* -- a wall the assignment failed to name --
#: sit around 0.25 to 0.67, since a surface borders the frame and the room's other
#: openings rather than being ringed by plane on every side.
_MIN_ENCLOSURE_FRACTION: Final[float] = 0.8

#: IoU with a fitted plane at or above which a candidate is treated as a
#: *restatement* of that plane rather than something enclosed by it. Enclosure
#: means lying within a surface, not repeating its outline, and the difference
#: matters because a point grid readily proposes the same floor twice -- once
#: whole, once as most of itself. The whole one wins the plane name; without this
#: guard the near-duplicate would satisfy every other foreground condition and
#: punch the floor out of the picture.
_MAX_PLANE_RESTATEMENT_IOU: Final[float] = 0.5

#: Top of the frame excluded from the foreground. The design's rule is that an
#: occluder sits in the lower two-thirds, which is a statement about where things
#: standing in a room appear: the top third of a room photograph is wall and
#: ceiling, and a region up there that failed the structural scores is a missed
#: surface -- a wall through a doorway, a ceiling -- not furniture. Cutting it out
#: of the foreground keeps that miss a labelling error instead of turning it into
#: a hole punched through the room.
_FOREGROUND_TOP_EXCLUSION: Final[float] = 1.0 / 3.0

#: onnx type strings to numpy dtypes. The feed tensors are built to whatever the
#: session declares rather than to a hardcoded ``float32``, because the SAM
#: exports disagree about ``point_labels`` -- some emit ``float32``, some
#: ``int64`` -- and onnxruntime rejects a mismatch outright.
_ONNX_DTYPES: Final[dict[str, np.dtype]] = {
    "tensor(float)": np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),
    "tensor(double)": np.dtype(np.float64),
    "tensor(int64)": np.dtype(np.int64),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(int8)": np.dtype(np.int8),
    "tensor(bool)": np.dtype(np.bool_),
}


class InferenceSessionLike(Protocol):
    """The slice of ``onnxruntime.InferenceSession`` this backend actually uses.

    Typing the sessions structurally rather than importing ``onnxruntime`` here
    is deliberate. It keeps this module free of the heavy optional dependency at
    import time, and -- more importantly -- it makes the backend constructible
    from *any* pair of objects with this shape, which is the only way the neural
    path can be exercised on a host that has no weights (Requirements 4.7, 13.1).
    """

    def get_inputs(self) -> Sequence[Any]: ...

    def get_outputs(self) -> Sequence[Any]: ...

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray]
    ) -> Sequence[np.ndarray]: ...


class UnsupportedModelSignature(RuntimeError):
    """Raised when a session's inputs are not a recognisable SAM signature.

    Separate from ``ModelUnavailable`` because it is discovered at inference time
    rather than at load time: the weights opened fine, they are simply not the
    model this backend knows how to drive. Handled the same way in the end --
    warn and fall back -- but distinguishable in a log.
    """


@dataclass(frozen=True, slots=True)
class _EncoderSpec:
    """How to shape a photograph for one particular encoder session."""

    input_name: str
    size: int
    channels_last: bool
    dtype: np.dtype
    normalise: bool


@dataclass(frozen=True, slots=True)
class _PreparedImage:
    """A photograph in the encoder's frame, plus the mapping back out of it.

    ``scale`` and ``valid`` are what let a prompt in image coordinates become a
    prompt in the padded square frame, and a mask in that frame become a mask over
    the original photograph. Losing either one silently misaligns every mask by
    the padding width, which is why they travel with the tensor rather than being
    recomputed at each use.
    """

    tensor: np.ndarray
    scale: float
    valid: tuple[int, int]  # (height, width) of the non-padded content
    size: int
    shape: tuple[int, int]  # original (height, width)


def _np_dtype(onnx_type: str | None) -> np.dtype:
    """numpy dtype for an onnx tensor type string, defaulting to ``float32``."""
    return _ONNX_DTYPES.get(str(onnx_type), np.dtype(np.float32))


def _static_dim(value: Any) -> int | None:
    """Return ``value`` as a positive int, or ``None`` when it is dynamic.

    onnxruntime reports a dynamic axis as ``None`` or as a symbolic name string,
    so anything not integral is treated as unknown and defaulted by the caller.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        return None
    return int(value) if int(value) > 0 else None


def _encoder_spec(encoder: InferenceSessionLike) -> _EncoderSpec:
    """Read the preprocessing contract off the encoder's input metadata.

    Both layouts in the wild are accepted -- ``(1, 3, S, S)`` and
    ``(1, S, S, 3)`` -- decided by which axis carries the 3. Normalisation is
    applied only for a floating-point input: an export that declares ``uint8``
    has SAM's channel statistics folded into the graph, and normalising before
    such a model would apply them twice.

    Raises:
        UnsupportedModelSignature: the session declares no inputs.
    """
    inputs = list(encoder.get_inputs())
    if not inputs:
        raise UnsupportedModelSignature("encoder session declares no inputs")

    meta = inputs[0]
    shape = list(getattr(meta, "shape", []) or [])
    dtype = _np_dtype(getattr(meta, "type", None))

    channels_last = len(shape) == 4 and _static_dim(shape[3]) == 3
    if len(shape) == 4:
        spatial = (shape[1], shape[2]) if channels_last else (shape[2], shape[3])
        sizes = [dim for dim in (_static_dim(spatial[0]), _static_dim(spatial[1])) if dim]
        size = max(sizes) if sizes else _SAM_INPUT_SIZE
    else:
        size = _SAM_INPUT_SIZE

    return _EncoderSpec(
        input_name=str(getattr(meta, "name", "image")),
        size=int(size),
        channels_last=channels_last,
        dtype=dtype,
        normalise=dtype.kind == "f",
    )


def _prepare_image(bgr: np.ndarray, spec: _EncoderSpec) -> _PreparedImage:
    """Resize-longest-edge, normalise, and zero-pad ``bgr`` into the encoder frame.

    Padding goes on the right and bottom only, so the valid content keeps the
    origin and the forward map is a single scale factor with no offset -- which is
    what makes :func:`_prompt_points` and :func:`_mask_to_image` inverses of each
    other.
    """
    height, width = bgr.shape[:2]
    size = spec.size
    scale = size / float(max(height, width))
    valid_h = max(1, min(size, int(round(height * scale))))
    valid_w = max(1, min(size, int(round(width * scale))))

    resized = cv2.resize(bgr, (valid_w, valid_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    if spec.normalise:
        plane = rgb.astype(np.float32)
        plane -= np.asarray(_SAM_PIXEL_MEAN_RGB, dtype=np.float32)
        plane /= np.asarray(_SAM_PIXEL_STD_RGB, dtype=np.float32)
    else:
        plane = rgb

    canvas = np.zeros((size, size, 3), dtype=plane.dtype)
    canvas[:valid_h, :valid_w] = plane
    tensor = canvas if spec.channels_last else canvas.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(tensor[None, ...].astype(spec.dtype, copy=False))

    return _PreparedImage(
        tensor=tensor,
        scale=scale,
        valid=(valid_h, valid_w),
        size=size,
        shape=(height, width),
    )


def _prompt_points(shape: tuple[int, int], side: int) -> np.ndarray:
    """Cell-centre grid of ``side * side`` prompt points in image coordinates.

    Centres rather than edges, so no prompt lands on a surface boundary where the
    mask it elicits is a coin flip between the two surfaces meeting there.
    """
    height, width = shape
    side = max(1, int(side))
    fractions = (np.arange(side, dtype=np.float32) + 0.5) / float(side)
    xs = fractions * float(width - 1)
    ys = fractions * float(height - 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(np.float32)


def _decoder_feed(
    decoder: InferenceSessionLike,
    embedding: np.ndarray,
    point_xy: np.ndarray,
    prepared: _PreparedImage,
) -> dict[str, np.ndarray]:
    """Build the decoder input feed for one foreground point prompt.

    Inputs are matched by name against the SAM decoder contract rather than by
    position, because the exports order them inconsistently. An input this
    function does not recognise is a signature it cannot drive, and guessing a
    zero tensor for it would produce silently meaningless masks -- so it raises
    instead and the caller falls back.

    Raises:
        UnsupportedModelSignature: the session declares an input this backend
            does not know how to fill.
    """
    height, width = prepared.shape
    # SAM prompts live in the resized frame, not the original one. Same scale as
    # the image, no offset, because the padding is bottom-right.
    scaled = point_xy.reshape(-1, 2) * prepared.scale

    # The export encodes "no box prompt" as a trailing padding point labelled -1,
    # and omitting it does not fail -- it quietly degrades every mask, because the
    # model then reads the prompt as an incomplete box. So the pad travels with
    # every point prompt.
    coords = np.concatenate([scaled, np.zeros((1, 2), dtype=np.float32)], axis=0)[None]
    labels = np.concatenate(
        [np.ones(len(scaled), dtype=np.float32), np.full(1, -1.0, dtype=np.float32)]
    )[None]

    feed: dict[str, np.ndarray] = {}
    for meta in decoder.get_inputs():
        name = str(getattr(meta, "name", ""))
        key = name.lower()
        dtype = _np_dtype(getattr(meta, "type", None))

        # Matched narrowly on purpose: a bare "embed" substring would also
        # swallow a text- or box-conditioned model's prompt inputs and quietly
        # feed them an image embedding, producing masks that mean nothing.
        if "image_embed" in key or key in {"embeddings", "embedding"}:
            feed[name] = embedding.astype(dtype, copy=False)
        elif "point_coord" in key or key in {"points", "coords"}:
            feed[name] = coords.astype(dtype, copy=False)
        elif "point_label" in key or key in {"labels", "label"}:
            # 1 is SAM's "foreground point", -1 the box padding above.
            feed[name] = labels.astype(dtype, copy=False)
        elif "has_mask" in key:
            feed[name] = np.zeros((1,), dtype=dtype)
        elif "mask_input" in key or key == "mask":
            feed[name] = np.zeros(
                (1, 1, _SAM_LOW_RES_SIZE, _SAM_LOW_RES_SIZE), dtype=dtype
            )
        elif "orig_im_size" in key or "orig_size" in key:
            # (H, W), the order the export documents.
            feed[name] = np.asarray([height, width], dtype=dtype)
        else:
            raise UnsupportedModelSignature(
                f"decoder input {name!r} is not part of the SAM point-prompt contract"
            )
    if not feed:
        raise UnsupportedModelSignature("decoder session declares no inputs")
    return feed


def _select_mask(
    decoder: InferenceSessionLike, outputs: Sequence[np.ndarray]
) -> np.ndarray:
    """Pick the best-scoring mask plane out of one decoder call.

    SAM returns several candidate masks per prompt -- an object, its part, and
    its whole -- alongside a predicted IoU for each. The IoU head is the model's
    own confidence, so it chooses; without it the first plane is taken.

    Returns:
        A 2-D float array of mask logits.

    Raises:
        UnsupportedModelSignature: no output has a spatial mask shape.
    """
    names = [str(getattr(meta, "name", "")).lower() for meta in decoder.get_outputs()]

    scores: np.ndarray | None = None
    masks: np.ndarray | None = None
    for index, array in enumerate(outputs):
        name = names[index] if index < len(names) else ""
        arr = np.asarray(array)
        if "iou" in name or "score" in name:
            scores = arr.reshape(-1).astype(np.float64, copy=False)
        elif masks is None and "low_res" not in name and arr.ndim >= 3:
            masks = arr

    if masks is None:
        # Unnamed or differently named outputs: fall back to the documented
        # positional order (masks, iou_predictions, low_res_masks).
        for array in outputs:
            arr = np.asarray(array)
            if arr.ndim >= 3:
                masks = arr
                break
        if masks is None:
            raise UnsupportedModelSignature("decoder produced no spatial mask output")
        if scores is None and len(outputs) > 1:
            candidate = np.asarray(outputs[1]).reshape(-1)
            if candidate.size and candidate.dtype.kind == "f":
                scores = candidate.astype(np.float64, copy=False)

    planes = np.asarray(masks, dtype=np.float32).reshape(-1, *masks.shape[-2:])
    if scores is not None and scores.size >= len(planes) and len(planes) > 1:
        return planes[int(np.argmax(scores[: len(planes)]))]
    return planes[0]


def _mask_to_image(logits: np.ndarray, prepared: _PreparedImage) -> np.ndarray:
    """Map one mask plane back onto the original photograph as a binary mask.

    Two decoder variants have to be handled. One takes ``orig_im_size`` and
    returns masks already at the photograph's resolution, in which case there is
    nothing to undo. The other returns a low-resolution mask still in the padded
    square frame; the padding is cropped off proportionally and the remainder is
    resized. Thresholding happens *after* the resize so the interpolation runs on
    logits rather than on a binary edge, which keeps the silhouette smooth.
    """
    height, width = prepared.shape
    plane = np.asarray(logits, dtype=np.float32)

    if plane.shape != (height, width):
        mask_h, mask_w = plane.shape
        valid_h, valid_w = prepared.valid
        crop_h = max(1, min(mask_h, int(round(mask_h * valid_h / prepared.size))))
        crop_w = max(1, min(mask_w, int(round(mask_w * valid_w / prepared.size))))
        plane = plane[:crop_h, :crop_w]
        plane = cv2.resize(plane, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.where(plane > _MASK_LOGIT_THRESHOLD, np.uint8(255), np.uint8(0))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two binary masks, ``0.0`` when both are empty."""
    first = a > 0
    second = b > 0
    union = int(np.count_nonzero(first | second))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(first & second)) / float(union)


def _dedup_masks(masks: Sequence[np.ndarray], iou_threshold: float) -> list[np.ndarray]:
    """Drop near-duplicate masks, keeping the largest of each cluster.

    Largest first is what makes the survivor the *whole* object: two prompts on
    one sofa return the sofa and a cushion, and keeping the sofa is what lets the
    enclosure test see one occluder instead of a fragment of one.
    """
    ordered = sorted(masks, key=lambda m: int(np.count_nonzero(m)), reverse=True)
    kept: list[np.ndarray] = []
    for mask in ordered:
        if all(_iou(mask, other) < iou_threshold for other in kept):
            kept.append(mask)
    return kept


def _neural_proposals(
    encoder: InferenceSessionLike,
    decoder: InferenceSessionLike,
    bgr: np.ndarray,
    segments: np.ndarray | None,
    *,
    grid_side: int = PROMPT_GRID_SIDE,
    min_fraction: float = FOREGROUND_MIN_COMPONENT_FRACTION,
    kernel_frac: float = _CLEAN_KERNEL_FRAC,
) -> list[Region]:
    """Class-agnostic region proposals from a grid of MobileSAM point prompts.

    The encoder runs once per photograph and the decoder once per prompt, which
    is the whole reason SAM is split into two graphs: the embedding is the
    expensive half and it is shared across all ``grid_side ** 2`` prompts.

    Every mask is morphologically cleaned and area-filtered before it becomes a
    proposal, so the 0.2 percent noise floor applies to plane candidates and
    foreground candidates alike (Requirement 3.2).

    Returns:
        Deduplicated proposals in descending area order, each carrying the
        photograph's line segments so the shared scorer has its orientation cue.

    Raises:
        UnsupportedModelSignature: either session's signature is not a SAM one.
    """
    spec = _encoder_spec(encoder)
    prepared = _prepare_image(bgr, spec)

    encoded = encoder.run(None, {spec.input_name: prepared.tensor})
    if not encoded:
        raise UnsupportedModelSignature("encoder produced no output")
    embedding = np.asarray(encoded[0])

    raw: list[np.ndarray] = []
    for point in _prompt_points(prepared.shape, grid_side):
        feed = _decoder_feed(decoder, embedding, point, prepared)
        mask = _mask_to_image(_select_mask(decoder, decoder.run(None, feed)), prepared)
        cleaned = clean_mask(
            mask, kernel_frac=kernel_frac, min_component_fraction=min_fraction
        )
        if cleaned.any():
            raw.append(cleaned)

    return [
        Region(mask, segments=segments, source="mobilesam_point_grid")
        for mask in _dedup_masks(raw, _PROPOSAL_DEDUP_IOU)
    ]


def _enclosure_ratio(
    candidate: np.ndarray,
    structural: np.ndarray,
    *,
    kernel_frac: float = _CLEAN_KERNEL_FRAC,
) -> float:
    """Share of ``candidate``'s outline that abuts structural surface.

    Enclosure is measured around the candidate's *boundary* rather than as
    containment inside the surfaces, because containment fails on the commonest
    occluder of all: a sofa whose base is clipped by the bottom of the frame
    leaves a notch in the floor's silhouette rather than a hole in it, so the
    floor mask -- hole-filled or not -- covers none of it. Its outline is still
    ringed by floor on every visible side, which is what "enclosed by a structural
    region" actually means.

    The ring is taken on a padded canvas so a candidate touching the frame edge
    is not silently credited with the boundary it has outside the picture; those
    ring pixels are excluded from the ratio entirely rather than counted against
    it, since what lies beyond the frame is unknown, not non-structural.

    Returns:
        Ratio in ``[0, 1]``; ``0.0`` for an empty candidate.
    """
    mask = binarize(candidate)
    kernel = _kernel_for(mask.shape, kernel_frac)
    pad = max(kernel.shape) // 2 + 1

    def _padded(array: np.ndarray) -> np.ndarray:
        return cv2.copyMakeBorder(
            array, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
        )

    padded = _padded(mask)
    ring = cv2.dilate(padded, kernel) & ~padded
    ring &= _padded(np.full(mask.shape, 255, dtype=np.uint8))
    total = int(np.count_nonzero(ring))
    if total == 0:
        return 0.0
    inside = int(np.count_nonzero(ring & _padded(binarize(structural))))
    return inside / float(total)


def _structural_silhouette(fitted: Mapping[PlaneName, np.ndarray]) -> np.ndarray:
    """Union of every fitted plane, as the enclosure test's reference.

    The planes arrive from :func:`_fitted_plane_masks` already hole-filled, so an
    occluder wholly inside one surface is inside this union too; one that breaks
    the silhouette instead is caught by the boundary measurement in
    :func:`_enclosure_ratio`, which is why no further filling happens here.
    """
    masks = [binarize(mask) for mask in fitted.values()]
    if not masks:
        return np.zeros((1, 1), dtype=np.uint8)
    union = np.zeros_like(masks[0])
    for mask in masks:
        union |= mask
    return union


def _mask_key(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """Cheap identity for a mask: set-pixel count plus bounding box.

    Used to find which proposal won which plane name without comparing every
    candidate pair pixelwise, which at 2048 px would cost more than the inference
    that produced them. Two masks agreeing on this key are then confirmed with an
    exact comparison, so the key only ever has to be a fast *filter*.
    """
    binary = binarize(mask)
    x, y, w, h = cv2.boundingRect(binary)
    return (int(np.count_nonzero(binary)), int(x), int(y), int(w), int(h))


def _awarded_region_indices(
    regions: Sequence[Region], assigned: Mapping[PlaneName, np.ndarray]
) -> dict[PlaneName, int]:
    """Map each awarded plane name back to the index of the region that won it.

    :func:`assign_structural_planes` returns masks rather than indices, but the
    masks it returns are copies of the winning regions' own masks, so identity
    recovers the mapping exactly. Recovering it is what lets the foreground rule
    below ask "did this proposal win anything?" without re-running the contest and
    risking a different answer than the assignment reached.
    """
    buckets: dict[tuple[int, int, int, int, int], list[int]] = {}
    for index, region in enumerate(regions):
        buckets.setdefault(_mask_key(region.mask), []).append(index)

    out: dict[PlaneName, int] = {}
    for plane, mask in assigned.items():
        for index in buckets.get(_mask_key(mask), ()):
            if index not in out.values() and np.array_equal(regions[index].mask, mask):
                out[plane] = index
                break
    return out


def _fails_every_plane(
    index: int,
    scores: Mapping[PlaneName, float],
    awarded: Mapping[PlaneName, int],
) -> bool:
    """Whether region ``index`` came away from the contest with no plane name.

    A region fails plane ``P`` when it either scored below :data:`SCORE_FLOOR` for
    ``P`` or ``P`` went to a better-scoring region. Failing all four in that sense
    is the design's "fails all four structural scores": the greedy contest awards
    each name once, so a proposal can score well and still be left with nothing --
    which is exactly what happens to a sofa, whose low, converging mask reads as a
    plausible floor right up until the actual floor outscores it.

    Reading the condition as "scored below the floor on all four" instead would
    make the foreground nearly always empty, because furniture standing on the
    floor scores *high* as floor. That reading would leave every occluder inside a
    plane mask and tiled over, defeating Requirement 3.2.
    """
    return all(
        scores.get(plane, 0.0) < SCORE_FLOOR or awarded.get(plane) != index
        for plane in PLANE_NAMES
    )


def _is_enclosed(
    mask: np.ndarray,
    silhouette: np.ndarray,
    fitted: Mapping[PlaneName, np.ndarray],
) -> bool:
    """Whether ``mask`` lies *within* the structural surfaces rather than being one.

    Two halves to the test. Most of the candidate's outline has to abut structural
    surface (:func:`_enclosure_ratio`) -- a surface runs off the edge of the
    picture, a thing in the room has room surface all around it -- and it must not
    simply restate a fitted plane, since a proposal identical to the floor is the
    floor and not something the floor encloses.
    """
    if _enclosure_ratio(mask, silhouette) < _MIN_ENCLOSURE_FRACTION:
        return False
    return all(
        _iou(mask, plane_mask) < _MAX_PLANE_RESTATEMENT_IOU
        for plane_mask in fitted.values()
    )


def _plane_candidate_indices(
    regions: Sequence[Region], min_area_fraction: float, shape: tuple[int, int]
) -> list[int]:
    """Indices of the proposals large enough to be worth a plane name.

    A proposal below ``min_area_fraction`` cannot survive
    :func:`enforce_plane_invariants`' area floor, so awarding it a plane name does
    not produce a small plane -- it produces *no* plane, because the greedy
    contest is one-to-one and the name is spent on a region that is then dropped.
    Holding those proposals out of the contest is therefore not a filter but a
    correction: it stops a low, compact occluder, which the vertical cue rates
    highly as floor, from taking the floor's name and taking it out of the result.

    The proposals themselves are kept -- being too small to be a surface is
    evidence *for* being an occluder, and they are still judged for the
    Foreground_Mask.
    """
    total = float(int(shape[0]) * int(shape[1]))
    floor_px = max(min_area_fraction, 0.0) * total
    return [
        index
        for index, region in enumerate(regions)
        if float(np.count_nonzero(region.mask)) >= floor_px
    ]


def _neural_foreground(
    regions: Sequence[Region],
    scores: Sequence[Mapping[PlaneName, float]],
    awarded: Mapping[PlaneName, int],
    fitted: Mapping[PlaneName, np.ndarray],
    shape: tuple[int, int],
    *,
    min_component_fraction: float = FOREGROUND_MIN_COMPONENT_FRACTION,
    kernel_frac: float = _CLEAN_KERNEL_FRAC,
) -> np.ndarray:
    """Union the proposals that read as occluders rather than surfaces. R3.2

    Three conditions, all of which a candidate must meet:

    1. It failed all four structural scores, in the sense
       :func:`_fails_every_plane` documents: the shared greedy contest left it
       with no plane name. The scores come from :func:`score_regions`, the same
       call that drove the assignment, so the two paths cannot disagree about
       which proposals were rejected. ``awarded`` maps each granted plane name to
       the index of the proposal that won it.
    2. It is enclosed by a structural region (:func:`_is_enclosed`), which is
       what separates a sofa from a wall the assignment simply failed to name.
    3. Its centroid sits in the lower two-thirds of the frame, where things
       standing on the floor appear.

    Components below ``min_component_fraction`` of the frame are discarded as
    noise, and interior holes are closed, since a gap in the middle of a sofa is
    always an artifact. Candidates above :data:`_MAX_FLOATING_AREA_FRACTION` of
    the frame are discarded too, for the reason recorded at that constant.

    Returns:
        ``(H, W) uint8`` ``{0, 255}`` mask; empty when no plane was fitted, since
        with no structural silhouette nothing can be enclosed by one.
    """
    height, width = int(shape[0]), int(shape[1])
    empty = np.zeros((height, width), dtype=np.uint8)
    if not fitted:
        return empty

    silhouette = _structural_silhouette(fitted)
    if silhouette.shape != (height, width):  # pragma: no cover - shapes agree upstream
        return empty

    min_row = _FOREGROUND_TOP_EXCLUSION * float(height - 1)
    max_area = _MAX_FLOATING_AREA_FRACTION * float(height * width)
    raw = np.zeros((height, width), dtype=bool)
    for index, region in enumerate(regions):
        mask = region.mask
        if index >= len(scores) or not mask.any():
            continue
        if not _fails_every_plane(index, scores[index], awarded):
            continue
        # Same ceiling the classical backend puts on an occluder candidate, and
        # for the same reason: a fifth of the picture is a generous wardrobe, and
        # past that the region is far more likely a surface the assignment failed
        # to name. Erring low is the right way round -- a missed occluder costs
        # one badly composited object, a surface mistaken for furniture costs the
        # whole surface.
        if float(np.count_nonzero(mask)) > max_area:
            continue
        if not _is_enclosed(mask, silhouette, fitted):
            continue
        rows = np.nonzero(mask)[0]
        if float(rows.mean()) < min_row:
            continue
        raw |= mask > 0

    if not raw.any():
        return empty
    return clean_mask(
        np.where(raw, np.uint8(255), np.uint8(0)),
        kernel_frac=kernel_frac,
        min_component_fraction=min_component_fraction,
        close_holes=True,
    )


class NeuralSegmenter(Segmenter):
    """MobileSAM segmentation backend over two onnxruntime sessions. R4.1, R4.6

    The sessions are *injected*, never built here: acquisition and provider
    selection belong to ``backend/utils/model_loader.py`` and the decision to use
    this backend at all belongs to ``build_segmenter`` in ``backend/app.py``
    (Requirement 4.5). That split has a second benefit -- the class is
    constructible from anything matching :class:`InferenceSessionLike`, so the
    neural path is exercisable on a host with no weights and this module reaches
    neither the network nor the filesystem at import or call time
    (Requirements 4.7, 13.1).

    The pipeline:

    1. One encoder pass over the resize-longest-edge, zero-padded photograph.
    2. One decoder pass per point in a ``PROMPT_GRID_SIDE`` square grid, giving
       class-agnostic region proposals (:func:`_neural_proposals`). Nothing here
       recognises furniture; the proposals are just "the object at this pixel".
    3. The *same* shared post-processing every backend uses --
       :func:`assign_structural_planes` for the planes,
       :func:`_fitted_plane_masks` to seal each one's silhouette, and
       :func:`finalize_segmentation` for the partition invariants.
    4. :func:`_neural_foreground` for the Foreground_Mask: the proposals that
       came away from the structural contest with no plane name, are enclosed by
       a structural region, and sit in the lower two-thirds of the frame.

    Because MobileSAM proposes objects and surfaces indiscriminately, the
    structural scoring is what tells the two apart -- which is precisely why the
    scorer is shared code and not a backend detail.

    A per-request inference failure -- a signature this backend cannot drive, a
    provider that dies mid-run -- degrades to :class:`ClassicalSegmenter` and
    returns *its* result, so the reported ``segmentation_backend`` stays truthful
    about what actually ran (Requirement 4.6) and the request still succeeds
    (Requirement 4.7).

    Known limits, none of which can break the partition invariants, since those
    come from :func:`finalize_segmentation`: an occluder no grid point lands on is
    never proposed and so never enters the Foreground_Mask, which puts a floor
    under the useful grid density; an occluder covering more than
    :data:`_MAX_FLOATING_AREA_FRACTION` of the frame is deliberately treated as a
    surface; and when the horizon hint degenerates -- pushed out of frame by a
    steep pitch -- the shared vertical cue can rank a low-sitting object above the
    floor it stands on, which inverts the two roles. That last one is a property of
    the shared scorer rather than of this backend, and it is why the area ceiling
    above is set where it is: a bad label costs one surface, while admitting a
    half-frame region to the foreground would cut a hole through the room.
    """

    __slots__ = ("_encoder", "_decoder", "_settings", "_logger", "_fallback", "_grid_side")

    def __init__(
        self,
        encoder: InferenceSessionLike,
        decoder: InferenceSessionLike,
        settings: Settings | None = None,
        *,
        logger: logging.Logger | None = None,
        fallback: Segmenter | None = None,
        grid_side: int = PROMPT_GRID_SIDE,
    ) -> None:
        """Bind the two sessions.

        Args:
            encoder: MobileSAM image-encoder session.
            decoder: MobileSAM mask-decoder session.
            settings: settings to read area thresholds from; defaults to
                :func:`get_settings`.
            logger: where inference-failure warnings go.
            fallback: segmenter to delegate to when inference fails. Defaults to
                a :class:`ClassicalSegmenter` built on the same settings.
            grid_side: points per axis in the prompt grid. Must be positive.

        Raises:
            ValueError: ``grid_side`` is not positive.
        """
        if int(grid_side) < 1:
            raise ValueError(f"grid_side must be positive, got {grid_side}")
        self._encoder = encoder
        self._decoder = decoder
        self._settings = settings or get_settings()
        self._logger = logger or _logger
        self._fallback = fallback or ClassicalSegmenter(self._settings)
        self._grid_side = int(grid_side)

    @property
    def backend_name(self) -> SegmentationBackend:
        """Always ``"mobilesam-onnx"``, which is what the API reports. R4.6"""
        return "mobilesam-onnx"

    @property
    def settings(self) -> Settings:
        """The settings this backend reads its area thresholds from."""
        return self._settings

    @property
    def fallback(self) -> Segmenter:
        """The backend a failed inference delegates to. R4.5"""
        return self._fallback

    def segment(self, image_bgr: np.ndarray) -> SegmentationResult:
        """Segment one photograph into Structural_Planes and a Foreground_Mask.

        Args:
            image_bgr: ``(H, W, 3) uint8`` BGR photograph. Grayscale and BGRA
                input is converted.

        Returns:
            A :class:`SegmentationResult` whose masks satisfy the partition
            invariants, because it comes from :func:`finalize_segmentation`.
            ``backend_name`` is ``"mobilesam-onnx"`` unless inference failed and
            the classical fallback produced the result, in which case it reports
            ``"classical"``.

        Raises:
            TypeError: the input is not a numpy array.
            ValueError: the input is empty or not an image-shaped array.
        """
        bgr = _as_bgr_u8(image_bgr)
        try:
            return self._segment_neural(bgr)
        except Exception as exc:  # noqa: BLE001 - any inference failure is a fallback
            # Deliberately broad. An unrecognised signature, a provider crash, a
            # corrupt embedding shape and an onnxruntime internal error all mean
            # the same thing to the caller, and none of them may fail a request
            # that the weightless backend can still serve (Requirement 4.7).
            self._logger.warning(
                "neural segmentation failed: %r; using %s backend",
                exc,
                self._fallback.backend_name,
            )
            return self._fallback.segment(bgr)

    def _segment_neural(self, bgr: np.ndarray) -> SegmentationResult:
        """The neural path proper, with no failure handling of its own."""
        height, width = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # One detection, two consumers: the horizon hint and the per-region
        # orientation cue. R5.1
        segments = detect_line_segments(gray)
        horizon_y = estimate_horizon_hint(gray, segments, self._settings)

        regions = _neural_proposals(
            self._encoder,
            self._decoder,
            bgr,
            segments,
            grid_side=self._grid_side,
        )

        # Scored once and used twice: the assignment consumes the winners, the
        # foreground consumes the proposals that no plane name would take.
        scores = score_regions(regions, horizon_y, (height, width))

        # Only proposals large enough to survive the plane area floor enter the
        # contest; the rest stay in `regions` for the foreground rule below.
        candidates = _plane_candidate_indices(
            regions, self._settings.min_plane_area_fraction, (height, width)
        )
        plane_regions = [regions[index] for index in candidates]
        assigned = assign_structural_planes(plane_regions, horizon_y, (height, width))
        fitted = _fitted_plane_masks(assigned)

        # Winners are reported back in whole-proposal index space, which is what
        # `_neural_foreground` scores against.
        awarded = {
            plane: candidates[local]
            for plane, local in _awarded_region_indices(plane_regions, assigned).items()
        }
        foreground = _neural_foreground(
            regions, scores, awarded, fitted, (height, width)
        )
        return finalize_segmentation(
            fitted, foreground, self.backend_name, settings=self._settings
        )
