"""Geometry_Engine -- lines, directional clusters, vanishing points, horizon.

Everything the calibration pipeline knows about a photograph's 3-D structure
starts as a bag of straight line segments. This module owns that first stage and
the two consumers that read it before any vanishing point is estimated:

1. :func:`detect_line_segments` -- OpenCV's LSD when the installed build ships
   it, a Canny + probabilistic Hough path when it does not, with both branches
   returning the identical ``(N,4)`` endpoint array so nothing downstream has to
   ask which detector ran (Requirement 5.1).
2. :func:`cluster_by_direction` -- the segments split into the directional groups
   a vanishing point can be fitted to: a vertical group taken by angle tolerance
   around ``pi/2``, plus two non-vertical groups derived from a 1-D mean-shift
   over the length-weighted angle histogram (Requirement 5.2).
3. :func:`estimate_horizon_hint` -- the cheap horizon row the Segmenter's
   structural scoring needs. It is derived from the same detection, so a request
   runs the line detector exactly once per photograph instead of once for
   segmentation and again for calibration.

On top of that sit the three stages that turn directional groups into a camera:

4. :func:`estimate_vanishing_point` -- RANSAC over pairs of a group's
   homogeneous lines, then a total-least-squares refit over the consensus set
   (Requirement 5.2).
5. :func:`enforce_orthogonality` -- recovers the focal length from the
   orthogonality constraint of each vanishing point pair, accepts only triples
   whose three constraints agree, and labels the survivors ``VPx``, ``VPy``,
   ``VPz`` (Requirement 5.2).
6. :func:`horizon_from_vps` and :func:`horizon_from_contours` -- the horizon as
   the join of the two horizontal vanishing points, or, when fewer than two are
   recovered, from the vertical extent of the structural contours
   (Requirements 5.3, 6.2).

Angle convention: a segment's direction is ``arctan2(dy, dx)`` reduced modulo
``pi``, so it lives in ``[0, pi)`` and a segment has one angle regardless of
which endpoint the detector reported first. Image ``y`` grows downward, so
``pi/2`` is vertical on screen and ``0`` is horizontal.

Line convention: a segment becomes the homogeneous line ``p1 x p2`` scaled so
``a^2 + b^2 = 1``. Then ``l . (x, y, 1)`` is the *signed pixel distance* from
``(x, y)`` to the line, which is the quantity every consensus test below is
stated in. A horizon is returned in the same normalisation, with the sign fixed
so ``b >= 0``, so two horizons for the same geometry compare equal componentwise
instead of differing by an arbitrary factor of -1.

Requirements: 5.1, 5.2, 5.3, 6.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import Final, Mapping, Sequence

import cv2
import numpy as np

from backend.config import Settings, get_settings
from backend.schemas import GeometryMode, PlaneName

#: An image-space vanishing point in pixels. Always finite: a direction whose
#: vanishing point lies at (or effectively at) infinity is reported as ``None``,
#: never as a huge coordinate, so callers cannot mistake one for the other.
VanishingPoint = tuple[float, float]

#: A homogeneous image line ``(a, b, c)`` normalised to ``a^2 + b^2 = 1``, so
#: ``a*x + b*y + c`` is a signed distance in pixels.
Line = tuple[float, float, float]

__all__ = [
    "detect_line_segments",
    "cluster_by_direction",
    "estimate_horizon_hint",
    "estimate_vanishing_point",
    "enforce_orthogonality",
    "horizon_from_vps",
    "horizon_from_contours",
    "principal_point",
    "default_focal_guess",
    "focal_from_vps",
    "plane_frame",
    "homography_from_frame",
    "homography_from_quad",
    "metric_quad_from_image_quad",
    "invert_homography",
    "reprojection_rmse",
    "calibrate",
    "Calibration",
    "PlaneFrame",
    "VanishingPoint",
    "Line",
    "MIN_SEGMENT_LENGTH_FRACTION",
    "VERTICAL_TOLERANCE_RAD",
    "ANGLE_BANDWIDTH_RAD",
    "VP_MAX_DIAGONALS",
    "FOCAL_GUESS_DIAGONAL_FRACTION",
    "PLANE_AXES",
    "PLANE_NORMAL_AXIS",
    "REPROJECTION_GRID_SAMPLES",
]

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

#: Segments shorter than this fraction of the image diagonal are dropped as
#: noise on both detector paths. Two percent of the diagonal is about 26 px on a
#: 1024x768 frame -- long enough that the endpoint positions pin down a
#: direction, short enough to keep the grout lines of a distant tile row.
MIN_SEGMENT_LENGTH_FRACTION: Final[float] = 0.02

#: Half-width of the vertical group, measured from ``pi/2``. Room verticals are
#: never exactly vertical on screen: they converge on ``VPy``, which tilts an
#: edge near the frame border by several degrees under a normal downward pitch.
#: Fifteen degrees covers that convergence while still excluding the steep
#: depth-going edges of a side wall, which sit far closer to 45 degrees.
VERTICAL_TOLERANCE_RAD: Final[float] = math.radians(15.0)

#: Flat-kernel bandwidth of the angle mean-shift. Two structural directions in a
#: rectangular room are separated by far more than this once projected, while the
#: spread *within* one direction's family is smaller, because convergence toward
#: a shared vanishing point fans the family out only gradually.
ANGLE_BANDWIDTH_RAD: Final[float] = math.radians(8.0)

#: Bin count of the angle histogram the mean-shift runs over: one bin per degree
#: across ``[0, pi)``. Seeding from occupied bins rather than from every segment
#: makes the cost independent of the detected segment count.
_ANGLE_HIST_BINS: Final[int] = 180

#: Mean-shift iteration cap and convergence radius. A flat kernel on a 180-bin
#: histogram settles in a handful of steps; the cap only guards a limit cycle
#: between two equally weighted bins.
_MEAN_SHIFT_MAX_ITERATIONS: Final[int] = 64
_MEAN_SHIFT_CONVERGENCE_RAD: Final[float] = 1e-5

#: Modes closer together than this are the same mode reached from two seeds.
_MODE_MERGE_RAD: Final[float] = math.radians(3.0)

#: Non-vertical directional groups returned. A rectangular room has exactly two
#: horizontal families, so this is a cap on the candidates, not a target.
_MAX_NON_VERTICAL_GROUPS: Final[int] = 2

#: Canny's upper threshold is set at this percentile of the image's own Sobel
#: gradient magnitude, so roughly the strongest three percent of gradient pixels
#: seed an edge. Keying off the gradient distribution rather than the intensity
#: median is what makes the fallback work on a low-contrast interior: a pale wall
#: photographed flat has a high median and a *small* gradient range, and the
#: common ``1.33 * median`` heuristic thresholds every one of its edges away.
_CANNY_GRADIENT_PERCENTILE: Final[float] = 97.0
_CANNY_LOW_RATIO: Final[float] = 0.4

#: Floor on the upper threshold. Without it a perfectly flat frame would compute
#: a threshold of zero and report noise as edges.
_CANNY_MIN_HIGH: Final[int] = 16

#: Hough accumulator votes required, as a fraction of ``minLineLength``. A
#: perfectly straight edge of length ``L`` casts about ``L`` votes, so half of
#: the minimum length tolerates a gappy edge without admitting noise.
_HOUGH_VOTE_FRACTION: Final[float] = 0.5
_HOUGH_MIN_VOTES: Final[int] = 15

#: Hough ``maxLineGap`` as a fraction of the image diagonal: roughly one percent,
#: enough to bridge a grout line crossed by a shadow edge.
_HOUGH_GAP_FRACTION: Final[float] = 0.01

#: Longest segments per group used for the horizon hint's pairwise
#: intersections. Capping the group bounds the pair count at a few hundred, and
#: the longest segments are the best-conditioned ones anyway.
_HINT_MAX_SEGMENTS: Final[int] = 40

#: Pairwise intersections farther than this many image diagonals from the frame
#: are treated as "at infinity" and excluded from the hint's median. Without the
#: cap, a near-frontal group whose vanishing point is genuinely at infinity would
#: drag the hint to an arbitrary row.
_HINT_MAX_DIAGONALS: Final[float] = 50.0

#: A recovered vanishing point farther than this many image diagonals from the
#: principal point is reported as ``None``, i.e. treated as lying at infinity.
#:
#: The cap is what discharges the "fewer than three vanishing points" branch of
#: Requirement 6.1 for a near-frontal camera. As the camera's yaw approaches
#: zero the lateral direction becomes parallel to the image plane and its true
#: vanishing point runs off to infinity -- on a 1600x1200 frame it passes 40
#: diagonals below one degree of yaw. Such a point has no consensus set at any
#: pixel threshold and carries no usable direction, so accepting it would hand
#: the plane frames a fabricated axis. Rejecting it instead routes the plane to
#: the planar fallback, which is the documented behaviour.
#:
#: Thirty diagonals is generous on purpose: at the synthetic fixture's default
#: eight degrees of yaw the true lateral vanishing point already sits five and a
#: half diagonals out, and it is perfectly recoverable there.
VP_MAX_DIAGONALS: Final[float] = 30.0

#: Assumed focal length as a fraction of the image diagonal, used only as a
#: prior. Corresponds to a diagonal field of view of about 71 degrees, which is
#: near the middle of the range phone cameras ship, and reproduces the synthetic
#: fixture's 1400 px focal length on its 1600x1200 frame exactly.
FOCAL_GUESS_DIAGONAL_FRACTION: Final[float] = 0.7

#: Smallest consensus set a vanishing point may be accepted from. Any pair of
#: lines intersects somewhere, so two inliers is no evidence at all; requiring a
#: third means at least one line the hypothesis did not come from agrees with it.
_VP_MIN_INLIERS: Final[int] = 3

#: Longest lines a group is trimmed to before estimation. Bounds the RANSAC
#: scoring matrix on a heavily textured photograph without discarding any
#: well-conditioned constraint, since a long line pins a direction down far
#: better than a short one.
_VP_MAX_LINES: Final[int] = 1500

#: Total-least-squares refits after RANSAC. The first moves the estimate onto
#: the consensus set, the second picks up the lines that only became inliers
#: because of the first; a third changes nothing measurable.
_VP_REFINE_PASSES: Final[int] = 2

#: Upper bound on the foreshortening factor by which the inlier threshold is
#: scaled. See :func:`_inlier_mask` for why the factor exists; the cap stops a
#: hypothesis hundreds of diagonals away from admitting the entire group.
_VP_THRESHOLD_SCALE_CAP: Final[float] = 2000.0

#: Seed of the RANSAC pair sampler. Fixed so a photograph always calibrates to
#: the same numbers -- a cached scene and a re-analysis of the same upload must
#: not disagree, and the tests need a stable reference.
_VP_RANSAC_SEED: Final[int] = 0

#: ``|sin|`` of the angle between two sampled lines, below which their
#: intersection is numerically meaningless and the pair is skipped.
_MIN_CROSS_SINE: Final[float] = 1e-9

#: How vertical a direction must look before it may be labelled ``VPy``: within
#: 45 degrees of the image vertical axis.
_VERTICAL_LABEL_MIN: Final[float] = math.cos(math.radians(45.0))

#: Second gate on the ``VPy`` label: ``tan`` of the smallest angle the direction
#: may make with the optical axis, so the vanishing point has to sit at least
#: ``0.3 * focal`` from the principal point.
#:
#: The first gate alone is not enough, and the failure is not a corner case. A
#: vanishing point close to the principal point has an *arbitrary* bearing from
#: it -- a few pixels of noise swing it through ninety degrees -- and the depth
#: vanishing point is close to the principal point by definition, because depth
#: is the direction the camera is pointing. So the depth point routinely reads as
#: "vertical" on bearing alone and steals the ``VPy`` label, which then costs the
#: run its horizon: :func:`horizon_from_vps` needs two *horizontal* labels, and
#: mislabelling one of them as vertical leaves only one.
#:
#: Requiring the direction to be genuinely steep fixes it, because a room's
#: verticals only vanish near the principal point when the camera is pointed
#: almost straight down. At 0.3 the gate admits pitches up to about 73 degrees,
#: well past anything a shopper photographing a room produces, while rejecting
#: the depth point of a normal view by a comfortable margin.
_VERTICAL_MIN_AXIS_TANGENT: Final[float] = 0.3

#: Where the contour-derived horizon sits inside a lone plane's vertical extent,
#: as a signed fraction of that extent measured from its midpoint (negative is
#: upward). The floor's value places the horizon on the floor's top edge, which
#: is where it has to be: the visible floor is the ground below the horizon, so
#: no floor pixel can sit above it. A wall spans the horizon, so its midpoint is
#: already close, nudged up because a photographer standing in a room holds the
#: camera above the halfway point of the wall's *visible* height.
_HORIZON_BIAS: Final[dict[str, float]] = {
    "floor": -0.5,
    "wall_back": 0.0,
    "wall_left": -0.10,
    "wall_right": -0.10,
}

#: Plane names treated as walls when deriving a horizon from contours.
_WALL_NAMES: Final[tuple[str, ...]] = ("wall_back", "wall_left", "wall_right")

_EPS: Final[float] = 1e-12
_HALF_PI: Final[float] = math.pi / 2.0


# --------------------------------------------------------------------------- #
# Segment primitives
# --------------------------------------------------------------------------- #


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    """Coerce an input to a contiguous single-channel ``uint8`` image.

    The detectors are documented to take a grayscale frame, but the callers hold
    a BGR photograph, so accepting either here keeps the conversion in one place
    instead of at every call site.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected a numpy array, got {type(image)!r}")
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError(f"unsupported channel count {image.shape[2]}")
    elif image.ndim != 2:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def _empty_segments() -> np.ndarray:
    """The canonical "no segments" value: shape ``(0,4)``, not ``(0,)``.

    Returning a correctly shaped empty array means callers can index columns
    unconditionally, which is why every early exit routes through here.
    """
    return np.empty((0, 4), dtype=np.float32)


def _as_segments(value: object) -> np.ndarray:
    """Validate and normalise an ``(N,4)`` endpoint array."""
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        return _empty_segments()
    if array.ndim == 0:
        raise ValueError("segments must be an (N,4) array, got a scalar")
    array = array.reshape(-1, array.shape[-1])
    if array.shape[1] != 4:
        raise ValueError(f"segments must have 4 columns, got {array.shape[1]}")
    return np.ascontiguousarray(array)


def _lengths(segments: np.ndarray) -> np.ndarray:
    """Euclidean length of each segment, as ``float64``."""
    if len(segments) == 0:
        return np.empty(0, dtype=np.float64)
    delta = segments[:, 2:4].astype(np.float64) - segments[:, 0:2].astype(np.float64)
    return np.hypot(delta[:, 0], delta[:, 1])


def _angles(segments: np.ndarray) -> np.ndarray:
    """Direction angle of each segment in ``[0, pi)``."""
    if len(segments) == 0:
        return np.empty(0, dtype=np.float64)
    delta = segments[:, 2:4].astype(np.float64) - segments[:, 0:2].astype(np.float64)
    return np.mod(np.arctan2(delta[:, 1], delta[:, 0]), math.pi)


def _angle_distance(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray:
    """Distance between two ``[0, pi)`` angles, respecting the wrap at ``pi``.

    Angles are direction-free, so 1 degree and 179 degrees are 2 degrees apart,
    not 178. Every comparison in this module goes through here for that reason.
    """
    delta = np.mod(np.abs(np.subtract(a, b)), math.pi)
    return np.minimum(delta, math.pi - delta)


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> float | None:
    """Weighted mean of ``[0, pi)`` angles, or ``None`` if it is undefined.

    Doubling maps the ``pi``-periodic angles onto the full circle, where the
    usual ``atan2`` of summed unit vectors is well defined; halving the result
    brings it back. Averaging the raw values instead would place the mean of 1
    and 179 degrees at 90.
    """
    doubled = 2.0 * angles
    sin_sum = float(np.sum(weights * np.sin(doubled)))
    cos_sum = float(np.sum(weights * np.cos(doubled)))
    if abs(sin_sum) < _EPS and abs(cos_sum) < _EPS:
        return None
    return float(np.mod(0.5 * math.atan2(sin_sum, cos_sum), math.pi))


def _diagonal(image_shape: Sequence[int]) -> float:
    """Image diagonal in pixels from a ``(height, width, ...)`` shape."""
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"degenerate image shape {(height, width)}")
    return math.hypot(float(width), float(height))


# --------------------------------------------------------------------------- #
# Line detection (Requirement 5.1)
# --------------------------------------------------------------------------- #


def _detect_lsd(gray: np.ndarray) -> np.ndarray:
    """Run OpenCV's line segment detector.

    Raises:
        cv2.error or AttributeError: when the installed build has no LSD. Both
            are caught by :func:`detect_line_segments`, which then falls back.
    """
    detector = cv2.createLineSegmentDetector()
    detected = detector.detect(gray)
    lines = detected[0] if isinstance(detected, tuple) else detected
    if lines is None:
        return _empty_segments()
    return _as_segments(lines)


def _detect_hough(gray: np.ndarray, min_length: float, diagonal: float) -> np.ndarray:
    """Canny edges plus a probabilistic Hough transform, the LSD-free path.

    Thresholds are keyed off the image median so a dim photograph is not edged
    away entirely, and ``minLineLength``/``maxLineGap`` scale with the diagonal
    so the fallback behaves the same at any capture resolution.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    # The same 3x3 Sobel and L2 norm Canny itself uses below, so the percentile
    # is measured in exactly the units the thresholds are compared against.
    high = float(np.percentile(magnitude, _CANNY_GRADIENT_PERCENTILE))
    high = max(float(_CANNY_MIN_HIGH), min(high, 255.0))
    edges = cv2.Canny(
        blurred,
        int(round(_CANNY_LOW_RATIO * high)),
        int(round(high)),
        apertureSize=3,
        L2gradient=True,
    )

    votes = max(_HOUGH_MIN_VOTES, int(round(_HOUGH_VOTE_FRACTION * min_length)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=math.pi / 180.0,
        threshold=votes,
        minLineLength=float(min_length),
        maxLineGap=max(2.0, _HOUGH_GAP_FRACTION * diagonal),
    )
    if lines is None:
        return _empty_segments()
    return _as_segments(lines)


def detect_line_segments(gray: np.ndarray) -> np.ndarray:
    """Extract straight line segments from a photograph (Requirement 5.1).

    ``cv2.createLineSegmentDetector`` is tried first because LSD gives
    sub-pixel, well-localised endpoints and needs no threshold tuning. Some
    OpenCV distributions ship without it, so a ``cv2.error`` or
    ``AttributeError`` from either the constructor or the ``detect`` call routes
    to a Canny + ``cv2.HoughLinesP`` fallback. Both paths return the same array
    layout, so no caller branches on which detector ran.

    Args:
        gray: grayscale or BGR photograph. Non-``uint8`` input is clipped and
            cast; BGR and BGRA input is converted.

    Returns:
        ``(N,4) float32`` endpoints ``(x1, y1, x2, y2)``, with every segment at
        least :data:`MIN_SEGMENT_LENGTH_FRACTION` of the image diagonal long.
        The array is ``(0,4)`` when nothing survives, never ``(0,)``.
    """
    image = _as_gray_u8(gray)
    diagonal = _diagonal(image.shape)
    min_length = MIN_SEGMENT_LENGTH_FRACTION * diagonal

    try:
        segments = _detect_lsd(image)
    except (cv2.error, AttributeError):
        segments = _detect_hough(image, min_length, diagonal)

    if len(segments) == 0:
        return _empty_segments()
    keep = _lengths(segments) >= min_length
    if not keep.any():
        return _empty_segments()
    return np.ascontiguousarray(segments[keep])


# --------------------------------------------------------------------------- #
# Directional clustering (Requirement 5.2)
# --------------------------------------------------------------------------- #


def _mean_shift_modes(angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Angle modes from a flat-kernel mean-shift over the angle histogram.

    Seeds are the occupied histogram bins, so the work scales with the 180-bin
    grid rather than with the segment count. Each seed climbs to a mode; modes
    that land within :data:`_MODE_MERGE_RAD` of one another are the same mode
    found twice and are collapsed, keeping the heavier one.
    """
    edges = np.linspace(0.0, math.pi, _ANGLE_HIST_BINS + 1)
    hist, _ = np.histogram(angles, bins=edges, weights=weights)
    centres = 0.5 * (edges[:-1] + edges[1:])
    occupied = np.flatnonzero(hist > 0.0)
    if len(occupied) == 0:  # pragma: no cover - callers filter empty input
        return np.empty(0, dtype=np.float64)

    modes: list[tuple[float, float]] = []  # (mode angle, kernel weight)
    for seed in centres[occupied]:
        mode = float(seed)
        weight = 0.0
        for _ in range(_MEAN_SHIFT_MAX_ITERATIONS):
            inside = _angle_distance(centres, mode) <= ANGLE_BANDWIDTH_RAD
            kernel = np.where(inside, hist, 0.0)
            weight = float(kernel.sum())
            if weight <= 0.0:  # pragma: no cover - the seed bin is always inside
                break
            shifted = _circular_mean(centres, kernel)
            if shifted is None:  # pragma: no cover - a positive kernel has a mean
                break
            moved = float(_angle_distance(shifted, mode))
            mode = shifted
            if moved < _MEAN_SHIFT_CONVERGENCE_RAD:
                break
        if weight > 0.0:
            modes.append((mode, weight))

    merged: list[tuple[float, float]] = []
    for mode, weight in sorted(modes, key=lambda item: item[1], reverse=True):
        if any(_angle_distance(mode, kept) <= _MODE_MERGE_RAD for kept, _ in merged):
            continue
        merged.append((mode, weight))
    return np.asarray([mode for mode, _ in merged], dtype=np.float64)


def _mode_assignment(angles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Index of each segment's nearest angle mode, or an empty array."""
    modes = _mean_shift_modes(angles, weights)
    if len(modes) == 0:  # pragma: no cover - callers filter empty input
        return np.empty(0, dtype=np.intp)
    distances = _angle_distance(angles[:, None], modes[None, :])
    return np.argmin(distances, axis=1)


def _dominant_and_residual(
    segments: np.ndarray, angles: np.ndarray, weights: np.ndarray
) -> list[np.ndarray]:
    """Split non-vertical segments into the dominant angle mode and the rest.

    The design's clustering reads as "the two largest non-vertical clusters", and
    that is what this returns *when* both horizontal families form compact angle
    modes -- which happens only while both horizontal vanishing points sit far
    outside the frame. In the common straight-on room view the second family's
    vanishing point lands near the picture, its lines radiate outward, and its
    angles spread across the whole histogram instead of piling into one mode. It
    is then recoverable as the *complement* of the dominant mode, not as a mode
    of its own, so the second group is the union of every non-dominant mode.

    That union carries outliers, but the vanishing point estimator is a RANSAC,
    so absorbing outliers is exactly what it is built to do; fragmenting the
    family across modes it cannot see past is not.

    Weighting is by summed normalised length, not by count: one long wall edge
    constrains a vanishing point more than several short furniture edges.
    """
    assignment = _mode_assignment(angles, weights)
    if len(assignment) == 0:  # pragma: no cover - callers filter empty input
        return []
    order = sorted(
        np.unique(assignment),
        key=lambda index: float(weights[assignment == index].sum()),
        reverse=True,
    )
    dominant = assignment == order[0]
    groups = [np.ascontiguousarray(segments[dominant])]
    if not dominant.all():
        groups.append(np.ascontiguousarray(segments[~dominant]))
    return groups


def _directional_groups(
    segments: np.ndarray,
    image_shape: Sequence[int],
    settings: Settings | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Split segments into non-vertical groups and the vertical group.

    Returns:
        ``(non_vertical, vertical)`` where ``non_vertical`` holds at most
        :data:`_MAX_NON_VERTICAL_GROUPS` groups ordered by descending weight and
        ``vertical`` is ``None`` when the vertical family is too small to fit a
        vanishing point to. Both are filtered by ``vp_min_cluster_size``.
    """
    resolved = settings if settings is not None else get_settings()
    min_size = int(resolved.vp_min_cluster_size)
    segments = _as_segments(segments)
    if len(segments) == 0:
        return [], None

    # Length weights are normalised by the diagonal so the histogram the
    # mean-shift runs over means the same thing at any capture resolution.
    diagonal = _diagonal(image_shape)
    weights = _lengths(segments) / diagonal
    angles = _angles(segments)

    is_vertical = _angle_distance(angles, _HALF_PI) <= VERTICAL_TOLERANCE_RAD
    vertical_idx = np.flatnonzero(is_vertical)
    vertical = (
        np.ascontiguousarray(segments[vertical_idx]) if len(vertical_idx) >= min_size else None
    )

    rest = np.flatnonzero(~is_vertical)
    non_vertical: list[np.ndarray] = []
    if len(rest) >= min_size:
        candidates = _dominant_and_residual(segments[rest], angles[rest], weights[rest])
        non_vertical = [
            group for group in candidates[:_MAX_NON_VERTICAL_GROUPS] if len(group) >= min_size
        ]
    return non_vertical, vertical


def cluster_by_direction(
    segments: np.ndarray,
    image_shape: Sequence[int],
    settings: Settings | None = None,
) -> list[np.ndarray]:
    """Group segments into candidate directional families (Requirement 5.2).

    Segments within :data:`VERTICAL_TOLERANCE_RAD` of ``pi/2`` form the vertical
    group directly -- a room's verticals are the one family whose screen
    orientation is known in advance, so guessing it from the data would only
    risk splitting it. The remainder is clustered by a 1-D mean-shift over the
    length-weighted angle histogram; the heaviest mode is the dominant
    horizontal candidate and everything else non-vertical is the secondary one,
    for the reason :func:`_dominant_and_residual` sets out.

    Args:
        segments: ``(N,4)`` endpoints from :func:`detect_line_segments`.
        image_shape: ``(height, width)`` of the source photograph; sets the
            length scale the histogram weights are normalised by.
        settings: overrides the process settings, whose ``vp_min_cluster_size``
            is the smallest group a vanishing point may be fitted to.

    Returns:
        Up to three ``(M,4) float32`` groups: the dominant non-vertical family,
        the secondary non-vertical family, then the vertical family. Ordering
        matches the design's ``VPx``, ``VPz``, ``VPy`` labelling order, though
        the labels themselves are settled later, by orthogonality. Groups
        smaller than ``vp_min_cluster_size`` are discarded, so the list may be
        shorter than three or empty.
    """
    non_vertical, vertical = _directional_groups(segments, image_shape, settings)
    clusters = list(non_vertical)
    if vertical is not None:
        clusters.append(vertical)
    return clusters


# --------------------------------------------------------------------------- #
# Horizon hint (Requirement 5.2, consumed by the Segmenter)
# --------------------------------------------------------------------------- #


def _homogeneous_lines(segments: np.ndarray) -> np.ndarray:
    """Each segment as a homogeneous line ``p1 x p2``, normalised.

    Scaling so ``a^2 + b^2 = 1`` makes the third component of a cross product
    between two such lines the sine of the angle between them, which is what
    lets :func:`_group_vanishing_point` recognise a near-parallel pair.
    """
    points = segments.astype(np.float64)
    p1 = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    p2 = np.column_stack((points[:, 2], points[:, 3], np.ones(len(points))))
    lines = np.cross(p1, p2)
    norm = np.hypot(lines[:, 0], lines[:, 1])
    norm[norm < _EPS] = 1.0
    return lines / norm[:, None]


def _group_vanishing_point(
    group: np.ndarray, diagonal: float
) -> tuple[float, float] | None:
    """Median pairwise intersection of a group's lines, or ``None``.

    This is deliberately not the RANSAC-and-refine estimator calibration uses:
    the hint only has to land the horizon in the right part of the frame, and a
    median over all pairs is both cheaper and, being order-statistical, immune to
    the handful of wild intersections a near-parallel pair produces. Candidates
    beyond :data:`_HINT_MAX_DIAGONALS` are dropped first, so a family whose
    vanishing point is genuinely at infinity reports no hint rather than an
    arbitrary one.
    """
    if len(group) < 2:
        return None
    if len(group) > _HINT_MAX_SEGMENTS:
        longest = np.argsort(_lengths(group))[::-1][:_HINT_MAX_SEGMENTS]
        group = group[longest]

    lines = _homogeneous_lines(group)
    i, j = np.triu_indices(len(lines), k=1)
    candidates = np.cross(lines[i], lines[j])
    w = candidates[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = candidates[:, :2] / w[:, None]
    limit = _HINT_MAX_DIAGONALS * diagonal
    usable = np.isfinite(xy).all(axis=1) & (np.abs(xy) <= limit).all(axis=1)
    if not usable.any():
        return None
    return (float(np.median(xy[usable, 0])), float(np.median(xy[usable, 1])))


def estimate_horizon_hint(
    gray: np.ndarray,
    segments: np.ndarray | None = None,
    settings: Settings | None = None,
) -> float:
    """Cheap horizon row the Segmenter scores structural planes against.

    The Segmenter needs to know roughly where the horizon sits before any plane
    exists, and calibration needs the same line segments afterwards. Passing
    ``segments`` in lets a request detect once and serve both, which is the
    whole reason this estimate lives in the Geometry_Engine rather than in the
    Segmenter.

    Each non-vertical group contributes the median of its pairwise line
    intersections -- its apparent horizontal vanishing point. With two of them
    the hint is where the joining line crosses the image's vertical centre
    column; with one it is that point's row; with none it is mid-height.

    Args:
        gray: grayscale or BGR photograph.
        segments: already-detected ``(N,4)`` segments. Detected here when
            omitted.
        settings: overrides the process settings.

    Returns:
        A horizon row in pixels, always finite and always inside
        ``[0, height - 1]``, so the caller never has to guard it.
    """
    image = _as_gray_u8(gray)
    height, width = image.shape[:2]
    midpoint = (height - 1) / 2.0
    if segments is None:
        segments = detect_line_segments(image)

    non_vertical, _vertical = _directional_groups(segments, image.shape, settings)
    diagonal = _diagonal(image.shape)
    points = [
        point
        for point in (_group_vanishing_point(group, diagonal) for group in non_vertical)
        if point is not None
    ]

    if not points:
        hint = midpoint
    elif len(points) == 1:
        hint = points[0][1]
    else:
        (x0, y0), (x1, y1) = points[0], points[1]
        centre_x = (width - 1) / 2.0
        if abs(x1 - x0) < _EPS:
            # Both vanishing points share a column: the joining line is vertical
            # and has no single row, so average the two rows instead.
            hint = 0.5 * (y0 + y1)
        else:
            hint = y0 + (y1 - y0) * (centre_x - x0) / (x1 - x0)

    if not math.isfinite(hint):  # pragma: no cover - guarded by the finite filter
        hint = midpoint
    return float(min(max(hint, 0.0), float(height - 1)))

# --------------------------------------------------------------------------- #
# Camera conventions
# --------------------------------------------------------------------------- #


def principal_point(image_shape: Sequence[int]) -> tuple[float, float]:
    """Assumed principal point of a photograph: the image centre.

    A single uncalibrated upload gives nothing to estimate a principal point
    offset from, so the whole engine assumes it sits at the centre of the frame
    and every consumer must agree on which "centre" that is. The synthetic
    fixture builds its intrinsics with the same ``(n - 1) / 2`` convention, so
    ground truth and recovered geometry are stated in one coordinate frame.

    Args:
        image_shape: ``(height, width, ...)``.

    Returns:
        ``(x, y)`` in pixels.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"degenerate image shape {(height, width)}")
    return ((width - 1) / 2.0, (height - 1) / 2.0)


def default_focal_guess(image_shape: Sequence[int]) -> float:
    """Prior on the focal length in pixels, from the image diagonal alone.

    Only ever used as a tiebreak: :func:`enforce_orthogonality` recovers the
    focal length from the vanishing points themselves and falls back on this
    prior solely to decide which vanishing point pair to keep when no triple is
    self-consistent. Nothing metric depends on it.
    """
    return FOCAL_GUESS_DIAGONAL_FRACTION * _diagonal(image_shape)


# --------------------------------------------------------------------------- #
# Vanishing point estimation (Requirement 5.2)
# --------------------------------------------------------------------------- #


def _midpoints(segments: np.ndarray) -> np.ndarray:
    """``(N,2) float64`` midpoint of each segment."""
    points = segments.astype(np.float64)
    return 0.5 * (points[:, 0:2] + points[:, 2:4])


def _threshold_scale(
    points: np.ndarray, midpoints: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    """Per-hypothesis, per-line factor the inlier threshold is scaled by.

    The design measures a line's disagreement with a vanishing point as the
    point-to-line distance ``|l . (x, y, 1)|``, and that is exactly what
    :func:`_inlier_mask` computes. Compared against a *fixed* pixel threshold,
    though, that quantity is unusable for any vanishing point outside the frame,
    and in a room photograph the lateral one normally is: a line whose direction
    is off by an angle ``d`` misses a vanishing point ``D`` pixels away by
    ``D sin(d)``, so at ``D = 11000`` px -- the synthetic fixture's true lateral
    vanishing point -- even a tenth of a degree of endpoint noise misses by
    19 px. Every line in a correct group would be an outlier and no vanishing
    point would ever be recovered.

    The fix keeps the design's distance but restores the meaning of
    ``vp_inlier_threshold_px`` as a tolerance on *observed pixels*. A segment of
    length ``L`` whose endpoints are misplaced by ``e`` perpendicular to itself
    tilts by ``sin(d) ~ 2e / L``, which throws the point-to-line distance out by
    ``2eD / L``. Scaling the threshold by ``2D / L`` therefore makes the test
    "the segment's endpoints agree with the vanishing point to within
    ``vp_inlier_threshold_px``", which is the tolerance an endpoint detector's
    accuracy can actually be stated in, and is scale-free in the bargain: the
    same setting behaves the same for a near vanishing point and a far one.

    Returns:
        ``(M, N)`` factors, at least 1.0 and at most
        :data:`_VP_THRESHOLD_SCALE_CAP`.
    """
    span = np.hypot(
        points[:, 0:1] - midpoints[None, :, 0], points[:, 1:2] - midpoints[None, :, 1]
    )
    scale = 2.0 * span / np.maximum(lengths, 1.0)[None, :]
    return np.clip(scale, 1.0, _VP_THRESHOLD_SCALE_CAP)


def _inlier_masks(
    points: np.ndarray,
    lines: np.ndarray,
    midpoints: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """``(M, N)`` boolean consensus matrix for ``M`` candidate vanishing points.

    ``distance`` is the design's point-to-line distance in pixels, exact because
    ``lines`` is normalised to ``a^2 + b^2 = 1``.
    """
    homogeneous = np.column_stack((points, np.ones(len(points))))
    distance = np.abs(homogeneous @ lines.T)
    return distance <= threshold * _threshold_scale(points, midpoints, lengths)


def _inlier_mask(
    point: np.ndarray,
    lines: np.ndarray,
    midpoints: np.ndarray,
    lengths: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Consensus set of a single candidate, as an ``(N,)`` boolean mask."""
    return _inlier_masks(
        np.asarray(point, dtype=np.float64).reshape(1, 2),
        lines,
        midpoints,
        lengths,
        threshold,
    )[0]


def _hypothesis_pairs(count: int, iterations: int) -> np.ndarray:
    """Index pairs RANSAC draws its two-line hypotheses from.

    Small groups are enumerated exhaustively rather than sampled: below the
    iteration budget that is both cheaper and strictly better, since it cannot
    miss the one pair that would have found the right consensus set. Larger
    groups are sampled from a fixed seed, so calibration is reproducible.
    """
    if count < 2:
        return np.empty((0, 2), dtype=np.intp)
    total = count * (count - 1) // 2
    if total <= max(iterations, 1):
        first, second = np.triu_indices(count, k=1)
        return np.column_stack((first, second)).astype(np.intp)

    rng = np.random.default_rng(_VP_RANSAC_SEED)
    first = rng.integers(0, count, size=iterations)
    # Offsetting by at least one and wrapping guarantees a distinct partner
    # without a rejection loop.
    second = (first + rng.integers(1, count, size=iterations)) % count
    return np.column_stack((first, second)).astype(np.intp)


def _total_least_squares_vp(
    lines: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> np.ndarray | None:
    """Refit a vanishing point over a consensus set (design: TLS refinement).

    A vanishing point lies on every one of its lines, so the homogeneous point
    ``v`` satisfies ``l . v = 0`` for all of them and is the null vector of the
    stacked line matrix. With noise there is no exact null vector, and the
    least-squares answer under ``|v| = 1`` is the right singular vector of the
    smallest singular value.

    Rows are weighted by segment length, so a long wall edge outvotes a short
    furniture edge in proportion to how much better it localises the direction.
    """
    rows = lines[mask] * weights[mask][:, None]
    if len(rows) < 2:
        return None
    _u, _s, vt = np.linalg.svd(rows, full_matrices=False)
    vector = vt[-1]
    if abs(vector[2]) < _EPS:
        return None  # the refit converged on a point at infinity
    return np.array([vector[0] / vector[2], vector[1] / vector[2]], dtype=np.float64)


def _vp_is_usable(point: np.ndarray, centre: tuple[float, float], limit: float) -> bool:
    """Whether a candidate is finite and inside the :data:`VP_MAX_DIAGONALS` cap."""
    if not np.isfinite(point).all():
        return False
    return bool(math.hypot(point[0] - centre[0], point[1] - centre[1]) <= limit)


def estimate_vanishing_point(
    cluster: np.ndarray,
    image_shape: Sequence[int],
    settings: Settings | None = None,
) -> tuple[VanishingPoint, float] | None:
    """Fit a vanishing point to one directional group (Requirement 5.2).

    Each segment becomes the homogeneous line through its endpoints, and the
    group's vanishing point is where those lines meet. RANSAC proposes
    intersections of line pairs and scores each by the consensus set it
    attracts; the winner is then refitted by total least squares over that set,
    twice. The split matters: RANSAC is what survives the furniture edges and
    the mis-grouped segments that :func:`cluster_by_direction` deliberately
    leaves in the secondary group, while the refit is what turns a two-line
    estimate into one accurate enough to compare against analytic ground truth.

    Args:
        cluster: ``(N,4)`` segment endpoints from :func:`cluster_by_direction`.
        image_shape: ``(height, width)`` of the source photograph.
        settings: overrides the process settings. Reads
            ``vp_ransac_iterations`` and ``vp_inlier_threshold_px``.

    Returns:
        ``((x, y), support)`` where ``support`` is the fraction of the group's
        total segment length that agrees with the fit, in ``(0, 1]``. ``None``
        when the group is too small, when no candidate attracts at least
        :data:`_VP_MIN_INLIERS` lines, or when the direction's vanishing point
        is at or effectively at infinity -- the last of which is a normal
        outcome for a near-frontal camera and is what routes a plane to the
        planar fallback of Requirement 6.1 rather than a fabricated direction.
    """
    resolved = settings if settings is not None else get_settings()
    segments = _as_segments(cluster)
    if len(segments) < 2:
        return None

    centre = principal_point(image_shape)
    limit = VP_MAX_DIAGONALS * _diagonal(image_shape)
    threshold = float(resolved.vp_inlier_threshold_px)

    lengths = _lengths(segments)
    if len(segments) > _VP_MAX_LINES:
        longest = np.argsort(lengths)[::-1][:_VP_MAX_LINES]
        segments, lengths = segments[longest], lengths[longest]
    total_length = float(lengths.sum())
    if total_length <= 0.0:
        return None

    lines = _homogeneous_lines(segments)
    mids = _midpoints(segments)
    # Weights are O(1) so the SVD is well scaled regardless of image size.
    weights = lengths / max(float(lengths.max()), _EPS)

    pairs = _hypothesis_pairs(len(lines), int(resolved.vp_ransac_iterations))
    if len(pairs) == 0:  # pragma: no cover - guarded by the len < 2 check above
        return None
    candidates = np.cross(lines[pairs[:, 0]], lines[pairs[:, 1]])
    # For normalised lines the third component is the sine of the angle between
    # them, so this drops exactly the near-parallel pairs.
    keep = np.abs(candidates[:, 2]) > _MIN_CROSS_SINE
    if not keep.any():
        return None
    candidates = candidates[keep]
    points = candidates[:, :2] / candidates[:, 2:3]
    usable = np.isfinite(points).all(axis=1) & (
        np.hypot(points[:, 0] - centre[0], points[:, 1] - centre[1]) <= limit
    )
    if not usable.any():
        return None
    points = points[usable]

    masks = _inlier_masks(points, lines, mids, lengths, threshold)
    support = masks @ lengths
    best = int(np.argmax(support))
    point, mask = points[best], masks[best]

    for _ in range(_VP_REFINE_PASSES):
        refined = _total_least_squares_vp(lines, weights, mask)
        if refined is None or not _vp_is_usable(refined, centre, limit):
            # The consensus set turned out to be a parallel family after all.
            return None
        refined_mask = _inlier_mask(refined, lines, mids, lengths, threshold)
        if int(refined_mask.sum()) < _VP_MIN_INLIERS:
            break
        point, mask = refined, refined_mask

    if int(mask.sum()) < _VP_MIN_INLIERS:
        return None
    if not _vp_is_usable(point, centre, limit):  # pragma: no cover - refit is checked
        return None
    return (
        (float(point[0]), float(point[1])),
        float(lengths[mask].sum() / total_length),
    )


# --------------------------------------------------------------------------- #
# Orthogonality and labelling (Requirement 5.2)
# --------------------------------------------------------------------------- #


def _finite_candidates(
    vps: Sequence[VanishingPoint | None],
) -> list[tuple[int, tuple[float, float]]]:
    """Drop ``None`` and non-finite entries, keeping each survivor's position.

    The position is the dominance rank :func:`cluster_by_direction` produced, and
    it is the tiebreak :func:`_label_vps` falls back on.
    """
    out: list[tuple[int, tuple[float, float]]] = []
    for index, vp in enumerate(vps):
        if vp is None:
            continue
        x, y = float(vp[0]), float(vp[1])
        if math.isfinite(x) and math.isfinite(y):
            out.append((index, (x, y)))
    return out


def _pair_focal_squared(
    a: tuple[float, float], b: tuple[float, float], centre: tuple[float, float]
) -> float:
    """``f^2 = -(v_i - pp) . (v_j - pp)`` for one vanishing point pair.

    Two orthogonal directions ``d_i``, ``d_j`` satisfy ``d_i . d_j = 0``, and
    with a centred principal point and unit aspect ratio the directions are
    ``[v - pp, f]``, so the constraint reads
    ``(v_i - pp) . (v_j - pp) + f^2 = 0``. A non-positive result means the pair
    cannot be the image of two orthogonal directions under any focal length.
    """
    ax, ay = a[0] - centre[0], a[1] - centre[1]
    bx, by = b[0] - centre[0], b[1] - centre[1]
    return -(ax * bx + ay * by)


def _orthogonality_residual(
    triple: Sequence[tuple[float, float]], centre: tuple[float, float]
) -> tuple[float, float] | None:
    """Focal length and orthogonality residual of a candidate triple.

    Three vanishing points give three independent estimates of one focal length.
    Their spread is the first half of the residual; the second is how far the
    directions those points imply, under the *shared* focal length, are from
    mutually perpendicular. Both are dimensionless, so a single tolerance covers
    them, and the worse of the two is reported so neither can hide behind the
    other.

    Returns:
        ``(focal_px, residual)``, or ``None`` when any pair fails the positive
        ``f^2`` test and the triple is therefore not the image of an orthogonal
        frame at all.
    """
    focals: list[float] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        squared = _pair_focal_squared(triple[i], triple[j], centre)
        if squared <= 0.0:
            return None
        focals.append(math.sqrt(squared))
    focal = float(np.mean(focals))
    if focal <= 0.0:  # pragma: no cover - a mean of positive roots is positive
        return None
    spread = float(max(abs(value - focal) for value in focals) / focal)

    directions = []
    for vp in triple:
        vector = np.array([vp[0] - centre[0], vp[1] - centre[1], focal], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < _EPS:  # pragma: no cover - focal > 0 keeps the norm positive
            return None
        directions.append(vector / norm)
    perpendicularity = max(
        abs(float(directions[i] @ directions[j])) for i, j in ((0, 1), (0, 2), (1, 2))
    )
    return focal, max(spread, perpendicularity)


def _verticality(vp: tuple[float, float], centre: tuple[float, float]) -> float:
    """How close a vanishing direction is to the image vertical axis, in ``[0,1]``.

    ``1`` means the point sits directly above or below the principal point, so
    its direction is the image's vertical; ``0`` means directly beside it.
    """
    dx, dy = vp[0] - centre[0], vp[1] - centre[1]
    span = math.hypot(dx, dy)
    if span < _EPS:
        return 0.0
    return abs(dy) / span


def _may_be_vertical(
    vp: tuple[float, float], centre: tuple[float, float], focal: float
) -> bool:
    """Whether a candidate is eligible for the ``VPy`` label.

    Both gates have to hold: the bearing from the principal point is near
    vertical, and the direction is steep enough off the optical axis for that
    bearing to mean anything. See :data:`_VERTICAL_MIN_AXIS_TANGENT`.
    """
    if _verticality(vp, centre) < _VERTICAL_LABEL_MIN:
        return False
    span = math.hypot(vp[0] - centre[0], vp[1] - centre[1])
    return span >= _VERTICAL_MIN_AXIS_TANGENT * max(focal, _EPS)


def _label_vps(
    items: Sequence[tuple[int, tuple[float, float]]],
    centre: tuple[float, float],
    focal: float,
    *,
    require_vertical: bool,
) -> dict[str, VanishingPoint | None]:
    """Assign ``VPx``, ``VPy``, ``VPz`` to at most three candidates.

    ``VPy`` is the most vertical direction, as the design specifies, restricted
    to the candidates :func:`_may_be_vertical` admits. Of the two that remain,
    ``VPz`` is the one whose direction is closest to the optical axis --
    equivalently, the one nearer the principal point -- and ``VPx`` is the other.

    That second rule is worth spelling out, because the design words it as
    "dominant" and "secondary" horizontal. The labels are not interchangeable
    downstream: the plane frame table gives ``wall_left`` and ``wall_right`` a
    ``u`` axis along ``VPz`` and calls it "into the room", and gives the floor a
    ``v`` axis along ``VPz`` and calls it "depth". ``VPz`` therefore has to *be*
    the depth axis, and depth is the axis pointing away from the camera, whose
    vanishing point consequently lands near the centre of the frame while the
    lateral one flies off toward the edge. Choosing by cluster weight instead
    would swap the two whenever the depth-going edges happened to be the longer
    family, which on a tiled floor they usually are, and every plank on the
    floor would come out crosswise. Dominance rank survives as the tiebreak for
    the measure-zero case where both sit equidistant from the centre.

    ``require_vertical`` is cleared only on the accepted-triple path, where all
    three labels have to be filled for the caller's "three labels means an
    orthogonal triple was accepted" test to hold. Even there the eligibility gate
    is tried first and relaxed to an unconditional pick only if it admits nobody,
    so a well-behaved triple is labelled by the strict rule.
    """
    result: dict[str, VanishingPoint | None] = {"VPx": None, "VPy": None, "VPz": None}
    remaining = list(items)
    if not remaining:
        return result

    pool = [item for item in remaining if _may_be_vertical(item[1], centre, focal)]
    if not pool and not require_vertical:
        pool = list(remaining)
    if pool:
        most_vertical = max(pool, key=lambda item: _verticality(item[1], centre))
        result["VPy"] = most_vertical[1]
        remaining.remove(most_vertical)

    if len(remaining) >= 2:
        first, second = remaining[0], remaining[1]
        span_first = math.hypot(first[1][0] - centre[0], first[1][1] - centre[1])
        span_second = math.hypot(second[1][0] - centre[0], second[1][1] - centre[1])
        if span_first < span_second:
            depth, lateral = first, second
        elif span_second < span_first:
            depth, lateral = second, first
        else:  # pragma: no cover - exact tie; fall back on dominance rank
            lateral, depth = sorted((first, second), key=lambda item: item[0])
        result["VPz"] = depth[1]
        result["VPx"] = lateral[1]
    elif len(remaining) == 1:
        only = remaining[0][1]
        span = math.hypot(only[0] - centre[0], only[1] - centre[1])
        # Within the focal length of the principal point is within 45 degrees of
        # the optical axis: depth-like rather than lateral.
        result["VPz" if span <= focal else "VPx"] = only
    return result


def _best_pair(
    candidates: Sequence[tuple[int, tuple[float, float]]],
    centre: tuple[float, float],
    focal_guess: float,
) -> list[tuple[int, tuple[float, float]]]:
    """The two candidates most likely to be a genuine orthogonal pair.

    Reached only when no triple is self-consistent, which means at least one
    candidate is wrong. The pair whose implied focal length is closest to the
    prior is kept and the odd one out is dropped, so the vanishing point count
    falls to two and the caller routes to the planar fallback -- while the two
    survivors are still good enough to carry a horizon.
    """
    best: list[tuple[int, tuple[float, float]]] = []
    best_error = math.inf
    for first, second in combinations(candidates, 2):
        squared = _pair_focal_squared(first[1], second[1], centre)
        if squared <= 0.0:
            continue
        error = abs(math.sqrt(squared) - focal_guess)
        if error < best_error:
            best_error, best = error, [first, second]
    if best:
        return best
    # No pair admits a positive f^2 at all: keep the two most dominant groups so
    # a horizon is still derivable.
    return sorted(candidates, key=lambda item: item[0])[:2]


def enforce_orthogonality(
    vps: Sequence[VanishingPoint | None],
    principal_point: tuple[float, float],
    focal_guess: float,
    settings: Settings | None = None,
) -> dict[str, VanishingPoint | None]:
    """Accept and label a mutually orthogonal vanishing point triple (R5.2).

    Every candidate pair implies a focal length through
    ``f^2 = -(v_i - pp) . (v_j - pp)``. A triple is the image of three
    orthogonal world directions only if all three pairs agree on that focal
    length and the directions it induces are mutually perpendicular, both within
    ``orthogonality_tolerance``. Among the triples that pass, the one with the
    lowest residual wins.

    When none passes, the candidate count is reduced instead of forcing a bad
    triple through: the best-conditioned pair is kept and labelled, leaving at
    least one label ``None``. That makes the return value unambiguous for the
    caller -- **three non-``None`` labels mean and only mean that an orthogonal
    triple was accepted**, which is exactly the test Requirement 6.1 needs to
    choose between ``vanishing_points`` and ``planar_fallback`` mode.

    Args:
        vps: candidate vanishing points in the dominance order
            :func:`cluster_by_direction` produced. ``None`` entries and
            non-finite coordinates are ignored, so the output of
            :func:`estimate_vanishing_point` can be passed straight through.
        principal_point: ``(x, y)``, normally :func:`principal_point`.
        focal_guess: focal length prior in pixels, normally
            :func:`default_focal_guess`. Used only to pick a pair when no triple
            is accepted.
        settings: overrides the process settings. Reads
            ``orthogonality_tolerance``.

    Returns:
        ``{"VPx": ..., "VPy": ..., "VPz": ...}`` with each value a finite
        vanishing point or ``None``. ``VPy`` is the vertical direction, ``VPz``
        the depth direction, ``VPx`` the lateral one.
    """
    resolved = settings if settings is not None else get_settings()
    tolerance = float(resolved.orthogonality_tolerance)
    centre = (float(principal_point[0]), float(principal_point[1]))
    focal_guess = float(focal_guess)

    candidates = _finite_candidates(vps)
    if not candidates:
        return {"VPx": None, "VPy": None, "VPz": None}

    best_triple: Sequence[tuple[int, tuple[float, float]]] | None = None
    best_focal = focal_guess
    best_residual = math.inf
    for triple in combinations(candidates, 3):
        scored = _orthogonality_residual([vp for _, vp in triple], centre)
        if scored is None:
            continue
        focal, residual = scored
        if residual <= tolerance and residual < best_residual:
            best_triple, best_focal, best_residual = triple, focal, residual

    if best_triple is not None:
        return _label_vps(best_triple, centre, best_focal, require_vertical=False)
    if len(candidates) >= 2:
        pair = _best_pair(candidates, centre, focal_guess)
        return _label_vps(pair, centre, focal_guess, require_vertical=True)
    return _label_vps(candidates, centre, focal_guess, require_vertical=True)


# --------------------------------------------------------------------------- #
# Horizon (Requirements 5.3, 6.2)
# --------------------------------------------------------------------------- #


def _normalise_line(a: float, b: float, c: float) -> Line | None:
    """Scale a homogeneous line to ``a^2 + b^2 = 1`` and fix its sign.

    Sign is a free parameter of a homogeneous line, so it is pinned to ``b >= 0``
    (falling back to ``a >= 0`` for a vertical line). Without that, two correct
    horizons for the same geometry can differ by a factor of -1 and no
    componentwise comparison against ground truth would hold.
    """
    norm = math.hypot(a, b)
    if not math.isfinite(norm) or norm < _EPS:
        return None
    a, b, c = a / norm, b / norm, c / norm
    if b < 0.0 or (abs(b) <= _EPS and a < 0.0):
        a, b, c = -a, -b, -c
    if not all(math.isfinite(value) for value in (a, b, c)):
        return None  # pragma: no cover - a finite norm keeps the quotient finite
    return (a, b, c)


def horizon_from_vps(vps: Mapping[str, VanishingPoint | None]) -> Line | None:
    """Horizon as the join of the two horizontal vanishing points (R5.3).

    Both horizontal directions lie in the ground plane, so both of their
    vanishing points lie on the image of that plane's line at infinity -- the
    horizon. Two points determine a line, and in homogeneous coordinates that
    line is simply their cross product.

    Only ``VPx`` and ``VPz`` are eligible. ``VPy`` is the vertical direction and
    does not lie on the horizon, so a run that recovered one horizontal and the
    vertical has *two* vanishing points but still no horizon from them, and must
    fall through to :func:`horizon_from_contours`.

    Args:
        vps: the labelled mapping from :func:`enforce_orthogonality`.

    Returns:
        The normalised horizon, or ``None`` when fewer than two horizontal
        vanishing points are available or they coincide.
    """
    finite = [
        (float(vp[0]), float(vp[1]))
        for vp in (vps.get("VPx"), vps.get("VPz"))
        if vp is not None and math.isfinite(float(vp[0])) and math.isfinite(float(vp[1]))
    ]
    if len(finite) < 2:
        return None
    (x0, y0), (x1, y1) = finite[0], finite[1]
    line = np.cross([x0, y0, 1.0], [x1, y1, 1.0])
    return _normalise_line(float(line[0]), float(line[1]), float(line[2]))


def _contour_rows(
    contours: Mapping[str, np.ndarray] | Sequence[np.ndarray] | None,
) -> tuple[dict[str, tuple[float, float]], list[tuple[float, float]]]:
    """Vertical extent of each contour as ``(top_row, bottom_row)``.

    Returns:
        ``(named, unnamed)`` -- the extents that carry a plane name, and those
        from a bare sequence of contours, which carries none.
    """
    named: dict[str, tuple[float, float]] = {}
    unnamed: list[tuple[float, float]] = []
    if contours is None:
        return named, unnamed

    items: list[tuple[str | None, object]]
    if isinstance(contours, Mapping):
        items = [(str(key), value) for key, value in contours.items()]
    else:
        items = [(None, value) for value in contours]

    for name, contour in items:
        points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) == 0:
            continue
        extent = (float(points[:, 1].min()), float(points[:, 1].max()))
        if name is None:
            unnamed.append(extent)
        else:
            named[name] = extent
    return named, unnamed


def _biased_row(name: str, extent: tuple[float, float]) -> float:
    """Horizon row inside one plane's vertical extent, biased by plane type."""
    top, bottom = extent
    midpoint = 0.5 * (top + bottom)
    return midpoint + _HORIZON_BIAS.get(name, 0.0) * (bottom - top)


def horizon_from_contours(
    contours: Mapping[str, np.ndarray] | Sequence[np.ndarray] | None,
    image_shape: Sequence[int],
) -> Line:
    """Horizon from the structural contours' vertical extent (R6.2).

    The fallback for when fewer than two horizontal vanishing points survive.
    It is a weaker estimate by construction -- contour extents locate the
    floor-wall junction, and the true horizon sits above that junction by
    however much the camera is pitched -- so it is used to *orient* downstream
    geometry, never to calibrate it. What it does guarantee is the part
    Requirement 6.2 asks for: a horizon always exists, and always inside the
    frame.

    Three cases, in order of how much the contours reveal:

    * floor plus at least one wall -- the horizon is placed on the floor-wall
      boundary, taken as the midpoint between the floor's top edge and the
      highest wall bottom edge. The highest one is used because a wall running
      away from the camera has its bottom edge clipped at the frame's lower
      border, which says nothing about the junction, while the wall whose bottom
      edge sits highest is the one facing the camera, whose bottom edge *is* the
      junction.
    * a single plane -- its vertical midpoint, shifted by the plane-type bias in
      :data:`_HORIZON_BIAS`.
    * nothing usable -- the image's vertical midpoint.

    Args:
        contours: contours keyed by Structural_Plane name. A bare sequence is
            accepted too and treated as unlabelled, in which case only the union
            extent's midpoint can be used.
        image_shape: ``(height, width)`` of the photograph.

    Returns:
        A horizontal normalised line ``(0.0, 1.0, -y)`` whose row lies inside
        ``[0, height - 1]``.
    """
    height = int(image_shape[0])
    if height <= 0:
        raise ValueError(f"degenerate image height {height}")
    midpoint = (height - 1) / 2.0

    named, unnamed = _contour_rows(contours)
    floor = named.get("floor")
    walls = {name: named[name] for name in _WALL_NAMES if name in named}

    if floor is not None and walls:
        highest_wall_bottom = min(extent[1] for extent in walls.values())
        row = 0.5 * (floor[0] + highest_wall_bottom)
    elif floor is not None:
        row = _biased_row("floor", floor)
    elif walls:
        # The tallest visible wall spans the most of the frame vertically, so its
        # midpoint is the least arbitrary anchor available.
        name, extent = max(walls.items(), key=lambda item: item[1][1] - item[1][0])
        row = _biased_row(name, extent)
    elif unnamed:
        row = 0.5 * (
            min(extent[0] for extent in unnamed) + max(extent[1] for extent in unnamed)
        )
    else:
        row = midpoint

    if not math.isfinite(row):  # pragma: no cover - extents are finite-filtered
        row = midpoint
    row = min(max(row, 0.0), float(height - 1))
    # Already in the module's normal form: hypot(0, 1) == 1 and b >= 0, so this
    # needs no trip through _normalise_line.
    return (0.0, 1.0, -row)

# --------------------------------------------------------------------------- #
# Metric plane frames, homographies, and calibration
# (Requirements 5.4, 5.5, 5.6, 6.1, 6.3, 6.4)
# --------------------------------------------------------------------------- #

#: The design's plane frame table, as ``(u axis, v axis)`` keys into the
#: recovered axis directions. ``x+`` is the lateral direction of ``VPx``, ``z+``
#: the depth direction of ``VPz``, and ``y-`` the *upward* direction, which is
#: the negation of ``VPy``'s direction because ``VPy`` is the vanishing point of
#: world *down* (see :func:`_axis_direction`).
#:
#: | Plane        | u axis        | v axis        |
#: |--------------|---------------|---------------|
#: | `floor`      | `VPx` lateral | `VPz` depth   |
#: | `wall_left`  | `VPz` depth   | `VPy` up      |
#: | `wall_right` | `VPz` depth   | `VPy` up      |
#: | `wall_back`  | `VPx` lateral | `VPy` up      |
PLANE_AXES: Final[dict[str, tuple[str, str]]] = {
    "floor": ("x+", "z+"),
    "wall_left": ("z+", "y-"),
    "wall_right": ("z+", "y-"),
    "wall_back": ("x+", "y-"),
}

#: Which recovered axis is each plane's normal. The floor's normal is the
#: vertical, a side wall's is the lateral direction, and the back wall's is the
#: depth direction.
PLANE_NORMAL_AXIS: Final[dict[str, str]] = {
    "floor": "y",
    "wall_left": "x",
    "wall_right": "x",
    "wall_back": "z",
}

#: Plane order every mapping this module returns is built in, matching
#: ``backend.schemas.PLANE_NAMES`` so a response's key order is stable.
_PLANE_REPORT_ORDER: Final[tuple[str, ...]] = (
    "floor",
    "wall_left",
    "wall_right",
    "wall_back",
)

#: Default metric sample count of :func:`reprojection_rmse`, rounded up to a
#: square grid.
REPROJECTION_GRID_SAMPLES: Final[int] = 256

#: Camera-space depth window a back-projected contour point must fall in, in
#: millimetres. The lower bound rejects a point that solves onto the camera
#: itself; the upper bound rejects one that solves a kilometre away, which only
#: happens for a point sitting on the plane's own vanishing line where the
#: solution carries no usable metric position.
_MIN_PLANE_DEPTH_MM: Final[float] = 1.0
_MAX_PLANE_DEPTH_MM: Final[float] = 1.0e6

#: Contour points within this fraction of the image diagonal of the plane they
#: are about to be back-projected onto are excluded. The exact rejection test is
#: the sign of the ray-normal dot product below, but a point *near* a plane's
#: vanishing line is numerically poor even when it is on the correct side,
#: because back-projection onto that plane degenerates on that line.
#:
#: Which line that is depends on which plane the rays are being intersected with,
#: not on which plane's frame is being built -- see
#: :func:`_plane_vanishing_line`.
_VANISHING_LINE_EXCLUSION_FRACTION: Final[float] = 0.0015

#: Percentile of the back-projected offsets a wall plane's distance is taken at.
#: The wall's base line is the *smallest* offset its contour can produce -- a
#: point higher up the wall, followed down to the floor plane, lands past the
#: wall, never in front of it -- so the true offset is the low tail. Taking the
#: tenth percentile rather than the strict minimum absorbs a contour that dips a
#: few pixels onto the floor without abandoning the estimate.
_WALL_OFFSET_PERCENTILE: Final[float] = 10.0

#: Smallest wall distance accepted, in millimetres. A wall closer than this is
#: passing through the camera, which no photograph of a room shows.
_MIN_WALL_OFFSET_MM: Final[float] = 50.0

#: Smallest metric span, in millimetres, a plane frame's visible extent may have
#: on either axis. Below one tile the frame carries no usable layout.
_MIN_EXTENT_SPAN_MM: Final[float] = 100.0

#: Metric extent bounds for the planar fallback, in millimetres. The fallback
#: infers scale from a single quad and an estimated horizon, so a degenerate
#: quad can imply a plane the size of a postage stamp or of a city block; these
#: clamp the result to something a room can plausibly contain.
_FALLBACK_MIN_EXTENT_MM: Final[float] = 200.0
_FALLBACK_MAX_EXTENT_MM: Final[float] = 30000.0

#: Largest metric aspect ratio the planar fallback will emit, either way up. A
#: quad seen almost edge-on implies an unbounded ratio, and mapping a tile grid
#: through that would smear one tile across the whole plane.
_FALLBACK_MAX_ASPECT: Final[float] = 20.0

#: Smallest distance from the horizon, as a fraction of the image diagonal, that
#: the planar fallback will divide by when converting pixels to millimetres.
_MIN_HORIZON_DISTANCE_FRACTION: Final[float] = 0.01

#: Condition number above which a homography is treated as degenerate rather
#: than inverted. ``float64`` carries about 16 digits, so at 1e12 the inverse
#: has lost all but four of them.
_MAX_CONDITION: Final[float] = 1.0e12


@dataclass(slots=True)
class PlaneFrame:
    """One Structural_Plane's metric coordinate frame, in camera space.

    The frame is what discharges Requirement 5.5: ``origin_cam`` is a point of
    the plane in millimetres, ``u_dir_cam`` and ``v_dir_cam`` are unit
    directions spanning it, so the plane point at metric ``(u_mm, v_mm)`` is
    ``origin_cam + u_mm * u_dir_cam + v_mm * v_dir_cam`` -- millimetres in,
    millimetres out, at every location on the plane.

    ``normal_cam`` and ``offset_mm`` record the plane the origin was solved on,
    ``normal_cam . p == offset_mm``, which is what makes the absolute scale
    auditable: for the floor ``offset_mm`` is exactly the assumed camera height,
    and for a wall it is the distance the floor-wall junction implied.

    That identity holds at the solved reference point but only within
    ``orthogonality_tolerance`` across the whole frame, because ``u_dir_cam`` and
    ``v_dir_cam`` are the *recovered* vanishing directions and a recovered triple
    is orthogonal only to within that tolerance. A point several metres out along
    ``u`` can therefore sit a few millimetres off the nominal plane. Keeping the
    axes exactly on the vanishing directions is the deliberate side of that
    trade: it is what makes each homography column an actual vanishing point, and
    so what makes the image-space foreshortening exact rather than merely close.
    Squaring the basis up instead would buy an identity nothing reads and rotate
    the axes away from the directions the image actually shows.

    ``homography`` and ``homography_inv`` are filled in on the first call to
    :func:`homography_from_frame` and reused afterwards, so a request that asks
    for the same frame's matrix more than once pays for one SVD-free assembly and
    one inversion.
    """

    name: str
    origin_cam: np.ndarray  # (3,) float64, millimetres
    u_dir_cam: np.ndarray  # (3,) float64, unit
    v_dir_cam: np.ndarray  # (3,) float64, unit
    normal_cam: np.ndarray  # (3,) float64, unit
    offset_mm: float
    focal_px: float
    intrinsics: np.ndarray  # (3,3) float64, the assumed K
    extent_mm: tuple[float, float, float, float]  # u_min, v_min, u_max, v_max
    homography: np.ndarray | None = field(default=None)
    homography_inv: np.ndarray | None = field(default=None)

    @property
    def extent_span_mm(self) -> tuple[float, float]:
        """``(u_span, v_span)`` of the visible metric extent."""
        u0, v0, u1, v1 = self.extent_mm
        return (u1 - u0, v1 - v0)


@dataclass(slots=True)
class Calibration:
    """Everything the Geometry_Engine recovers for one photograph.

    ``vanishing_points`` and ``geometry_mode`` describe the whole scene;
    ``homographies``, ``homography_inverses``, ``plane_extents_mm``, and
    ``reprojection_rmse_px`` are per plane and share one key set. A plane whose
    geometry could not be established by *either* path is absent from all four,
    the same way an undetected plane is absent upstream (Requirement 3.5), so a
    caller never has to distinguish "missing" from "meaningless".
    """

    vanishing_points: dict[str, VanishingPoint | None]
    horizon: Line
    geometry_mode: GeometryMode
    homographies: dict[PlaneName, np.ndarray]
    homography_inverses: dict[PlaneName, np.ndarray]
    plane_extents_mm: dict[PlaneName, tuple[float, float, float, float]]
    reprojection_rmse_px: dict[PlaneName, float]

    @property
    def plane_names(self) -> tuple[str, ...]:
        """Planes that received a homography, in reporting order."""
        return tuple(self.homographies)


# --------------------------------------------------------------------------- #
# Camera recovery
# --------------------------------------------------------------------------- #


def focal_from_vps(
    vps: Mapping[str, VanishingPoint | None],
    principal_point: tuple[float, float],
    fallback: float,
) -> float:
    """Focal length in pixels implied by the labelled vanishing points.

    Each orthogonal pair satisfies ``f^2 = -(v_i - pp) . (v_j - pp)``, so a
    recovered triple gives three estimates of one number and their mean is the
    least-squares answer under equal weighting. :func:`enforce_orthogonality`
    has already rejected any triple whose three estimates disagree beyond
    ``orthogonality_tolerance``, so the spread here is small by construction.

    Args:
        vps: the labelled mapping from :func:`enforce_orthogonality`.
        principal_point: ``(x, y)``, normally :func:`principal_point`.
        fallback: focal length to return when no pair admits a positive ``f^2``,
            normally :func:`default_focal_guess`.

    Returns:
        A positive focal length in pixels.

    Raises:
        ValueError: if ``fallback`` is not positive, since every consumer
            divides by the result.
    """
    if not (math.isfinite(fallback) and fallback > 0.0):
        raise ValueError(f"fallback focal length must be positive, got {fallback!r}")
    centre = (float(principal_point[0]), float(principal_point[1]))
    finite = [vp for _, vp in _finite_candidates([vps.get(key) for key in ("VPx", "VPy", "VPz")])]
    estimates = [
        math.sqrt(squared)
        for a, b in combinations(finite, 2)
        if (squared := _pair_focal_squared(a, b, centre)) > 0.0
    ]
    if not estimates:
        return float(fallback)
    return float(np.mean(estimates))


def _intrinsic_matrix(centre: tuple[float, float], focal: float) -> np.ndarray:
    """The assumed intrinsics: unit aspect ratio, no skew, centred principal point.

    Matches the synthetic fixture's ``_intrinsics`` exactly, including its
    ``(n - 1) / 2`` principal point convention, so recovered and ground-truth
    geometry are stated in one coordinate frame.
    """
    return np.array(
        [
            [focal, 0.0, centre[0]],
            [0.0, focal, centre[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _axis_direction(
    vp: VanishingPoint, centre: tuple[float, float], focal: float, component: int
) -> np.ndarray:
    """Unit camera-space direction of the world axis that vanishes at ``vp``.

    A vanishing point is the image of a direction's point at infinity, so
    ``v ~ K R d`` and therefore ``d ~ K^-1 [v, 1] = [(x - cx)/f, (y - cy)/f, 1]``
    once normalised. That recovers the direction up to sign, and sign is not free
    downstream: it decides which corner of the plane the metric origin lands on
    and which way a plank runs.

    The sign is pinned by requiring ``component`` of the direction to be
    positive, and the choice of component per axis is what makes the recovered
    frame agree with the world frame the synthetic fixture builds:

    * lateral (``component=0``) points to the camera's right, matching world
      ``+X``, for any ``|yaw| < 90``;
    * vertical (``component=1``) points image-down, matching world ``+Y``
      (down), for any ``|pitch| < 90``;
    * depth (``component=2``) points away from the camera, matching world
      ``+Z``, for any ``|yaw|, |pitch| < 90``.

    Those bounds cover every pose a photograph of a room's interior can have, so
    the convention is unambiguous in practice rather than merely canonical.
    """
    vector = np.array(
        [float(vp[0]) - centre[0], float(vp[1]) - centre[1], float(focal)],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if norm < _EPS:  # pragma: no cover - focal > 0 keeps the norm positive
        raise ValueError("degenerate vanishing direction")
    vector /= norm
    if vector[component] < 0.0:
        vector = -vector
    return vector


def _axis_directions(
    vps: Mapping[str, VanishingPoint | None], centre: tuple[float, float], focal: float
) -> dict[str, np.ndarray] | None:
    """The three recovered axis directions, or ``None`` if any is missing.

    Every entry of :data:`PLANE_AXES` needs two of the three, and every plane's
    normal needs the third, so a frame is only ever built from a complete
    triple -- which is the same condition Requirement 6.1 uses to stay out of
    the planar fallback.
    """
    lateral, vertical, depth = (vps.get("VPx"), vps.get("VPy"), vps.get("VPz"))
    if lateral is None or vertical is None or depth is None:
        return None
    for vp in (lateral, vertical, depth):
        if not (math.isfinite(float(vp[0])) and math.isfinite(float(vp[1]))):
            return None
    down = _axis_direction(vertical, centre, focal, 1)
    return {
        "x+": _axis_direction(lateral, centre, focal, 0),
        "y+": down,
        "y-": -down,
        "z+": _axis_direction(depth, centre, focal, 2),
    }


# --------------------------------------------------------------------------- #
# Plane frames (Requirements 5.4, 5.5)
# --------------------------------------------------------------------------- #


def _contour_points(contour: object) -> np.ndarray:
    """Coerce a contour to a finite ``(N,2) float64`` point array.

    Accepts OpenCV's ``(N,1,2)`` layout as well as a plain ``(N,2)``, since the
    Segmenter's simplified contours travel in the former and hand-built test
    contours in the latter.
    """
    array = np.asarray(contour, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 2:
        raise ValueError(f"contour must be an (N,2) point array, got shape {array.shape}")
    points = array.reshape(-1, 2)
    return np.ascontiguousarray(points[np.isfinite(points).all(axis=1)])


def _camera_rays(points: np.ndarray, centre: tuple[float, float], focal: float) -> np.ndarray:
    """``(N,3)`` camera-space rays through image points, scaled to ``z == 1``.

    Because the third component is exactly one, the scale factor that lands a
    ray on a plane *is* that plane point's camera-space depth in millimetres,
    which is what every guard below is expressed in.
    """
    return np.column_stack(
        (
            (points[:, 0] - centre[0]) / focal,
            (points[:, 1] - centre[1]) / focal,
            np.ones(len(points), dtype=np.float64),
        )
    )


def _plane_vanishing_line(normal_cam: np.ndarray, intrinsics: np.ndarray) -> Line | None:
    """The image line a plane sends its own points at infinity to.

    Every direction ``v`` lying in the plane ``normal . p = d`` satisfies
    ``normal . v == 0`` and images at ``K v``, so ``(K^-T normal) . (K v)``
    reduces to ``normal . v`` and vanishes: all of the plane's vanishing points
    lie on the single line ``K^-T normal``, which is what makes that line the
    plane's vanishing line.

    Naming it per plane matters because the three room planes have three
    different ones. For the floor it is the horizon. For a side wall it is the
    join of ``VPz`` and ``VPy`` -- a steep line that crosses the frame nowhere
    near the horizon, which the wall's visible surface reaches and passes through
    quite legitimately. Filtering a wall contour against the horizon instead
    would discard the wall's whole upper edge whenever the frame happens to clip
    it at eye level, and with it the wall's entire vertical extent.

    Returns:
        The normalised line, or ``None`` when the plane's vanishing line is
        itself at infinity -- a plane exactly perpendicular to the optical axis,
        whose parallels stay parallel in the image and so need no filter at all.
    """
    matrix = np.asarray(intrinsics, dtype=np.float64)
    try:
        line = np.linalg.inv(matrix).T @ np.asarray(normal_cam, dtype=np.float64)
    except np.linalg.LinAlgError:  # pragma: no cover - K has a positive focal length
        return None
    return _normalise_line(float(line[0]), float(line[1]), float(line[2]))


def _off_vanishing_line(points: np.ndarray, line: Line | None, diagonal: float) -> np.ndarray:
    """Mask of points far enough from ``line`` to back-project.

    The exact rejection test is the sign of the ray-normal product in
    :func:`_intersect_plane`; this is the numerical one. A point within a few
    pixels of a plane's vanishing line solves to a position hundreds of metres
    away whose metric coordinates are dominated by endpoint noise.
    """
    if line is None:
        return np.ones(len(points), dtype=bool)
    a, b, c = (float(line[0]), float(line[1]), float(line[2]))
    band = _VANISHING_LINE_EXCLUSION_FRACTION * diagonal
    return np.abs(a * points[:, 0] + b * points[:, 1] + c) >= band


def _usable_for_plane(points: np.ndarray, line: Line | None, diagonal: float) -> np.ndarray:
    """``points`` with those sitting on ``line`` dropped, unless too few remain.

    Three points is the minimum a metric extent can be measured from, so a
    contour that would be cut below that is passed through whole and left to
    :func:`_intersect_plane`'s exact depth guards. Dropping to two points and
    reporting the resulting sliver extent as real is the worse failure: it is
    indistinguishable from a genuinely edge-on plane.
    """
    if len(points) < 3:
        return points
    keep = _off_vanishing_line(points, line, diagonal)
    if int(keep.sum()) < 3:
        return points
    return points[keep]


def _intersect_plane(
    rays: np.ndarray, normal: np.ndarray, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect camera rays with the plane ``normal . p == offset``.

    Returns:
        ``(points, mask)`` where ``mask`` selects the rays that met the plane in
        front of the camera at a plausible depth, and ``points`` is the ``(M,3)``
        millimetre intersection for those rays only.
    """
    if len(rays) == 0:
        return np.empty((0, 3), dtype=np.float64), np.zeros(0, dtype=bool)
    denominator = rays @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = float(offset) / denominator
    mask = (
        (np.abs(denominator) > _EPS)
        & np.isfinite(depth)
        & (depth >= _MIN_PLANE_DEPTH_MM)
        & (depth <= _MAX_PLANE_DEPTH_MM)
    )
    if not mask.any():
        return np.empty((0, 3), dtype=np.float64), mask
    return rays[mask] * depth[mask][:, None], mask


def _wall_offset(ground_points: np.ndarray, normal: np.ndarray) -> float | None:
    """Signed distance from the camera to a wall plane, in millimetres.

    This is how a wall inherits the floor's absolute scale, which is the only
    scale a single photograph has (Requirement 5.5). Every visible wall point
    lies above the wall's base line, so following its ray down to the *floor*
    plane lands at or beyond the wall, never in front of it: the wall's own
    offset is therefore the smallest offset its contour can produce, and it is
    reached exactly at the base line where wall and floor meet.

    Returns:
        The signed offset, or ``None`` when no contour point reaches the floor
        plane or the implied wall passes through the camera.
    """
    if len(ground_points) == 0:
        return None
    values = ground_points @ normal
    values = values[np.isfinite(values)]
    if len(values) == 0:  # pragma: no cover - inputs are finite-filtered
        return None
    sign = 1.0 if float(np.median(values)) >= 0.0 else -1.0
    magnitudes = values[values * sign > 0.0] * sign
    if len(magnitudes) == 0:
        return None
    if len(magnitudes) >= 4:
        offset = float(np.percentile(magnitudes, _WALL_OFFSET_PERCENTILE))
    else:
        offset = float(magnitudes.min())
    if not math.isfinite(offset) or offset < _MIN_WALL_OFFSET_MM:
        return None
    return sign * offset


def plane_frame(
    name: str,
    vps: Mapping[str, VanishingPoint | None],
    horizon: Line | None,
    contour: np.ndarray,
    image_shape: Sequence[int],
    settings: Settings | None = None,
) -> PlaneFrame | None:
    """Build one Structural_Plane's metric frame (Requirements 5.4, 5.5).

    Three things have to be pinned down, and each comes from a different place:

    1. **Orientation** from the vanishing points, per :data:`PLANE_AXES`. The
       ``u`` and ``v`` axes are the recovered directions themselves, so a metric
       step along ``u`` is the image of a real-world step along that world axis
       and perspective foreshortening falls out of the homography rather than
       being applied on top of it (Requirement 5.7).
    2. **Absolute scale** from ``assumed_camera_height_mm``. Scale is
       unobservable from one uncalibrated photograph, so the design fixes it by
       convention: the camera sits 1500 mm above the floor plane. That single
       number places the floor, and :func:`_wall_offset` propagates it to the
       walls through the floor-wall junction, which is why a 600 mm tile is the
       same metric size on the floor and on the wall.
    3. **Origin** by back-projecting the plane's contour and taking the corner of
       the resulting metric bounding box, so every visible point of the plane has
       non-negative ``(u, v)``.

    Step 3 is a deliberate, and purely translational, reading of the design's
    origin column. The table names a specific contour corner per plane -- lowest
    -leftmost for ``wall_left`` and ``wall_back``, lowest-rightmost for
    ``wall_right`` -- and the metric bounding-box corner *is* that corner under
    each plane's own axis pair, while being insensitive to which single contour
    vertex the simplification happened to emit. For the floor the table names the
    point nearest the image bottom-centre where this yields the near-left corner
    instead; the two differ by a metric translation, which no tiling, extent, or
    reprojection figure can distinguish, and the near-left corner is what the
    synthetic fixture's floor frame uses.

    Args:
        name: Structural_Plane name; must be a key of :data:`PLANE_AXES`.
        vps: the labelled mapping from :func:`enforce_orthogonality`. All three
            labels must be present, which is the same condition that keeps the
            scene out of the planar fallback.
        horizon: the estimated horizon, used to discard contour points sitting on
            the *ground* plane's vanishing line, where the back-projection that
            places the plane degenerates. A wall's own extent is filtered against
            its own vanishing line instead, per :func:`_plane_vanishing_line`.
            ``None`` skips the horizon filter; the exact in-front-of-camera test
            still applies.
        contour: ``(N,2)`` image-space plane contour.
        image_shape: ``(height, width)`` of the photograph.
        settings: overrides the process settings. Reads
            ``assumed_camera_height_mm``.

    Returns:
        The frame, or ``None`` when the vanishing point triple is incomplete, the
        contour cannot be back-projected onto the plane, or the visible metric
        extent is degenerate. Every ``None`` routes the plane to the planar
        fallback of Requirement 6.1.

    Raises:
        ValueError: for an unknown plane name or a malformed contour.
    """
    if name not in PLANE_AXES:
        raise ValueError(f"unknown plane {name!r}; expected one of {tuple(PLANE_AXES)!r}")
    resolved = settings if settings is not None else get_settings()
    camera_height_mm = float(resolved.assumed_camera_height_mm)
    if not (math.isfinite(camera_height_mm) and camera_height_mm > 0.0):
        raise ValueError(f"assumed_camera_height_mm must be positive, got {camera_height_mm!r}")

    centre = principal_point(image_shape)
    diagonal = _diagonal(image_shape)
    focal = focal_from_vps(vps, centre, default_focal_guess(image_shape))
    axes = _axis_directions(vps, centre, focal)
    if axes is None:
        return None

    points = _contour_points(contour)
    if len(points) < 3:
        return None

    u_hat = axes[PLANE_AXES[name][0]]
    v_hat = axes[PLANE_AXES[name][1]]
    normal = axes[f"{PLANE_NORMAL_AXIS[name]}+"]
    intrinsics = _intrinsic_matrix(centre, focal)

    # Two back-projections happen below, onto two different planes, so each one
    # is filtered against the vanishing line of the plane *it* intersects.
    #
    # The ground pass is what recovers a wall's distance, so it degenerates on
    # the *horizon* whatever plane's frame is being built. The plane pass is
    # what measures the visible metric extent, so it degenerates on the plane's
    # own vanishing line -- which for a wall is the join of `VPz` and `VPy`, not
    # the horizon. Filtering a wall against the horizon would delete its top
    # edge whenever the frame clips the wall at eye level, leaving a contour that
    # hugs the floor junction and a `v` extent of a few millimetres: the wall
    # then fails the extent-span guard below, loses its metric frame, and falls
    # back to a fabricated quad rectangle that carries none of the floor's
    # millimetre scale.
    #
    # For the floor the two planes are the same plane and the two lines are the
    # same line, so nothing about the floor path changes.
    ground_points = _usable_for_plane(points, horizon, diagonal)
    ground, _ = _intersect_plane(
        _camera_rays(ground_points, centre, focal), axes["y+"], camera_height_mm
    )
    if name == "floor":
        offset_mm: float | None = camera_height_mm
        plane_points = ground
    else:
        offset_mm = _wall_offset(ground, normal)
        if offset_mm is None:
            return None
        extent_points = _usable_for_plane(
            points, _plane_vanishing_line(normal, intrinsics), diagonal
        )
        plane_points, _ = _intersect_plane(
            _camera_rays(extent_points, centre, focal), normal, offset_mm
        )
    if len(plane_points) < 3:
        return None

    # Any point of the plane works as the reference the extent is measured from,
    # because u_hat and v_hat span the plane; the centroid is used because it is
    # the best-conditioned choice, and the origin is then moved to the extent's
    # own corner, which cancels the choice entirely.
    reference = plane_points.mean(axis=0)
    relative = plane_points - reference
    u = relative @ u_hat
    v = relative @ v_hat
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    if (u_max - u_min) < _MIN_EXTENT_SPAN_MM or (v_max - v_min) < _MIN_EXTENT_SPAN_MM:
        return None

    origin = reference + u_min * u_hat + v_min * v_hat
    return PlaneFrame(
        name=name,
        origin_cam=origin,
        u_dir_cam=u_hat,
        v_dir_cam=v_hat,
        normal_cam=normal,
        offset_mm=float(offset_mm),
        focal_px=float(focal),
        intrinsics=intrinsics,
        extent_mm=(0.0, 0.0, u_max - u_min, v_max - v_min),
    )


# --------------------------------------------------------------------------- #
# Homographies (Requirements 5.4, 6.4)
# --------------------------------------------------------------------------- #


def _canonical_homography(matrix: np.ndarray, probe: Sequence[float]) -> np.ndarray | None:
    """Scale a homography to unit Frobenius norm and pin its sign.

    ``p_img ~ H p`` is projective, so ``H`` and ``k H`` describe the same
    mapping for any non-zero ``k``. Two conventions are fixed here for reasons
    that are not cosmetic: unit norm keeps the matrix and its inverse in the same
    numerical range whatever the image resolution, and the sign is chosen so the
    homogeneous divisor is *positive* over the plane's interior, which is what
    lets the Compositor's inverse warp divide by ``w`` without tracking a sign.

    Args:
        matrix: the ``(3,3)`` candidate.
        probe: a metric ``(u, v)`` known to lie on the visible plane, normally
            the centre of its extent.

    Returns:
        The normalised matrix, or ``None`` when it is not finite or the probe
        lands on the plane's vanishing line.
    """
    candidate = np.asarray(matrix, dtype=np.float64)
    if candidate.shape != (3, 3) or not np.isfinite(candidate).all():
        return None
    norm = float(np.linalg.norm(candidate))
    if norm < _EPS:
        return None
    candidate = candidate / norm
    w = float(
        candidate[2, 0] * float(probe[0]) + candidate[2, 1] * float(probe[1]) + candidate[2, 2]
    )
    if abs(w) < _EPS:
        return None
    if w < 0.0:
        candidate = -candidate
    return candidate


def homography_from_frame(frame: PlaneFrame) -> np.ndarray | None:
    """Assemble ``H`` from a metric plane frame (Requirement 5.4).

    A plane point is ``O + u * u_hat + v * v_hat``, so projecting it gives
    ``p_img ~ K (O + u * u_hat + v * v_hat) = [K u_hat | K v_hat | K O] [u, v, 1]``
    -- the columns are the image-space homogeneous representations of the u axis
    direction, the v axis direction, and the origin, exactly as the design
    states. The first two columns are points at infinity, which is another way
    of saying they are the plane's two vanishing points, and that is why the
    resulting mapping foreshortens correctly with no further work.

    The result and its inverse are cached on ``frame``, so repeat calls are free.

    Returns:
        The ``(3,3) float64`` homography, or ``None`` when the frame is
        degenerate enough that it or its inverse is not usable.
    """
    if frame.homography is not None:
        return frame.homography
    matrix = frame.intrinsics @ np.column_stack(
        (frame.u_dir_cam, frame.v_dir_cam, frame.origin_cam)
    )
    u0, v0, u1, v1 = frame.extent_mm
    canonical = _canonical_homography(matrix, (0.5 * (u0 + u1), 0.5 * (v0 + v1)))
    if canonical is None:
        return None
    inverse = invert_homography(canonical)
    if inverse is None:
        return None
    frame.homography = canonical
    frame.homography_inv = inverse
    return canonical


def invert_homography(homography: np.ndarray) -> np.ndarray | None:
    """Invert a homography, rejecting the ill-conditioned ones.

    The inverse is scaled to unit Frobenius norm by a *positive* factor, so the
    sign convention :func:`_canonical_homography` established survives: a metric
    point with positive ``w`` under ``H`` maps back with positive ``w`` under
    ``H^-1``.

    Returns:
        The ``(3,3) float64`` inverse, or ``None`` when ``homography`` is not a
        finite ``(3,3)`` matrix or its condition number exceeds
        :data:`_MAX_CONDITION`.
    """
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        return None
    try:
        condition = float(np.linalg.cond(matrix))
    except np.linalg.LinAlgError:  # pragma: no cover - cond falls back to inf
        return None
    if not math.isfinite(condition) or condition > _MAX_CONDITION:
        return None
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded by the cond test
        return None
    if not np.isfinite(inverse).all():  # pragma: no cover - guarded by the cond test
        return None
    norm = float(np.linalg.norm(inverse))
    if norm < _EPS:  # pragma: no cover - an inverse of a finite matrix is non-zero
        return None
    return inverse / norm


def _quad_points(quad: object) -> np.ndarray:
    """Coerce a bounding quad to a finite ``(4,2) float64`` array."""
    array = np.asarray(quad, dtype=np.float64)
    if array.ndim < 2 or array.shape[-1] != 2:
        raise ValueError(f"quad must be a (4,2) point array, got shape {array.shape}")
    points = array.reshape(-1, 2)
    if len(points) != 4:
        raise ValueError(f"quad must have exactly 4 points, got {len(points)}")
    if not np.isfinite(points).all():
        raise ValueError("quad contains non-finite coordinates")
    return np.ascontiguousarray(points)


def _order_quad_indices(quad: np.ndarray) -> np.ndarray:
    """Permutation putting a quad in ``(top-left, top-right, bottom-right, bottom-left)``.

    Sorting by bearing from the centroid orders the vertices around the polygon,
    and with image ``y`` growing downward increasing bearing runs clockwise on
    screen. Rotating so the vertex minimising ``x + y`` comes first fixes which
    of the four rotations is used. The result is a permutation rather than a
    reordered array so the *same* permutation can be applied to a corresponding
    metric quad, which is what keeps the two in row correspondence however the
    caller happened to order them.
    """
    centroid = quad.mean(axis=0)
    bearings = np.arctan2(quad[:, 1] - centroid[1], quad[:, 0] - centroid[0])
    ring = np.argsort(bearings, kind="stable")
    ordered = quad[ring]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ring, -start)


def _quad_from_contour(contour: np.ndarray) -> np.ndarray:
    """Four-point bounding quad of a contour, for callers that supply none.

    The Segmenter already computes a better quad -- it distinguishes
    near-rectangular regions from L-shaped ones -- and :func:`calibrate` uses
    that one whenever it is offered. This is the minimum needed to keep the
    fallback available when only a contour was passed in.
    """
    points = _contour_points(contour)
    if len(points) < 3:
        raise ValueError(f"a bounding quad needs at least 3 contour points, got {len(points)}")
    box = cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32)))
    return np.asarray(box, dtype=np.float64)


def metric_quad_from_image_quad(
    quad_img: np.ndarray,
    horizon: Line | None,
    image_shape: Sequence[int],
    *,
    focal_px: float | None = None,
    settings: Settings | None = None,
) -> np.ndarray | None:
    """Metric rectangle a bounding quad is the perspective image of (R6.4).

    This is the half of the planar fallback that keeps Requirement 6.4 honest.
    Mapping the quad to an arbitrary square would let a 600x1200 plank come out
    square, so the rectangle's metric dimensions are derived from the quad's
    relationship to the horizon instead.

    For a plane on the ground, a point at metric depth ``Z`` sits ``f h / Z``
    pixels from the horizon and a metric length ``W`` at that depth spans
    ``f W / Z`` pixels, with ``h`` the assumed camera height. Eliminating ``Z``
    from a pair of quad edges gives both dimensions in millimetres::

        W = L_edge * h / d_edge
        D = f * h * (1 / d_far - 1 / d_near)

    where ``d`` is an edge's pixel distance from the horizon. That branch is
    taken when the quad lies wholly on one side of the horizon, which is what a
    floor region does.

    A wall region straddles the horizon, so the far edge's distance goes to zero
    and through it. The relation above then degenerates, but a vertical plane has
    a simpler answer: both of its extents are measured at the *same* depth as its
    base line, so their metric ratio is just their pixel ratio, scaled by
    ``h / d_base``. That is the second branch, and the sign test that selects
    between the two is exactly the floor-versus-wall distinction, recovered from
    the geometry rather than from the plane's name.

    Args:
        quad_img: ``(4,2)`` image-space bounding quad, in any order.
        horizon: the estimated horizon. ``None`` forces the vertical-plane
            branch with a mid-frame reference distance.
        image_shape: ``(height, width)`` of the photograph.
        focal_px: focal length in pixels; :func:`default_focal_guess` when
            omitted, which is the normal case since the fallback runs precisely
            when the vanishing points did not pin a focal length down.
        settings: overrides the process settings. Reads
            ``assumed_camera_height_mm``.

    Returns:
        A ``(4,2) float64`` metric rectangle in the *same row order* as
        ``quad_img``, so the two can be handed straight to
        :func:`homography_from_quad`, or ``None`` when the quad is degenerate.
        The rectangle spans ``[0, width_mm] x [0, depth_mm]`` with its origin at
        the near-lower corner, matching the orientation a
        vanishing-point-derived frame would have used for the same plane.
    """
    resolved = settings if settings is not None else get_settings()
    camera_height_mm = float(resolved.assumed_camera_height_mm)
    if not (math.isfinite(camera_height_mm) and camera_height_mm > 0.0):
        raise ValueError(f"assumed_camera_height_mm must be positive, got {camera_height_mm!r}")

    quad = _quad_points(quad_img)
    order = _order_quad_indices(quad)
    top_left, top_right, bottom_right, bottom_left = quad[order]

    far_px = float(np.hypot(*(top_right - top_left)))
    near_px = float(np.hypot(*(bottom_right - bottom_left)))
    side_px = 0.5 * (
        float(np.hypot(*(bottom_left - top_left))) + float(np.hypot(*(bottom_right - top_right)))
    )
    if min(far_px, near_px, side_px) <= _EPS:
        return None

    diagonal = _diagonal(image_shape)
    focal = float(focal_px) if focal_px is not None else default_focal_guess(image_shape)
    if not (math.isfinite(focal) and focal > 0.0):
        raise ValueError(f"focal_px must be positive, got {focal_px!r}")
    floor_distance = _MIN_HORIZON_DISTANCE_FRACTION * diagonal

    if horizon is not None:
        a, b, c = (float(horizon[0]), float(horizon[1]), float(horizon[2]))
        far_d = 0.5 * (
            (a * top_left[0] + b * top_left[1] + c) + (a * top_right[0] + b * top_right[1] + c)
        )
        near_d = 0.5 * (
            (a * bottom_left[0] + b * bottom_left[1] + c)
            + (a * bottom_right[0] + b * bottom_right[1] + c)
        )
    else:
        far_d = near_d = 0.0

    same_side = far_d * near_d > 0.0
    far_abs, near_abs = abs(far_d), abs(near_d)
    if same_side and near_abs > far_abs >= floor_distance:
        width_mm = 0.5 * (
            near_px * camera_height_mm / near_abs + far_px * camera_height_mm / far_abs
        )
        depth_mm = focal * camera_height_mm * (1.0 / far_abs - 1.0 / near_abs)
    else:
        # Vertical (or unresolvable) plane: both extents are read at the depth of
        # the edge furthest from the horizon, so their ratio is the pixel ratio.
        reference = max(near_abs, far_abs, floor_distance)
        mm_per_px = camera_height_mm / reference
        width_mm = max(near_px, far_px) * mm_per_px
        depth_mm = side_px * mm_per_px

    if not (math.isfinite(width_mm) and math.isfinite(depth_mm)):
        return None
    width_mm = min(max(width_mm, _FALLBACK_MIN_EXTENT_MM), _FALLBACK_MAX_EXTENT_MM)
    depth_mm = min(max(depth_mm, _FALLBACK_MIN_EXTENT_MM), _FALLBACK_MAX_EXTENT_MM)
    # An edge-on quad implies an unbounded ratio; clamping keeps one tile from
    # being smeared across the whole plane.
    depth_mm = min(max(depth_mm, width_mm / _FALLBACK_MAX_ASPECT), width_mm * _FALLBACK_MAX_ASPECT)

    # v grows with distance from the camera, so the near (bottom) edge of the
    # image quad carries v = 0 and the origin is its lower-left corner.
    ordered_metric = np.array(
        [[0.0, depth_mm], [width_mm, depth_mm], [width_mm, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    metric = np.empty((4, 2), dtype=np.float64)
    metric[order] = ordered_metric
    return metric


def homography_from_quad(quad_img: np.ndarray, quad_mm: np.ndarray) -> np.ndarray | None:
    """Four-point planar homography for the degenerate case (R6.1, R6.4).

    Both quads are reordered by the *same* permutation, the one that puts the
    image quad in ``(top-left, top-right, bottom-right, bottom-left)``, so the
    caller only has to pass the two quads in matching row order and not worry
    about which corner either of them starts at.

    Args:
        quad_img: ``(4,2)`` image-space quad.
        quad_mm: ``(4,2)`` metric quad, row-corresponding to ``quad_img``.
            Normally from :func:`metric_quad_from_image_quad`.

    Returns:
        The ``(3,3) float64`` homography with ``p_img ~ H @ [u_mm, v_mm, 1]``,
        normalised like every other homography this module returns, or ``None``
        when the correspondence is degenerate.

    Raises:
        ValueError: if either quad is not four finite points.
    """
    image_quad = _quad_points(quad_img)
    metric_quad = _quad_points(quad_mm)
    order = _order_quad_indices(image_quad)
    try:
        matrix = cv2.getPerspectiveTransform(
            metric_quad[order].astype(np.float32), image_quad[order].astype(np.float32)
        )
    except cv2.error:
        return None
    probe = metric_quad.mean(axis=0)
    canonical = _canonical_homography(matrix, (probe[0], probe[1]))
    if canonical is None:
        return None
    if invert_homography(canonical) is None:
        return None
    return canonical


# --------------------------------------------------------------------------- #
# Reprojection error (Requirement 5.6)
# --------------------------------------------------------------------------- #


def _project(homography: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map ``(N,2)`` points through a homography.

    Returns:
        ``(mapped, mask)`` where ``mask`` selects the points whose homogeneous
        divisor was usable and ``mapped`` holds only those, so a caller can drop
        the samples that landed on the vanishing line.
    """
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    projected = homogeneous @ np.asarray(homography, dtype=np.float64).T
    w = projected[:, 2]
    mask = np.isfinite(w) & (np.abs(w) > _EPS)
    if not mask.any():
        return np.empty((0, 2), dtype=np.float64), mask
    out = projected[mask, :2] / w[mask][:, None]
    finite = np.isfinite(out).all(axis=1)
    if not finite.all():
        indices = np.flatnonzero(mask)[~finite]
        mask[indices] = False
        out = out[finite]
    return out, mask


def reprojection_rmse(
    homography: np.ndarray,
    extent_mm: Sequence[float],
    n_samples: int = REPROJECTION_GRID_SAMPLES,
) -> float:
    """Round-trip reprojection error of a homography, in pixels (R5.6).

    A metric grid is spread over the plane's extent, mapped forward to image
    space, back through ``H^-1`` to millimetres, and forward again. The residual
    is reported between the two image-space positions rather than between the two
    metric ones, because Requirement 5.6 states the bound in pixels and a
    millimetre residual would mean something different at the near and far ends
    of the same plane.

    Args:
        homography: the ``(3,3)`` matrix to measure.
        extent_mm: ``(u_min, v_min, u_max, v_max)`` metric extent to sample.
        n_samples: approximate sample count, rounded up to a square grid.

    Returns:
        The RMS residual in pixels, or ``math.inf`` when the homography is not
        invertible or no sample survives the round trip -- a value that fails
        Requirement 5.6's bound loudly instead of reporting a small number for a
        plane that has no usable geometry.
    """
    inverse = invert_homography(homography)
    if inverse is None:
        return math.inf
    u0, v0, u1, v1 = (float(value) for value in extent_mm)
    if not all(math.isfinite(value) for value in (u0, v0, u1, v1)):
        return math.inf
    side = max(2, int(math.ceil(math.sqrt(max(int(n_samples), 4)))))
    us, vs = np.meshgrid(
        np.linspace(u0, u1, side, dtype=np.float64),
        np.linspace(v0, v1, side, dtype=np.float64),
        indexing="xy",
    )
    metric = np.column_stack((us.ravel(), vs.ravel()))

    forward, forward_mask = _project(homography, metric)
    if len(forward) < 4:
        return math.inf
    recovered, back_mask = _project(inverse, forward)
    if len(recovered) < 4:
        return math.inf
    again, again_mask = _project(homography, recovered)
    if len(again) < 4:
        return math.inf

    # Each _project drops a different subset, so the forward positions have to be
    # narrowed by the same two masks before the difference lines up row for row.
    reference = forward[back_mask][again_mask]
    residual = again - reference
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


# --------------------------------------------------------------------------- #
# Orchestration (Requirements 6.1, 6.3)
# --------------------------------------------------------------------------- #


def _plane_geometry_inputs(planes: object) -> dict[str, tuple[np.ndarray, np.ndarray | None]]:
    """Normalise whatever the caller passed into contours plus optional quads.

    ``calibrate`` is called with the Segmenter's result, but this module cannot
    import ``backend.core.segmenter`` -- the Segmenter imports *this* module for
    its horizon hint, so the dependency only runs one way. Duck typing keeps that
    direction intact and, as a side effect, lets a test drive calibration from a
    plain ``{name: contour}`` mapping with no Segmenter at all.

    Accepted shapes:

    * an object with a ``contours`` mapping, optionally also ``bounding_points``
      (a ``SegmentationResult``);
    * a mapping of plane name to contour array;
    * a mapping of plane name to an object or mapping carrying ``contour`` and
      optionally ``bounding_points`` (a ``PlaneMetadata``).

    Planes whose contour has fewer than three finite points are dropped, since
    nothing can be fitted to them.
    """
    if planes is None:
        return {}

    source: Mapping[str, object]
    quads_by_name: Mapping[str, object] = {}
    contours = getattr(planes, "contours", None)
    if isinstance(contours, Mapping):
        source = contours
        bounding = getattr(planes, "bounding_points", None)
        if isinstance(bounding, Mapping):
            quads_by_name = bounding
    elif isinstance(planes, Mapping):
        source = planes
    else:
        raise TypeError(
            "planes must be a mapping of plane name to contour, or an object with a "
            f"'contours' mapping, got {type(planes)!r}"
        )

    out: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    for name in _PLANE_REPORT_ORDER:
        if name not in source:
            continue
        entry = source[name]
        contour_value: object = entry
        quad_value: object = quads_by_name.get(name)
        if isinstance(entry, Mapping):
            contour_value = entry.get("contour")
            quad_value = entry.get("bounding_points", quad_value)
        elif not isinstance(entry, np.ndarray) and hasattr(entry, "contour"):
            contour_value = entry.contour
            quad_value = getattr(entry, "bounding_points", quad_value)
        if contour_value is None:
            continue
        try:
            contour = _contour_points(contour_value)
        except ValueError:
            continue
        if len(contour) < 3:
            continue
        quad: np.ndarray | None = None
        if quad_value is not None:
            try:
                quad = _quad_points(quad_value)
            except ValueError:
                quad = None
        out[name] = (contour, quad)
    return out


def calibrate(
    image_bgr: np.ndarray,
    planes: object,
    *,
    segments: np.ndarray | None = None,
    settings: Settings | None = None,
) -> Calibration:
    """Calibrate a photograph and every Structural_Plane in it.

    The stages run in the order the design lays out, each one degrading rather
    than failing (see the module docstring): detect segments, cluster them by
    direction, fit a vanishing point per cluster, accept and label an orthogonal
    triple, derive a horizon, then build one homography per plane.

    ``geometry_mode`` is decided once for the scene, by whether
    :func:`enforce_orthogonality` accepted a triple -- three non-``None`` labels
    mean exactly that. With a triple, each plane gets a metric frame from the
    vanishing directions; without one, each plane gets the four-point planar
    homography of Requirement 6.1. A plane may still fall back individually if
    its own contour is degenerate while the scene's mode stays
    ``vanishing_points``, because the mode reports what the *camera* recovery
    achieved, which is what Requirement 6.3 surfaces to the client.

    Args:
        image_bgr: the photograph. Grayscale is accepted too.
        planes: the Segmenter's result, or any of the shapes
            :func:`_plane_geometry_inputs` documents.
        segments: already-detected ``(N,4)`` line segments. Passing the array
            that produced ``estimate_horizon_hint`` means a request runs the line
            detector exactly once.
        settings: overrides the process settings.

    Returns:
        A :class:`Calibration`. Its four per-plane mappings share one key set,
        which holds the planes whose geometry could be established; a plane that
        failed both paths is absent from all four.
    """
    resolved = settings if settings is not None else get_settings()
    gray = _as_gray_u8(image_bgr)
    shape = (int(gray.shape[0]), int(gray.shape[1]))

    if segments is None:
        segments = detect_line_segments(gray)
    candidates: list[VanishingPoint | None] = []
    for cluster in cluster_by_direction(segments, shape, resolved):
        estimated = estimate_vanishing_point(cluster, shape, resolved)
        candidates.append(None if estimated is None else estimated[0])

    centre = principal_point(shape)
    vps = enforce_orthogonality(candidates, centre, default_focal_guess(shape), resolved)
    inputs = _plane_geometry_inputs(planes)
    contours = {name: contour for name, (contour, _) in inputs.items()}

    horizon = horizon_from_vps(vps)
    if horizon is None:
        horizon = horizon_from_contours(contours, shape)
    mode: GeometryMode = (
        "vanishing_points"
        if all(vps.get(label) is not None for label in ("VPx", "VPy", "VPz"))
        else "planar_fallback"
    )

    homographies: dict[PlaneName, np.ndarray] = {}
    inverses: dict[PlaneName, np.ndarray] = {}
    extents: dict[PlaneName, tuple[float, float, float, float]] = {}
    errors: dict[PlaneName, float] = {}

    for name, (contour, quad) in inputs.items():
        homography: np.ndarray | None = None
        extent: tuple[float, float, float, float] | None = None

        if mode == "vanishing_points":
            frame = plane_frame(name, vps, horizon, contour, shape, resolved)
            if frame is not None:
                homography = homography_from_frame(frame)
                extent = frame.extent_mm

        if homography is None:
            try:
                image_quad = quad if quad is not None else _quad_from_contour(contour)
            except (ValueError, cv2.error):
                continue
            metric_quad = metric_quad_from_image_quad(
                image_quad, horizon, shape, settings=resolved
            )
            if metric_quad is None:
                continue
            homography = homography_from_quad(image_quad, metric_quad)
            if homography is None:
                continue
            extent = (
                0.0,
                0.0,
                float(metric_quad[:, 0].max()),
                float(metric_quad[:, 1].max()),
            )

        inverse = invert_homography(homography)
        if inverse is None or extent is None:  # pragma: no cover - both are checked above
            continue
        key: PlaneName = name  # type: ignore[assignment]
        homographies[key] = homography
        inverses[key] = inverse
        extents[key] = extent
        errors[key] = reprojection_rmse(homography, extent)

    return Calibration(
        vanishing_points=dict(vps),
        horizon=horizon,
        geometry_mode=mode,
        homographies=homographies,
        homography_inverses=inverses,
        plane_extents_mm=extents,
        reprojection_rmse_px=errors,
    )
