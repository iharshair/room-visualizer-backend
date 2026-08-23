"""Tests for `backend.core.geometry` (Requirements 5.1 to 5.7, 6.1 to 6.4, 13.3, 13.4).

The Geometry_Engine is the one module in the service whose output has a closed
form, so every property here is scored against `tests/fixtures/synthetic.py`'s
analytic camera rather than against a remembered number:

* **Property 11** compares recovered vanishing points to `K R d` for each world
  axis. Its pixel tolerance *widens with each point's distance from the principal
  point*, which is not a convenience -- see `VP_PIXEL_TOLERANCE` for why no fixed
  pixel bound, and no fixed fraction of that distance either, can hold for a
  depth point five pixels out and a lateral point six diagonals out at once. Past
  a few diagonals no position bound discriminates at all, so the property carries
  a second, conditioned bound on the recovered *direction*.
* **Property 12** compares the horizon derived from those points against the
  image of the ground plane's line at infinity, and asserts the two horizontal
  points actually lie on it. The second half is exact to floating point, because
  the horizon *is* their cross product -- so it is asserted at 1e-6 px, not at a
  fudged pixel bound, and it would catch a horizon derived from the wrong pair.
* **Property 13** covers the case Property 12 cannot: fewer than two horizontal
  vanishing points, where the horizon has to come from contour extents instead.
  What is asserted there is only what Requirement 6.2 promises -- a horizon
  exists and crosses the frame -- because a contour-derived horizon is an
  orientation, not a calibration.

* **Properties 14, 15, and 16** cover the homographies those points imply, and
  all three are stated over quantities an image can show -- a round trip, a
  length ratio, a projected area -- rather than over the matrices themselves.
  Recovered and analytic homographies parametrise the same plane from different
  metric origins, so an elementwise comparison would score a translation nothing
  downstream can observe; the absolute scale that *is* observable is pinned by an
  anchored test on the documented fixture pose instead.

The LSD-absent unit tests pin Requirement 5.1's fallback: both detector paths
have to hand back the same array layout, or every consumer would need to know
which one ran. The mode and clamp unit tests pin Requirement 6.1's other half --
a pose that loses a vanishing point has to say so and still serve geometry.

Layout: measured tolerances first, then shared helpers and strategies, then one
banner-delimited section per property. Tolerances and helpers are shared, so a
new section adds only its own assertions.

Poses are drawn at `supersample=2`. At `supersample=1` distant checkerboard rows
alias into short false edges that the detector reports as real, which triples the
observed vanishing point error -- a fixture artifact, not a property of the
engine, so it is rendered away rather than absorbed into the tolerance.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Sequence

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
    target,
)

from backend.config import get_settings
from backend.core import geometry
from backend.core.geometry import (
    MIN_SEGMENT_LENGTH_FRACTION,
    VP_MAX_DIAGONALS,
    Calibration,
    Line,
    PlaneFrame,
    VanishingPoint,
    calibrate,
    detect_line_segments,
    homography_from_frame,
    horizon_from_contours,
    horizon_from_vps,
    metric_quad_from_image_quad,
    plane_frame,
    principal_point,
    reprojection_rmse,
)
from tests.fixtures.synthetic import SyntheticRoom

#: The three labels `Calibration.vanishing_points` always carries.
VP_LABELS: tuple[str, ...] = ("VPx", "VPy", "VPz")

#: Every Structural_Plane name the fixture can produce, in the design's order.
PLANE_NAMES: tuple[str, ...] = ("floor", "wall_back", "wall_left", "wall_right")


# --------------------------------------------------------------------------- #
# Documented tolerances
# --------------------------------------------------------------------------- #
#
# Every bound below was measured over a 1920-configuration sweep of the fixture
# -- yaw +/-5 to +/-24 deg, pitch -4 to -28 deg, focal 0.7 to 1.3 x image width,
# all four two-or-more wall subsets, two occluder seeds -- at the same 640x480
# `supersample=2` render the properties draw over. The measured maximum is quoted
# next to each bound so a regression that widens one is visible as a behaviour
# change rather than as a rounding nuisance.

#: Per-label `(absolute floor px, angular slack in radians)` for Property 11. The
#: pixel budget each pair yields is assembled by
#: :func:`allowed_vanishing_point_error_px`.
#:
#: A single absolute pixel bound is impossible here, and the reason is geometric
#: rather than numerical. A vanishing point sits at `pp + f * tan(theta) * u` for
#: the angle `theta` between its world direction and the optical axis, so its
#: *position* is a tangent of the quantity actually recovered. Near-frontal
#: directions land a few hundred pixels out and are pinned to well under a pixel;
#: near-lateral ones land thousands of pixels out, where the same angular
#: accuracy costs `f * sec^2(theta)` pixels per radian -- seventy times more.
#: Scoring both against one number would either wave the depth point through or
#: fail the lateral point for being correct.
#:
#: A bound stated as a plain *fraction* of the distance from the principal point
#: does not fix that either, and the sweep showed why: `sec^2` grows faster than
#: the distance itself, so the fraction a fixed direction error costs keeps
#: climbing -- past 0.4 at six diagonals and past 0.8 at eight, on poses where the
#: recovered direction is within three degrees of truth. The slack is therefore
#: applied to the *derivative*, `f * (1 + (d/f)^2)` pixels per radian, which is
#: the position sensitivity at that distance. The result is still a pixel bound
#: that widens with distance from the principal point, but it widens at the rate
#: the geometry actually dictates.
#:
#: Measured maxima over the sweep and over 800 additional random draws from the
#: same strategy, per label:
#:
#: | label | max err/sensitivity | max absolute | max direction error |
#: |-------|---------------------|--------------|---------------------|
#: | VPx   | 0.143 rad           | 3031 px      | 5.1 deg             |
#: | VPy   | 0.051 rad           | 650 px       | 2.7 deg             |
#: | VPz   | 0.009 rad           | 5.4 px       | 0.5 deg             |
#:
#: `VPz` is the depth direction, which sits near the principal point and is
#: carried by the longest converging edges in the frame, so it is held an order of
#: magnitude tighter -- and for it, at low yaw, the absolute floor is what binds.
#: The worst `VPx` configurations are all the left-plus-right wall set with no
#: back wall, where the lateral direction is evidenced only by floor grout lines.
#:
#: Note on what the slack is not: because it multiplies the *linearised*
#: sensitivity it over-estimates the true direction error at large `theta` --
#: 0.143 linearised radians was 4.7 degrees of actual direction error, not 8.2.
#: That is deliberate. The exact form, `f * (tan(theta + slack) - tan(theta))`,
#: passes through infinity for any `theta` within `slack` of 90 degrees, so it
#: stops bounding anything at all exactly where a bound is hardest to meet.
#:
#: And this is where a position bound runs out: past about four diagonals the
#: budget the engine genuinely needs exceeds the *separation between two different
#: world axes*, so no position bound can both hold and rule out a mislabelled
#: axis. `VP_DIRECTION_TOLERANCE_DEG` is what carries that half of the claim.
VP_PIXEL_TOLERANCE: Mapping[str, tuple[float, float]] = {
    "VPx": (4.0, 0.30),
    "VPy": (4.0, 0.30),
    "VPz": (4.0, 0.05),
}

#: Property 11's companion bound: the angle, in degrees, between the recovered
#: vanishing direction and the analytic one. Where `VP_PIXEL_TOLERANCE` is the
#: pixel statement the design asks for, this is the *conditioned* one -- the angle
#: is what the estimator actually recovers, and unlike a position it stays
#: bounded no matter how far out the vanishing point lands.
#:
#: It is also what gives the property teeth. World axes are 90 degrees apart, so a
#: recovered triple with two labels transposed misses by tens of degrees; the
#: provisional labels a `planar_fallback` run reports miss by more still. A bound
#: at 15 degrees rules both out by a wide margin while sitting comfortably clear
#: of what the engine needs.
#:
#: Measured maxima over 1200 random draws from the property's own strategy:
#: `VPx` 4.19 deg (99th percentile 2.29, median 0.099), `VPy` 2.50 deg (1.06,
#: 0.075), `VPz` 0.45 deg (0.21, 0.042). A dense probe of the worst corner --
#: steep pitch, long focal, left-and-right walls only, so the visible verticals
#: are few and short -- pushed `VPy` to 8 degrees, which is what the 15 is sized
#: against rather than the random-draw maximum.
VP_DIRECTION_TOLERANCE_DEG: Mapping[str, float] = {
    "VPx": 15.0,
    "VPy": 15.0,
    "VPz": 4.0,
}

#: Property 12, rows between the derived horizon and the analytic one, sampled at
#: the left border, the vertical centre line, and the right border. Measured
#: maximum 14.4 px over the sweep and over 745 further random draws, with a median
#: of 0.65 px and a 95th percentile of 3.1 px; 7.2 px was the worst at the centre
#: column alone. The tail is driven by the same shallow-pitch wide-field poses that
#: stretch `VPx`: the horizon is the join of two points, so an angular error in
#: either one levers the row hardest at the far border.
HORIZON_ROW_TOLERANCE_PX = 28.0

#: Property 12, distance from a recovered horizontal vanishing point to the
#: derived horizon. This one is not a measurement allowance: the horizon is the
#: cross product of those two points, so incidence is exact and the only slack
#: needed is float64 round-off. Measured maximum 8.5e-14 px. Keeping it at 1e-6
#: rather than at a pixel is what makes the assertion able to fail -- a horizon
#: built from the wrong pair, or from `VPy`, misses by hundreds of pixels.
HORIZON_INCIDENCE_TOLERANCE_PX = 1e-6

#: Size and render quality of the rooms the properties are drawn over. 640x480
#: at `supersample=2` costs about 60 ms to generate and another 15 ms to
#: calibrate, so a 75-example property runs in about six seconds; the fixed
#: 1600x1200 fixture would cost twenty times that per example.
PROPERTY_ROOM_WIDTH = 640
PROPERTY_ROOM_HEIGHT = 480
PROPERTY_SUPERSAMPLE = 2


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def room_contours(room: SyntheticRoom) -> dict[str, np.ndarray]:
    """The room's exact plane outlines, in the shape `calibrate` accepts.

    `calibrate` duck-types its `planes` argument, so handing it the fixture's
    analytic polygons keeps the geometry properties independent of the Segmenter:
    a segmentation regression cannot make a calibration property fail, and a
    calibration regression cannot hide behind a lucky mask.
    """
    return {name: polygon.astype(np.float64) for name, polygon in room.plane_polygons.items()}


def calibrated(room: SyntheticRoom) -> Calibration:
    """Calibrate a fixture room against its own analytic plane outlines."""
    return calibrate(room.image, room_contours(room), settings=get_settings())


def image_diagonal(shape: Sequence[int]) -> float:
    """Image diagonal in pixels from a `(height, width)` shape."""
    return math.hypot(float(shape[1]), float(shape[0]))


def distance_from_principal_point(point: Sequence[float], centre: Sequence[float]) -> float:
    """How far a vanishing point sits from the principal point, in pixels.

    This is the conditioning measure Property 11's tolerance scales with, and it
    is also what decides whether a *ground-truth* point is near enough to
    infinity to be excluded from the comparison.
    """
    return math.hypot(float(point[0]) - float(centre[0]), float(point[1]) - float(centre[1]))


def vanishing_point_sensitivity_px_per_rad(distance_px: float, focal_px: float) -> float:
    """How far a vanishing point moves per radian of direction error.

    A direction `theta` off the optical axis images at `d = f * tan(theta)`, so
    `dd/dtheta = f * sec^2(theta) = f * (1 + (d/f)^2)`. This is the whole reason
    Property 11's tolerance cannot be a constant: the same recovery quality shows
    up as sub-pixel agreement for a near-frontal direction and as thousands of
    pixels for a near-lateral one.
    """
    return focal_px * (1.0 + (distance_px / focal_px) ** 2)


def allowed_vanishing_point_error_px(
    label: str, truth_distance_px: float, focal_px: float
) -> float:
    """Property 11's pixel budget for one label at one ground-truth position.

    The floor covers the regime where the sensitivity term vanishes -- a direction
    on the optical axis, where the vanishing point is pinned but the derivative
    says nothing -- and the sensitivity term covers everything else.
    """
    floor_px, slack_rad = VP_PIXEL_TOLERANCE[label]
    return floor_px + slack_rad * vanishing_point_sensitivity_px_per_rad(
        truth_distance_px, focal_px
    )


def vanishing_direction(
    point: Sequence[float], centre: Sequence[float], focal_px: float
) -> np.ndarray:
    """Unit camera-space direction a vanishing point stands for.

    With a centred principal point and unit aspect ratio the direction is simply
    `[v - pp, f]` normalised, which is the same relation the engine's own
    orthogonality constraint is derived from.
    """
    vector = np.array(
        [float(point[0]) - float(centre[0]), float(point[1]) - float(centre[1]), focal_px],
        dtype=np.float64,
    )
    return vector / float(np.linalg.norm(vector))


def direction_error_deg(
    recovered: Sequence[float],
    truth: Sequence[float],
    centre: Sequence[float],
    focal_px: float,
) -> float:
    """Angle in degrees between a recovered and an analytic vanishing direction.

    Compared on `abs` of the dot product because a direction and its negation
    share a vanishing point -- the engine has no way to tell "along +X" from
    "along -X" and is not asked to.
    """
    cosine = abs(
        float(
            vanishing_direction(recovered, centre, focal_px)
            @ vanishing_direction(truth, centre, focal_px)
        )
    )
    return math.degrees(math.acos(min(1.0, cosine)))


def point_line_distance(line: Line, point: Sequence[float]) -> float:
    """Perpendicular distance in pixels from a point to a normalised line.

    Valid only because the module normalises every line it returns to
    `a^2 + b^2 = 1`, which makes `a*x + b*y + c` a signed pixel distance rather
    than an arbitrarily scaled residual.
    """
    a, b, c = line
    return abs(a * float(point[0]) + b * float(point[1]) + c)


def horizon_row_at(line: Line, x: float) -> float:
    """Row where a non-vertical line crosses image column `x`."""
    a, b, c = line
    if abs(b) < 1e-12:
        raise AssertionError(f"horizon {line!r} is vertical; it crosses no image column")
    return -(a * float(x) + c) / b


def sample_columns(width: int) -> tuple[float, ...]:
    """Left border, vertical centre line, right border.

    The centre column is the one Requirement 6.2 states its in-frame guarantee
    over; the borders are where an angular error in the horizon is most visible,
    so sampling all three makes the Property 12 row comparison sensitive to tilt
    as well as to offset.
    """
    return (0.0, (width - 1) / 2.0, float(width - 1))


def orthogonality_residual(
    triple: Sequence[VanishingPoint], centre: Sequence[float]
) -> tuple[float, float] | None:
    """Focal length and orthogonality residual implied by three vanishing points.

    Written out here rather than imported from the module under test: the
    residual bound is the second half of Property 11, and importing the
    implementation's own private helper would reduce that half to a tautology.

    Three orthogonal world directions give three independent readings of one
    focal length through `f^2 = -(v_i - pp) . (v_j - pp)`. The residual is the
    worse of two dimensionless defects -- the spread of those readings, and how
    far from perpendicular the directions they induce actually are.

    Returns:
        `(focal_px, residual)`, or `None` when any pair admits no positive
        `f^2`, in which case the triple is not the image of an orthogonal frame
        under any focal length at all.
    """
    focals: list[float] = []
    for i, j in ((0, 1), (0, 2), (1, 2)):
        ax = float(triple[i][0]) - float(centre[0])
        ay = float(triple[i][1]) - float(centre[1])
        bx = float(triple[j][0]) - float(centre[0])
        by = float(triple[j][1]) - float(centre[1])
        squared = -(ax * bx + ay * by)
        if squared <= 0.0:
            return None
        focals.append(math.sqrt(squared))

    focal = float(np.mean(focals))
    spread = max(abs(value - focal) for value in focals) / focal

    directions = []
    for vp in triple:
        vector = np.array(
            [float(vp[0]) - float(centre[0]), float(vp[1]) - float(centre[1]), focal],
            dtype=np.float64,
        )
        directions.append(vector / float(np.linalg.norm(vector)))
    perpendicularity = max(
        abs(float(directions[i] @ directions[j])) for i, j in ((0, 1), (0, 2), (1, 2))
    )
    return focal, max(spread, perpendicularity)


def truth_is_effectively_infinite(
    truth: VanishingPoint | None, centre: Sequence[float], diagonal: float
) -> bool:
    """Whether a ground-truth vanishing point is beyond what is recoverable.

    A direction parallel to the image plane has its vanishing point genuinely at
    infinity, and the fixture reports that as `None`. Approaching it, the point
    is still finite but carries no usable direction -- a pixel of noise on a
    contributing line moves it thousands of pixels -- so the engine caps it at
    `VP_MAX_DIAGONALS` and reports `None` instead of a fabricated axis. Comparing
    a recovered point against such a truth value measures the fixture's tangent,
    not the engine, so those labels are excluded from Property 11's bound.
    """
    if truth is None:
        return True
    return distance_from_principal_point(truth, centre) > VP_MAX_DIAGONALS * diagonal


# --------------------------------------------------------------------------- #
# Shared strategies
# --------------------------------------------------------------------------- #
#
# The pose space is the field-of-view and attitude regime the fixture documents,
# with one deliberate restriction: `|yaw| >= 5` degrees. Below that the lateral
# direction turns parallel to the image plane and its true vanishing point runs
# off past `VP_MAX_DIAGONALS` -- at zero yaw the fixture reports it as `None`
# outright -- so the engine correctly declines to recover it and routes the scene
# to the planar fallback. Those poses are real and are covered by Property 13 and
# by task 7.5's mode unit tests; including them in Property 11 would only measure
# how a tangent diverges.

#: Focal length as a fraction of image width. 0.7 to 1.3 spans roughly 75 down to
#: 45 degrees of horizontal field of view, which brackets what phone cameras
#: ship; the fixture's own default is 0.875.
_focal_ratio = st.floats(min_value=0.70, max_value=1.30)
_yaw_magnitude = st.floats(min_value=5.0, max_value=24.0)
_yaw_sign = st.sampled_from((-1.0, 1.0))
_pitch_deg = st.floats(min_value=-28.0, max_value=-4.0)

#: Visible wall subsets. Single-wall rooms are excluded because two structural
#: surfaces are the minimum from which two horizontal directions can be read at
#: all; the four subsets below are exactly the two-or-more-wall cases.
_WALL_SETS: tuple[tuple[str, ...], ...] = (
    ("left", "right", "back"),
    ("left", "back"),
    ("right", "back"),
    ("left", "right"),
)
_walls = st.sampled_from(_WALL_SETS)

#: Seeds occluder placement and sensor noise, so a counterexample reproduces from
#: the reported draws alone.
_seed = st.integers(min_value=0, max_value=2**16)

_ROOM_PROPERTY_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    # A supersampled room plus a full calibration is ~75 ms of honest work per
    # example, so these are legitimately slow rather than accidentally slow.
    suppress_health_check=[HealthCheck.too_slow],
)


def _draw_room(
    factory: Callable[..., SyntheticRoom],
    focal_ratio: float,
    yaw_magnitude: float,
    yaw_sign: float,
    pitch_deg: float,
    walls: Iterable[str],
    seed: int,
) -> SyntheticRoom:
    """Turn one set of drawn camera parameters into a rendered room."""
    return factory(
        focal_px=focal_ratio * PROPERTY_ROOM_WIDTH,
        yaw_deg=yaw_sign * yaw_magnitude,
        pitch_deg=pitch_deg,
        walls=tuple(walls),
        width=PROPERTY_ROOM_WIDTH,
        height=PROPERTY_ROOM_HEIGHT,
        seed=seed,
        supersample=PROPERTY_SUPERSAMPLE,
    )


# --------------------------------------------------------------------------- #
# Property 11 -- vanishing points match analytic ground truth
# (Requirements 5.2, 13.3)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 11: Vanishing points match analytic
# ground truth
@given(
    focal_ratio=_focal_ratio,
    yaw_magnitude=_yaw_magnitude,
    yaw_sign=_yaw_sign,
    pitch_deg=_pitch_deg,
    walls=_walls,
    seed=_seed,
)
@_ROOM_PROPERTY_SETTINGS
def test_property_11_vanishing_points_match_analytic_ground_truth(
    randomized_room: Callable[..., SyntheticRoom],
    focal_ratio: float,
    yaw_magnitude: float,
    yaw_sign: float,
    pitch_deg: float,
    walls: tuple[str, ...],
    seed: int,
) -> None:
    """Property 11: recovered vanishing points match `K R d`, and are orthogonal.

    Three claims, all scored against the fixture's exact camera:

    * each recovered point is within `VP_PIXEL_TOLERANCE` of the analytic point
      for its world axis, with the budget widening with that point's distance from
      the principal point for the reason documented on the table;
    * each recovered *direction* is within `VP_DIRECTION_TOLERANCE_DEG` of the
      analytic one, which is the conditioned half of the same claim and the half
      that rules out a transposed label;
    * the recovered triple's mutual orthogonality residual is inside
      `orthogonality_tolerance`, computed here from first principles rather than
      taken from the implementation.

    The bound is asserted in `vanishing_points` mode, which is precisely the mode
    in which an orthogonal triple was accepted and all three labels therefore
    carry a recovered direction. In `planar_fallback` mode the labels are a
    provisional best pair, not a calibrated frame -- the documented contract
    there is only that fewer than three survive, which is what is asserted
    instead. Both branches are reported through `event`, so
    `--hypothesis-show-statistics` shows the split and the property cannot go
    quietly vacuous.

    **Validates: Requirements 5.2, 13.3**
    """
    room = _draw_room(
        randomized_room, focal_ratio, yaw_magnitude, yaw_sign, pitch_deg, walls, seed
    )
    calibration = calibrated(room)
    centre = principal_point(room.shape)
    diagonal = image_diagonal(room.shape)
    recovered = calibration.vanishing_points
    event(f"geometry_mode={calibration.geometry_mode}")

    if calibration.geometry_mode != "vanishing_points":
        # The documented meaning of the fallback mode: no orthogonal triple was
        # accepted, so at least one label is unset.
        assert sum(recovered.get(label) is not None for label in VP_LABELS) < 3
        return

    triple = [recovered[label] for label in VP_LABELS]
    assert all(vp is not None for vp in triple), (
        f"vanishing_points mode must label all three axes, got {recovered!r}"
    )

    scored = orthogonality_residual(triple, centre)  # type: ignore[arg-type]
    assert scored is not None, (
        f"recovered triple {triple!r} admits no positive f^2 about {centre!r}, "
        "so it is not the image of an orthogonal frame"
    )
    _, residual = scored
    tolerance = float(get_settings().orthogonality_tolerance)
    target(residual, label="orthogonality residual")
    assert residual <= tolerance, (
        f"orthogonality residual {residual:.4f} exceeds {tolerance} for {triple!r}"
    )

    checked = 0
    for label in VP_LABELS:
        truth = room.truth_vps[label]
        if truth_is_effectively_infinite(truth, centre, diagonal):
            event(f"{label} truth effectively at infinity")
            continue
        assert truth is not None  # narrowed by the guard above
        point = recovered[label]
        assert point is not None  # narrowed by the mode check above

        truth_distance = distance_from_principal_point(truth, centre)
        error_px = math.hypot(point[0] - truth[0], point[1] - truth[1])
        focal_px = float(room.camera.focal_px)
        allowed_px = allowed_vanishing_point_error_px(label, truth_distance, focal_px)

        target(
            error_px / vanishing_point_sensitivity_px_per_rad(truth_distance, focal_px),
            label=f"{label} position error in sensitivity radians",
        )
        assert error_px <= allowed_px, (
            f"{label} recovered at {point!r} is {error_px:.2f} px from the analytic "
            f"{truth!r}, which is {truth_distance:.1f} px "
            f"({truth_distance / diagonal:.2f} diagonals) from the principal point at "
            f"focal {focal_px:.1f} px; allowed {allowed_px:.2f} px"
        )

        angle_deg = direction_error_deg(point, truth, centre, focal_px)
        target(angle_deg, label=f"{label} direction error deg")
        assert angle_deg <= VP_DIRECTION_TOLERANCE_DEG[label], (
            f"{label} recovered direction is {angle_deg:.2f} deg from the analytic one "
            f"(recovered {point!r}, truth {truth!r}); allowed "
            f"{VP_DIRECTION_TOLERANCE_DEG[label]} deg"
        )
        checked += 1

    assert checked > 0, "no label had a recoverable ground-truth vanishing point to compare"


def test_property_11_direction_bound_rejects_a_transposed_label(
    synthetic_room: SyntheticRoom,
) -> None:
    """Guard for Property 11: the direction bound is wide but not vacuous.

    `VP_PIXEL_TOLERANCE` has to widen with distance from the principal point, and
    past a few diagonals the budget it grants exceeds the gap between two
    different world axes -- so on its own it would pass a calibration that had
    swapped two labels. This drives exactly that corruption and shows the
    direction bound catches every transposition, by a factor of at least four.

    Written against the analytic points rather than a recovered calibration, so it
    measures the tolerance table itself and cannot be perturbed by an estimator
    change.
    """
    centre = principal_point(synthetic_room.shape)
    focal_px = float(synthetic_room.camera.focal_px)
    truth = {label: synthetic_room.truth_vps[label] for label in VP_LABELS}
    assert all(point is not None for point in truth.values())

    for label in VP_LABELS:
        for other in VP_LABELS:
            if other == label:
                continue
            # Truth for `other` presented as if it were `label`: the exact mistake
            # a mislabelled triple makes.
            angle_deg = direction_error_deg(truth[other], truth[label], centre, focal_px)
            allowed = VP_DIRECTION_TOLERANCE_DEG[label]
            assert angle_deg > 4.0 * allowed, (
                f"claiming {other}'s direction as {label} is only {angle_deg:.1f} deg "
                f"out, against a {allowed} deg bound -- the bound no longer "
                "distinguishes the axes"
            )


def test_property_11_holds_on_the_documented_fixture_pose(
    synthetic_room: SyntheticRoom,
) -> None:
    """Property 11 at the fixed 1600x1200 fixture, as a non-random anchor.

    Property 11 itself draws small, fast rooms across a hard pose range. This
    pins the same claim at the documented full-size pose, where the render is
    supersampled and the segments are long, so the recovery is the best the engine
    manages -- and it comes in within one degree, two orders of magnitude inside
    the drawn bound. That gap is the evidence that the drawn bound is paying for
    pose difficulty rather than for a sloppy estimator.
    """
    calibration = calibrated(synthetic_room)
    assert calibration.geometry_mode == "vanishing_points"
    centre = principal_point(synthetic_room.shape)
    diagonal = image_diagonal(synthetic_room.shape)

    for label in VP_LABELS:
        truth = synthetic_room.truth_vps[label]
        assert truth is not None, f"the documented pose should make {label} finite"
        point = calibration.vanishing_points[label]
        assert point is not None
        error_px = math.hypot(point[0] - truth[0], point[1] - truth[1])
        distance = distance_from_principal_point(truth, centre)
        focal_px = float(synthetic_room.camera.focal_px)
        allowed_px = allowed_vanishing_point_error_px(label, distance, focal_px)
        assert error_px <= allowed_px, (
            f"{label}: {error_px:.2f} px from truth at {distance / diagonal:.2f} "
            f"diagonals, allowed {allowed_px:.2f} px"
        )
        angle_deg = direction_error_deg(point, truth, centre, focal_px)
        assert angle_deg <= 1.0, f"{label} direction is {angle_deg:.3f} deg out"

    scored = orthogonality_residual(
        [calibration.vanishing_points[label] for label in VP_LABELS],  # type: ignore[misc]
        centre,
    )
    assert scored is not None
    assert scored[1] <= float(get_settings().orthogonality_tolerance)


# --------------------------------------------------------------------------- #
# Property 12 -- horizon is consistent with the recovered vanishing points
# (Requirement 5.3)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 12: Horizon is consistent with the
# recovered vanishing points
@given(
    focal_ratio=_focal_ratio,
    yaw_magnitude=_yaw_magnitude,
    yaw_sign=_yaw_sign,
    pitch_deg=_pitch_deg,
    walls=_walls,
    seed=_seed,
)
@_ROOM_PROPERTY_SETTINGS
def test_property_12_horizon_is_consistent_with_recovered_vanishing_points(
    randomized_room: Callable[..., SyntheticRoom],
    focal_ratio: float,
    yaw_magnitude: float,
    yaw_sign: float,
    pitch_deg: float,
    walls: tuple[str, ...],
    seed: int,
) -> None:
    """Property 12: the derived horizon matches truth and carries both VPs.

    The precondition is two recovered *horizontal* vanishing points, not two
    recovered points of any kind: `VPy` is the vertical direction and does not
    lie on the horizon, so a run holding one horizontal point and the vertical
    has two points and still no horizon from them. `horizon_from_vps` returning a
    line is exactly that precondition, so it is what the `assume` tests.

    **Validates: Requirements 5.3**
    """
    room = _draw_room(
        randomized_room, focal_ratio, yaw_magnitude, yaw_sign, pitch_deg, walls, seed
    )
    calibration = calibrated(room)
    from_vps = horizon_from_vps(calibration.vanishing_points)
    assume(from_vps is not None)
    assert from_vps is not None  # narrowed for type checkers
    event(f"geometry_mode={calibration.geometry_mode}")

    # `calibrate` prefers the vanishing-point horizon whenever it exists, so this
    # is also an assertion that the fallback did not shadow the better estimate.
    assert calibration.horizon == pytest.approx(from_vps, abs=1e-12), (
        f"calibrate reported {calibration.horizon!r} while the vanishing points "
        f"give {from_vps!r}"
    )

    height, width = room.shape
    worst_row_error = 0.0
    for x in sample_columns(width):
        error = abs(horizon_row_at(calibration.horizon, x) - room.horizon_y_at(x))
        worst_row_error = max(worst_row_error, error)
    target(worst_row_error, label="horizon row error px")
    assert worst_row_error <= HORIZON_ROW_TOLERANCE_PX, (
        f"derived horizon {calibration.horizon!r} is {worst_row_error:.2f} rows from "
        f"the analytic {room.truth_horizon!r} on a {width}x{height} frame"
    )

    incident = 0
    for label in ("VPx", "VPz"):
        point = calibration.vanishing_points.get(label)
        if point is None:
            continue
        offset = point_line_distance(calibration.horizon, point)
        assert offset <= HORIZON_INCIDENCE_TOLERANCE_PX, (
            f"{label} at {point!r} lies {offset:.3e} px off the derived horizon "
            f"{calibration.horizon!r}"
        )
        incident += 1
    assert incident >= 2, (
        "a horizon derived from vanishing points implies two finite horizontal "
        f"points, found {incident}"
    )


def test_horizon_from_vps_needs_two_horizontal_points() -> None:
    """`VPy` is not on the horizon, so one horizontal point plus it yields none.

    The guard that makes Property 12's precondition meaningful: without it, a
    horizon fitted through the vertical vanishing point would satisfy the
    incidence half of the property while being wrong by construction.
    """
    assert horizon_from_vps({"VPx": (900.0, 300.0), "VPz": (-400.0, 305.0)}) is not None
    assert horizon_from_vps({"VPx": (900.0, 300.0), "VPy": (320.0, 9000.0)}) is None
    assert horizon_from_vps({"VPz": (900.0, 300.0), "VPy": (320.0, 9000.0)}) is None
    assert horizon_from_vps({"VPx": None, "VPy": None, "VPz": None}) is None
    # Coincident points span no line.
    assert horizon_from_vps({"VPx": (900.0, 300.0), "VPz": (900.0, 300.0)}) is None


# --------------------------------------------------------------------------- #
# Property 13 -- a horizon is always produced and lies within the image
# (Requirement 6.2)
# --------------------------------------------------------------------------- #


def _contour_strategy(width: int, height: int) -> st.SearchStrategy[np.ndarray]:
    """Point clouds spanning well past the frame in both directions.

    Contours are drawn far outside the image on purpose: a Segmenter contour is
    clipped to the frame, but the horizon this feeds is *derived* from vertical
    extents, and clamping is the mechanism Requirement 6.2's in-frame guarantee
    rests on. Out-of-frame extents are what exercise it.
    """
    coordinate = st.floats(
        min_value=-4.0 * max(width, height),
        max_value=4.0 * max(width, height),
        allow_nan=False,
        allow_infinity=False,
    )
    return st.lists(
        st.tuples(coordinate, coordinate), min_size=1, max_size=8
    ).map(lambda points: np.asarray(points, dtype=np.float64))


# Feature: ai-room-tile-visualizer, Property 13: A horizon is always produced and
# lies within the image
@given(
    named=st.dictionaries(
        keys=st.sampled_from(PLANE_NAMES),
        values=_contour_strategy(PROPERTY_ROOM_WIDTH, PROPERTY_ROOM_HEIGHT),
        max_size=len(PLANE_NAMES),
    ),
    height=st.integers(min_value=1, max_value=2048),
    width=st.integers(min_value=1, max_value=2048),
)
@hypothesis_settings(max_examples=200, deadline=None)
def test_property_13_a_horizon_is_always_produced_inside_the_image(
    named: dict[str, np.ndarray], height: int, width: int
) -> None:
    """Property 13: any contour set yields an in-frame horizon.

    Driven directly against `horizon_from_contours`, because that is the function
    Requirement 6.2 places the guarantee on -- `horizon_from_vps` returns the
    *true* horizon, which for a strongly pitched camera legitimately falls
    outside the frame, and clamping it would be a calibration error rather than a
    robustness measure.

    The empty contour set is included, so the "no usable contour at all" branch
    is covered rather than assumed.

    **Validates: Requirements 6.2**
    """
    horizon = horizon_from_contours(named, (height, width))
    a, b, c = horizon

    assert all(math.isfinite(value) for value in horizon), f"non-finite horizon {horizon!r}"
    assert math.hypot(a, b) == pytest.approx(1.0, abs=1e-12), (
        f"horizon {horizon!r} is not normalised to a^2 + b^2 == 1"
    )
    assert b > 0.0, (
        f"a contour-derived horizon must cross every image column, got {horizon!r}"
    )

    row = horizon_row_at(horizon, (width - 1) / 2.0)
    assert 0.0 <= row <= float(height - 1), (
        f"horizon {horizon!r} crosses the vertical centre line at row {row:.2f}, "
        f"outside a {width}x{height} frame"
    )


@given(
    unnamed=st.lists(
        _contour_strategy(PROPERTY_ROOM_WIDTH, PROPERTY_ROOM_HEIGHT), max_size=4
    ),
)
@hypothesis_settings(max_examples=100, deadline=None)
def test_property_13_holds_for_unlabelled_contour_sequences(
    unnamed: list[np.ndarray],
) -> None:
    """Property 13 again, for a bare sequence carrying no plane names.

    The named and unnamed inputs take different branches -- the named path can
    find a floor-wall junction, the unnamed one only has a union extent -- so the
    in-frame guarantee has to be established on both.

    **Validates: Requirements 6.2**
    """
    horizon = horizon_from_contours(unnamed, (PROPERTY_ROOM_HEIGHT, PROPERTY_ROOM_WIDTH))
    row = horizon_row_at(horizon, (PROPERTY_ROOM_WIDTH - 1) / 2.0)
    assert 0.0 <= row <= float(PROPERTY_ROOM_HEIGHT - 1), (
        f"horizon {horizon!r} crosses the centre line at row {row:.2f}"
    )


@pytest.mark.parametrize("contours", [None, {}, [], [np.empty((0, 2), dtype=np.float32)]])
def test_horizon_from_contours_survives_an_empty_input(contours: object) -> None:
    """No contours at all still produces a usable, in-frame horizon."""
    height, width = 480, 640
    horizon = horizon_from_contours(contours, (height, width))  # type: ignore[arg-type]
    row = horizon_row_at(horizon, (width - 1) / 2.0)
    assert 0.0 <= row <= float(height - 1)
    assert row == pytest.approx((height - 1) / 2.0)


@pytest.mark.parametrize("yaw_deg", [0.0, 0.4, 1.0])
def test_property_13_horizon_survives_a_pose_with_too_few_vanishing_points(
    randomized_room: Callable[..., SyntheticRoom], yaw_deg: float
) -> None:
    """Property 13 end to end: the near-frontal pose Property 12 cannot cover.

    At a near-zero yaw the lateral direction turns parallel to the image plane,
    so its true vanishing point is at (or effectively at) infinity, the engine
    reports `None` for it, and `horizon_from_vps` has fewer than two horizontal
    points to join. `calibrate` must still hand back a horizon that crosses the
    frame, which is the whole of Requirement 6.2.

    **Validates: Requirements 6.2**
    """
    room = randomized_room(
        focal_px=0.875 * PROPERTY_ROOM_WIDTH,
        yaw_deg=yaw_deg,
        pitch_deg=-12.0,
        walls=("left", "right", "back"),
        width=PROPERTY_ROOM_WIDTH,
        height=PROPERTY_ROOM_HEIGHT,
        seed=0,
        supersample=PROPERTY_SUPERSAMPLE,
    )
    calibration = calibrated(room)

    assert horizon_from_vps(calibration.vanishing_points) is None, (
        "this pose is meant to exercise the contour fallback; "
        f"got vanishing points {calibration.vanishing_points!r}"
    )
    height, width = room.shape
    row = horizon_row_at(calibration.horizon, (width - 1) / 2.0)
    assert 0.0 <= row <= float(height - 1), (
        f"fallback horizon {calibration.horizon!r} crosses the centre line at {row:.2f}"
    )


# --------------------------------------------------------------------------- #
# Line detection -- the LSD-absent fallback (Requirement 5.1)
# --------------------------------------------------------------------------- #


def _raise_cv2_error(*args: object, **kwargs: object) -> object:
    """Stand-in for a build whose `cv2` ships no line segment detector."""
    raise cv2.error("createLineSegmentDetector is unavailable in this build")


def _raise_attribute_error(*args: object, **kwargs: object) -> object:
    """The other way a stripped `cv2` fails: the symbol is simply missing."""
    raise AttributeError("createLineSegmentDetector")


@pytest.mark.parametrize("failure", [_raise_cv2_error, _raise_attribute_error])
def test_hough_fallback_returns_the_same_segment_array_as_lsd(
    synthetic_room: SyntheticRoom,
    monkeypatch: pytest.MonkeyPatch,
    failure: Callable[..., object],
) -> None:
    """Requirement 5.1: both detector paths return one array layout.

    `detect_line_segments` is the only place in the service that knows which
    detector ran, and it earns that by making the two paths indistinguishable
    downstream. So the fallback is checked for the same rank, column count,
    dtype, and minimum-length filtering as LSD -- not for the same segment count,
    which two different detectors have no reason to agree on.

    Both failure modes the module documents are driven, because they arrive from
    different places: `cv2.error` from a constructor that exists but refuses,
    `AttributeError` from a build where the symbol was never bound.
    """
    lsd_segments = detect_line_segments(synthetic_room.image)
    assert len(lsd_segments) > 0, "the fixture room should give LSD plenty to detect"

    monkeypatch.setattr(geometry.cv2, "createLineSegmentDetector", failure)
    hough_segments = detect_line_segments(synthetic_room.image)

    assert hough_segments.ndim == lsd_segments.ndim == 2
    assert hough_segments.shape[1] == lsd_segments.shape[1] == 4
    assert hough_segments.dtype == lsd_segments.dtype == np.float32
    assert len(hough_segments) > 0, "the Hough fallback found no segments at all"

    minimum = MIN_SEGMENT_LENGTH_FRACTION * image_diagonal(synthetic_room.shape)
    deltas = hough_segments[:, 2:4] - hough_segments[:, 0:2]
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    assert lengths.min() >= minimum - 1e-3, (
        f"fallback returned a segment {lengths.min():.2f} px long, under the "
        f"{minimum:.2f} px floor"
    )


def test_hough_fallback_still_calibrates_the_fixture_room(
    synthetic_room: SyntheticRoom, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback's segments are good enough to calibrate from.

    Same array layout is necessary but not sufficient: what Requirement 5.1
    actually buys is that a build without LSD still recovers a camera. Driving
    `calibrate` through the fallback segments is what shows that.
    """
    monkeypatch.setattr(geometry.cv2, "createLineSegmentDetector", _raise_cv2_error)
    segments = detect_line_segments(synthetic_room.image)
    calibration = calibrate(
        synthetic_room.image,
        room_contours(synthetic_room),
        segments=segments,
        settings=get_settings(),
    )
    assert calibration.geometry_mode == "vanishing_points"
    centre = principal_point(synthetic_room.shape)
    for label in VP_LABELS:
        truth = synthetic_room.truth_vps[label]
        point = calibration.vanishing_points[label]
        assert truth is not None and point is not None
        distance = distance_from_principal_point(truth, centre)
        error_px = math.hypot(point[0] - truth[0], point[1] - truth[1])
        focal_px = float(synthetic_room.camera.focal_px)
        allowed_px = allowed_vanishing_point_error_px(label, distance, focal_px)
        assert error_px <= allowed_px, (
            f"{label} recovered {error_px:.2f} px from truth through the Hough "
            f"fallback, allowed {allowed_px:.2f} px"
        )
        angle_deg = direction_error_deg(point, truth, centre, focal_px)
        assert angle_deg <= VP_DIRECTION_TOLERANCE_DEG[label], (
            f"{label} direction is {angle_deg:.2f} deg out through the Hough fallback"
        )

# --------------------------------------------------------------------------- #
# Homography correctness -- documented tolerances
# (Requirements 5.4, 5.5, 5.6, 5.7, 6.1, 6.3, 13.4)
# --------------------------------------------------------------------------- #
#
# Properties 14 to 16 are stated over *derived* quantities rather than over the
# recovered matrices themselves, and that is a deliberate reading of what a
# homography means. `H` and the fixture's `truth_homographies[name]` describe the
# same physical plane in two metric parametrisations that differ by a
# translation: the engine puts the metric origin at the corner of the plane's
# *visible* extent, the fixture puts it at the corner of the whole surface quad.
# Comparing the two elementwise would score that translation, which no tiling,
# extent, or reprojection figure can observe. So the properties compare what is
# observable -- a round trip, a length ratio, an image area -- and the anchored
# unit test at the end of this section is what pins the absolute scale against
# ground truth.

#: Property 14, the bound Requirement 5.6 states: RMS round-trip residual in
#: image pixels over a metric grid spanning the plane extent.
REPROJECTION_TOLERANCE_PX = 1.0

#: Property 14's second, much tighter bound, and the one that can actually fail.
#: `H^-1` is a true matrix inverse, so a round trip is an identity up to float64
#: conditioning, not a fitted approximation -- asserting it at 1.0 px would pass
#: a homography five orders of magnitude worse than the engine's. Measured
#: maximum 8e-11 px, over a 70-pose sweep (both geometry modes, all four planes,
#: nine sample positions each) and over the property's own targeted runs, so 1e-6
#: leaves four decades of headroom while still failing on a matrix that has lost
#: its conditioning.
ROUND_TRIP_TOLERANCE_PX = 1e-6

#: Property 15: relative difference between two equal-millimetre segments'
#: lengths, measured in the fixture's own metric units, at different locations on
#: one plane.
#:
#: The residual is not measurement noise. The recovered u and v axes are the
#: recovered *vanishing* directions, which are orthogonal only to within
#: `orthogonality_tolerance`, so the recovered plane is very slightly tilted
#: against the true one and the composite recovered-to-truth map is a genuine
#: projective transform rather than an affine one. A projective map does not
#: preserve length ratios, so a residual that grows with distance across the
#: plane is expected and bounded rather than absent.
#:
#: Measured over 1950 uniform draws from this property's own pose strategy:
#: median 0.05%, 99th percentile 1.1%, maximum 4.6% -- the tail belonging to the
#: floor, whose visible extent runs the full 6 m of the room and so reaches
#: closest to the plane's vanishing line. The property also `target`s this
#: quantity, which means Hypothesis actively searches for the worst pose rather
#: than sampling one; across twelve targeted runs of 75 examples the worst any run
#: reached was 6.0%, with most runs landing between 1% and 5%.
#:
#: The bound is set at 20%, which is three times that measured worst case and
#: still an order of magnitude clear of the regression it exists to catch. Zeroing
#: a recovered homography's projective row -- the exact "foreshortening applied on
#: top instead of falling out" mistake Requirement 5.7 rules out -- moves this
#: figure to 41% on the floor and about 70% on each side wall.
METRIC_SCALE_TOLERANCE = 0.20

#: The absolute-scale bound the anchored fixture test uses. Requirement 5.5 fixes
#: scale by convention -- `assumed_camera_height_mm` -- so on the documented pose,
#: where the fixture's camera really is at that height, a recovered millimetre has
#: to be a true millimetre. Measured maximum 0.23% over the fixture's four planes.
FIXTURE_METRIC_SCALE_TOLERANCE = 0.01

#: Property 16 is an inequality between two areas, so the only slack it needs is
#: float64 round-off on the shoelace sums. Measured zero violations at this slack
#: over 11880 ordered pairs.
AREA_MONOTONICITY_SLACK = 1e-9

#: Probe geometry for Properties 15 and 16, in millimetres and as a fraction of
#: the plane extent. 600 mm is the fixture's tile pitch, so the probes are the
#: size of the thing the Compositor will actually lay down; the fraction caps
#: them on a plane too small to hold one.
PROBE_SEGMENT_MM = 600.0
PROBE_FOOTPRINT_FRACTION = 0.15


# --------------------------------------------------------------------------- #
# Homography helpers
# --------------------------------------------------------------------------- #


def map_through(homography: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map `(N,2)` points through a homography.

    Returns:
        `(mapped, w)` with the homogeneous divisors alongside the mapped points,
        because a caller has to know whether a sample landed on the plane's
        vanishing line -- where the mapped position is meaningless -- rather than
        silently comparing an infinity.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((pts, np.ones(len(pts), dtype=np.float64)))
    projected = homogeneous @ np.asarray(homography, dtype=np.float64).T
    w = projected[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        mapped = projected[:, :2] / w[:, None]
    return mapped, w


def on_the_visible_side(w: np.ndarray) -> bool:
    """Whether every sample kept a positive, usable homogeneous divisor.

    Every homography this module returns is signed so `w` is positive over the
    plane's visible interior, so a non-positive `w` means the sample is behind the
    camera or on the vanishing line -- not that the mapping is wrong.
    """
    return bool(np.all(np.isfinite(w)) and np.all(w > 1e-12))


def polygon_area_px(polygon: np.ndarray) -> float:
    """Absolute shoelace area of a closed image-space polygon, in square pixels."""
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))


def interior_uv(
    extent: Sequence[float], fractions: Sequence[float], delta: Sequence[float] = (0.0, 0.0)
) -> np.ndarray | None:
    """A metric point placed by fraction inside an extent, allowing for an offset.

    `delta` is the vector a probe will be extended by, so the point is drawn from
    the sub-box that keeps *both* ends inside the extent. That matters more than
    it looks: outside the visible extent the plane's parametrisation is still
    defined but the surface is not, and a probe that leaves it is measuring the
    fixture's extrapolation rather than the engine.

    Returns:
        The `(2,)` metric point, or `None` when the extent is too small to hold
        the offset at all.
    """
    u0, v0, u1, v1 = (float(value) for value in extent)
    du, dv = float(delta[0]), float(delta[1])
    span_u, span_v = (u1 - u0) - abs(du), (v1 - v0) - abs(dv)
    if span_u <= 0.0 or span_v <= 0.0:
        return None
    return np.array(
        [
            u0 + max(0.0, -du) + float(fractions[0]) * span_u,
            v0 + max(0.0, -dv) + float(fractions[1]) * span_v,
        ],
        dtype=np.float64,
    )


def probe_size_mm(extent: Sequence[float]) -> float:
    """Side of the metric probe a plane of this extent can hold."""
    u0, v0, u1, v1 = (float(value) for value in extent)
    return min(PROBE_SEGMENT_MM, PROBE_FOOTPRINT_FRACTION * min(u1 - u0, v1 - v0))


def recovered_plane_frames(
    room: SyntheticRoom, calibration: Calibration
) -> dict[str, PlaneFrame]:
    """Rebuild the metric frames behind a calibration's homographies.

    `Calibration` reports matrices, not frames, but Property 16 is stated over
    *distance from the camera*, which only the frame carries: `origin_cam` and the
    two unit axes put a metric `(u, v)` at an actual camera-space millimetre
    position. Rebuilding from the same vanishing points, horizon, and contour
    reproduces exactly the frame `calibrate` used, which the property asserts by
    comparing the reassembled matrix against the reported one.

    Frames only exist in `vanishing_points` mode -- a fallback plane has a
    four-point homography and no camera-space frame at all -- so this returns an
    empty mapping there.
    """
    contours = room_contours(room)
    frames: dict[str, PlaneFrame] = {}
    for name in calibration.homographies:
        frame = plane_frame(
            name,
            calibration.vanishing_points,
            calibration.horizon,
            contours[name],
            room.shape,
            get_settings(),
        )
        if frame is None or homography_from_frame(frame) is None:
            continue
        frames[name] = frame
    return frames


def camera_position_mm(frame: PlaneFrame, uv: Sequence[float]) -> np.ndarray:
    """Camera-space millimetre position of a metric plane point.

    This is the frame's defining identity, `O + u * u_hat + v * v_hat`, and it is
    what lets Property 16 order two footprints by distance from the camera without
    consulting the fixture's camera at all.
    """
    return (
        frame.origin_cam + float(uv[0]) * frame.u_dir_cam + float(uv[1]) * frame.v_dir_cam
    )


def truth_length_mm(
    room: SyntheticRoom,
    name: str,
    homography: np.ndarray,
    start_uv: np.ndarray,
    delta: np.ndarray,
) -> float | None:
    """Length of a recovered-frame segment, measured in the fixture's own millimetres.

    The segment is stated in the *recovered* frame, projected to image space
    through the recovered homography, then read back through the *analytic*
    plane's inverse. So the number returned is how long the engine's segment
    really is, in ground truth units -- which is the only way to compare a
    recovered scale with truth without being confounded by the translation
    between the two metric origins.

    `numpy` inverts the truth matrix here rather than
    :func:`~backend.core.geometry.invert_homography`, so the reference side of the
    comparison owes nothing to the module under test.

    Returns:
        The length in millimetres, or `None` when either end failed to map --
        which for a probe drawn inside the visible extent means the plane is
        degenerate, not that the probe was badly placed.
    """
    truth = room.truth_homographies.get(name)
    if truth is None:
        return None
    ends = np.vstack((np.asarray(start_uv, dtype=np.float64), start_uv + delta))
    image, w = map_through(homography, ends)
    if not on_the_visible_side(w) or not np.isfinite(image).all():
        return None
    metric, truth_w = map_through(np.linalg.inv(truth), image)
    if not np.all(np.isfinite(truth_w)) or not np.isfinite(metric).all():
        return None
    return float(np.linalg.norm(metric[1] - metric[0]))


#: Fractional positions inside a plane extent, and a probe bearing. Drawing the
#: bearing rather than fixing it is what makes Property 15 cover the axis-skew
#: direction as well as the two axes themselves.
_fraction = st.floats(min_value=0.0, max_value=1.0)
_bearing = st.floats(min_value=0.0, max_value=2.0 * math.pi)

#: `_ROOM_PROPERTY_SETTINGS` with the example count the spec's 100-example floor
#: asks for. Same rooms, same cost per example -- about 65 ms of render plus
#: calibration -- so each of the three properties below runs in roughly seven
#: seconds. The pose strategy is shared with Properties 11 to 13, deliberately:
#: a homography is only as good as the vanishing points it was assembled from, so
#: both halves of the engine are scored over one pose space.
_HOMOGRAPHY_PROPERTY_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Property 14 -- plane homography round-trip reprojection error is bounded
# (Requirements 5.4, 5.6, 13.4)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 14: Plane homography round-trip
# reprojection error is bounded
@given(
    focal_ratio=_focal_ratio,
    yaw_magnitude=_yaw_magnitude,
    yaw_sign=_yaw_sign,
    pitch_deg=_pitch_deg,
    walls=_walls,
    seed=_seed,
    fractions=st.tuples(_fraction, _fraction),
)
@_HOMOGRAPHY_PROPERTY_SETTINGS
def test_property_14_round_trip_reprojection_error_is_bounded(
    randomized_room: Callable[..., SyntheticRoom],
    focal_ratio: float,
    yaw_magnitude: float,
    yaw_sign: float,
    pitch_deg: float,
    walls: tuple[str, ...],
    seed: int,
    fractions: tuple[float, float],
) -> None:
    """Property 14: forward then inverse returns the metric point it started from.

    Asserted three ways over every plane the calibration reported, at both the
    grid level Requirement 5.6 states and the single-point level Property 14 does:

    * `reprojection_rmse` over the plane extent is inside
      `REPROJECTION_TOLERANCE_PX`, and equals what `calibrate` reported -- so the
      figure the client is handed is the figure this property measured;
    * a drawn interior point and all four extent corners survive
      metric-to-image-to-metric-to-image within `ROUND_TRIP_TOLERANCE_PX`;
    * the recovered metric point matches the original in millimetres too,
      relative to the extent it was drawn from.

    No geometry mode is excluded. A `planar_fallback` plane has a four-point
    homography rather than a metric frame, and Requirement 5.6's bound applies to
    it just the same; both modes are reported through `event` so the split is
    visible under `--hypothesis-show-statistics`.

    **Validates: Requirements 5.4, 5.6, 13.4**
    """
    room = _draw_room(
        randomized_room, focal_ratio, yaw_magnitude, yaw_sign, pitch_deg, walls, seed
    )
    calibration = calibrated(room)
    event(f"geometry_mode={calibration.geometry_mode}")
    assert calibration.homographies, "no plane received a homography from either path"

    worst_rmse = 0.0
    worst_pixel_residual = 0.0
    for name, homography in calibration.homographies.items():
        extent = calibration.plane_extents_mm[name]
        u0, v0, u1, v1 = extent
        inverse = calibration.homography_inverses[name]

        rmse = reprojection_rmse(homography, extent)
        worst_rmse = max(worst_rmse, rmse)
        assert rmse <= REPROJECTION_TOLERANCE_PX, (
            f"{name} round-trip RMSE is {rmse:.3e} px over extent {extent!r}, "
            f"allowed {REPROJECTION_TOLERANCE_PX} px"
        )
        assert calibration.reprojection_rmse_px[name] == pytest.approx(rmse, rel=1e-9), (
            f"{name} reported RMSE {calibration.reprojection_rmse_px[name]:.3e} px "
            f"disagrees with the measured {rmse:.3e} px"
        )

        drawn = interior_uv(extent, fractions)
        assert drawn is not None, f"{name} extent {extent!r} is degenerate"
        samples = np.vstack((drawn, [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]))
        forward, forward_w = map_through(homography, samples)
        assert on_the_visible_side(forward_w), (
            f"{name} maps a point of its own extent to w={forward_w!r}, so the "
            "extent reaches the plane's vanishing line"
        )
        back, back_w = map_through(inverse, forward)
        assert on_the_visible_side(back_w)
        again, _ = map_through(homography, back)

        pixel_residual = float(np.max(np.linalg.norm(again - forward, axis=1)))
        worst_pixel_residual = max(worst_pixel_residual, pixel_residual)
        assert pixel_residual <= ROUND_TRIP_TOLERANCE_PX, (
            f"{name} round trip moved a sample {pixel_residual:.3e} px, over the "
            f"{ROUND_TRIP_TOLERANCE_PX:.0e} px an exact inverse allows"
        )

        # The same claim in millimetres, scaled by the extent so it means the same
        # thing on a 1 m wall and a 6 m floor.
        span = max(u1 - u0, v1 - v0)
        metric_residual = float(np.max(np.linalg.norm(back - samples, axis=1))) / span
        assert metric_residual <= 1e-9, (
            f"{name} recovered its own metric samples {metric_residual:.3e} of an "
            f"extent span out"
        )

    # Reported once per example rather than per plane: `target` takes one
    # observation per label, and the worst plane is the one worth shrinking toward.
    target(worst_rmse, label="reprojection rmse px")
    target(worst_pixel_residual, label="worst point round trip px")


def test_property_14_holds_on_the_documented_fixture_pose(
    synthetic_room: SyntheticRoom,
) -> None:
    """Property 14 at the fixed 1600x1200 fixture, as a non-random anchor.

    The drawn property covers a hard pose range at 640x480; this pins the same
    claim at the documented pose, where every plane is large and well conditioned.

    **Validates: Requirements 5.4, 5.6, 13.4**
    """
    calibration = calibrated(synthetic_room)
    assert calibration.geometry_mode == "vanishing_points"
    assert set(calibration.homographies) == set(PLANE_NAMES)

    for name, homography in calibration.homographies.items():
        extent = calibration.plane_extents_mm[name]
        rmse = reprojection_rmse(homography, extent)
        assert rmse <= ROUND_TRIP_TOLERANCE_PX, (
            f"{name} RMSE {rmse:.3e} px on the documented pose"
        )


def test_homography_from_frame_caches_its_matrix_and_inverse(
    synthetic_room: SyntheticRoom,
) -> None:
    """A frame's homography and inverse are assembled once and reused.

    Requirement 9.3's render budget rests on this: the Compositor asks each frame
    for its matrix per plane per request, and an SVD-free assembly plus an
    inversion per call would be paid every time. Identity, not equality, is what
    shows the cache is real.
    """
    calibration = calibrated(synthetic_room)
    frames = recovered_plane_frames(synthetic_room, calibration)
    assert frames, "the documented pose should yield a metric frame per plane"

    for name, frame in frames.items():
        first = homography_from_frame(frame)
        assert first is not None
        assert homography_from_frame(frame) is first, f"{name} reassembled its homography"
        assert frame.homography is first
        assert frame.homography_inv is not None
        # The cached matrix is the one `calibrate` reported for the same plane.
        assert first == pytest.approx(calibration.homographies[name], abs=1e-12)


# --------------------------------------------------------------------------- #
# Property 15 -- metric scale is consistent across each plane
# (Requirement 5.5)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 15: Metric scale is consistent
# across each plane
@given(
    focal_ratio=_focal_ratio,
    yaw_magnitude=_yaw_magnitude,
    yaw_sign=_yaw_sign,
    pitch_deg=_pitch_deg,
    walls=_walls,
    seed=_seed,
    first=st.tuples(_fraction, _fraction),
    second=st.tuples(_fraction, _fraction),
    bearing=_bearing,
)
@_HOMOGRAPHY_PROPERTY_SETTINGS
def test_property_15_metric_scale_is_consistent_across_each_plane(
    randomized_room: Callable[..., SyntheticRoom],
    focal_ratio: float,
    yaw_magnitude: float,
    yaw_sign: float,
    pitch_deg: float,
    walls: tuple[str, ...],
    seed: int,
    first: tuple[float, float],
    second: tuple[float, float],
    bearing: float,
) -> None:
    """Property 15: equal-millimetre segments measure equal wherever they sit.

    Two segments of the same drawn length and bearing are placed at two drawn
    locations on one plane, and each is measured in the *fixture's* metric units
    by mapping it forward through the recovered homography and back through the
    analytic one. Equal recovered millimetres have to mean equal real
    millimetres, or the plane's scale varies across it and a tile grid laid in
    that frame would drift.

    Both segments carry the same bearing, and that is the claim rather than a
    convenience. The recovered u and v axes are the recovered vanishing
    directions, which are mutually orthogonal only to within
    `orthogonality_tolerance`, so the frame is very slightly skewed and a
    millimetre along one bearing is not identical to a millimetre along another.
    That skew is a property of the *frame*, identical everywhere on the plane;
    what Requirement 5.5 promises, and what this measures, is that nothing
    changes with *location*. The bearing is drawn rather than fixed so the
    skewed directions are covered too.

    The bound is wide, and `METRIC_SCALE_TOLERANCE` documents why: the recovered
    plane is very slightly tilted against the true one, so the composite map is
    projective and the residual grows toward the plane's far end. It is still an
    order of magnitude tighter than the failure it is here to catch.

    Restricted to `vanishing_points` mode. A fallback homography maps a
    fabricated metric rectangle onto a bounding quad, so it is not a metric
    parametrisation of the plane at all and lengths in it legitimately vary with
    position -- Requirement 6.4 asks only that its *aspect ratio* survive, which
    is Property 17's subject in task 10.3.

    **Validates: Requirements 5.5**
    """
    room = _draw_room(
        randomized_room, focal_ratio, yaw_magnitude, yaw_sign, pitch_deg, walls, seed
    )
    calibration = calibrated(room)
    event(f"geometry_mode={calibration.geometry_mode}")
    assume(calibration.geometry_mode == "vanishing_points")

    checked = 0
    worst_relative = 0.0
    for name, homography in calibration.homographies.items():
        if name not in room.truth_homographies:
            continue
        extent = calibration.plane_extents_mm[name]
        length_mm = probe_size_mm(extent)
        if length_mm <= 1.0:
            continue
        delta = length_mm * np.array([math.cos(bearing), math.sin(bearing)], dtype=np.float64)

        starts = [interior_uv(extent, fractions, delta) for fractions in (first, second)]
        if any(start is None for start in starts):
            continue
        measured = [
            truth_length_mm(room, name, homography, start, delta)
            for start in starts
            if start is not None
        ]
        if any(value is None or value <= 0.0 for value in measured):
            continue
        near_mm, far_mm = float(measured[0]), float(measured[1])  # type: ignore[arg-type]

        relative = abs(near_mm - far_mm) / max(near_mm, far_mm)
        worst_relative = max(worst_relative, relative)
        assert relative <= METRIC_SCALE_TOLERANCE, (
            f"{name}: a {length_mm:.0f} mm segment at {starts[0]!r} measures "
            f"{near_mm:.1f} mm and the same segment at {starts[1]!r} measures "
            f"{far_mm:.1f} mm in truth units, a {100.0 * relative:.2f} percent "
            f"spread over extent {extent!r}"
        )
        checked += 1

    assert checked > 0, (
        "no plane offered a comparable pair of segments; "
        f"planes were {tuple(calibration.homographies)!r}"
    )
    target(worst_relative, label="metric scale relative spread")


def test_property_15_recovers_absolute_scale_on_the_documented_fixture(
    synthetic_room: SyntheticRoom,
) -> None:
    """Requirement 5.5's absolute half, anchored at the documented pose.

    Property 15 says equal segments measure equal, which a uniformly wrong scale
    would also satisfy. What makes the frame *metric* rather than merely
    self-consistent is `assumed_camera_height_mm`: the fixture's camera really is
    at that height, so on this pose a recovered millimetre has to be a true
    millimetre. Three independent readings of that:

    * a 600 mm recovered step measures 600 mm in truth units, on every plane;
    * every plane frame's metric origin sits on the floor, which is what makes a
      tile the same size on the floor and on the wall it meets;
    * the three walls independently agree on how far away the room's far end is,
      which they can only do if the scale they inherited through the floor-wall
      junction is the same one.

    Written against the fixture's camera extrinsics rather than against its room
    dimensions, so it states the geometry rather than restating the generator's
    default arguments.
    """
    calibration = calibrated(synthetic_room)
    assert calibration.geometry_mode == "vanishing_points"
    frames = recovered_plane_frames(synthetic_room, calibration)
    assert set(frames) == set(PLANE_NAMES)

    rotation = np.asarray(synthetic_room.camera.R, dtype=np.float64)
    translation = np.asarray(synthetic_room.camera.t, dtype=np.float64)

    def to_world(point_cam: np.ndarray) -> np.ndarray:
        """Camera millimetres back to the fixture's world frame."""
        return rotation.T @ (np.asarray(point_cam, dtype=np.float64) - translation)

    far_depths_mm: list[float] = []
    for name, frame in frames.items():
        homography = calibration.homographies[name]
        extent = frame.extent_mm
        for bearing in (0.0, 0.5 * math.pi, 0.25 * math.pi):
            delta = PROBE_SEGMENT_MM * np.array([math.cos(bearing), math.sin(bearing)])
            start = interior_uv(extent, (0.5, 0.5), delta)
            assert start is not None
            measured = truth_length_mm(synthetic_room, name, homography, start, delta)
            assert measured is not None
            error = abs(measured - PROBE_SEGMENT_MM) / PROBE_SEGMENT_MM
            assert error <= FIXTURE_METRIC_SCALE_TOLERANCE, (
                f"{name}: a {PROBE_SEGMENT_MM:.0f} mm step at bearing "
                f"{math.degrees(bearing):.0f} deg measures {measured:.1f} mm in "
                f"truth units, {100.0 * error:.2f} percent out"
            )

        # The floor is world Y = 0 in the fixture, and every plane's metric origin
        # is the corner of its visible extent nearest the floor.
        origin_world = to_world(frame.origin_cam)
        assert abs(float(origin_world[1])) <= 5.0, (
            f"{name} metric origin sits {origin_world[1]:.1f} mm off the floor plane"
        )

        if name in ("wall_left", "wall_right"):
            # u runs along depth on a side wall, so the extent's far end is the
            # room's far end, reached without either wall knowing about the other.
            u1 = frame.extent_mm[2]
            far_depths_mm.append(float(to_world(camera_position_mm(frame, (u1, 0.0)))[2]))
        elif name == "wall_back":
            far_depths_mm.append(float(to_world(frame.origin_cam)[2]))

    assert len(far_depths_mm) == 3, "the documented pose shows both side walls and the back"
    spread = (max(far_depths_mm) - min(far_depths_mm)) / max(far_depths_mm)
    assert spread <= FIXTURE_METRIC_SCALE_TOLERANCE, (
        f"the walls disagree about the room's far depth: {far_depths_mm!r}, a "
        f"{100.0 * spread:.2f} percent spread"
    )


# --------------------------------------------------------------------------- #
# Property 16 -- projected tile size decreases with depth
# (Requirement 5.7)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 16: Projected tile size decreases
# with depth
@given(
    focal_ratio=_focal_ratio,
    yaw_magnitude=_yaw_magnitude,
    yaw_sign=_yaw_sign,
    pitch_deg=_pitch_deg,
    walls=_walls,
    seed=_seed,
    first=st.tuples(_fraction, _fraction),
    second=st.tuples(_fraction, _fraction),
)
@_HOMOGRAPHY_PROPERTY_SETTINGS
def test_property_16_projected_footprint_area_decreases_with_depth(
    randomized_room: Callable[..., SyntheticRoom],
    focal_ratio: float,
    yaw_magnitude: float,
    yaw_sign: float,
    pitch_deg: float,
    walls: tuple[str, ...],
    seed: int,
    first: tuple[float, float],
    second: tuple[float, float],
) -> None:
    """Property 16: the farther of two equal footprints projects no larger.

    Two square metric footprints of identical side are placed at two drawn
    locations on one plane, ordered by their centres' distance from the camera --
    which the plane frame gives directly, since a metric `(u, v)` *is* a
    camera-space millimetre position in it -- and their projected image polygon
    areas are compared. This is what Requirement 5.7 asks for: foreshortening has
    to fall out of the homography, not be applied on top of it.

    One class of pair is excluded, and the reason is geometric rather than
    convenient. A fixed footprint at camera-space point `p` projects to an area
    proportional to `offset / (|p| * z^2)`, where `z` is depth along the optical
    axis and `offset = n . p` is constant over the plane. Distance from the camera
    pins `|p|` but not `z`: a footprint far out to the side is a long way away
    while sitting *shallow* in depth, and it legitimately projects larger than a
    nearer one straight ahead -- up to 3.3 times larger, measured. Those are
    lateral pairs, not the receding pairs "at increasing distance from the camera"
    describes, so a pair whose depth ordering contradicts its distance ordering is
    excluded and reported through `event`. Over an 11880-pair sweep the excluded
    set was *exactly* the set of pairs that would otherwise have failed: 1320
    excluded, 1320 raw violations, none left over.

    Frames only exist in `vanishing_points` mode, so that is where the property is
    asserted; the fallback's round trip is covered by Property 14 and its aspect
    ratio by Property 17.

    **Validates: Requirements 5.7**
    """
    room = _draw_room(
        randomized_room, focal_ratio, yaw_magnitude, yaw_sign, pitch_deg, walls, seed
    )
    calibration = calibrated(room)
    event(f"geometry_mode={calibration.geometry_mode}")
    frames = recovered_plane_frames(room, calibration)
    assume(bool(frames))

    compared = 0
    worst_ratio = 0.0
    for name, frame in frames.items():
        homography = calibration.homographies[name]
        extent = frame.extent_mm
        side_mm = probe_size_mm(extent)
        if side_mm <= 1.0:
            continue

        footprints = []
        for fractions in (first, second):
            corner = interior_uv(extent, fractions, (side_mm, side_mm))
            if corner is None:
                continue
            quad = corner + np.array(
                [[0.0, 0.0], [side_mm, 0.0], [side_mm, side_mm], [0.0, side_mm]]
            )
            image, w = map_through(homography, quad)
            if not on_the_visible_side(w) or not np.isfinite(image).all():
                continue
            centre = corner + 0.5 * side_mm
            position = camera_position_mm(frame, centre)
            footprints.append(
                (
                    float(np.linalg.norm(position)),
                    float(position[2]),
                    polygon_area_px(image),
                    centre,
                )
            )
        if len(footprints) < 2:
            continue

        near, far = sorted(footprints, key=lambda entry: entry[0])
        if far[1] < near[1]:
            # A lateral pair: farther in distance, nearer in depth. See the
            # docstring -- the requirement's claim is not stated over these.
            event(f"{name}: lateral pair excluded")
            continue

        allowed = near[2] * (1.0 + AREA_MONOTONICITY_SLACK)
        if near[2] > 0.0:
            worst_ratio = max(worst_ratio, far[2] / near[2])
        assert far[2] <= allowed, (
            f"{name}: the footprint at {far[3]!r}, {far[0]:.0f} mm from the camera "
            f"at depth {far[1]:.0f} mm, projects to {far[2]:.2f} px^2 while the one "
            f"at {near[3]!r}, {near[0]:.0f} mm away at depth {near[1]:.0f} mm, "
            f"projects to {near[2]:.2f} px^2"
        )
        compared += 1

    target(worst_ratio, label="far/near projected area ratio")
    assume(compared > 0)


def test_property_16_footprint_area_falls_monotonically_along_the_floor(
    synthetic_room: SyntheticRoom,
) -> None:
    """Property 16 as a receding sequence, not just a pair.

    The drawn property compares two footprints; this walks one across the whole
    visible floor from the near edge to the far edge and asserts the projected
    area falls at *every* step. A pairwise claim can be satisfied by a mapping
    that is monotone only on average, and the near-to-far ratio recorded here --
    an order of magnitude over a 6 m room -- is the evidence that Requirement
    5.7's foreshortening is present at full strength rather than merely
    signed correctly.

    **Validates: Requirements 5.7**
    """
    calibration = calibrated(synthetic_room)
    frames = recovered_plane_frames(synthetic_room, calibration)
    frame = frames["floor"]
    homography = calibration.homographies["floor"]
    u0, v0, u1, v1 = frame.extent_mm
    side_mm = probe_size_mm(frame.extent_mm)

    areas: list[float] = []
    for step in range(12):
        v = v0 + (v1 - v0 - side_mm) * step / 11.0
        corner = np.array([0.5 * (u0 + u1 - side_mm), v])
        quad = corner + np.array(
            [[0.0, 0.0], [side_mm, 0.0], [side_mm, side_mm], [0.0, side_mm]]
        )
        image, w = map_through(homography, quad)
        assert on_the_visible_side(w)
        areas.append(polygon_area_px(image))

    for nearer, farther in zip(areas, areas[1:]):
        assert farther < nearer, (
            f"projected floor footprint areas are not strictly decreasing: {areas!r}"
        )
    assert areas[0] / areas[-1] > 4.0, (
        f"a {side_mm:.0f} mm footprint shrinks by only {areas[0] / areas[-1]:.2f}x "
        f"across the visible floor, which is too little foreshortening to be real"
    )


# --------------------------------------------------------------------------- #
# Geometry mode and the planar fallback (Requirements 6.1, 6.3)
# --------------------------------------------------------------------------- #


def test_geometry_mode_is_vanishing_points_when_an_orthogonal_triple_survives(
    randomized_room: Callable[..., SyntheticRoom],
) -> None:
    """Requirement 6.3: a recoverable triple is reported as `vanishing_points`.

    The mode describes what the *camera* recovery achieved, so the assertion is
    tied to the three labels being set rather than to any per-plane outcome.
    """
    room = _draw_room(randomized_room, 0.875, 8.0, 1.0, -12.0, ("left", "right", "back"), 0)
    calibration = calibrated(room)

    assert calibration.geometry_mode == "vanishing_points"
    assert all(calibration.vanishing_points[label] is not None for label in VP_LABELS)
    assert set(calibration.homographies) == set(PLANE_NAMES)


@pytest.mark.parametrize("yaw_deg", [0.0, 0.4])
def test_geometry_mode_is_planar_fallback_when_a_vanishing_point_is_missing(
    randomized_room: Callable[..., SyntheticRoom], yaw_deg: float
) -> None:
    """Requirements 6.1, 6.3: a frontal pose falls back and still serves geometry.

    At a near-zero yaw the lateral direction turns parallel to the image plane and
    its vanishing point runs to infinity, so no orthogonal triple can be accepted
    and the mode has to say so. What matters as much as the label is that the
    fallback is not a failure state: every plane still receives a homography whose
    round trip is inside Requirement 5.6's bound, and its metric extent still sits
    inside the bounds the fallback documents rather than being unbounded.
    """
    walls = ("left", "right", "back")
    room = _draw_room(randomized_room, 0.875, yaw_deg, 1.0, -12.0, walls, 0)
    calibration = calibrated(room)

    assert calibration.geometry_mode == "planar_fallback"
    assert sum(calibration.vanishing_points[label] is not None for label in VP_LABELS) < 3
    assert calibration.vanishing_points["VPx"] is None, (
        "a frontal pose is meant to lose the lateral direction, got "
        f"{calibration.vanishing_points!r}"
    )
    assert recovered_plane_frames(room, calibration) == {}, (
        "no metric frame can exist without a complete triple"
    )

    assert calibration.homographies, "the fallback must still hand back geometry"
    for name, homography in calibration.homographies.items():
        extent = calibration.plane_extents_mm[name]
        rmse = reprojection_rmse(homography, extent)
        assert rmse <= REPROJECTION_TOLERANCE_PX, f"{name} fallback RMSE {rmse:.3e} px"
        u_span, v_span = extent[2] - extent[0], extent[3] - extent[1]
        assert 200.0 <= u_span <= 30000.0, f"{name} fallback width {u_span:.1f} mm"
        assert 200.0 <= v_span <= 30000.0, f"{name} fallback depth {v_span:.1f} mm"
        assert max(u_span, v_span) / min(u_span, v_span) <= 20.0 + 1e-9, (
            f"{name} fallback extent {extent!r} is more elongated than the "
            "documented 20:1 cap"
        )


@pytest.mark.parametrize(
    "quad, label",
    [
        (
            np.array([[300.0, 300.0], [304.0, 300.0], [304.0, 304.0], [300.0, 304.0]]),
            "a four-pixel quad, which implies a sub-millimetre plane",
        ),
        (
            np.array([[100.0, 400.0], [600.0, 400.0], [600.0, 402.0], [100.0, 402.0]]),
            "a two-pixel-tall sliver, seen almost edge-on",
        ),
        (
            np.array([[20.0, 100.0], [620.0, 90.0], [620.0, 470.0], [20.0, 460.0]]),
            "a quad straddling the horizon, which a wall region does",
        ),
    ],
)
def test_metric_quad_extents_stay_inside_the_documented_fallback_bounds(
    quad: np.ndarray, label: str
) -> None:
    """Requirement 6.4: the fallback's metric rectangle is bounded, both ways.

    The fallback derives millimetres from pixel distances to the horizon, and
    those distances go to zero on the horizon itself, so an unguarded division
    produces a plane either microscopic or kilometres across -- and an aspect
    ratio to match, which would smear one tile over a whole wall. The degenerate
    quads here drive each of those routes.
    """
    horizon: Line = (0.0, 1.0, -200.0)
    metric = metric_quad_from_image_quad(quad, horizon, (480, 640))
    assert metric is not None, f"{label} produced no metric rectangle at all"

    width_mm = float(metric[:, 0].max() - metric[:, 0].min())
    depth_mm = float(metric[:, 1].max() - metric[:, 1].min())
    for span, axis in ((width_mm, "width"), (depth_mm, "depth")):
        assert 200.0 <= span <= 30000.0, f"{label}: {axis} {span:.1f} mm is out of bounds"
    assert max(width_mm, depth_mm) / min(width_mm, depth_mm) <= 20.0 + 1e-9, (
        f"{label}: {width_mm:.1f} x {depth_mm:.1f} mm exceeds the 20:1 aspect cap"
    )

    # A rectangle is only useful if it maps back through a homography, which is
    # what the fallback immediately does with it.
    homography = geometry.homography_from_quad(quad, metric)
    assert homography is not None, f"{label}: the metric rectangle is not usable"
    assert reprojection_rmse(homography, (0.0, 0.0, width_mm, depth_mm)) <= (
        REPROJECTION_TOLERANCE_PX
    )
