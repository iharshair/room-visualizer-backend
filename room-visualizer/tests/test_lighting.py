"""Tests for `backend.core.lighting` (Requirements 7.1, 7.2, 7.3, 7.4, 12.4).

The Lighting_Engine's contract is a three-way split of one photograph, so the
module is organised as one section per stage plus one for the per-plane medians
the Compositor branches on:

* **Property 18** pins `to_lab_l` to the reference CIELAB conversion. Because the
  implementation and the reference are the same OpenCV call, a unit test derives
  `L*` independently from the CIE D65 formula first, so the property cannot pass
  by comparing a wrong answer against itself.
* **Property 19** holds `low_frequency` to being a genuine low-pass. Sizes are
  drawn down to a single pixel on purpose: the kernels clamp to what the image
  can be reflect-padded against, and on the smallest inputs every filter is
  skipped and the map comes back equal to its source -- so the bound has to be
  satisfied with equality, not strict inequality.
* **Property 20** pins the reconstruction relation `shading + (detail - 128)`.
  The strong half is exact equality wherever the signed residual fitted in
  `uint8`; the residual clipping that happens on high-frequency inputs is what
  the mean-error and detail-mean tolerances cover.
* **Per-plane medians** are checked against a hand-built two-band image where a
  global median would give a visibly different answer, so "per plane, not
  globally" is actually falsifiable.

Both maps are asserted `uint8` throughout, which is the cached-artifact bound of
Requirement 12.4.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings as hypothesis_settings, strategies as st

from backend.config import Settings
from backend.core.lighting import (
    NEUTRAL_DETAIL,
    LightingMaps,
    decompose,
    high_frequency,
    low_frequency,
    to_lab_l,
)

# --------------------------------------------------------------------------- #
# Documented tolerances
# --------------------------------------------------------------------------- #
#
# Every number here was measured against the implementation rather than guessed,
# and each one is attached to a specific mechanism so a future regression that
# widens it is visible as a behaviour change and not as a rounding nuisance.

#: Slack on the Property 19 gradient bound. The comparison is between two
#: float32 Sobel reductions, so only float accumulation noise is forgiven --
#: nowhere near enough to hide a filter that actually sharpened its input.
_GRADIENT_SLACK_ABS = 1e-4
_GRADIENT_SLACK_REL = 1e-6

#: Property 20, mean absolute reconstruction error over the whole frame, in
#: 8-bit `L*` units. Zero on smooth and photograph-like inputs; strictly
#: positive only where the signed residual left [-128, 127] and clipped, which a
#: full-range checkerboard does at a few percent of its pixels.
_RECONSTRUCTION_MEAN_TOLERANCE = 1.0

#: Property 20, deviation of the detail map's mean from `NEUTRAL_DETAIL`. The
#: same clipping biases the mean slightly low: about 0.02 on the synthetic room,
#: up to ~1.2 on full-range noise, where nearly every pixel is a large residual.
_DETAIL_MEAN_TOLERANCE = 2.0

#: Agreement between OpenCV's 8-bit `L*` and the CIE formula computed in float64.
#: OpenCV evaluates the sRGB transfer function through an 8-bit lookup table, so
#: a little over one 8-bit step of disagreement is expected and correct.
_ANALYTIC_L_TOLERANCE = 2.0

#: Shortest edge at which the detail-mean bound is asserted. Below this a
#: handful of clipped pixels is a large share of the frame -- a 3x3 patch of
#: noise can pull the mean 7 units off neutral -- and nothing that small is a
#: photograph.
_MIN_PHOTOGRAPH_EDGE = 32


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #


def mean_gradient_magnitude(plane: np.ndarray) -> float:
    """Mean Sobel gradient magnitude of a single-channel map.

    This is the "how much high frequency is in here" measure Property 19 is
    stated over. Sobel rather than a plain difference so the measure is
    isotropic: a map smoothed only horizontally must not read as smooth.
    """
    as_float = plane.astype(np.float32)
    dx = cv2.Sobel(as_float, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(as_float, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(np.sqrt(dx * dx + dy * dy)))


def analytic_l_star_8bit(image_bgr: np.ndarray) -> np.ndarray:
    """CIE `L*` from sRGB, in OpenCV's 0-255 scaling, computed from scratch.

    Deliberately independent of OpenCV: the sRGB transfer function, the D65
    luminance row, and the `L*` companding are all written out in float64. This
    is what makes the Property 18 comparison meaningful -- without it the
    property would only assert that `cv2.cvtColor` equals `cv2.cvtColor`.
    """
    rgb = image_bgr[..., ::-1].astype(np.float64) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    luminance = (
        0.2126729 * linear[..., 0] + 0.7151522 * linear[..., 1] + 0.0721750 * linear[..., 2]
    )
    # CIE 15:2004 companding, using the exact 216/24389 and 24389/27 constants
    # rather than the rounded 0.008856 / 903.3 so the two branches meet.
    companded = np.where(
        luminance > 216.0 / 24389.0,
        np.cbrt(luminance),
        (24389.0 / 27.0 * luminance + 16.0) / 116.0,
    )
    return (116.0 * companded - 16.0) * 255.0 / 100.0


def reference_lab_l(image_bgr: np.ndarray) -> np.ndarray:
    """The reference CIELAB conversion's lightness channel."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]


# --------------------------------------------------------------------------- #
# Photograph builders
# --------------------------------------------------------------------------- #
#
# Four synthetic inputs spanning the frequency content the engine has to cope
# with, plus the analytic room fixture. Each is deterministic given its
# arguments, so a Hypothesis counterexample reproduces from the reported values
# alone.


def _noise_photo(height: int, width: int, seed: int) -> np.ndarray:
    """Full-range uncorrelated noise: all high frequency, no illumination."""
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _gradient_photo(height: int, width: int, seed: int) -> np.ndarray:
    """A smooth diagonal ramp: all low frequency, so the residual never clips."""
    rng = np.random.default_rng(seed)
    ramp_x = np.linspace(0.0, 255.0, width, dtype=np.float32)[np.newaxis, :]
    ramp_y = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, np.newaxis]
    value = 0.5 * (ramp_x + ramp_y)
    # Per-channel gain so the ramp is chromatic and the L* conversion has real
    # colour to fold down rather than a grey ramp it can pass through.
    gains = rng.uniform(0.7, 1.0, 3).astype(np.float32)
    return np.clip(value[:, :, np.newaxis] * gains, 0, 255).astype(np.uint8)


def _checkerboard_photo(height: int, width: int, seed: int, cell: int = 3) -> np.ndarray:
    """Black-and-white cells: the worst case for residual clipping."""
    ys, xs = np.mgrid[0:height, 0:width]
    dark = ((ys // cell) + (xs // cell)) % 2 == 1
    return np.repeat(np.where(dark, 0, 255).astype(np.uint8)[:, :, np.newaxis], 3, axis=2)


def _shadowed_photo(height: int, width: int, seed: int) -> np.ndarray:
    """A lit surface with a soft cast shadow and fine grain.

    The closest of the four to what the engine actually sees: a low-frequency
    illumination envelope the shading map should keep, plus film-grain-scale
    texture it should hand to the detail map.
    """
    rng = np.random.default_rng(seed)
    envelope = np.linspace(90.0, 210.0, height, dtype=np.float32)[:, np.newaxis]
    surface = np.repeat(envelope, width, axis=1)

    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = np.sqrt(
        ((ys - 0.6 * height) / (0.35 * height + 1.0)) ** 2
        + ((xs - 0.35 * width) / (0.30 * width + 1.0)) ** 2
    )
    surface *= 0.45 + 0.55 * np.clip(radius, 0.0, 1.0)
    surface += rng.normal(0.0, 14.0, surface.shape).astype(np.float32)

    tint = rng.uniform(0.85, 1.0, 3).astype(np.float32)
    return np.clip(surface[:, :, np.newaxis] * tint, 0, 255).astype(np.uint8)


#: Room images come from the shared analytic fixture rather than a builder, so
#: the sentinel is handled by :func:`_photograph` instead of living in here.
_SYNTHETIC_BUILDERS: dict[str, Callable[[int, int, int], np.ndarray]] = {
    "noise": _noise_photo,
    "gradient": _gradient_photo,
    "checkerboard": _checkerboard_photo,
    "shadowed": _shadowed_photo,
}

#: Every photograph kind the properties are driven over, `"room"` included.
PHOTO_KINDS: tuple[str, ...] = (*sorted(_SYNTHETIC_BUILDERS), "room")


def _photograph(
    kind: str,
    height: int,
    width: int,
    seed: int,
    room_factory: Callable[..., object],
    *,
    focal_px: float,
    yaw_deg: float,
    pitch_deg: float,
) -> np.ndarray:
    """Build one photograph of the requested kind.

    The camera arguments are consumed only by `"room"`; drawing them
    unconditionally keeps the strategy flat and lets Hypothesis shrink a room
    counterexample down to a simple pose.
    """
    if kind == "room":
        room = room_factory(focal_px=focal_px, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
        return room.image  # type: ignore[attr-defined]
    return _SYNTHETIC_BUILDERS[kind](height, width, seed)


# Shared draws. Sizes bottom out at a single pixel so the clamped-kernel path in
# `low_frequency` is exercised, and the camera ranges match the field-of-view
# and pose regime the geometry fixture documents.
_kind = st.sampled_from(PHOTO_KINDS)
_seed = st.integers(min_value=0, max_value=2**16)
_focal_px = st.floats(min_value=400.0, max_value=900.0)
_yaw_deg = st.floats(min_value=-25.0, max_value=25.0)
_pitch_deg = st.floats(min_value=-30.0, max_value=-3.0)

_PROPERTY_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    # A room example costs ~20 ms to generate and the large-sigma Gaussian
    # another ~40 ms, so individual examples are legitimately slow.
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Property 18 -- L* extraction (Requirement 7.1)
# --------------------------------------------------------------------------- #


def test_analytic_l_star_agrees_with_the_reference_conversion_on_known_colours():
    """Guard for Property 18: the reference conversion is itself correct.

    `to_lab_l` is a thin wrapper over `cv2.cvtColor`, so comparing it against
    that same call proves only self-consistency. Anchoring the reference to `L*`
    values derived independently from the CIE formula -- pure black at 0, pure
    white at the top of the 8-bit range, and four coloured corners in between --
    is what gives the property something to be wrong about.
    """
    swatches = np.array(
        [[(0, 0, 0), (255, 255, 255), (128, 128, 128), (255, 0, 0), (0, 255, 0), (0, 0, 255)]],
        dtype=np.uint8,
    )

    measured = to_lab_l(swatches).astype(np.float64)
    expected = analytic_l_star_8bit(swatches)

    assert measured[0, 0] == 0, "pure black must sit at L* = 0"
    assert measured[0, 1] == 255, "pure white must sit at the top of the 8-bit L* range"
    np.testing.assert_allclose(measured, expected, atol=_ANALYTIC_L_TOLERANCE)


# Feature: ai-room-tile-visualizer, Property 18: L* extraction matches the
# reference CIELAB conversion
@_PROPERTY_SETTINGS
@given(
    kind=_kind,
    height=st.integers(min_value=1, max_value=96),
    width=st.integers(min_value=1, max_value=96),
    seed=_seed,
    focal_px=_focal_px,
    yaw_deg=_yaw_deg,
    pitch_deg=_pitch_deg,
)
def test_property_18_lab_l_equals_the_reference_lightness_channel(
    randomized_room, kind, height, width, seed, focal_px, yaw_deg, pitch_deg
):
    """For any BGR image, the isolated `L*` channel equals the lightness channel
    of the reference CIELAB conversion of that image and every value lies within
    the valid 8-bit range.

    **Validates: Requirements 7.1**
    """
    image = _photograph(
        kind, height, width, seed, randomized_room,
        focal_px=focal_px, yaw_deg=yaw_deg, pitch_deg=pitch_deg,
    )

    l_channel = to_lab_l(image)

    assert np.array_equal(l_channel, reference_lab_l(image))
    assert l_channel.shape == image.shape[:2]
    # Requirement 12.4: the cached map is 8 bits per channel, so the whole
    # 0-255 range is representable and nothing can sit outside it.
    assert l_channel.dtype == np.uint8
    assert int(l_channel.min()) >= 0
    assert int(l_channel.max()) <= 255


def test_lab_l_returns_a_buffer_the_caller_owns():
    """The result must not alias the interleaved Lab temporary.

    A strided view would keep the three-channel conversion alive for as long as
    the Scene_State holds the map, tripling the cached cost Requirement 12.4
    budgets for.
    """
    image = _shadowed_photo(48, 64, seed=3)

    l_channel = to_lab_l(image)
    before = l_channel.copy()
    image[:] = 0

    assert np.array_equal(l_channel, before)


# --------------------------------------------------------------------------- #
# Property 19 -- low-frequency shading map (Requirement 7.2)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 19: The shading map is
# lower-frequency than its source
@_PROPERTY_SETTINGS
@given(
    kind=_kind,
    height=st.integers(min_value=1, max_value=96),
    width=st.integers(min_value=1, max_value=96),
    seed=_seed,
    use_bilateral=st.booleans(),
    focal_px=_focal_px,
    yaw_deg=_yaw_deg,
    pitch_deg=_pitch_deg,
)
def test_property_19_shading_map_is_lower_frequency_than_its_source(
    randomized_room, kind, height, width, seed, use_bilateral, focal_px, yaw_deg, pitch_deg
):
    """For any photograph, the mean gradient magnitude of the produced shading
    map is no greater than that of the source `L*` channel, and the shading map
    has the same shape and `uint8` dtype as the source.

    Both settings of `use_bilateral_shading` are drawn, since Requirement 7.2
    is satisfied either way and the operator is free to switch the pass off.

    **Validates: Requirements 7.2**
    """
    image = _photograph(
        kind, height, width, seed, randomized_room,
        focal_px=focal_px, yaw_deg=yaw_deg, pitch_deg=pitch_deg,
    )
    overrides = Settings(use_bilateral_shading=use_bilateral)
    l_channel = to_lab_l(image)

    shading = low_frequency(l_channel, overrides)

    source_gradient = mean_gradient_magnitude(l_channel)
    shading_gradient = mean_gradient_magnitude(shading)
    # `<=`, not `<`: on inputs a few pixels across every kernel clamps below 3
    # taps and is skipped, so the map comes back equal to its source and the
    # bound is met with equality.
    slack = max(_GRADIENT_SLACK_ABS, _GRADIENT_SLACK_REL * source_gradient)
    assert shading_gradient <= source_gradient + slack

    assert shading.shape == l_channel.shape
    assert shading.dtype == np.uint8


@pytest.mark.parametrize("use_bilateral", [True, False], ids=["bilateral", "gaussian_only"])
def test_shading_map_strictly_smooths_a_full_size_textured_photograph(use_bilateral):
    """The non-vacuous half of Property 19.

    The property is satisfied with equality on tiny inputs, so on its own it
    would pass for an identity filter. At a realistic size the shading map has
    to actually discard the surface texture, which is the whole reason the
    Compositor can multiply by it without reprinting the old floor's grain.
    """
    l_channel = to_lab_l(_shadowed_photo(240, 320, seed=11))

    shading = low_frequency(l_channel, Settings(use_bilateral_shading=use_bilateral))

    source_gradient = mean_gradient_magnitude(l_channel)
    assert mean_gradient_magnitude(shading) < 0.25 * source_gradient


def test_shading_map_does_not_alias_its_input_when_every_filter_is_skipped():
    """A 1x1 image clamps every kernel away, and must still come back a copy.

    Handing the caller's own buffer back would let a later in-place blend
    corrupt the `L*` channel the detail map was derived from.
    """
    l_channel = to_lab_l(np.full((1, 1, 3), 200, np.uint8))

    shading = low_frequency(l_channel)

    assert np.array_equal(shading, l_channel)
    assert shading is not l_channel
    assert not np.shares_memory(shading, l_channel)


# --------------------------------------------------------------------------- #
# Property 20 -- reconstruction (Requirement 7.3)
# --------------------------------------------------------------------------- #


def _reconstruct(shading: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """`shading + (detail - 128)`, the relation the Compositor relies on."""
    return shading.astype(np.int16) + (detail.astype(np.int16) - NEUTRAL_DETAIL)


def _residual_fits_uint8(l_channel: np.ndarray, shading: np.ndarray) -> np.ndarray:
    """Mask of pixels whose biased residual stayed inside the 8-bit range.

    Computed from `L*` and the shading map directly rather than read back off
    the detail map, so it is an independent statement about where the encoding
    was lossless.
    """
    unclipped = l_channel.astype(np.int16) - shading.astype(np.int16) + NEUTRAL_DETAIL
    return (unclipped >= 0) & (unclipped <= 255)


# Feature: ai-room-tile-visualizer, Property 20: The lighting decomposition
# reconstructs the source
@_PROPERTY_SETTINGS
@given(
    kind=_kind,
    height=st.integers(min_value=_MIN_PHOTOGRAPH_EDGE, max_value=128),
    width=st.integers(min_value=_MIN_PHOTOGRAPH_EDGE, max_value=128),
    seed=_seed,
    focal_px=_focal_px,
    yaw_deg=_yaw_deg,
    pitch_deg=_pitch_deg,
)
def test_property_20_decomposition_reconstructs_the_source_l_channel(
    randomized_room, kind, height, width, seed, focal_px, yaw_deg, pitch_deg
):
    """For any photograph, recombining the shading map with the detail map's
    signed residual reconstructs the source `L*` channel within the documented
    tolerance, and the detail map's mean is within tolerance of its neutral
    midpoint.

    The tolerance has one source: the residual is stored biased by 128 in
    `uint8`, so a residual outside [-128, 127] clips. Everywhere it fitted, the
    reconstruction is *exact* -- that is asserted with no tolerance at all --
    and the two tolerances below bound the error and the mean bias that the
    clipped minority contributes.

    **Validates: Requirements 7.3**
    """
    image = _photograph(
        kind, height, width, seed, randomized_room,
        focal_px=focal_px, yaw_deg=yaw_deg, pitch_deg=pitch_deg,
    )
    l_channel = to_lab_l(image)
    shading = low_frequency(l_channel)

    detail = high_frequency(l_channel, shading)

    reconstructed = _reconstruct(shading, detail)
    error = np.abs(reconstructed - l_channel.astype(np.int16))

    lossless = _residual_fits_uint8(l_channel, shading)
    assert np.all(error[lossless] == 0), "reconstruction must be exact where nothing clipped"
    assert float(np.mean(error)) <= _RECONSTRUCTION_MEAN_TOLERANCE
    assert abs(float(np.mean(detail)) - NEUTRAL_DETAIL) <= _DETAIL_MEAN_TOLERANCE

    assert detail.shape == l_channel.shape
    assert detail.dtype == np.uint8


def test_decomposition_reconstructs_the_synthetic_room_exactly(synthetic_room):
    """On the fixture the whole suite calibrates against, nothing clips at all.

    A room photograph's residual is surface texture, which is small; the
    tolerances in Property 20 exist for adversarial inputs, not for this one, so
    here the reconstruction is bit-exact and the detail mean sits on neutral.
    """
    maps = decompose(synthetic_room.image)

    l_channel = to_lab_l(synthetic_room.image)
    reconstructed = _reconstruct(maps.shading, maps.detail)

    assert np.array_equal(reconstructed, l_channel.astype(np.int16))
    assert float(np.mean(maps.detail)) == pytest.approx(NEUTRAL_DETAIL, abs=0.5)


def test_detail_map_is_exactly_neutral_where_the_source_has_no_local_variation():
    """A uniform image has no residual, so the detail map is 128 everywhere.

    This pins the *meaning* of the 128 bias rather than just its arithmetic: the
    Compositor's highlight term is `detail - 128`, so a flat surface must
    contribute no highlight at all.
    """
    l_channel = to_lab_l(np.full((64, 96, 3), 140, np.uint8))
    shading = low_frequency(l_channel)

    detail = high_frequency(l_channel, shading)

    assert np.all(detail == NEUTRAL_DETAIL)


def test_detail_map_is_signed_about_neutral():
    """Brighter-than-shading pixels land above 128 and darker ones below.

    Without this the reconstruction assertions would still pass for a detail map
    that stored `abs(residual)`, which would make every shadow read as a
    highlight downstream.
    """
    l_channel = to_lab_l(_shadowed_photo(120, 160, seed=5))
    shading = low_frequency(l_channel)

    detail = high_frequency(l_channel, shading)

    brighter = l_channel.astype(np.int16) > shading.astype(np.int16)
    darker = l_channel.astype(np.int16) < shading.astype(np.int16)
    assert brighter.any() and darker.any(), "fixture must contain both signs of residual"
    assert np.all(detail[brighter] > NEUTRAL_DETAIL)
    assert np.all(detail[darker] < NEUTRAL_DETAIL)


# --------------------------------------------------------------------------- #
# Per-plane medians (Requirement 7.4) and cached dtypes (Requirement 12.4)
# --------------------------------------------------------------------------- #


def _two_band_image(height: int = 160, width: int = 160) -> np.ndarray:
    """A dark upper band over a bright lower band, with grain in both.

    Two bands with widely separated luminance is the configuration that
    distinguishes a per-plane median from a global one: the global median falls
    between the bands and belongs to neither.
    """
    rng = np.random.default_rng(21)
    image = np.empty((height, width, 3), np.uint8)
    split = height // 2
    image[:split] = 60
    image[split:] = 205
    grain = rng.normal(0.0, 4.0, image.shape)
    return np.clip(image.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def _band_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Masks selecting the two bands, well inside each so blur bleed is excluded."""
    height, width = shape
    split = height // 2
    margin = height // 8

    upper = np.zeros(shape, np.uint8)
    upper[margin : split - margin] = 255
    lower = np.zeros(shape, np.uint8)
    lower[split + margin : height - margin] = 1  # any non-zero value counts as set
    return {"wall_back": upper, "floor": lower}


def test_plane_medians_are_taken_per_mask_and_not_globally():
    """Requirement 7.4's neutral point is per plane.

    Each median must land inside its own band, and the two must straddle the
    global median of the shading map. A single global value would put an entire
    plane on one side of the Compositor's multiply/soft-light branch and darken
    or blow out the whole surface.
    """
    image = _two_band_image()
    masks = _band_masks(image.shape[:2])

    maps = decompose(image, masks)

    global_median = float(np.median(maps.shading))
    dark_median = maps.plane_medians["wall_back"]
    bright_median = maps.plane_medians["floor"]

    # Each plane's median is the median of the shading map over that plane only.
    for plane, mask in masks.items():
        expected = float(np.median(maps.shading[mask > 0]))
        assert maps.plane_medians[plane] == pytest.approx(expected)

    assert dark_median < global_median < bright_median
    assert bright_median - dark_median > 100.0


def test_plane_with_an_empty_mask_is_absent_from_the_medians():
    """An unoccupied plane is omitted, not reported with a placeholder.

    A placeholder neutral point would be indistinguishable from a measured one,
    and the Compositor would branch its blend on a number that means nothing.
    """
    image = _two_band_image(96, 96)
    masks = _band_masks(image.shape[:2])
    masks["wall_left"] = np.zeros(image.shape[:2], np.uint8)

    maps = decompose(image, masks)

    assert "wall_left" not in maps.plane_medians
    assert set(maps.plane_medians) == {"wall_back", "floor"}


def test_decompose_without_masks_reports_no_medians():
    image = _two_band_image(64, 64)

    maps = decompose(image)

    assert isinstance(maps, LightingMaps)
    assert maps.plane_medians == {}


def test_cached_lighting_maps_are_single_channel_uint8(synthetic_room):
    """Requirement 12.4: both maps cache at one byte per pixel."""
    masks = {"floor": (synthetic_room.occluder_mask == 0).astype(np.uint8)}

    maps = decompose(synthetic_room.image, masks)

    expected_shape = synthetic_room.image.shape[:2]
    for name, plane in (("shading", maps.shading), ("detail", maps.detail)):
        assert plane.dtype == np.uint8, f"{name} must cache as uint8"
        assert plane.shape == expected_shape, f"{name} must be single channel"
    assert set(maps.plane_medians) == {"floor"}


def test_mask_shape_mismatch_is_rejected():
    """A mask from a different frame is a caller bug, not a silent no-op."""
    image = _two_band_image(64, 64)
    wrong_shape = np.ones((32, 32), np.uint8)

    with pytest.raises(ValueError, match="floor"):
        decompose(image, {"floor": wrong_shape})
