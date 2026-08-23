"""Tests for `backend.utils.texture_helper` (Requirements 8.1, 8.6, 8.7, 11.2).

Two concerns share this module, in the order the pipeline uses them:

* **Seamless synthesis.** Property 25 drives `make_seamless` with hypothesis over
  noise, gradient, and high-contrast checkerboard inputs and holds the result to
  the `edge_continuity` bound Requirement 8.1 defines (Requirement 13.6). A pair
  of unit tests pin `edge_continuity` itself first, so the property cannot pass
  by measuring nothing.
* **Procedural generators.** The four finishes the Setup_Tool ships are checked
  for shape, dtype, per-seed determinism, seed sensitivity, and -- composed the
  way the Setup_Tool composes them -- the same continuity bound
  (Requirement 11.2).
* **Metric scaling.** `to_metric_texture` is held to both halves of the contract
  Requirements 8.6 and 8.7 state: the output pixel ratio matches the declared
  millimetre ratio to within 0.1 percent, and the single recorded `px_per_mm`
  reproduces both pixel dimensions from the millimetre dimensions -- so no
  stretch can enter the render between the catalog and the Compositor. The
  regression these guard is a 600x1200 plank arriving square.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from backend.utils.texture_helper import (
    MAX_GENERATED_EDGE,
    MAX_TEXTURE_EDGE,
    METRIC_RATIO_TOLERANCE,
    MIN_TEXTURE_EDGE,
    SEAMLESS_TOLERANCE,
    SeamlessTexture,
    edge_continuity,
    generate_concrete,
    generate_marble,
    generate_terrazzo,
    generate_wood_plank,
    make_seamless,
    to_metric_texture,
)

#: The four finishes Requirement 11.2 names, as ``(name, callable)``.
GENERATORS = (
    ("marble", generate_marble),
    ("wood_plank", generate_wood_plank),
    ("concrete", generate_concrete),
    ("terrazzo", generate_terrazzo),
)

#: Generator sizes used throughout. Small on purpose -- these tests assert
#: contracts (shape, dtype, determinism, continuity) that hold at any edge, and
#: a 64 px tile exercises them in about a millisecond.
_SQUARE_PX = 64
_PLANK_PX = (48, 96)  # (width, height), the 1:2 format from Requirement 11.2


# --------------------------------------------------------------------------- #
# Input tile builders
# --------------------------------------------------------------------------- #
#
# Three shapes of input, matching the three the task names. Each is
# deterministic given its parameters, so a hypothesis counterexample is
# reproducible from the reported arguments alone.


def _noise_tile(height: int, width: int, seed: int) -> np.ndarray:
    """Full-range uncorrelated noise -- the worst case for edge matching."""
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def _gradient_tile(height: int, width: int, seed: int) -> np.ndarray:
    """A diagonal ramp: smooth everywhere except across the wrapped edges."""
    rng = np.random.default_rng(seed)
    ramp_x = np.linspace(0.0, 255.0, width, dtype=np.float32)[np.newaxis, :]
    ramp_y = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, np.newaxis]
    value = 0.5 * (ramp_x + ramp_y)
    # Per-channel gain, so the ramp is not achromatic and the luminance
    # conversion in edge_continuity has real colour to fold down.
    gains = rng.uniform(0.75, 1.0, 3).astype(np.float32)
    return np.clip(value[:, :, np.newaxis] * gains, 0, 255).astype(np.uint8)


def _checkerboard_tile(height: int, width: int, seed: int, cell: int = 3) -> np.ndarray:
    """Black-and-white cells: maximum contrast right at the edges."""
    ys, xs = np.mgrid[0:height, 0:width]
    dark = ((ys // cell) + (xs // cell)) % 2 == 1
    tile = np.where(dark, 0, 255).astype(np.uint8)
    return np.repeat(tile[:, :, np.newaxis], 3, axis=2)


_TILE_BUILDERS = {
    "noise": _noise_tile,
    "gradient": _gradient_tile,
    "checkerboard": _checkerboard_tile,
}


def _input_tile(kind: str, height: int, width: int, seed: int) -> np.ndarray:
    return _TILE_BUILDERS[kind](height, width, seed)


# --------------------------------------------------------------------------- #
# edge_continuity (Requirement 8.1)
# --------------------------------------------------------------------------- #


def test_edge_continuity_is_zero_for_a_pattern_whose_edges_already_match():
    # Opposite edges are identical by construction: every row and column is
    # constant, so the wrapped difference is exactly zero.
    flat = np.full((32, 48, 3), 120, np.uint8)

    assert edge_continuity(flat) == pytest.approx(0.0)


def test_edge_continuity_reports_a_large_mismatch_for_a_full_range_ramp():
    """A black-to-white ramp is the worst wrapped edge there is.

    This is the guard that keeps Property 25 from being vacuous: the metric must
    be capable of reporting a violation, and the inputs the property feeds
    `make_seamless` must actually violate the bound before the call.
    """
    ramp = _gradient_tile(64, 64, seed=0)

    # Both wrapped edges span most of the range, so the mean mismatch is far
    # above the 2 percent tolerance the property then has to reach.
    assert edge_continuity(ramp) > 10 * SEAMLESS_TOLERANCE


# --------------------------------------------------------------------------- #
# Seamless synthesis (Requirements 8.1, 13.6)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 25: Synthesized patterns are
# seamless
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    kind=st.sampled_from(sorted(_TILE_BUILDERS)),
    height=st.integers(min_value=4, max_value=96),
    width=st.integers(min_value=4, max_value=96),
    blend_frac=st.floats(min_value=0.05, max_value=0.5),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_property_25_make_seamless_closes_the_wrapped_edges(
    kind, height, width, blend_frac, seed
):
    """For any input tile image, the wrapped edge continuity metric of the
    synthesized seamless pattern -- the mean absolute luminance difference
    across wrapped opposite edge pixels -- is no more than 2 percent of the full
    luminance range.

    **Validates: Requirements 8.1, 13.6**
    """
    tile = _input_tile(kind, height, width, seed)

    pattern = make_seamless(tile, blend_frac=blend_frac)

    assert edge_continuity(pattern) <= SEAMLESS_TOLERANCE
    # Synthesis resamples nothing: the pattern must stay drop-in for its source.
    assert pattern.shape == tile.shape
    assert pattern.dtype == np.uint8


# --------------------------------------------------------------------------- #
# Procedural generators (Requirement 11.2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(("name", "generate"), GENERATORS, ids=[n for n, _ in GENERATORS])
@pytest.mark.parametrize(
    "size_px",
    [_SQUARE_PX, _PLANK_PX],
    ids=["square", "plank_1to2"],
)
def test_generators_return_the_requested_shape_as_uint8_bgr(name, generate, size_px):
    expected_width, expected_height = (
        (size_px, size_px) if isinstance(size_px, int) else size_px
    )

    pattern = generate(size_px, seed=5)

    assert pattern.shape == (expected_height, expected_width, 3)
    assert pattern.dtype == np.uint8


@pytest.mark.parametrize(("name", "generate"), GENERATORS, ids=[n for n, _ in GENERATORS])
def test_generators_are_deterministic_for_a_fixed_seed(name, generate):
    """Requirement 11.5's reproducibility half: same seed, same bytes.

    The Setup_Tool ships whatever these produce, so a seed that drifted between
    runs would make the asset tree unreproducible.
    """
    first = generate(_PLANK_PX, seed=7)
    second = generate(_PLANK_PX, seed=7)

    assert np.array_equal(first, second)


@pytest.mark.parametrize(("name", "generate"), GENERATORS, ids=[n for n, _ in GENERATORS])
def test_generators_differ_across_seeds(name, generate):
    # Not merely unequal: the seed must change the pattern substantially, or the
    # eight shipped tiles would be near-duplicates of each other.
    first = generate(_PLANK_PX, seed=7)
    other = generate(_PLANK_PX, seed=8)

    assert not np.array_equal(first, other)
    mean_delta = float(np.mean(np.abs(first.astype(np.int16) - other.astype(np.int16))))
    assert mean_delta > 1.0


@pytest.mark.parametrize(("name", "generate"), GENERATORS, ids=[n for n, _ in GENERATORS])
@pytest.mark.parametrize(
    "size_px",
    [_SQUARE_PX, _PLANK_PX],
    ids=["square", "plank_1to2"],
)
def test_generated_patterns_are_seamless_after_make_seamless(name, generate, size_px):
    """The composition the Setup_Tool uses satisfies Requirement 8.1.

    Raw generator output is not seamless by contract -- the generators only keep
    their noise primitives periodic where that is cheap -- so the bound is
    asserted on `make_seamless(generate_x(...))`, which is what the Setup_Tool
    writes to `assets/tiles/`.
    """
    pattern = make_seamless(generate(size_px, seed=11))

    assert edge_continuity(pattern) <= SEAMLESS_TOLERANCE


@pytest.mark.parametrize(("name", "generate"), GENERATORS, ids=[n for n, _ in GENERATORS])
@pytest.mark.parametrize(
    "size_px",
    [MIN_TEXTURE_EDGE - 1, MAX_GENERATED_EDGE + 1],
    ids=["below_min_edge", "above_max_edge"],
)
def test_generators_reject_edges_outside_the_supported_band(name, generate, size_px):
    with pytest.raises(ValueError):
        generate(size_px, seed=0)

# --------------------------------------------------------------------------- #
# Metric scaling (Requirements 8.6, 8.7)
# --------------------------------------------------------------------------- #
#
# Requirement 8.7 forbids any stretch that changes a tile's metric aspect ratio,
# and Requirement 8.6 fixes the tolerance at 0.1 percent. `to_metric_texture` is
# the single place that is decided, so the assertions below pin both halves of
# what it promises the Compositor:
#
#   1. the output *pixel* ratio matches the declared *millimetre* ratio, and
#   2. one shared `px_per_mm` reproduces both pixel dimensions exactly, so the
#      Compositor's metric-to-texture conversion is a multiply with no drift.
#
# Holding (2) as well as (1) is what forces the long edge to stay negotiable: a
# fixed long edge rounded to an integer short edge can miss the ratio by far more
# than 0.1 percent, so the implementation scans candidate pixel pairs outward
# from a target instead. These tests therefore never assert an exact output size.

#: The three formats Requirement 8.7 names, as ``(width_mm, height_mm)``. 1:1 and
#: 1:2 are the shipped tile formats; 200x1200 is the plank.
_METRIC_FORMATS = ((600.0, 600.0), (600.0, 1200.0), (200.0, 1200.0))
_METRIC_FORMAT_IDS = ("1to1_600x600", "1to2_600x1200", "plank_200x1200")

#: Source patterns fed to the resampler. Sizes straddle the resolved output on
#: purpose, so both the downsampling and the upsampling branch are exercised, and
#: the non-square entry proves the declared ratio -- not the source ratio -- wins.
_METRIC_SOURCES = (("noise", 64, 64), ("checkerboard", 33, 96), ("gradient", 128, 40))
_METRIC_SOURCE_IDS = ("square_64", "tall_33x96", "wide_128x40")


def _metric_source(kind: str, height: int, width: int, seed: int = 3) -> np.ndarray:
    """A seamless source pattern, composed the way the Catalog_Loader composes it."""
    return make_seamless(_input_tile(kind, height, width, seed))


def _ratio_error(texture: SeamlessTexture) -> float:
    """Relative gap between the pixel ratio and the declared millimetre ratio."""
    declared = texture.width_mm / texture.height_mm
    return abs((texture.width_px / texture.height_px) / declared - 1.0)


@pytest.mark.parametrize(
    ("width_mm", "height_mm"), _METRIC_FORMATS, ids=_METRIC_FORMAT_IDS
)
@pytest.mark.parametrize(
    ("kind", "height", "width"), _METRIC_SOURCES, ids=_METRIC_SOURCE_IDS
)
def test_to_metric_texture_preserves_the_declared_millimetre_ratio(
    width_mm, height_mm, kind, height, width
):
    """Requirement 8.6: pixel ratio matches millimetre ratio within 0.1 percent.

    Asserted across all three formats and a source whose own aspect ratio
    disagrees with the declared one, because that disagreement is exactly what
    the resample has to overrule.
    """
    texture = to_metric_texture(_metric_source(kind, height, width), width_mm, height_mm)

    assert _ratio_error(texture) <= METRIC_RATIO_TOLERANCE
    # The declared dimensions travel with the pattern, so the Compositor reads
    # the tile's real size rather than re-deriving it from pixels.
    assert texture.width_mm == pytest.approx(width_mm)
    assert texture.height_mm == pytest.approx(height_mm)


@pytest.mark.parametrize(
    ("width_mm", "height_mm"), _METRIC_FORMATS, ids=_METRIC_FORMAT_IDS
)
@pytest.mark.parametrize(
    ("kind", "height", "width"), _METRIC_SOURCES, ids=_METRIC_SOURCE_IDS
)
def test_px_per_mm_reproduces_the_pattern_pixel_dimensions(
    width_mm, height_mm, kind, height, width
):
    """Requirement 8.7: one scale reproduces *both* pixel dimensions.

    A ratio that merely holds within tolerance is not enough -- the Compositor
    converts millimetres to texture pixels with a single `px_per_mm`, so if the
    two axes needed different scales the second axis would stretch.
    """
    texture = to_metric_texture(_metric_source(kind, height, width), width_mm, height_mm)

    assert round(texture.width_mm * texture.px_per_mm) == texture.width_px
    assert round(texture.height_mm * texture.px_per_mm) == texture.height_px
    # The reported pixel dimensions must describe the array the Compositor samples.
    assert (texture.height_px, texture.width_px) == texture.pattern.shape[:2]
    assert texture.px_per_mm > 0.0
    assert texture.pattern.dtype == np.uint8


@pytest.mark.parametrize(
    ("width_mm", "height_mm", "expect_taller"),
    [(600.0, 1200.0, True), (200.0, 1200.0, True), (1200.0, 600.0, False)],
    ids=["plank_600x1200", "plank_200x1200", "landscape_1200x600"],
)
def test_a_rectangular_tile_is_not_squashed_towards_a_square(
    width_mm, height_mm, expect_taller
):
    """The regression Requirement 8.7 exists for: a 600x1200 plank rendered square.

    A square source is the case that would produce it -- nothing in the pixels
    hints at the tile's real proportions, so only the declared millimetres can
    put them back.
    """
    texture = to_metric_texture(_metric_source("noise", 64, 64), width_mm, height_mm)

    assert (texture.height_px > texture.width_px) is expect_taller
    assert texture.height_px != texture.width_px
    assert texture.width_px / texture.height_px == pytest.approx(
        width_mm / height_mm, rel=METRIC_RATIO_TOLERANCE
    )


def test_a_square_tile_stays_square():
    texture = to_metric_texture(_metric_source("gradient", 128, 40), 600.0, 600.0)

    assert texture.width_px == texture.height_px


@pytest.mark.parametrize(
    ("width_mm", "height_mm"), _METRIC_FORMATS, ids=_METRIC_FORMAT_IDS
)
@pytest.mark.parametrize("max_edge_px", [MIN_TEXTURE_EDGE, 32, 128, MAX_TEXTURE_EDGE])
def test_the_long_edge_tracks_the_source_within_the_resolution_budget(
    width_mm, height_mm, max_edge_px
):
    """Resolution follows the source, clamped into the supported band.

    The long edge is only ever approximate -- holding the ratio takes priority,
    so a candidate a few pixels either side of the target may be the one that
    lands -- hence a 15 percent band rather than equality. What matters is that a
    6000 px product shot is not sampled at full size and an 8 px thumbnail is not
    left below the floor.
    """
    for source_long in (8, 96, 4000):
        source = _metric_source("noise", source_long, source_long)

        texture = to_metric_texture(
            source, width_mm, height_mm, max_edge_px=max_edge_px
        )

        target = min(max(source_long, MIN_TEXTURE_EDGE), max_edge_px)
        long_px = max(texture.width_px, texture.height_px)
        assert long_px == pytest.approx(target, rel=0.15)
        assert min(texture.width_px, texture.height_px) >= 1
        assert _ratio_error(texture) <= METRIC_RATIO_TOLERANCE


def test_an_awkward_ratio_sacrifices_the_target_size_rather_than_the_ratio():
    """A 197.5x1200 mm tile cannot honour 0.1 percent at a small long edge.

    13x79 is the smallest pixel pair that does, so the resolver overshoots a
    16 px budget to reach it. This is the documented trade: Requirement 8.6 is a
    hard bound, the resolution budget is a preference.
    """
    texture = to_metric_texture(
        _metric_source("noise", 16, 16), 197.5, 1200.0, max_edge_px=MIN_TEXTURE_EDGE
    )

    assert max(texture.width_px, texture.height_px) > MIN_TEXTURE_EDGE
    assert _ratio_error(texture) <= METRIC_RATIO_TOLERANCE
    assert round(texture.width_mm * texture.px_per_mm) == texture.width_px
    assert round(texture.height_mm * texture.px_per_mm) == texture.height_px


# Generalizes the parametrized cases above over arbitrary tile dimensions and
# source sizes. Not one of the design's numbered properties -- Requirements 8.6
# and 8.7 are stated as bounds on `to_metric_texture` itself, and this is the
# widest form of those bounds.
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    kind=st.sampled_from(sorted(_TILE_BUILDERS)),
    source_height=st.integers(min_value=2, max_value=96),
    source_width=st.integers(min_value=2, max_value=96),
    width_mm=st.floats(min_value=50.0, max_value=2400.0),
    height_mm=st.floats(min_value=50.0, max_value=2400.0),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_metric_scaling_holds_the_ratio_for_any_realistic_tile_format(
    kind, source_height, source_width, width_mm, height_mm, seed
):
    """For any tile within the ratio range real catalogs carry, the resampled
    pixel ratio matches the declared millimetre ratio within 0.1 percent and one
    `px_per_mm` reproduces both pixel dimensions.

    Ratios beyond 8:1 are excluded deliberately: past that the resolver may find
    no pixel pair that honours the bound at a sane resolution and raises rather
    than shipping a distorted tile, which the rejection tests below cover.
    """
    ratio = width_mm / height_mm
    assume(1.0 / 8.0 <= ratio <= 8.0)
    source = _input_tile(kind, source_height, source_width, seed)

    texture = to_metric_texture(source, width_mm, height_mm)

    assert _ratio_error(texture) <= METRIC_RATIO_TOLERANCE
    assert round(texture.width_mm * texture.px_per_mm) == texture.width_px
    assert round(texture.height_mm * texture.px_per_mm) == texture.height_px
    assert texture.pattern.dtype == np.uint8
    assert texture.pattern.shape[:2] == (texture.height_px, texture.width_px)


@pytest.mark.parametrize(
    ("width_mm", "height_mm"), _METRIC_FORMATS, ids=_METRIC_FORMAT_IDS
)
@pytest.mark.parametrize(("name", "generate"), GENERATORS, ids=[n for n, _ in GENERATORS])
def test_the_setup_tool_composition_is_seamless_at_its_metric_size(
    width_mm, height_mm, name, generate
):
    """`to_metric_texture(make_seamless(generate_x(...)), w, h)` -- the exact
    composition the Setup_Tool and Catalog_Loader write to `assets/tiles/`.

    Resampling is where Requirement 8.1 is easiest to lose: resize kernels clamp
    at the border instead of wrapping, so an anisotropic rescale can reopen a
    seam the source had closed. The shipped asset must satisfy both requirements
    at once, not each in isolation.
    """
    texture = to_metric_texture(
        make_seamless(generate(_SQUARE_PX, seed=13)), width_mm, height_mm
    )

    assert edge_continuity(texture.pattern) <= SEAMLESS_TOLERANCE
    assert _ratio_error(texture) <= METRIC_RATIO_TOLERANCE


def test_ensure_seamless_repairs_a_seam_the_resample_reopened():
    """The default repairs; `ensure_seamless=False` is a pure resample.

    A high-contrast checkerboard squeezed to a 1:6 plank is the case that breaks:
    the source wraps cleanly, and the rescale alone leaves a visible seam. The
    two calls are compared against each other so the repair is shown to be the
    only difference, and to be confined to a narrow border band -- the pattern's
    interior comes through untouched either way.
    """
    source = _metric_source("checkerboard", 96, 96)
    assert edge_continuity(source) <= SEAMLESS_TOLERANCE

    resampled = to_metric_texture(source, 200.0, 1200.0, ensure_seamless=False)
    repaired = to_metric_texture(source, 200.0, 1200.0, ensure_seamless=True)

    assert edge_continuity(resampled.pattern) > SEAMLESS_TOLERANCE
    assert edge_continuity(repaired.pattern) <= SEAMLESS_TOLERANCE
    # Same geometry: the repair changes pixels, never the metric contract.
    assert repaired.pattern.shape == resampled.pattern.shape
    assert repaired.px_per_mm == resampled.px_per_mm

    band = 2
    height, width = resampled.pattern.shape[:2]
    assert np.array_equal(
        repaired.pattern[band : height - band, band : width - band],
        resampled.pattern[band : height - band, band : width - band],
    )


def test_a_pattern_already_at_its_metric_size_passes_through_unchanged():
    """No resample, so no resample artifacts -- the bytes are the source's."""
    source = _metric_source("noise", 64, 64)

    texture = to_metric_texture(source, 600.0, 600.0)

    assert texture.pattern.shape == source.shape
    assert np.array_equal(texture.pattern, source)


@pytest.mark.parametrize(
    ("width_mm", "height_mm"),
    [
        (0.0, 600.0),
        (600.0, 0.0),
        (-600.0, 1200.0),
        (600.0, -1200.0),
        (float("nan"), 600.0),
        (600.0, float("nan")),
        (float("inf"), 600.0),
        (600.0, float("inf")),
    ],
    ids=[
        "zero_width",
        "zero_height",
        "negative_width",
        "negative_height",
        "nan_width",
        "nan_height",
        "inf_width",
        "inf_height",
    ],
)
def test_to_metric_texture_rejects_unusable_millimetre_dimensions(width_mm, height_mm):
    """A tile with no real size has no metric scale, so there is nothing to
    fall back to -- the Catalog_Loader excludes the entry instead."""
    with pytest.raises(ValueError):
        to_metric_texture(_metric_source("noise", 32, 32), width_mm, height_mm)


@pytest.mark.parametrize(
    "shape", [(1, 1, 3), (1, 8, 3), (8, 1, 3)], ids=["1x1", "single_row", "single_column"]
)
def test_to_metric_texture_rejects_a_source_smaller_than_two_pixels(shape):
    # Below 2x2 there are no opposite edges to wrap, so the pattern cannot be a
    # tiling texture whatever it is scaled to.
    with pytest.raises(ValueError):
        to_metric_texture(np.zeros(shape, np.uint8), 600.0, 600.0)


def test_to_metric_texture_rejects_a_budget_below_the_minimum_edge():
    with pytest.raises(ValueError):
        to_metric_texture(
            _metric_source("noise", 32, 32), 600.0, 600.0, max_edge_px=MIN_TEXTURE_EDGE - 1
        )


def test_to_metric_texture_raises_when_no_candidate_honours_the_ratio():
    """A 1x1000 mm sliver has no small pixel pair within 0.1 percent of its ratio.

    Raising is the right answer: silently widening the tolerance would ship the
    distortion Requirement 8.7 forbids.
    """
    with pytest.raises(ValueError, match="aspect ratio"):
        to_metric_texture(_metric_source("noise", 32, 32), 1.0, 1000.0)


def test_seamless_texture_is_immutable():
    # The Catalog_Loader memoises these, so a shared instance must not be
    # rewritable by one caller mid-render.
    texture = to_metric_texture(_metric_source("noise", 32, 32), 600.0, 1200.0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        texture.px_per_mm = 1.0  # type: ignore[misc]
