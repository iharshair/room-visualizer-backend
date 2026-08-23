"""Tests for `backend.utils.imageio` (Requirements 2.2, 2.6).

Covers the decode contract that gates every upload, the longest-edge clamp, and
the output encoder. The clamp is additionally exercised with hypothesis as the
module-level half of Property 8; the HTTP-level half lands in `test_api.py`.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from backend.utils.imageio import (
    DecodeError,
    EncodeError,
    clamp_longest_edge,
    decode_image,
    encode_image,
    mime_for_format,
)


def _sample_bgr(height: int = 40, width: int = 60) -> np.ndarray:
    """A deterministic, non-flat BGR image so encoders have real content."""
    ys, xs = np.mgrid[0:height, 0:width]
    blue = (xs * 4 % 256).astype(np.uint8)
    green = (ys * 6 % 256).astype(np.uint8)
    red = ((xs + ys) * 3 % 256).astype(np.uint8)
    return np.ascontiguousarray(np.dstack([blue, green, red]))


def _encode(img: np.ndarray, ext: str = ".png", params: list[int] | None = None) -> bytes:
    ok, buf = cv2.imencode(ext, img, params or [])
    assert ok
    return buf.tobytes()


# --- decode_image (Requirement 2.2) --------------------------------------


@pytest.mark.parametrize("ext", [".png", ".jpg", ".webp"])
def test_decode_returns_bgr_uint8_for_each_allowed_format(ext):
    original = _sample_bgr()

    decoded = decode_image(_encode(original, ext))

    assert decoded.shape == original.shape
    assert decoded.dtype == np.uint8
    assert decoded.flags["C_CONTIGUOUS"]
    if ext == ".png":
        # Lossless, so the round trip must be exact.
        assert np.array_equal(decoded, original)


def test_decode_expands_grayscale_to_three_channels():
    gray = (np.mgrid[0:20, 0:30][1] * 8 % 256).astype(np.uint8)

    decoded = decode_image(_encode(gray))

    assert decoded.shape == (20, 30, 3)
    assert np.array_equal(decoded[:, :, 0], decoded[:, :, 2])


def test_decode_drops_alpha_channel():
    bgra = np.dstack([_sample_bgr(), np.full((40, 60), 128, np.uint8)])

    decoded = decode_image(_encode(bgra))

    assert decoded.shape == (40, 60, 3)


def test_decode_reduces_sixteen_bit_samples_to_uint8():
    deep = (_sample_bgr().astype(np.uint16) * 257)

    decoded = decode_image(_encode(deep))

    assert decoded.dtype == np.uint8
    assert decoded.shape == (40, 60, 3)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"not an image at all", id="text"),
        pytest.param(b"\x00" * 4096, id="zero_bytes"),
        pytest.param(b"%PDF-1.7\n%\xc7\xec\x8f\xa2\n", id="pdf_header"),
        pytest.param(b"#!/bin/sh\necho hi\n", id="script"),
    ],
)
def test_decode_raises_decode_error_on_non_raster_payloads(payload):
    with pytest.raises(DecodeError):
        decode_image(payload)


def test_decode_raises_decode_error_on_truncated_image():
    truncated = _encode(_sample_bgr(200, 200), ".jpg")[:64]

    with pytest.raises(DecodeError):
        decode_image(truncated)


def test_decode_error_is_catchable_as_value_of_module_base():
    from backend.utils.imageio import ImageIOError

    with pytest.raises(ImageIOError):
        decode_image(b"")


# --- clamp_longest_edge (Requirement 2.6) --------------------------------


@pytest.mark.parametrize(
    "shape",
    [(100, 100), (2048, 1536), (1536, 2048), (10, 2048)],
    ids=["square_small", "landscape_at_limit", "portrait_at_limit", "sliver_at_limit"],
)
def test_clamp_is_identity_at_or_below_the_limit(shape):
    img = np.zeros((*shape, 3), np.uint8)

    result = clamp_longest_edge(img, 2048)

    # Same object: no copy, no resample when nothing needs shrinking.
    assert result is img


def test_clamp_never_upscales():
    img = np.zeros((300, 400, 3), np.uint8)

    result = clamp_longest_edge(img, 4096)

    assert result is img
    assert result.shape == (300, 400, 3)


@pytest.mark.parametrize(
    ("height", "width", "limit", "expected"),
    [
        (3000, 4000, 2048, (1536, 2048)),
        (4000, 3000, 2048, (2048, 1536)),
        (3000, 3000, 2048, (2048, 2048)),
        (1200, 3600, 1200, (400, 1200)),
    ],
)
def test_clamp_downscales_to_expected_dimensions(height, width, limit, expected):
    img = np.zeros((height, width, 3), np.uint8)

    result = clamp_longest_edge(img, limit)

    assert result.shape[:2] == expected
    assert max(result.shape[:2]) == limit
    assert result.dtype == np.uint8


def test_clamp_preserves_dtype_channels_and_handles_two_dimensional_masks():
    colour = _sample_bgr(1200, 1600)
    mask = np.full((1200, 1600), 255, np.uint8)

    small_colour = clamp_longest_edge(colour, 400)
    small_mask = clamp_longest_edge(mask, 400)

    assert small_colour.shape == (300, 400, 3)
    assert small_colour.dtype == np.uint8
    assert small_mask.shape == (300, 400)
    assert small_mask.dtype == np.uint8


def test_clamp_uses_area_averaging_rather_than_nearest_neighbour():
    # A one-pixel checkerboard averages to a flat mid-grey under INTER_AREA and
    # stays fully bimodal under nearest-neighbour, so this distinguishes them.
    checker = np.indices((400, 400)).sum(axis=0) % 2
    img = (checker * 255).astype(np.uint8)

    result = clamp_longest_edge(img, 200)

    assert result.shape == (200, 200)
    assert np.all(np.abs(result.astype(np.int16) - 128) <= 1)


@pytest.mark.parametrize("limit", [0, -1])
def test_clamp_rejects_non_positive_limit(limit):
    with pytest.raises(ValueError):
        clamp_longest_edge(np.zeros((10, 10, 3), np.uint8), limit)


def test_clamp_rejects_empty_array():
    with pytest.raises(ValueError):
        clamp_longest_edge(np.zeros((0, 0, 3), np.uint8), 100)


# Feature: ai-room-tile-visualizer, Property 8: Downscaling clamps the longest
# edge and preserves aspect ratio
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    limit=st.integers(min_value=400, max_value=1024),
    long_edge=st.integers(min_value=400, max_value=2500),
    short_frac=st.floats(min_value=0.25, max_value=1.0),
    landscape=st.booleans(),
)
def test_property_8_clamp_bounds_longest_edge_and_preserves_aspect(
    limit, long_edge, short_frac, landscape
):
    """For any input dimensions, the output longest edge equals
    ``min(limit, input longest edge)`` and the aspect ratio is preserved to
    within 0.5 percent.

    **Validates: Requirements 2.6**
    """
    long_edge = max(long_edge, limit // 2)
    short_edge = max(1, int(round(long_edge * short_frac)))
    height, width = (short_edge, long_edge) if landscape else (long_edge, short_edge)
    img = np.zeros((height, width, 3), np.uint8)

    result = clamp_longest_edge(img, limit)
    out_h, out_w = result.shape[:2]

    assert max(out_h, out_w) == min(limit, max(height, width))
    assert min(out_h, out_w) >= 1
    assert result.dtype == np.uint8
    assert result.shape[2] == 3

    in_ratio = width / height
    out_ratio = out_w / out_h
    assert abs(out_ratio - in_ratio) / in_ratio <= 0.005


# --- encode_image --------------------------------------------------------


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [("png", "image/png"), ("jpeg", "image/jpeg"), ("jpg", "image/jpeg"), ("webp", "image/webp")],
)
def test_encode_returns_bytes_and_mime_that_round_trip(fmt, mime):
    original = _sample_bgr()

    data, returned_mime = encode_image(original, fmt, jpeg_quality=90)

    assert returned_mime == mime
    assert isinstance(data, bytes) and len(data) > 0
    assert decode_image(data).shape == original.shape


def test_encode_defaults_to_lossless_png():
    original = _sample_bgr()

    data, mime = encode_image(original)

    assert mime == "image/png"
    assert np.array_equal(decode_image(data), original)


def test_encode_honours_jpeg_quality():
    original = _sample_bgr(200, 200)

    low, _ = encode_image(original, "jpeg", jpeg_quality=15)
    high, _ = encode_image(original, "jpeg", jpeg_quality=95)

    assert len(low) < len(high)


def test_encode_ignores_quality_for_png():
    original = _sample_bgr()

    low, _ = encode_image(original, "png", jpeg_quality=10)
    high, _ = encode_image(original, "png", jpeg_quality=100)

    assert low == high


@pytest.mark.parametrize("fmt", ["gif", "tiff", "", "exe"])
def test_encode_rejects_unsupported_format(fmt):
    with pytest.raises(EncodeError):
        encode_image(_sample_bgr(), fmt)


@pytest.mark.parametrize("quality", [0, 101, -5])
def test_encode_rejects_out_of_range_quality(quality):
    with pytest.raises(EncodeError):
        encode_image(_sample_bgr(), "jpeg", jpeg_quality=quality)


def test_encode_rejects_empty_array():
    with pytest.raises(EncodeError):
        encode_image(np.zeros((0, 0, 3), np.uint8), "png")


def test_mime_for_format_is_case_and_dot_insensitive():
    assert mime_for_format("PNG") == "image/png"
    assert mime_for_format(".jpeg") == "image/jpeg"
    with pytest.raises(EncodeError):
        mime_for_format("bmp")
