"""Image decode, downscale, and encode helpers.

Every byte that enters the pipeline passes through :func:`decode_image`, and
every byte that leaves it passes through :func:`encode_image`. Keeping both in
one leaf module means the upload-hardening rules (Requirements 2.2, 2.6) and the
output-format rules live in exactly one place instead of being re-derived at
each call site.

The decode path is deliberately strict and total: any input OpenCV cannot turn
into a raster image raises :class:`DecodeError`, which `app.py` maps to HTTP 415
before any segmentation, geometry, or lighting stage is entered.
"""

from __future__ import annotations

from typing import Final

import cv2
import numpy as np

__all__ = [
    "ImageIOError",
    "DecodeError",
    "EncodeError",
    "decode_image",
    "clamp_longest_edge",
    "encode_image",
    "mime_for_format",
    "MIME_BY_FORMAT",
]

# Encoder formats the service supports, mapped to the OpenCV file extension used
# for `cv2.imencode` and the MIME type returned to the caller. `render_format`
# in Settings is a Literal["png", "jpeg"], and `jpg` is accepted as a spelling
# of `jpeg` so a hand-set RV_RENDER_FORMAT=jpg does not fail at render time.
_EXT_BY_FORMAT: Final[dict[str, str]] = {
    "png": ".png",
    "jpeg": ".jpg",
    "jpg": ".jpg",
    "webp": ".webp",
}

MIME_BY_FORMAT: Final[dict[str, str]] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
}


class ImageIOError(Exception):
    """Base class for image conversion failures raised by this module."""


class DecodeError(ImageIOError):
    """Raised when submitted bytes do not decode as a raster image (R2.2)."""


class EncodeError(ImageIOError):
    """Raised when an array cannot be encoded in the requested format."""


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into a contiguous 3-channel BGR ``uint8`` array.

    `cv2.IMREAD_COLOR` normalises the awkward cases for us: alpha channels are
    dropped, 16-bit samples are reduced to 8-bit, and grayscale is expanded to
    three channels, so every downstream module can assume one shape and dtype.

    Raises:
        DecodeError: the payload is empty, truncated, or not a raster image.
    """
    if not data:
        raise DecodeError("empty payload")

    buffer = np.frombuffer(data, dtype=np.uint8)
    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as exc:  # pragma: no cover - OpenCV normally returns None
        raise DecodeError(f"image decode failed: {exc}") from exc

    if image is None:
        raise DecodeError("bytes do not decode as a raster image")
    if image.ndim != 3 or image.shape[2] != 3:  # pragma: no cover - defensive
        raise DecodeError(f"unexpected decoded shape {image.shape!r}")
    if image.size == 0:  # pragma: no cover - defensive
        raise DecodeError("decoded image has zero pixels")

    # `imdecode` already returns uint8 under IMREAD_COLOR; assert it rather than
    # silently casting, so a future flag change surfaces here instead of as
    # garbage pixels three modules downstream.
    if image.dtype != np.uint8:  # pragma: no cover - defensive
        raise DecodeError(f"unexpected decoded dtype {image.dtype!r}")

    return np.ascontiguousarray(image)


def _short_edge_for(long_in: int, short_in: int, long_out: int) -> int:
    """Pick the integer short edge whose aspect ratio best matches the input.

    Rounding the scaled short edge to the nearer integer is usually optimal, but
    not always -- comparing both neighbours costs nothing and keeps the aspect
    error at the minimum any integer size can achieve (Requirement 2.6).
    """
    exact = short_in * long_out / long_in
    target_ratio = long_in / short_in
    candidates = {max(1, int(np.floor(exact))), max(1, int(np.ceil(exact)))}
    return min(candidates, key=lambda short: abs(long_out / short - target_ratio))


def clamp_longest_edge(img: np.ndarray, limit: int) -> np.ndarray:
    """Downscale ``img`` so its longest edge equals ``min(limit, longest)``.

    Images already at or under the limit are returned unchanged (same object),
    so the common case costs nothing. This is a pure downscale: the function
    never enlarges, which is what bounds every downstream allocation regardless
    of the dimensions an upload declares.

    `cv2.INTER_AREA` is the correct kernel for shrinking -- it averages the
    source pixels covered by each destination pixel, so tile grout lines and
    room edges do not alias into the analysis stages.

    Args:
        img: 2-D or 3-D array to downscale.
        limit: maximum permitted longest edge in pixels; must be positive.

    Returns:
        The original array when no downscale is needed, else a new array whose
        aspect ratio matches the input to within 0.5 percent for any input whose
        aspect ratio permits it at integer sizes.
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if img.ndim < 2 or img.size == 0:
        raise ValueError(f"expected a non-empty image array, got shape {img.shape!r}")

    height, width = img.shape[:2]
    longest = max(height, width)
    target = min(limit, longest)
    if target == longest:
        return img

    if width >= height:
        new_width = target
        new_height = _short_edge_for(width, height, target)
    else:
        new_height = target
        new_width = _short_edge_for(height, width, target)

    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


def mime_for_format(fmt: str) -> str:
    """Return the MIME type for an output format name.

    Raises:
        EncodeError: the format is not one this module can encode.
    """
    key = fmt.strip().lower().lstrip(".")
    try:
        return MIME_BY_FORMAT[key]
    except KeyError:
        supported = ", ".join(sorted(MIME_BY_FORMAT))
        raise EncodeError(f"unsupported output format {fmt!r}; expected one of {supported}") from None


def encode_image(
    img: np.ndarray,
    fmt: str = "png",
    jpeg_quality: int = 90,
) -> tuple[bytes, str]:
    """Encode ``img`` and return ``(bytes, mime_type)``.

    PNG is the default because it is lossless: JPEG ringing around flat grout
    lines is the one artifact a tile preview cannot afford. ``jpeg_quality`` is
    ignored for lossless formats.

    Raises:
        EncodeError: the format is unsupported or OpenCV refused the array.
    """
    key = fmt.strip().lower().lstrip(".")
    mime = mime_for_format(key)
    if img.ndim < 2 or img.size == 0:
        raise EncodeError(f"expected a non-empty image array, got shape {img.shape!r}")
    if not 1 <= jpeg_quality <= 100:
        raise EncodeError(f"jpeg_quality must be in [1, 100], got {jpeg_quality}")

    params: list[int] = []
    if key in {"jpeg", "jpg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    elif key == "webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, int(jpeg_quality)]

    try:
        ok, encoded = cv2.imencode(_EXT_BY_FORMAT[key], img, params)
    except cv2.error as exc:
        raise EncodeError(f"{key} encode failed: {exc}") from exc
    if not ok or encoded is None or encoded.size == 0:
        raise EncodeError(f"{key} encode failed")

    return encoded.tobytes(), mime
