"""Lighting_Engine -- CIELAB illumination decomposition.

The Compositor needs to know two independent things about every pixel it is
about to cover with a tile: how much light falls there (so a cast shadow keeps
falling across the new floor) and how much local texture sits there (so a
specular glint stays a glint). Those two live at different spatial frequencies,
so this module splits the photograph's ``L*`` channel into a low-frequency
**shading map** and a high-frequency **detail map** (Requirements 7.1, 7.2,
7.3).

Both outputs are single-channel ``uint8``, which is the whole point: a
Scene_State caches them for the lifetime of the scene and Requirement 12.4 caps
cached artifacts at 8 bits per channel. The detail map therefore stores its
*signed* residual biased by :data:`NEUTRAL_DETAIL` (128) instead of carrying a
separate sign array, which makes the Compositor's highlight term just
``detail - 128`` and makes reconstruction ``shading + (detail - 128) == L*``
everywhere the residual did not clip.

The third output is :attr:`LightingMaps.plane_medians`: the median of the
shading map taken *per plane mask*, never globally. Requirement 7.4 selects
between a multiply blend and a soft-light blend by comparing shading against
this median, and a dim floor and a sunlit back wall have very different neutral
points -- a single global median would push an entire plane onto one side of
the branch and darken or blow out the whole surface.

Requirements: 7.1, 7.2, 7.3, 7.4, 12.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping

import cv2
import numpy as np

from backend.config import Settings, get_settings
from backend.schemas import PlaneName

__all__ = [
    "NEUTRAL_DETAIL",
    "LightingMaps",
    "to_lab_l",
    "low_frequency",
    "high_frequency",
    "decompose",
]

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

#: Neutral value of the detail map. A residual of zero is stored as 128, so the
#: Compositor's highlight term is ``int16(detail) - NEUTRAL_DETAIL`` and no
#: separate sign array is cached (Requirement 12.4).
NEUTRAL_DETAIL: Final[int] = 128

#: Bilateral neighbourhood diameter, fixed by the design. ``d`` (not
#: ``sigmaSpace``) governs the window OpenCV actually visits, so this is the
#: value that bounds the pass's cost.
_BILATERAL_DIAMETER: Final[int] = 9

#: ``sigmaColor`` is tied to the image's ``L*`` standard deviation so the pass
#: adapts to contrast: a flat, evenly lit wall gets a tight range kernel and a
#: high-contrast room a loose one. A floor keeps the filter well-defined on a
#: perfectly uniform image, where the standard deviation is zero.
_MIN_SIGMA_COLOR: Final[float] = 1.0

#: Gaussian kernel width in standard deviations. Six sigma captures the kernel
#: to about 0.3 percent of its mass, past which widening it only costs time.
_KERNEL_SIGMAS: Final[float] = 6.0


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class LightingMaps:
    """The illumination decomposition of one photograph.

    Attributes:
        shading: ``(H, W)`` ``uint8`` low-frequency illumination (Requirement
            7.2).
        detail: ``(H, W)`` ``uint8`` high-frequency residual centred on
            :data:`NEUTRAL_DETAIL` (Requirement 7.3).
        plane_medians: median of ``shading`` over each plane's mask
            (Requirement 7.4). A plane whose mask selects no pixels is absent
            from the mapping rather than present with a placeholder, so a caller
            cannot mistake "no data" for a real neutral point.
    """

    shading: np.ndarray
    detail: np.ndarray
    plane_medians: dict[PlaneName, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Shared validation
# --------------------------------------------------------------------------- #


def _as_l_channel(array: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a single-channel ``uint8`` plane and return it as an ndarray.

    Every stage after :func:`to_lab_l` consumes an 8-bit single-channel map, and
    a float or three-channel array slipped in here would otherwise fail deep
    inside OpenCV with an opaque message.
    """
    plane = np.asarray(array)
    if plane.ndim != 2:
        raise ValueError(f"expected a 2-D {name}, got shape {plane.shape!r}")
    if plane.size == 0:
        raise ValueError(f"expected a non-empty {name}")
    if plane.dtype != np.uint8:
        raise ValueError(f"expected a uint8 {name}, got dtype {plane.dtype!r}")
    return plane


# --------------------------------------------------------------------------- #
# Stage 1 -- L* extraction
# --------------------------------------------------------------------------- #


def to_lab_l(image_bgr: np.ndarray) -> np.ndarray:
    """Isolate the ``L*`` channel of ``image_bgr``. R7.1

    Converts through ``cv2.COLOR_BGR2LAB`` and returns channel 0 in OpenCV's
    8-bit scaling of ``L*``, where perceptual 0-100 lightness is mapped onto
    0-255. The scaling is kept rather than rescaled to 0-100 so the residual in
    :func:`high_frequency` and the blend arithmetic downstream stay in the same
    8-bit domain as the tile pixels they modulate.

    Args:
        image_bgr: ``(H, W, 3)`` ``uint8`` BGR image.

    Returns:
        A fresh contiguous ``(H, W)`` ``uint8`` array.

    Raises:
        ValueError: the input is not a non-empty ``(H, W, 3)`` ``uint8`` image.
    """
    array = np.asarray(image_bgr)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) BGR image, got shape {array.shape!r}")
    if array.size == 0:
        raise ValueError("expected a non-empty image")
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got dtype {array.dtype!r}")

    lab = cv2.cvtColor(np.ascontiguousarray(array), cv2.COLOR_BGR2LAB)
    # `lab[:, :, 0]` is a strided view over the interleaved buffer; copy so the
    # caller owns a compact plane and the three-channel temporary can be freed.
    return np.ascontiguousarray(lab[:, :, 0])


# --------------------------------------------------------------------------- #
# Stage 2 -- low-frequency shading map
# --------------------------------------------------------------------------- #


def _odd_at_most(value: int, limit: int) -> int:
    """Largest odd integer that is both ``<= value`` and ``<= limit``, min 1."""
    size = min(int(value), int(limit))
    if size < 1:
        return 1
    return size if size % 2 == 1 else size - 1


def _border_safe_extent(shape: tuple[int, int]) -> int:
    """Largest kernel extent OpenCV can reflect-pad ``shape`` against.

    ``BORDER_REFLECT_101`` requires the pad on each side to be strictly smaller
    than the corresponding image dimension, so a kernel of ``2 * min(H, W) - 1``
    is the widest one a filter can apply without OpenCV raising. Filters here
    are square, hence the shorter edge governs.
    """
    return 2 * min(shape[0], shape[1]) - 1


def low_frequency(l_channel: np.ndarray, settings: Settings | None = None) -> np.ndarray:
    """Low-frequency shading map of ``l_channel``. R7.2

    Two passes, in this order:

    1. A bilateral pass (``d=9``, ``sigmaColor`` tied to the ``L*`` standard
       deviation, ``sigmaSpace`` from ``settings.shading_sigma_px``), skipped
       when ``settings.use_bilateral_shading`` is false. It runs *first* because
       it smooths within a shadow without smoothing across its boundary; a lone
       Gaussian would drag a hard shadow edge over the surrounding surface and
       the smear reads as a blur artifact once tiles are composited under it.
    2. A large-sigma Gaussian blur, which erases the surface texture the
       bilateral pass deliberately preserved and leaves only the illumination
       envelope. Running it last also makes this function a genuine low-pass:
       the returned map is never sharper than its source.

    Kernel sizes are clamped to what the image can be reflect-padded against, so
    a thumbnail-sized input blurs with the widest kernel that fits instead of
    raising. Turning the bilateral pass off costs edge fidelity but roughly
    halves the stage's runtime, which is the tradeoff Requirement 7.2 leaves to
    the operator.

    Args:
        l_channel: ``(H, W)`` ``uint8`` ``L*`` channel from :func:`to_lab_l`.
        settings: source of ``shading_sigma_px`` and ``use_bilateral_shading``;
            defaults to :func:`get_settings`.

    Returns:
        A ``(H, W)`` ``uint8`` array with the same shape as ``l_channel``.

    Raises:
        ValueError: the input is not a non-empty 2-D ``uint8`` array.
    """
    source = _as_l_channel(l_channel, name="l_channel")
    cfg = settings or get_settings()
    sigma_px = float(cfg.shading_sigma_px)
    max_extent = _border_safe_extent(source.shape)

    smoothed = source
    if cfg.use_bilateral_shading:
        diameter = _odd_at_most(_BILATERAL_DIAMETER, max_extent)
        if diameter >= 3:
            sigma_color = max(float(source.std()), _MIN_SIGMA_COLOR)
            smoothed = cv2.bilateralFilter(
                smoothed,
                d=diameter,
                sigmaColor=sigma_color,
                sigmaSpace=sigma_px,
            )

    kernel = _odd_at_most(int(round(_KERNEL_SIGMAS * sigma_px)) | 1, max_extent)
    if kernel >= 3:
        smoothed = cv2.GaussianBlur(smoothed, (kernel, kernel), sigmaX=sigma_px, sigmaY=sigma_px)

    # A no-op filter chain (a 1x1 image, or a kernel clamped below 3) would
    # otherwise hand back the caller's own buffer.
    if smoothed is source:
        smoothed = source.copy()
    return smoothed


# --------------------------------------------------------------------------- #
# Stage 3 -- high-frequency detail map
# --------------------------------------------------------------------------- #


def high_frequency(l_channel: np.ndarray, shading: np.ndarray) -> np.ndarray:
    """Detail map: the ``L*`` residual biased onto :data:`NEUTRAL_DETAIL`. R7.3

    ``clip(int16(L*) - int16(shading) + 128, 0, 255)``. The subtraction is done
    in ``int16`` because the residual is signed and ranges over [-255, 255]; the
    128 bias then places "no local variation" at 128 and lets the result cache
    as ``uint8`` (Requirement 12.4). ``shading + (detail - 128)`` recovers ``L*``
    exactly wherever the residual stayed inside [-128, 127], which is the
    reconstruction relation the property tests assert.

    Args:
        l_channel: ``(H, W)`` ``uint8`` ``L*`` channel.
        shading: ``(H, W)`` ``uint8`` shading map of the same shape.

    Returns:
        A ``(H, W)`` ``uint8`` array centred on :data:`NEUTRAL_DETAIL`.

    Raises:
        ValueError: either input is not a non-empty 2-D ``uint8`` array, or
            their shapes differ.
    """
    source = _as_l_channel(l_channel, name="l_channel")
    low = _as_l_channel(shading, name="shading")
    if source.shape != low.shape:
        raise ValueError(
            f"l_channel shape {source.shape!r} does not match shading shape {low.shape!r}"
        )

    residual = source.astype(np.int16)
    residual -= low.astype(np.int16)
    residual += NEUTRAL_DETAIL
    return np.clip(residual, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Full decomposition
# --------------------------------------------------------------------------- #


def _mask_median(shading: np.ndarray, mask: np.ndarray, *, plane: str) -> float | None:
    """Median of ``shading`` over ``mask``, or ``None`` when nothing is set."""
    binary = np.asarray(mask)
    if binary.ndim != 2:
        raise ValueError(f"expected a 2-D mask for plane {plane!r}, got shape {binary.shape!r}")
    if binary.shape != shading.shape:
        raise ValueError(
            f"plane {plane!r} mask shape {binary.shape!r} does not match "
            f"shading shape {shading.shape!r}"
        )
    selected = shading[binary > 0]
    if selected.size == 0:
        return None
    return float(np.median(selected))


def decompose(
    image_bgr: np.ndarray,
    plane_masks: Mapping[PlaneName, np.ndarray] | None = None,
    settings: Settings | None = None,
) -> LightingMaps:
    """Split ``image_bgr`` into shading, detail, and per-plane medians.

    Runs the three stages above once over the whole frame -- the maps are
    frame-global because the Compositor may be asked for any subset of planes on
    any later render, and re-deriving them per request would put a
    seconds-scale cost inside the render budget (Requirement 9.3).

    Args:
        image_bgr: ``(H, W, 3)`` ``uint8`` BGR photograph.
        plane_masks: Structural_Plane masks; any non-zero pixel counts as set.
            Planes selecting no pixels are omitted from
            :attr:`LightingMaps.plane_medians`.
        settings: settings for the shading stage; defaults to
            :func:`get_settings`.

    Returns:
        A :class:`LightingMaps` whose maps are both ``(H, W)`` ``uint8``.

    Raises:
        ValueError: the image is not a non-empty ``(H, W, 3)`` ``uint8`` image,
            or a mask's shape does not match it.
    """
    l_channel = to_lab_l(image_bgr)
    shading = low_frequency(l_channel, settings)
    detail = high_frequency(l_channel, shading)

    medians: dict[PlaneName, float] = {}
    for plane, mask in (plane_masks or {}).items():
        median = _mask_median(shading, mask, plane=str(plane))
        if median is not None:
            medians[plane] = median

    return LightingMaps(shading=shading, detail=detail, plane_medians=medians)
