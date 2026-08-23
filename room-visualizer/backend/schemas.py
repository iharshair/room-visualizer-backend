"""Shared data contracts for the AI Room & Tile Visualizer.

Two families of types live here:

* **Internal dataclasses** (`PlaneMetadata`, `SceneState`, `TileDefinition`,
  `PlaneRenderSpec`) carry numpy arrays between the pipeline modules. They are
  never serialised directly; they use ``slots=True`` because a Scene_State holds
  tens of megabytes of arrays and thousands of them may be constructed over a
  process lifetime.
* **Pydantic models** describe the HTTP surface exactly as documented in the
  design's API Contracts section, including the single error envelope every
  failure path returns.

Requirements: 1.3, 8.3, 9.1, 12.3, 12.4, 12.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "PlaneName",
    "GeometryMode",
    "SegmentationBackend",
    "RenderFormat",
    "PLANE_NAMES",
    "PlaneMetadata",
    "SceneState",
    "TileDefinition",
    "PlaneRenderSpec",
    "HorizonResponse",
    "PlaneResponse",
    "SegmentResponse",
    "RenderRequest",
    "RenderResponse",
    "TileResponse",
    "TilesResponse",
    "HealthResponse",
    "ErrorDetail",
    "ErrorEnvelope",
]

# --------------------------------------------------------------------------- #
# Shared literals
# --------------------------------------------------------------------------- #

PlaneName = Literal["floor", "wall_left", "wall_right", "wall_back"]
GeometryMode = Literal["vanishing_points", "planar_fallback"]
SegmentationBackend = Literal["mobilesam-onnx", "classical"]
RenderFormat = Literal["png", "jpeg"]

#: Every Structural_Plane name, in the order the design uses for reporting.
PLANE_NAMES: tuple[PlaneName, ...] = ("floor", "wall_left", "wall_right", "wall_back")


# --------------------------------------------------------------------------- #
# Internal dataclasses
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class PlaneMetadata:
    """Per-plane analysis result carried in a Scene_State.

    ``bounding_points`` is deliberately a four-point quad rather than the full
    contour: the frontend uses it for cheap point-in-quad hit-testing and for
    drawing a selection outline, while ``contour`` drives the filled highlight
    overlay. Both are reported by ``/api/segment`` (Requirement 1.3).
    """

    name: PlaneName
    contour: np.ndarray  # (N,2) int32, simplified, image pixels        R3.6
    bounding_points: np.ndarray  # (4,2) int32, convex quad for UI hit-test R1.3
    area_fraction: float  # mask pixels / (H*W), in (0,1]               R3.6
    centroid: tuple[float, float]
    homography: np.ndarray  # (3,3) float64, p_img ~ H @ p_plane        R5.4
    homography_inv: np.ndarray  # (3,3) float64, cached inverse
    plane_extent_mm: tuple[float, float, float, float]  # xmin,ymin,xmax,ymax
    reprojection_rmse_px: float  # R5.6
    geometry_mode: GeometryMode  # R6.3
    luminance_median: float  # plane median of L*, for blend selection   R7.4


@dataclass(slots=True)
class SceneState:
    """Cached analysis artifacts for one uploaded photograph.

    Every mask and lighting map is stored at 8-bit precision per channel
    (Requirement 12.4). ``detail_map`` uses 128 as its neutral midpoint so a
    signed high-frequency residual fits in ``uint8`` without a sign array.
    """

    scene_id: str  # uuid4 hex
    created_at: float  # wall clock at creation, for TTL
    image: np.ndarray  # (H,W,3) uint8 BGR, post-downscale original
    width: int
    height: int
    planes: dict[PlaneName, PlaneMetadata]
    plane_masks: dict[PlaneName, np.ndarray]  # (H,W) uint8 {0,255}  R3.1, R12.4
    foreground_mask: np.ndarray  # (H,W) uint8 {0,255}               R3.2, R12.4
    shading_map: np.ndarray  # (H,W) uint8                           R7.2, R12.4
    detail_map: np.ndarray  # (H,W) uint8, 128 = neutral             R7.3
    horizon: tuple[float, float, float]  # homogeneous line (a,b,c)  R5.3
    vanishing_points: dict[str, tuple[float, float] | None]  # VPx, VPy, VPz R5.2
    geometry_mode: GeometryMode
    segmentation_backend: SegmentationBackend  # R4.6
    #: Feathered per-plane composite alpha, populated by the Compositor on the
    #: first render that touches each plane and reused by every later render of
    #: this scene. It is derived state rather than analysis output -- the
    #: distance transform behind it costs 5-15 ms per plane, which Requirement
    #: 9.3's budget does not have room to pay twice -- so it is `float32` rather
    #: than 8-bit and is deliberately excluded from the 8-bit cached-artifact
    #: contract of Requirement 12.4, which is stated over masks and lighting
    #: maps. :meth:`release` clears it along with everything else, so it cannot
    #: outlive its scene.
    plane_alpha: dict[PlaneName, np.ndarray] = field(default_factory=dict)

    def nbytes(self) -> int:
        """Total bytes held by this state's arrays.

        Arrays are optional at runtime because :meth:`release` nulls them, so a
        released state reports 0.
        """
        total = 0
        for array in (
            self.image,
            self.foreground_mask,
            self.shading_map,
            self.detail_map,
        ):
            if array is not None:
                total += int(array.nbytes)
        for mask in self.plane_masks.values():
            if mask is not None:
                total += int(mask.nbytes)
        for alpha in self.plane_alpha.values():
            if alpha is not None:
                total += int(alpha.nbytes)
        for plane in self.planes.values():
            for array in (
                plane.contour,
                plane.bounding_points,
                plane.homography,
                plane.homography_inv,
            ):
                if array is not None:
                    total += int(array.nbytes)
        return total

    def release(self) -> None:
        """Drop every array reference so eviction frees memory promptly. R12.3

        Sets each array attribute to ``None`` and empties the mask, plane, and
        composite-alpha mappings, so the last strong reference to every array
        held by this state goes away as soon as the cache drops the state
        itself. Idempotent: calling it twice is harmless. A released state must
        not be rendered from.
        """
        self.image = None  # type: ignore[assignment]
        self.foreground_mask = None  # type: ignore[assignment]
        self.shading_map = None  # type: ignore[assignment]
        self.detail_map = None  # type: ignore[assignment]
        self.plane_masks.clear()
        self.plane_alpha.clear()
        self.planes.clear()


@dataclass(frozen=True, slots=True)
class TileDefinition:
    """One validated Asset_Catalog entry (Requirement 8.3)."""

    id: str
    name: str
    image_path: Path
    width_mm: float  # R8.3
    height_mm: float  # R8.3
    finish: str  # R8.3
    gloss: float  # 0.0-1.0, R8.3
    grout_mm: float | None = None


@dataclass(slots=True)
class PlaneRenderSpec:
    """Per-plane tile selection inside a Render_Request (Requirement 9.1).

    ``grout_mm`` and ``grout_rgb`` of ``None`` mean "inherit": the Compositor
    falls back to the Tile_Definition value and then to the configured
    ``default_grout_mm`` / ``default_grout_rgb`` settings.
    """

    tile_id: str
    rotation_deg: float = 0.0  # rotation within metric plane space
    grout_mm: Annotated[float, Field(ge=0.0)] | None = None
    grout_rgb: tuple[
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
        Annotated[int, Field(ge=0, le=255)],
    ] | None = None
    offset_mm: tuple[float, float] = (0.0, 0.0)


# --------------------------------------------------------------------------- #
# HTTP models — POST /api/segment
# --------------------------------------------------------------------------- #


class HorizonResponse(BaseModel):
    """Estimated horizon as a homogeneous line plus a convenience sample."""

    a: float
    b: float
    c: float
    y_at_center: float


class PlaneResponse(BaseModel):
    """One detected Structural_Plane. Undetected planes are omitted. R3.5"""

    name: PlaneName
    area_fraction: float = Field(gt=0.0, le=1.0)
    contour: list[list[int]] = Field(min_length=3)
    bounding_points: list[list[int]] = Field(min_length=4, max_length=4)
    centroid: tuple[float, float]
    reprojection_rmse_px: float = Field(ge=0.0)


class SegmentResponse(BaseModel):
    """Analysis result for one uploaded photograph (Requirement 1.3)."""

    scene_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    segmentation_backend: SegmentationBackend
    geometry_mode: GeometryMode
    horizon: HorizonResponse
    vanishing_points: dict[str, tuple[float, float] | None]
    planes: list[PlaneResponse]
    analysis_ms: int = Field(ge=0)


# --------------------------------------------------------------------------- #
# HTTP models — POST /api/render
# --------------------------------------------------------------------------- #


class RenderRequest(BaseModel):
    """Tile selections to composite onto a cached scene (Requirement 9.1).

    Planes absent from ``planes`` keep their original photographic appearance.
    ``format`` of ``None`` means "use the configured ``render_format``".
    """

    scene_id: str = Field(min_length=1)
    planes: dict[PlaneName, PlaneRenderSpec] = Field(default_factory=dict)
    format: RenderFormat | None = None


class RenderResponse(BaseModel):
    """Composited image, base64-encoded in the JSON body."""

    scene_id: str
    mime: str
    image: str  # base64-encoded encoded image bytes
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    render_ms: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# HTTP models — GET /api/tiles
# --------------------------------------------------------------------------- #


class TileResponse(BaseModel):
    """One catalog entry as published by ``/api/tiles`` (Requirement 8.4)."""

    id: str
    name: str
    width_mm: float = Field(gt=0.0)
    height_mm: float = Field(gt=0.0)
    finish: str = Field(min_length=1)
    gloss: float = Field(ge=0.0, le=1.0)
    thumbnail_url: str


class TilesResponse(BaseModel):
    tiles: list[TileResponse] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# HTTP models — GET /api/health
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    """Liveness plus the operator-facing runtime facts (Requirement 12.5)."""

    status: Literal["ok"] = "ok"
    segmentation_backend: SegmentationBackend
    onnx_provider: str  # provider name, or "n/a" when the neural backend is off
    scene_cache_entries: int = Field(ge=0)
    scene_cache_max_entries: int = Field(gt=0)
    scene_cache_ttl_seconds: int = Field(gt=0)


# --------------------------------------------------------------------------- #
# Shared error envelope
# --------------------------------------------------------------------------- #


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    """The one body shape every failure returns: ``{"error": {...}}``.

    Requirements 1.6 and 10.6: the frontend renders ``message`` verbatim and
    branches on ``code``, so no endpoint needs a bespoke error shape.
    """

    error: ErrorDetail

    @classmethod
    def of(cls, code: str, message: str) -> "ErrorEnvelope":
        """Build an envelope from a code and a human-readable message."""
        return cls(error=ErrorDetail(code=code, message=message))
