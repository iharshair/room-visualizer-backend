"""Analytic synthetic perspective room generator (Requirements 11.4, 13.2).

Every geometry, lighting, and compositing test needs an input whose correct
answer is known in closed form. This module renders one: a rectangular room in
millimetre world coordinates, photographed by a pinhole camera with known
intrinsics and pose, with a checkerboard floor, up to three checkerboard walls,
and solid boxes standing on the floor as occluders.

Because the camera is exact, the ground truth is *computed* rather than
*measured*:

* vanishing points are ``K @ R @ d`` for each world axis direction ``d``;
* the horizon is the image of the ground plane's line at infinity,
  ``K^-T @ R @ n`` for the floor normal ``n``;
* each plane homography is ``K [r_u | r_v | t_O]`` for that plane's metric
  frame, so ``p_img ~ H @ [u_mm, v_mm, 1]`` holds to floating point.

That exactness is what gives Requirement 13.3's pixel tolerance and Requirement
13.4's 1.0 px round-trip bound a meaningful reference.

Coordinate conventions
----------------------
World axes follow the camera convention so no reflection is ever needed:
**X right, Y down, Z forward (depth)**, in millimetres. The floor is the plane
``Y = 0`` and the camera centre sits at ``(0, -camera_height_mm, 0)``, i.e.
``camera_height_mm`` above the floor. ``pitch_deg`` is the camera elevation, so
the default ``-12`` looks downward; ``yaw_deg`` turns the camera to its right.

Per-plane metric frames match the design's plane frame table --- floor
``u = +X`` / ``v = +Z``, side walls ``u = +Z`` / ``v = up``, back wall
``u = +X`` / ``v = up`` --- with each origin at the plane's near-lower corner,
so metric coordinates inside ``plane_extents_mm`` are non-negative.

The Setup_Tool (``scripts/setup_assets.py``) imports this same module through
:func:`write_room`, so the shipped sample image and the test fixture cannot
drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

__all__ = [
    "CameraParams",
    "SyntheticRoom",
    "make_synthetic_room",
    "write_room",
    "WALL_KEYS",
    "WALL_TO_PLANE",
]

#: Accepted members of the ``walls`` argument, in reporting order.
WALL_KEYS: tuple[str, ...] = ("left", "right", "back")

#: Mapping from a ``walls`` member to its Structural_Plane name.
WALL_TO_PLANE: dict[str, str] = {
    "left": "wall_left",
    "right": "wall_right",
    "back": "wall_back",
}

# Overlap priority from the design's plane invariant pass. Adjacent planes share
# a room edge, so a boundary pixel has to be awarded to exactly one of them.
_PLANE_PRIORITY: tuple[str, ...] = ("floor", "wall_back", "wall_left", "wall_right")

# Draw order, lowest priority first, so the same plane wins a shared boundary
# pixel in the rendered image as in :meth:`SyntheticRoom.plane_masks`.
_DRAW_ORDER: tuple[str, ...] = ("ceiling",) + tuple(reversed(_PLANE_PRIORITY))

# Surface colours as (light, dark) BGR pairs. The floor carries strong
# checkerboard contrast because the line detector needs long, crisp edges toward
# two horizontal vanishing points; the walls carry a subtler grid so each wall
# still reads as one dominant colour cluster to the classical segmenter.
_SURFACE_COLOURS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "floor": ((208, 202, 194), (158, 152, 145)),
    "wall_left": ((176, 186, 198), (156, 166, 178)),
    "wall_right": ((190, 199, 210), (172, 181, 192)),
    "wall_back": ((198, 204, 209), (180, 186, 191)),
    "ceiling": ((238, 239, 240), (238, 239, 240)),
}

# Background fill for any pixel no surface covers (only reachable with an
# extreme field of view). Neutral, so it never reads as a structural plane.
_BACKGROUND_BGR: tuple[int, int, int] = (232, 233, 234)

# Camera-space near plane in millimetres. Surface points closer than this always
# project far outside the image, so clipping here changes no visible pixel while
# keeping the projected polygon coordinates numerically tame.
_NEAR_MM = 100.0

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CameraParams:
    """The exact pinhole camera the room was photographed with.

    ``R`` and ``t`` map world to camera (``p_cam = R @ p_world + t``), which is
    the convention every ground-truth quantity in :class:`SyntheticRoom` is
    derived under.
    """

    width: int
    height: int
    focal_px: float
    principal_point: tuple[float, float]
    yaw_deg: float
    pitch_deg: float
    camera_height_mm: float
    K: np.ndarray  # (3,3) float64 intrinsics
    R: np.ndarray  # (3,3) float64 world -> camera rotation
    t: np.ndarray  # (3,)  float64 world -> camera translation
    position_mm: np.ndarray  # (3,) float64 camera centre in world millimetres

    def projection_matrix(self) -> np.ndarray:
        """``K [R | t]``, the full 3x4 world-to-image projection."""
        return self.K @ np.column_stack((self.R, self.t))

    def to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """Transform ``(N,3)`` world millimetre points into camera space."""
        pts = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        return pts @ self.R.T + self.t

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project ``(N,3)`` world points to ``(N,2)`` image pixels.

        Points at or behind the camera plane are returned as NaN rather than
        silently wrapping to a mirrored position.
        """
        cam = self.to_camera(points_world)
        img = cam @ self.K.T
        z = img[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = img[:, :2] / z[:, None]
        out[cam[:, 2] <= 0.0] = np.nan
        return out

    def vanishing_point(self, direction: Sequence[float]) -> tuple[float, float] | None:
        """Image of the point at infinity along a world ``direction``.

        Returns ``None`` when the direction is parallel to the image plane, i.e.
        when its vanishing point genuinely lies at infinity.
        """
        v = self.K @ self.R @ np.asarray(direction, dtype=np.float64)
        if abs(v[2]) < 1e-9 * max(1.0, float(np.abs(v[:2]).max())):
            return None
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable view, used by :func:`write_room`."""
        return {
            "width": self.width,
            "height": self.height,
            "focal_px": self.focal_px,
            "principal_point": list(self.principal_point),
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
            "camera_height_mm": self.camera_height_mm,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "position_mm": self.position_mm.tolist(),
        }


# --------------------------------------------------------------------------- #
# Room
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SyntheticRoom:
    """A rendered room plus its analytic ground truth.

    Keys of ``truth_vps`` are ``VPx`` (horizontal, along world X), ``VPz``
    (horizontal, along the depth axis), and ``VPy`` (vertical), matching
    ``Calibration.vanishing_points``. A value is ``None`` only for a pose whose
    vanishing point is genuinely at infinity (for example ``VPy`` at zero
    pitch), so callers must handle ``None`` exactly as they do for a recovered
    calibration.

    ``truth_horizon`` is the homogeneous line ``(a, b, c)`` normalised to
    ``a^2 + b^2 = 1``, so ``a*x + b*y + c`` is a signed pixel distance.

    ``plane_extents_mm`` gives each plane's ``(u_min, v_min, u_max, v_max)``
    metric box over the *visible* part of that surface, so every point inside an
    extent projects into the frame from in front of the camera.

    ``plane_polygons`` holds each visible plane's exact projected outline as
    ``(N,2) float32`` image coordinates, already clipped to the image
    rectangle. Planes that fall outside the frame are omitted, as are walls not
    requested. Occluders are *not* subtracted --- use :meth:`plane_mask` for the
    visible surface, and ``occluder_mask`` for the foreground itself.
    """

    image: np.ndarray  # (H,W,3) uint8 BGR
    truth_vps: dict[str, tuple[float, float] | None]
    truth_horizon: tuple[float, float, float]
    truth_homographies: dict[str, np.ndarray]  # (3,3) float64, p_img ~ H @ [u,v,1]
    plane_polygons: dict[str, np.ndarray]  # (N,2) float32 image polygons
    occluder_mask: np.ndarray  # (H,W) uint8 {0,255}
    camera: CameraParams
    plane_extents_mm: dict[str, tuple[float, float, float, float]]  # u0,v0,u1,v1
    tile_mm: float
    walls: tuple[str, ...]

    # -- derived views ----------------------------------------------------- #

    @property
    def shape(self) -> tuple[int, int]:
        """``(height, width)`` of the rendered image."""
        return self.image.shape[0], self.image.shape[1]

    def plane_names(self) -> tuple[str, ...]:
        """Visible Structural_Plane names, in the design's priority order."""
        return tuple(name for name in _PLANE_PRIORITY if name in self.plane_polygons)

    def plane_mask(self, name: str, *, subtract_occluders: bool = True) -> np.ndarray:
        """Rasterise one plane polygon as a ``(H,W) uint8`` ``{0,255}`` mask.

        By default the occluders are removed, so the result is the plane's
        *visible* surface --- the thing a Segmenter could plausibly return.
        """
        if name not in self.plane_polygons:
            raise KeyError(f"plane {name!r} is not visible in this room")
        mask = np.zeros(self.shape, dtype=np.uint8)
        polygon = np.round(self.plane_polygons[name]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
        if subtract_occluders:
            mask[self.occluder_mask > 0] = 0
        return mask

    def plane_masks(
        self, *, subtract_occluders: bool = True, resolve_overlaps: bool = True
    ) -> dict[str, np.ndarray]:
        """Every visible plane's mask, keyed by plane name.

        Adjacent planes share a room edge, so rasterising their polygons
        independently double-counts the boundary pixels. With
        ``resolve_overlaps`` those pixels are awarded by the same
        ``floor > wall_back > wall_left > wall_right`` priority the Segmenter's
        invariant pass uses, making the returned masks a genuine partition.
        """
        masks = {
            name: self.plane_mask(name, subtract_occluders=subtract_occluders)
            for name in self.plane_names()
        }
        if resolve_overlaps:
            claimed = np.zeros(self.shape, dtype=bool)
            for name in _PLANE_PRIORITY:
                mask = masks.get(name)
                if mask is None:
                    continue
                mask[claimed] = 0
                claimed |= mask > 0
        return masks

    def horizon_y_at(self, x: float) -> float:
        """Horizon row at image column ``x``, from the ground-truth line."""
        a, b, c = self.truth_horizon
        if abs(b) < _EPS:
            raise ValueError("horizon is vertical; y is undefined")
        return float(-(a * x + c) / b)

    def truth_dict(self) -> dict[str, object]:
        """JSON-serialisable ground truth, used by :func:`write_room`."""
        return {
            "generator": "tests/fixtures/synthetic.py",
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "tile_mm": self.tile_mm,
            "walls": list(self.walls),
            "camera": self.camera.to_dict(),
            "truth_vps": {
                key: (list(value) if value is not None else None)
                for key, value in self.truth_vps.items()
            },
            "truth_horizon": list(self.truth_horizon),
            "truth_homographies": {
                name: matrix.tolist() for name, matrix in self.truth_homographies.items()
            },
            "plane_polygons": {
                name: polygon.astype(float).tolist()
                for name, polygon in self.plane_polygons.items()
            },
            "plane_extents_mm": {
                name: list(extent) for name, extent in self.plane_extents_mm.items()
            },
        }


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _rotation_x(radians: float) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rotation_y(radians: float) -> np.ndarray:
    c, s = np.cos(radians), np.sin(radians)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _intrinsics(focal_px: float, width: int, height: int) -> np.ndarray:
    """Intrinsics with a centred principal point, as the Geometry_Engine assumes."""
    return np.array(
        [
            [focal_px, 0.0, (width - 1) / 2.0],
            [0.0, focal_px, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _clip_by_signed_distance(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a convex polygon to ``distance >= 0``.

    Works for both 2-D and 3-D vertices; the crossing point is linear in the
    vertex coordinates either way, which is what lets the same routine clip
    against the camera near plane and against the image border.
    """
    kept: list[np.ndarray] = []
    count = len(points)
    for i in range(count):
        j = (i + 1) % count
        d_i, d_j = float(distance[i]), float(distance[j])
        if d_i >= 0.0:
            kept.append(points[i])
        if (d_i > 0.0 and d_j < 0.0) or (d_i < 0.0 and d_j > 0.0):
            ratio = d_i / (d_i - d_j)
            kept.append(points[i] + ratio * (points[j] - points[i]))
    if not kept:
        return np.empty((0, points.shape[1]), dtype=np.float64)
    return np.asarray(kept, dtype=np.float64)


def _clip_near(points_cam: np.ndarray, near_mm: float) -> np.ndarray:
    """Clip a camera-space polygon to the half-space in front of the camera."""
    if len(points_cam) < 3:
        return np.empty((0, 3), dtype=np.float64)
    return _clip_by_signed_distance(points_cam, points_cam[:, 2] - near_mm)


def _clip_to_image(points_img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip an image-space polygon to ``[0, width-1] x [0, height-1]``.

    Clipping after projection is valid because a perspective projection maps
    lines to lines, so the clipped polygon is still the exact outline of the
    visible surface.
    """
    poly = np.asarray(points_img, dtype=np.float64)
    borders = (
        (1.0, 0.0, 0.0),  # x >= 0
        (-1.0, 0.0, float(width - 1)),  # x <= width-1
        (0.0, 1.0, 0.0),  # y >= 0
        (0.0, -1.0, float(height - 1)),  # y <= height-1
    )
    for a, b, c in borders:
        if len(poly) < 3:
            return np.empty((0, 2), dtype=np.float64)
        poly = _clip_by_signed_distance(poly, a * poly[:, 0] + b * poly[:, 1] + c)
    return _drop_degenerate(poly)


def _drop_degenerate(poly: np.ndarray) -> np.ndarray:
    """Remove duplicate consecutive vertices and reject sliver polygons."""
    if len(poly) < 3:
        return np.empty((0, 2), dtype=np.float64)
    keep = [poly[0]]
    for point in poly[1:]:
        if np.hypot(*(point - keep[-1])) > 1e-6:
            keep.append(point)
    if len(keep) >= 2 and np.hypot(*(keep[0] - keep[-1])) <= 1e-6:
        keep.pop()
    if len(keep) < 3:
        return np.empty((0, 2), dtype=np.float64)
    out = np.asarray(keep, dtype=np.float64)
    area = 0.5 * abs(
        float(
            np.dot(out[:, 0], np.roll(out[:, 1], -1))
            - np.dot(out[:, 1], np.roll(out[:, 0], -1))
        )
    )
    if area < 1.0:  # under one square pixel: nothing to rasterise
        return np.empty((0, 2), dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Surface description
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Surface:
    """One planar quad with its metric frame.

    ``origin`` is the metric ``(0, 0)`` corner in world millimetres, and the
    quad spans ``u`` over ``[0, u_extent]`` and ``v`` over ``[0, v_extent]``.
    """

    name: str
    origin: np.ndarray
    u_dir: np.ndarray
    v_dir: np.ndarray
    u_extent: float
    v_extent: float

    def corners(self) -> np.ndarray:
        """The quad's four world corners, in metric frame order."""
        return np.array(
            [
                self.origin,
                self.origin + self.u_dir * self.u_extent,
                self.origin + self.u_dir * self.u_extent + self.v_dir * self.v_extent,
                self.origin + self.v_dir * self.v_extent,
            ],
            dtype=np.float64,
        )


def _surfaces(
    *,
    walls: Sequence[str],
    half_width_mm: float,
    depth_near_mm: float,
    depth_far_mm: float,
    height_mm: float,
) -> list[_Surface]:
    """Build the room's quads, following the design's plane frame table."""
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, -1.0, 0.0])  # Y points down, so "up" is negative Y
    forward = np.array([0.0, 0.0, 1.0])
    depth_span = depth_far_mm - depth_near_mm
    width_span = 2.0 * half_width_mm

    out = [
        # Floor: u along +X, v along +Z (depth), origin at the near-left corner.
        _Surface(
            "floor",
            np.array([-half_width_mm, 0.0, depth_near_mm]),
            right,
            forward,
            width_span,
            depth_span,
        ),
        # Ceiling is background only: it is never a Structural_Plane, it just
        # keeps the top of the frame from showing bare fill when the field of
        # view reaches above the wall tops.
        _Surface(
            "ceiling",
            np.array([-half_width_mm, -height_mm, depth_near_mm]),
            right,
            forward,
            width_span,
            depth_span,
        ),
    ]
    if "left" in walls:
        out.append(
            _Surface(
                "wall_left",
                np.array([-half_width_mm, 0.0, depth_near_mm]),
                forward,
                up,
                depth_span,
                height_mm,
            )
        )
    if "right" in walls:
        out.append(
            _Surface(
                "wall_right",
                np.array([half_width_mm, 0.0, depth_near_mm]),
                forward,
                up,
                depth_span,
                height_mm,
            )
        )
    if "back" in walls:
        out.append(
            _Surface(
                "wall_back",
                np.array([-half_width_mm, 0.0, depth_far_mm]),
                right,
                up,
                width_span,
                height_mm,
            )
        )
    return out


def _homography(K: np.ndarray, R: np.ndarray, t: np.ndarray, surface: _Surface) -> np.ndarray:
    """``K [r_u | r_v | t_O]`` for one metric plane frame.

    Scaled to unit Frobenius norm, and signed so the plane centre maps to a
    positive homogeneous ``w``. Both are pure conventions --- ``p_img ~ H p`` is
    projective --- but they keep the matrix numerically tame and make the
    inverse mapping's ``w`` positive across the visible surface, matching what
    the Compositor's inverse warp expects.
    """
    H = K @ np.column_stack(
        (R @ surface.u_dir, R @ surface.v_dir, R @ surface.origin + t)
    )
    norm = float(np.linalg.norm(H))
    if norm < _EPS:  # pragma: no cover - a degenerate frame is a programming error
        raise ValueError(f"degenerate plane frame for {surface.name!r}")
    H = H / norm
    centre_w = float(
        H[2, 0] * (surface.u_extent / 2.0) + H[2, 1] * (surface.v_extent / 2.0) + H[2, 2]
    )
    if centre_w < 0.0:
        H = -H
    return H


# --------------------------------------------------------------------------- #
# Rasterisation
# --------------------------------------------------------------------------- #


def _metric_coordinates(
    homography_inv: np.ndarray, x0: int, y0: int, cols: int, rows: int
) -> tuple[np.ndarray, np.ndarray]:
    """Metric ``(u_mm, v_mm)`` for every pixel of an image bounding box.

    Broadcast against a row vector and a column vector rather than a full
    meshgrid, and accumulate in ``float32``: at the supersampled sizes this
    generator renders, the memory traffic of the intermediate grids is what
    dominates its runtime. ``float32`` still resolves a millimetre position to
    roughly a micron, orders of magnitude finer than a tile edge.

    The homogeneous divisor is clamped away from zero, so pixels on the plane's
    horizon (where the divisor vanishes) yield a large finite coordinate instead
    of a NaN. Those pixels always lie outside the plane polygon, so the value
    they take is never painted.
    """
    xs = np.arange(x0, x0 + cols, dtype=np.float32)[None, :]
    ys = np.arange(y0, y0 + rows, dtype=np.float32)[:, None]
    inv = homography_inv.astype(np.float32)
    u = (inv[0, 0] * xs) + (inv[0, 1] * ys + inv[0, 2])
    v = (inv[1, 0] * xs) + (inv[1, 1] * ys + inv[1, 2])
    w = (inv[2, 0] * xs) + (inv[2, 1] * ys + inv[2, 2])
    w[np.abs(w) < 1e-12] = 1e-12
    u /= w
    v /= w
    return u, v


def _fill_checkerboard(
    canvas: np.ndarray,
    polygon: np.ndarray,
    homography: np.ndarray,
    tile_mm: float,
    light: tuple[int, int, int],
    dark: tuple[int, int, int],
) -> None:
    """Paint a metric checkerboard inside ``polygon`` by inverse mapping.

    Sampling through ``H^-1`` is what makes the rendered pattern the *exact*
    perspective image of a metric checkerboard: the tile edges converge on the
    plane's true vanishing points, so a line detector run over this image is
    being scored against the same geometry the ground truth describes.
    """
    if len(polygon) < 3:
        return
    height, width = canvas.shape[:2]
    poly_i = np.round(polygon).astype(np.int32)
    if light == dark:
        # A flat surface needs no inverse mapping at all. The ceiling takes this
        # path, and it is the largest bounding box in a downward-pitched frame.
        cv2.fillPoly(canvas, [poly_i], light)
        return

    bx, by, bw, bh = cv2.boundingRect(poly_i)
    x0, y0 = max(bx, 0), max(by, 0)
    x1, y1 = min(bx + bw, width), min(by + bh, height)
    if x1 <= x0 or y1 <= y0:
        return

    local_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(local_mask, [poly_i - np.array([[x0, y0]], dtype=np.int32)], 255)
    if not local_mask.any():
        return

    u, v = _metric_coordinates(np.linalg.inv(homography), x0, y0, x1 - x0, y1 - y0)
    u /= tile_mm
    v /= tile_mm
    np.floor(u, out=u)
    np.floor(v, out=v)
    u += v
    parity = np.remainder(u, 2.0, out=u) >= 1.0

    region = canvas[y0:y1, x0:x1]
    inside = local_mask > 0
    region[inside & ~parity] = light
    region[inside & parity] = dark


def _visible_extent_mm(
    homography: np.ndarray,
    polygon: np.ndarray,
    u_extent: float,
    v_extent: float,
) -> tuple[float, float, float, float]:
    """Metric bounding box of the plane's *visible* region.

    The surface quad generally runs past the frame and, for the floor, behind
    the camera as well. Mapping the clipped image polygon back through ``H^-1``
    and intersecting with the full quad extent yields an extent every point of
    which is genuinely in front of the camera and inside the picture --- which
    is what the homography and aspect-ratio properties want to sample over.
    """
    inv = np.linalg.inv(homography)
    pts = np.column_stack((polygon.astype(np.float64), np.ones(len(polygon))))
    metric = pts @ inv.T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = metric[:, :2] / metric[:, 2:3]
    uv = uv[np.isfinite(uv).all(axis=1)]
    if len(uv) == 0:  # pragma: no cover - clipped polygons always map back
        return (0.0, 0.0, float(u_extent), float(v_extent))
    u_min = float(np.clip(uv[:, 0].min(), 0.0, u_extent))
    u_max = float(np.clip(uv[:, 0].max(), 0.0, u_extent))
    v_min = float(np.clip(uv[:, 1].min(), 0.0, v_extent))
    v_max = float(np.clip(uv[:, 1].max(), 0.0, v_extent))
    return (u_min, v_min, u_max, v_max)


def _apply_shading(image: np.ndarray, rng: np.random.Generator, noise_sigma: float) -> np.ndarray:
    """Add a smooth illumination falloff plus sensor noise.

    Purely image-space, so it changes no geometry: the Lighting_Engine gets a
    low-frequency gradient to recover and the line detector gets a mild noise
    floor to cope with, while the checkerboard edges stay where the camera put
    them.
    """
    height, width = image.shape[:2]
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = (xs - 0.38 * width) / float(width)
    dy = (ys - 0.10 * height) / float(height)
    falloff = np.clip(np.sqrt(dx * dx + dy * dy) / 1.15, 0.0, 1.0)
    factor = (1.06 - 0.34 * falloff)[..., None]
    out = image.astype(np.float32) * factor
    if noise_sigma > 0.0:
        out += rng.normal(0.0, noise_sigma, size=out.shape).astype(np.float32)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Occluders
# --------------------------------------------------------------------------- #

# Unit-cube face descriptions as (corner index quad, outward normal). Indices
# 0-3 are the base ring at y=0, 4-7 the top ring, in the same winding. The
# bottom face is omitted: a box standing on the floor never shows it.
_BOX_FACES: tuple[tuple[tuple[int, int, int, int], tuple[float, float, float]], ...] = (
    ((4, 5, 6, 7), (0.0, -1.0, 0.0)),  # top
    ((0, 1, 5, 4), (0.0, 0.0, -1.0)),  # near
    ((3, 2, 6, 7), (0.0, 0.0, 1.0)),  # far
    ((0, 3, 7, 4), (-1.0, 0.0, 0.0)),  # left
    ((1, 2, 6, 5), (1.0, 0.0, 0.0)),  # right
)

_FACE_SHADE: dict[tuple[float, float, float], float] = {
    (0.0, -1.0, 0.0): 1.20,
    (0.0, 0.0, -1.0): 1.00,
    (0.0, 0.0, 1.0): 0.88,
    (-1.0, 0.0, 0.0): 0.80,
    (1.0, 0.0, 0.0): 0.92,
}


def _box_corners(
    centre_x: float, centre_z: float, size_x: float, size_z: float, height: float
) -> np.ndarray:
    """Eight world corners of an axis-aligned box standing on ``Y = 0``."""
    hx, hz = size_x / 2.0, size_z / 2.0
    base = np.array(
        [
            [centre_x - hx, 0.0, centre_z - hz],
            [centre_x + hx, 0.0, centre_z - hz],
            [centre_x + hx, 0.0, centre_z + hz],
            [centre_x - hx, 0.0, centre_z + hz],
        ],
        dtype=np.float64,
    )
    top = base.copy()
    top[:, 1] = -height
    return np.vstack((base, top))


def _box_face_polygons(
    corners: np.ndarray,
    camera: CameraParams,
    K: np.ndarray,
    shape: tuple[int, int],
    base_bgr: np.ndarray,
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    """Visible face polygons of a box, with their shaded colours.

    Faces are culled by the sign of ``normal . view``, so a convex box yields
    exactly its silhouette --- which is why the union of these polygons is a
    sound occluder mask. Polygons are returned rather than drawn so placement
    can be accepted or rejected without touching the canvas.
    """
    height, width = shape
    out: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for indices, normal in _BOX_FACES:
        quad = corners[list(indices)]
        normal_cam = camera.R @ np.asarray(normal, dtype=np.float64)
        centroid_cam = camera.R @ quad.mean(axis=0) + camera.t
        if float(np.dot(normal_cam, centroid_cam)) >= 0.0:
            continue  # back face
        clipped = _clip_near(quad @ camera.R.T + camera.t, _NEAR_MM)
        if len(clipped) < 3:
            continue
        homogeneous = clipped @ K.T
        polygon = _clip_to_image(
            homogeneous[:, :2] / homogeneous[:, 2:3], width, height
        )
        if len(polygon) < 3:
            continue
        colour = np.clip(base_bgr * _FACE_SHADE[normal], 0.0, 255.0)
        out.append(
            (
                np.round(polygon).astype(np.int32),
                (int(colour[0]), int(colour[1]), int(colour[2])),
            )
        )
    return out


def _place_occluders(
    canvas: np.ndarray,
    camera: CameraParams,
    K: np.ndarray,
    *,
    n_occluders: int,
    half_width_mm: float,
    depth_far_mm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stand ``n_occluders`` solid boxes on the floor; return their union mask.

    Placement is rejection-sampled so each box covers a usable share of the
    frame and does not hide behind another, giving Property 5 a foreground
    target with real recall to measure. Fewer boxes than requested are placed
    if the camera simply cannot see that many, rather than looping forever.
    """
    height, width = canvas.shape[:2]
    frame_area = float(height * width)
    union = np.zeros((height, width), dtype=np.uint8)
    placed = 0
    attempts = 0
    max_attempts = 40 * max(n_occluders, 1)
    z_far_limit = max(depth_far_mm - 700.0, 1400.0)

    while placed < n_occluders and attempts < max_attempts:
        attempts += 1
        size_x = float(rng.uniform(380.0, 820.0))
        size_z = float(rng.uniform(380.0, 820.0))
        box_height = float(rng.uniform(520.0, 1320.0))
        centre_x = float(rng.uniform(-half_width_mm + 450.0, half_width_mm - 450.0))
        centre_z = float(rng.uniform(1300.0, z_far_limit))
        grey = float(rng.uniform(58.0, 112.0))
        base_bgr = np.clip(
            np.array([grey + 12.0, grey, grey - 8.0], dtype=np.float64), 0.0, 255.0
        )
        faces = _box_face_polygons(
            _box_corners(centre_x, centre_z, size_x, size_z, box_height),
            camera,
            K,
            (height, width),
            base_bgr,
        )
        if not faces:
            continue

        candidate_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(candidate_mask, [polygon for polygon, _ in faces], 255)
        area = float(np.count_nonzero(candidate_mask))
        if area < 0.004 * frame_area:
            continue  # too small or mostly off-frame to be a useful occluder
        if float(np.count_nonzero(candidate_mask & union)) > 0.25 * area:
            continue  # hiding behind an already-placed box

        for polygon, colour in faces:
            cv2.fillPoly(canvas, [polygon], colour)
        union |= candidate_mask
        placed += 1

    return union


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #


def make_synthetic_room(
    width: int = 1600,
    height: int = 1200,
    focal_px: float = 1400.0,
    camera_height_mm: float = 1500.0,
    yaw_deg: float = 8.0,
    pitch_deg: float = -12.0,
    tile_mm: float = 600.0,
    walls: Iterable[str] = ("left", "right", "back"),
    n_occluders: int = 2,
    seed: int = 0,
    *,
    room_width_mm: float = 3000.0,
    room_height_mm: float = 2700.0,
    depth_near_mm: float = -1200.0,
    depth_far_mm: float = 6000.0,
    supersample: int = 2,
    noise_sigma: float = 2.0,
) -> SyntheticRoom:
    """Render a perspective room and return it with its analytic ground truth.

    Args:
        width, height: output image size in pixels.
        focal_px: focal length in pixels; the principal point is the image
            centre, which is what the Geometry_Engine assumes.
        camera_height_mm: camera centre height above the floor. The same value
            fixes absolute metric scale in the design's plane frames, so a test
            can compare recovered scale against it directly.
        yaw_deg: camera turn to its right. A non-zero yaw is what makes both
            horizontal vanishing points finite and therefore recoverable.
        pitch_deg: camera elevation; negative looks down.
        tile_mm: checkerboard pitch on every surface, in millimetres.
        walls: any subset of ``("left", "right", "back")``.
        n_occluders: solid boxes to stand on the floor. Fewer may be placed if
            the camera cannot see that many.
        seed: seeds occluder placement and sensor noise, so the whole image is
            deterministic.
        room_width_mm, room_height_mm: room dimensions. The defaults describe a
            3.0 m by 6.0 m room with a 2.7 m ceiling, chosen so that at the
            default pose all four Structural_Planes land well clear of the
            ``min_plane_area_fraction`` floor a Segmenter would drop them at.
        depth_near_mm, depth_far_mm: the floor's depth span in camera-forward
            millimetres. ``depth_near_mm`` is negative so the floor continues
            behind the camera and fills the bottom of the frame.
        supersample: integer factor the image is rendered at before being
            reduced with ``INTER_AREA``. The default 2 keeps distant
            checkerboard rows from aliasing into false edges; property tests
            that call this hundreds of times can pass 1 for speed.
        noise_sigma: standard deviation of additive sensor noise, in levels.

    Raises:
        ValueError: for a non-positive image size, focal length, tile pitch, or
            supersample factor, or for an unknown wall key.
    """
    if width < 16 or height < 16:
        raise ValueError("width and height must each be at least 16 pixels")
    if focal_px <= 0.0:
        raise ValueError("focal_px must be positive")
    if tile_mm <= 0.0:
        raise ValueError("tile_mm must be positive")
    if camera_height_mm <= 0.0:
        raise ValueError("camera_height_mm must be positive")
    if room_width_mm <= 0.0 or room_height_mm <= 0.0:
        raise ValueError("room dimensions must be positive")
    if depth_far_mm <= depth_near_mm:
        raise ValueError("depth_far_mm must exceed depth_near_mm")
    if supersample < 1:
        raise ValueError("supersample must be at least 1")
    if n_occluders < 0:
        raise ValueError("n_occluders must not be negative")

    wall_keys = tuple(dict.fromkeys(walls))
    unknown = [key for key in wall_keys if key not in WALL_TO_PLANE]
    if unknown:
        raise ValueError(f"unknown wall(s) {unknown!r}; expected any of {WALL_KEYS!r}")

    width, height = int(width), int(height)
    rng = np.random.default_rng(seed)

    # --- camera ---------------------------------------------------------- #
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    R_cam_to_world = _rotation_y(yaw) @ _rotation_x(pitch)
    R = R_cam_to_world.T
    position = np.array([0.0, -float(camera_height_mm), 0.0], dtype=np.float64)
    t = -R @ position
    K = _intrinsics(float(focal_px), width, height)
    camera = CameraParams(
        width=width,
        height=height,
        focal_px=float(focal_px),
        principal_point=(float(K[0, 2]), float(K[1, 2])),
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        camera_height_mm=float(camera_height_mm),
        K=K,
        R=R,
        t=t,
        position_mm=position,
    )

    # --- analytic ground truth ------------------------------------------- #
    truth_vps = {
        "VPx": camera.vanishing_point((1.0, 0.0, 0.0)),
        "VPy": camera.vanishing_point((0.0, 1.0, 0.0)),
        "VPz": camera.vanishing_point((0.0, 0.0, 1.0)),
    }
    # The image of a plane's line at infinity is K^-T R n for plane normal n:
    # every direction d in the plane satisfies n.d = 0, and (K R d) lies on this
    # line by construction. For the floor (Y = 0) the normal is the world Y axis.
    horizon_vec = np.linalg.inv(K).T @ R @ np.array([0.0, 1.0, 0.0])
    scale = float(np.hypot(horizon_vec[0], horizon_vec[1]))
    if scale < _EPS:  # pragma: no cover - only for a camera looking along Y
        raise ValueError("degenerate horizon for this camera pose")
    horizon_vec = horizon_vec / scale
    truth_horizon = (
        float(horizon_vec[0]),
        float(horizon_vec[1]),
        float(horizon_vec[2]),
    )

    surfaces = {
        surface.name: surface
        for surface in _surfaces(
            walls=wall_keys,
            half_width_mm=room_width_mm / 2.0,
            depth_near_mm=depth_near_mm,
            depth_far_mm=depth_far_mm,
            height_mm=room_height_mm,
        )
    }
    truth_homographies: dict[str, np.ndarray] = {}
    plane_extents: dict[str, tuple[float, float, float, float]] = {}
    plane_polygons: dict[str, np.ndarray] = {}

    # --- render ----------------------------------------------------------- #
    ss = int(supersample)
    K_ss = K.copy()
    if ss > 1:
        # Scale intrinsics to the supersampled raster. The 0.5 shift keeps the
        # scaled principal point on the same physical location once INTER_AREA
        # folds each ss x ss block back into one pixel.
        K_ss[0, 0] *= ss
        K_ss[1, 1] *= ss
        K_ss[0, 2] = (K[0, 2] + 0.5) * ss - 0.5
        K_ss[1, 2] = (K[1, 2] + 0.5) * ss - 0.5
    canvas = np.empty((height * ss, width * ss, 3), dtype=np.uint8)
    canvas[:, :] = np.array(_BACKGROUND_BGR, dtype=np.uint8)

    for name in _DRAW_ORDER:
        surface = surfaces.get(name)
        if surface is None:
            continue
        cam_corners = surface.corners() @ R.T + t
        clipped = _clip_near(cam_corners, _NEAR_MM)
        if len(clipped) < 3:
            continue

        homogeneous_ss = clipped @ K_ss.T
        projected_ss = homogeneous_ss[:, :2] / homogeneous_ss[:, 2:3]
        polygon_ss = _clip_to_image(projected_ss, width * ss, height * ss)
        light, dark = _SURFACE_COLOURS[name]
        _fill_checkerboard(
            canvas, polygon_ss, _homography(K_ss, R, t, surface), tile_mm, light, dark
        )

        if name == "ceiling":
            continue  # background only, never a Structural_Plane
        homogeneous = clipped @ K.T
        projected = homogeneous[:, :2] / homogeneous[:, 2:3]
        polygon = _clip_to_image(projected, width, height)
        if len(polygon) < 3:
            continue  # entirely outside the frame: report the plane as absent
        homography = _homography(K, R, t, surface)
        plane_polygons[name] = polygon.astype(np.float32)
        truth_homographies[name] = homography
        plane_extents[name] = _visible_extent_mm(
            homography, polygon, surface.u_extent, surface.v_extent
        )

    occluder_ss = _place_occluders(
        canvas,
        camera,
        K_ss,
        n_occluders=int(n_occluders),
        half_width_mm=room_width_mm / 2.0,
        depth_far_mm=depth_far_mm,
        rng=rng,
    )

    if ss > 1:
        image = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
        reduced = cv2.resize(occluder_ss, (width, height), interpolation=cv2.INTER_AREA)
        occluder_mask = np.where(reduced > 127, 255, 0).astype(np.uint8)
    else:
        image = canvas
        occluder_mask = occluder_ss

    image = np.ascontiguousarray(_apply_shading(image, rng, float(noise_sigma)))

    return SyntheticRoom(
        image=image,
        truth_vps=truth_vps,
        truth_horizon=truth_horizon,
        truth_homographies=truth_homographies,
        plane_polygons=plane_polygons,
        occluder_mask=occluder_mask,
        camera=camera,
        plane_extents_mm=plane_extents,
        tile_mm=float(tile_mm),
        walls=wall_keys,
    )


def write_room(
    room: SyntheticRoom,
    out_dir: str | Path,
    *,
    stem: str = "synthetic_room",
    force: bool = True,
) -> dict[str, Path]:
    """Write ``room`` to ``out_dir`` as an image plus a ground-truth sidecar.

    Three files are produced, all named from ``stem``: ``<stem>.png``, the
    ``<stem>.truth.json`` sidecar carrying the camera, vanishing points,
    horizon, homographies, plane polygons, and metric extents, and
    ``<stem>.occluders.png``, the occluder mask that a JSON file cannot
    reasonably hold.

    With ``force=False`` existing files are left untouched, which is how the
    Setup_Tool preserves assets across re-runs.

    Returns:
        The three paths, keyed ``image``, ``truth``, and ``occluders``.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "image": directory / f"{stem}.png",
        "truth": directory / f"{stem}.truth.json",
        "occluders": directory / f"{stem}.occluders.png",
    }

    if force or not paths["image"].exists():
        if not cv2.imwrite(str(paths["image"]), room.image):
            raise OSError(f"failed to write {paths['image']}")
    if force or not paths["occluders"].exists():
        if not cv2.imwrite(str(paths["occluders"]), room.occluder_mask):
            raise OSError(f"failed to write {paths['occluders']}")
    if force or not paths["truth"].exists():
        paths["truth"].write_text(
            json.dumps(room.truth_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return paths
