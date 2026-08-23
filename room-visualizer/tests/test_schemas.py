"""Unit tests for the shared data contracts in `backend/schemas.py`.

Scene_State memory release is property-tested through the Scene_Cache
(Property 32) and the cached-artifact dtype contract through the API
(Property 33); these tests cover the dataclass mechanics and the HTTP model
validation directly so a contract break surfaces without the whole pipeline.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from backend.schemas import (
    ErrorEnvelope,
    HealthResponse,
    PlaneMetadata,
    PlaneRenderSpec,
    RenderRequest,
    SceneState,
    SegmentResponse,
    TileDefinition,
    TileResponse,
    TilesResponse,
)


def _plane_metadata(name: str = "floor") -> PlaneMetadata:
    return PlaneMetadata(
        name=name,
        contour=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32),
        bounding_points=np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.int32),
        area_fraction=0.25,
        centroid=(5.0, 5.0),
        homography=np.eye(3, dtype=np.float64),
        homography_inv=np.eye(3, dtype=np.float64),
        plane_extent_mm=(0.0, 0.0, 1000.0, 1000.0),
        reprojection_rmse_px=0.4,
        geometry_mode="vanishing_points",
        luminance_median=120.0,
    )


def _scene_state(h: int = 8, w: int = 6) -> SceneState:
    return SceneState(
        scene_id="a" * 32,
        created_at=1000.0,
        image=np.zeros((h, w, 3), dtype=np.uint8),
        width=w,
        height=h,
        planes={"floor": _plane_metadata()},
        plane_masks={"floor": np.zeros((h, w), dtype=np.uint8)},
        foreground_mask=np.zeros((h, w), dtype=np.uint8),
        shading_map=np.zeros((h, w), dtype=np.uint8),
        detail_map=np.full((h, w), 128, dtype=np.uint8),
        horizon=(0.0, 1.0, -4.0),
        vanishing_points={"VPx": (-100.0, 4.0), "VPy": None, "VPz": (200.0, 4.0)},
        geometry_mode="vanishing_points",
        segmentation_backend="classical",
    )


class TestSceneState:
    def test_nbytes_counts_every_held_array(self):
        state = _scene_state(h=8, w=6)
        expected = (
            state.image.nbytes
            + state.foreground_mask.nbytes
            + state.shading_map.nbytes
            + state.detail_map.nbytes
            + state.plane_masks["floor"].nbytes
            + sum(
                a.nbytes
                for a in (
                    state.planes["floor"].contour,
                    state.planes["floor"].bounding_points,
                    state.planes["floor"].homography,
                    state.planes["floor"].homography_inv,
                )
            )
        )
        assert state.nbytes() == expected

    def test_release_drops_every_array_reference(self):
        import weakref

        state = _scene_state()
        refs = [
            weakref.ref(state.image),
            weakref.ref(state.foreground_mask),
            weakref.ref(state.shading_map),
            weakref.ref(state.detail_map),
            weakref.ref(state.plane_masks["floor"]),
        ]

        state.release()

        assert state.image is None
        assert state.foreground_mask is None
        assert state.shading_map is None
        assert state.detail_map is None
        assert state.plane_masks == {}
        assert state.planes == {}
        assert all(ref() is None for ref in refs)
        assert state.nbytes() == 0

    def test_release_is_idempotent(self):
        state = _scene_state()
        state.release()
        state.release()
        assert state.nbytes() == 0

    def test_cached_artifacts_are_uint8(self):
        state = _scene_state()
        assert state.image.dtype == np.uint8
        assert state.foreground_mask.dtype == np.uint8
        assert state.shading_map.dtype == np.uint8
        assert state.detail_map.dtype == np.uint8
        assert all(m.dtype == np.uint8 for m in state.plane_masks.values())


class TestTileDefinition:
    def test_is_frozen_and_hashable(self):
        tile = TileDefinition(
            id="marble-carrara-600",
            name="Carrara Marble 600x600",
            image_path=Path("marble_carrara_600x600.png"),
            width_mm=600.0,
            height_mm=600.0,
            finish="polished",
            gloss=0.85,
        )
        assert tile.grout_mm is None
        assert hash(tile) == hash(tile)
        with pytest.raises(Exception):
            tile.gloss = 0.1  # type: ignore[misc]


class TestPlaneRenderSpec:
    def test_defaults_mean_inherit(self):
        spec = PlaneRenderSpec(tile_id="t1")
        assert spec.rotation_deg == 0.0
        assert spec.grout_mm is None
        assert spec.grout_rgb is None
        assert spec.offset_mm == (0.0, 0.0)


class TestRenderRequest:
    def test_parses_documented_payload_into_specs(self):
        req = RenderRequest.model_validate(
            {
                "scene_id": "8f14e45fceea167a5a36dedd4bea2543",
                "planes": {
                    "floor": {
                        "tile_id": "marble-carrara-600",
                        "rotation_deg": 0,
                        "grout_mm": 3,
                    },
                    "wall_back": {
                        "tile_id": "concrete-matte-600",
                        "rotation_deg": 45,
                    },
                },
                "format": "png",
            }
        )
        assert isinstance(req.planes["floor"], PlaneRenderSpec)
        assert req.planes["floor"].grout_mm == 3.0
        assert req.planes["wall_back"].rotation_deg == 45.0
        assert req.planes["wall_back"].grout_mm is None
        assert req.format == "png"

    def test_planes_and_format_are_optional(self):
        req = RenderRequest.model_validate({"scene_id": "abc"})
        assert req.planes == {}
        assert req.format is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"scene_id": "abc", "planes": {"ceiling": {"tile_id": "t1"}}},
            {"scene_id": "abc", "planes": {"floor": {}}},
            {"scene_id": "abc", "planes": {"floor": {"tile_id": "t", "grout_mm": -1}}},
            {"scene_id": "abc", "format": "gif"},
            {"planes": {}},
        ],
    )
    def test_rejects_invalid_payloads(self, payload):
        with pytest.raises(ValidationError):
            RenderRequest.model_validate(payload)


class TestResponseModels:
    def test_segment_response_round_trips_documented_shape(self):
        payload = {
            "scene_id": "8f14e45fceea167a5a36dedd4bea2543",
            "width": 1600,
            "height": 1200,
            "segmentation_backend": "classical",
            "geometry_mode": "vanishing_points",
            "horizon": {"a": 0.0, "b": 1.0, "c": -512.4, "y_at_center": 512.4},
            "vanishing_points": {
                "VPx": [-2140.5, 511.8],
                "VPy": [802.1, -9840.2],
                "VPz": None,
            },
            "planes": [
                {
                    "name": "floor",
                    "area_fraction": 0.312,
                    "contour": [[12, 780], [1588, 802], [1140, 1199], [430, 1199]],
                    "bounding_points": [
                        [12, 780],
                        [1588, 802],
                        [1140, 1199],
                        [430, 1199],
                    ],
                    "centroid": [800.0, 980.4],
                    "reprojection_rmse_px": 0.41,
                }
            ],
            "analysis_ms": 2140,
        }
        parsed = SegmentResponse.model_validate(payload)
        assert parsed.model_dump()["horizon"]["y_at_center"] == 512.4
        assert parsed.vanishing_points["VPz"] is None
        assert parsed.planes[0].name == "floor"

    def test_bounding_points_must_be_exactly_four(self):
        with pytest.raises(ValidationError):
            SegmentResponse.model_validate(
                {
                    "scene_id": "s",
                    "width": 10,
                    "height": 10,
                    "segmentation_backend": "classical",
                    "geometry_mode": "planar_fallback",
                    "horizon": {"a": 0.0, "b": 1.0, "c": -5.0, "y_at_center": 5.0},
                    "vanishing_points": {},
                    "planes": [
                        {
                            "name": "floor",
                            "area_fraction": 0.5,
                            "contour": [[0, 0], [1, 0], [1, 1]],
                            "bounding_points": [[0, 0], [1, 0], [1, 1]],
                            "centroid": [0.5, 0.5],
                            "reprojection_rmse_px": 0.1,
                        }
                    ],
                    "analysis_ms": 1,
                }
            )

    def test_tiles_response_defaults_to_empty(self):
        assert TilesResponse().tiles == []

    def test_tile_response_rejects_out_of_range_gloss(self):
        with pytest.raises(ValidationError):
            TileResponse(
                id="t",
                name="T",
                width_mm=600,
                height_mm=600,
                finish="matte",
                gloss=1.5,
                thumbnail_url="/assets/tiles/t.png",
            )

    def test_health_response_matches_documented_shape(self):
        health = HealthResponse(
            segmentation_backend="classical",
            onnx_provider="n/a",
            scene_cache_entries=3,
            scene_cache_max_entries=32,
            scene_cache_ttl_seconds=1800,
        )
        assert health.model_dump() == {
            "status": "ok",
            "segmentation_backend": "classical",
            "onnx_provider": "n/a",
            "scene_cache_entries": 3,
            "scene_cache_max_entries": 32,
            "scene_cache_ttl_seconds": 1800,
        }


class TestErrorEnvelope:
    def test_serialises_to_the_shared_shape(self):
        envelope = ErrorEnvelope.of("no_usable_plane", "No wall or floor found.")
        assert envelope.model_dump() == {
            "error": {"code": "no_usable_plane", "message": "No wall or floor found."}
        }

    def test_parses_a_server_error_body(self):
        parsed = ErrorEnvelope.model_validate(
            {"error": {"code": "scene_expired", "message": "Re-upload the photo."}}
        )
        assert parsed.error.code == "scene_expired"
