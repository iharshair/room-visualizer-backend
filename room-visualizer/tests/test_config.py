"""Unit tests for `backend.config.Settings`.

Covers the documented defaults (Requirements 2.4, 2.6, 2.7, 5.5, 6.5, 7.7, 9.5,
9.6), `RV_`-prefixed environment overrides (Requirement 11.7), the cached
accessor, and the field validation rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_documented_defaults():
    settings = Settings()

    # Upload hardening
    assert settings.max_upload_bytes == 12 * 1024 * 1024
    assert settings.max_longest_edge == 2048
    assert settings.allowed_mime_types == ("image/jpeg", "image/png", "image/webp")
    assert settings.allowed_extensions == (".jpg", ".jpeg", ".png", ".webp")
    assert settings.cors_allow_origins == ("*",)

    # Scene cache
    assert settings.scene_cache_max_entries == 32
    assert settings.scene_cache_ttl_seconds == 1800

    # Geometry
    assert settings.min_plane_area_fraction == 0.02
    assert settings.vp_min_cluster_size == 8
    assert settings.vp_ransac_iterations == 400
    assert settings.vp_inlier_threshold_px == 2.0
    assert settings.orthogonality_tolerance == 0.25
    assert settings.assumed_camera_height_mm == 1500.0

    # Lighting / compositing
    assert settings.shading_sigma_px == 31
    assert settings.use_bilateral_shading is True
    assert settings.feather_width_px == 2
    assert settings.default_grout_mm == 3.0
    assert settings.default_grout_rgb == (168, 168, 164)

    # Model loader
    assert settings.weights_dir.parts[-3:] == (".cache", "room-visualizer", "weights")
    assert settings.mobilesam_encoder_url.startswith("https://")
    assert settings.mobilesam_decoder_url.startswith("https://")
    assert len(settings.mobilesam_encoder_sha256) == 64
    assert len(settings.mobilesam_decoder_sha256) == 64
    assert settings.model_download_timeout_s == 30.0
    assert settings.enable_neural_backend is True

    # Assets and output
    assert settings.assets_dir.name == "assets"
    assert settings.tiles_manifest_name == "manifest.json"
    assert settings.tiles_dir == settings.assets_dir / "tiles"
    assert settings.tiles_manifest_path == settings.assets_dir / "tiles" / "manifest.json"
    assert settings.render_format == "png"
    assert settings.render_jpeg_quality == 90


def test_rv_prefixed_env_vars_override_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("RV_MAX_UPLOAD_BYTES", "2048")
    monkeypatch.setenv("RV_MAX_LONGEST_EDGE", "1024")
    monkeypatch.setenv("RV_ENABLE_NEURAL_BACKEND", "false")
    monkeypatch.setenv("RV_WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("RV_RENDER_FORMAT", "jpeg")
    monkeypatch.setenv("RV_SCENE_CACHE_TTL_SECONDS", "60")

    settings = get_settings()

    assert settings.max_upload_bytes == 2048
    assert settings.max_longest_edge == 1024
    assert settings.enable_neural_backend is False
    assert settings.weights_dir == Path(tmp_path / "weights")
    assert settings.render_format == "jpeg"
    assert settings.scene_cache_ttl_seconds == 60


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("image/png,image/webp", ("image/png", "image/webp")),
        ("image/PNG, image/WEBP", ("image/png", "image/webp")),
        ('["image/png", "image/webp"]', ("image/png", "image/webp")),
    ],
)
def test_sequence_env_vars_accept_csv_and_json(monkeypatch, raw, expected):
    monkeypatch.setenv("RV_ALLOWED_MIME_TYPES", raw)
    assert get_settings().allowed_mime_types == expected


def test_extensions_are_normalised_to_lowercase_dotted():
    settings = Settings(allowed_extensions="JPG, .PNG")
    assert settings.allowed_extensions == (".jpg", ".png")


def test_grout_rgb_accepts_csv_env(monkeypatch):
    monkeypatch.setenv("RV_DEFAULT_GROUT_RGB", "10, 20, 30")
    assert get_settings().default_grout_rgb == (10, 20, 30)


def test_get_settings_is_cached_until_cleared(monkeypatch):
    first = get_settings()
    assert get_settings() is first

    monkeypatch.setenv("RV_MAX_UPLOAD_BYTES", "4096")
    assert get_settings() is first, "cached instance must survive an env change"

    get_settings.cache_clear()
    assert get_settings().max_upload_bytes == 4096


def test_shading_sigma_is_coerced_to_odd():
    assert Settings(shading_sigma_px=30).shading_sigma_px == 31
    assert Settings(shading_sigma_px=31).shading_sigma_px == 31


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_upload_bytes": 0},
        {"max_longest_edge": 16},
        {"scene_cache_max_entries": 0},
        {"scene_cache_ttl_seconds": 0},
        {"min_plane_area_fraction": 1.5},
        {"vp_inlier_threshold_px": 0.0},
        {"model_download_timeout_s": 0.0},
        {"render_jpeg_quality": 101},
        {"render_format": "gif"},
        {"default_grout_rgb": (300, 0, 0)},
        {"default_grout_mm": -1.0},
        {"mobilesam_encoder_url": "http://example.com/encoder.onnx"},
        {"mobilesam_decoder_sha256": "not-a-digest"},
    ],
)
def test_invalid_values_are_rejected(overrides):
    with pytest.raises(ValueError):
        Settings(**overrides)


def test_digests_are_normalised_to_lowercase_hex():
    digest = "AB" * 32
    settings = Settings(mobilesam_encoder_sha256=digest)
    assert settings.mobilesam_encoder_sha256 == digest.lower()
