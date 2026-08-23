"""Application configuration for the AI Room & Tile Visualizer.

A single :class:`Settings` object holds every tunable named in the requirements.
Fields are populated from ``RV_``-prefixed environment variables (or a ``.env``
file next to the process working directory), so an operator can retune the
service without editing Python.

Tests import :func:`get_settings`, call ``get_settings.cache_clear()``, and/or
construct ``Settings(...)`` directly with overrides (for example lowering
``max_upload_bytes`` so the 413 path is exercisable without a 12 MB payload).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

__all__ = [
    "Settings",
    "get_settings",
    "MOBILESAM_ENCODER_URL",
    "MOBILESAM_DECODER_URL",
]

# Pinned MobileSAM ONNX artifacts (Requirement 4.2, Requirement 12.2). The
# encoder/decoder pair is roughly 27 MB combined -- a mobile-scale model, not a
# full SAM ViT-H checkpoint. Both the URL and the digest are overridable with
# RV_MOBILESAM_ENCODER_URL / RV_MOBILESAM_ENCODER_SHA256 (and the decoder
# equivalents) so a host can mirror the weights internally.
_HF_BASE = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main"
MOBILESAM_ENCODER_URL = f"{_HF_BASE}/mobile_sam.encoder.onnx"
MOBILESAM_DECODER_URL = f"{_HF_BASE}/mobile_sam.decoder.onnx"

# SHA-256 pins for the artifacts above. Set these to the digests measured on the
# mirror the deployment actually uses; a mismatch makes the Model_Loader discard
# the download and the service falls back to the classical segmentation backend
# (Requirement 4.3, Requirement 4.5), so an unverified pin degrades gracefully
# rather than serving unvetted weights.
MOBILESAM_ENCODER_SHA256 = "6004fd02ef6cec4098f0949c191e48c44fa696bbffa1ab6e21583c0ab578b541"
MOBILESAM_DECODER_SHA256 = "09a53a0b95e756480e63bcabe4603f664d53ecb6f16c2294f757dd31bf4b01d6"

_BACKEND_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _BACKEND_DIR.parent


def _split_str_sequence(value: object) -> object:
    """Accept ``a,b,c``, a JSON array, or an already-parsed sequence.

    ``pydantic-settings`` would otherwise only understand JSON for these tuple
    fields, making ``RV_ALLOWED_MIME_TYPES='["image/png"]'`` the sole spelling.
    Operators reach for a comma-separated list, so ``NoDecode`` hands the raw
    string here and both spellings work.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ()
    if text[0] in "[(":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _split_int_sequence(value: object) -> object:
    """Same leniency as :func:`_split_str_sequence`, for numeric triples."""
    parsed = _split_str_sequence(value)
    if isinstance(parsed, (list, tuple)):
        return tuple(int(part) if isinstance(part, str) else part for part in parsed)
    return parsed


StrTuple = Annotated[tuple[str, ...], NoDecode, BeforeValidator(_split_str_sequence)]
RgbTuple = Annotated[tuple[int, int, int], NoDecode, BeforeValidator(_split_int_sequence)]


class Settings(BaseSettings):
    """Env-driven settings, read once and injected everywhere."""

    model_config = SettingsConfigDict(
        env_prefix="RV_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Field names such as `model_download_timeout_s` are part of the design's
        # published contract; clearing the protected namespace keeps them without
        # tripping pydantic's `model_` shadow warning.
        protected_namespaces=(),
    )

    # --- Upload hardening (Requirements 2.4, 2.6, 2.7) --------------------
    max_upload_bytes: int = Field(default=12 * 1024 * 1024, gt=0)
    max_longest_edge: int = Field(default=2048, ge=64)
    allowed_mime_types: StrTuple = ("image/jpeg", "image/png", "image/webp")
    allowed_extensions: StrTuple = (".jpg", ".jpeg", ".png", ".webp")
    cors_allow_origins: StrTuple = ("*",)

    # --- Scene cache (Requirements 9.5, 9.6) ------------------------------
    scene_cache_max_entries: int = Field(default=32, ge=1)
    scene_cache_ttl_seconds: int = Field(default=1800, gt=0)

    # --- Geometry (Requirements 5.5, 6.5) ---------------------------------
    min_plane_area_fraction: float = Field(default=0.02, gt=0.0, lt=1.0)
    vp_min_cluster_size: int = Field(default=8, ge=2)
    vp_ransac_iterations: int = Field(default=400, ge=1)
    vp_inlier_threshold_px: float = Field(default=2.0, gt=0.0)
    orthogonality_tolerance: float = Field(default=0.25, gt=0.0)
    assumed_camera_height_mm: float = Field(default=1500.0, gt=0.0)

    # --- Lighting / compositing (Requirement 7.7) -------------------------
    shading_sigma_px: int = Field(default=31, ge=1)
    use_bilateral_shading: bool = True
    feather_width_px: int = Field(default=2, ge=0)
    default_grout_mm: float = Field(default=3.0, ge=0.0)
    default_grout_rgb: RgbTuple = (168, 168, 164)

    # --- Model loader (Requirements 4.2, 12.2) ----------------------------
    weights_dir: Path = Path.home() / ".cache" / "room-visualizer" / "weights"
    mobilesam_encoder_url: str = MOBILESAM_ENCODER_URL
    mobilesam_encoder_sha256: str = MOBILESAM_ENCODER_SHA256
    mobilesam_decoder_url: str = MOBILESAM_DECODER_URL
    mobilesam_decoder_sha256: str = MOBILESAM_DECODER_SHA256
    model_download_timeout_s: float = Field(default=30.0, gt=0.0)
    enable_neural_backend: bool = True

    # --- Assets -----------------------------------------------------------
    assets_dir: Path = _PROJECT_DIR / "assets"
    tiles_manifest_name: str = "manifest.json"

    # --- Output -----------------------------------------------------------
    render_format: Literal["png", "jpeg"] = "png"
    render_jpeg_quality: int = Field(default=90, ge=1, le=100)

    @field_validator("shading_sigma_px")
    @classmethod
    def _shading_sigma_must_be_odd(cls, value: int) -> int:
        """OpenCV Gaussian kernels derived from this sigma require odd sizes."""
        return value if value % 2 == 1 else value + 1

    @field_validator("default_grout_rgb")
    @classmethod
    def _grout_channels_in_range(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if any(channel < 0 or channel > 255 for channel in value):
            raise ValueError("default_grout_rgb channels must be in [0, 255]")
        return value

    @field_validator("mobilesam_encoder_url", "mobilesam_decoder_url")
    @classmethod
    def _weights_url_must_be_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("model weight URLs must use https://")
        return value

    @field_validator("mobilesam_encoder_sha256", "mobilesam_decoder_sha256")
    @classmethod
    def _digest_must_be_sha256_hex(cls, value: str) -> str:
        digest = value.strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("expected a 64-character hexadecimal SHA-256 digest")
        return digest

    @field_validator("allowed_extensions")
    @classmethod
    def _extensions_are_lowercase_dotted(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalised = []
        for ext in value:
            ext = ext.strip().lower()
            normalised.append(ext if ext.startswith(".") else f".{ext}")
        return tuple(normalised)

    @field_validator("allowed_mime_types")
    @classmethod
    def _mime_types_are_lowercase(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(item.strip().lower() for item in value)

    @property
    def tiles_dir(self) -> Path:
        """Directory holding tile images and the manifest."""
        return self.assets_dir / "tiles"

    @property
    def tiles_manifest_path(self) -> Path:
        """Full path of the tile manifest (Requirement 8.2)."""
        return self.tiles_dir / self.tiles_manifest_name


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, built once and cached.

    Tests call ``get_settings.cache_clear()`` after mutating the environment so
    the next call re-reads it.
    """
    return Settings()
