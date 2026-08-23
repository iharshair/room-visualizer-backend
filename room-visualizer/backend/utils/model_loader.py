"""Model_Loader -- acquisition and initialisation of the MobileSAM ONNX pair.

The loader is deliberately narrow: it locates (or downloads and verifies) the two
pinned ONNX artifacts, picks an onnxruntime execution provider, and builds the
two inference sessions. Every failure -- transport error, checksum mismatch, or
session initialisation error -- surfaces as :class:`ModelUnavailable` carrying a
single documented reason string, so the caller in ``backend/app.py`` can log the
concrete reason at ``WARNING`` and fall back to the classical segmentation
backend without inspecting exception internals (Requirement 4.5).

The loader also logs that same reason at ``WARNING`` itself before raising, so
the operator sees which fallback fired even if a future caller swallows the
exception.

Weight acquisition happens once at startup, never per request, so no request
pays download latency (Requirement 4.2).

Network use is confined to :meth:`ModelLoader.ensure_weights`; nothing else in
this module touches the network, and a cached, digest-matching pair short-circuits
the download entirely, which is what lets an offline host boot (Requirement 4.7).
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, Sequence
from urllib.parse import urlparse

import httpx

from backend.config import Settings

try:  # pragma: no cover - exercised only on hosts without onnxruntime
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

__all__ = [
    "ModelArtifacts",
    "ModelUnavailable",
    "ModelLoader",
    "CPU_PROVIDER",
    "CUDA_PROVIDER",
    "REASON_DOWNLOAD_FAILED",
    "REASON_CHECKSUM_MISMATCH",
    "REASON_SESSION_INIT_FAILED",
    "MOBILESAM_ENCODER_EXPECTED_BYTES",
    "MOBILESAM_DECODER_EXPECTED_BYTES",
    "MOBILESAM_EXPECTED_BYTES",
    "MOBILE_SCALE_MAX_BYTES",
    "MODEL_SIZE_TOLERANCE_FRACTION",
    "is_mobile_scale",
    "size_is_plausible",
]

CPU_PROVIDER: Final = "CPUExecutionProvider"
CUDA_PROVIDER: Final = "CUDAExecutionProvider"

# The three documented fallback reasons (Requirement 4.5). `str(ModelUnavailable)`
# is exactly one of these, so callers can match on it without parsing prose.
REASON_DOWNLOAD_FAILED: Final = "weights_download_failed"
REASON_CHECKSUM_MISMATCH: Final = "checksum_mismatch"
REASON_SESSION_INIT_FAILED: Final = "onnx_session_init_failed"

# Pinned artifact scale (Requirement 12.2). These are the expected on-disk sizes
# of the MobileSAM encoder/decoder pair -- a mobile-scale model of roughly 27 MB
# combined -- recorded to the nearest 100 KB because a mirror may re-export with
# trivially different padding. They are a *scale* pin, not an integrity pin: the
# SHA-256 digests in `backend/config.py` are the authoritative integrity gate.
# Their purpose is to make substituting a full SAM ViT-H checkpoint (~2.4 GB, or
# ~375 MB for ViT-B) detectable, both by the test suite and by the runtime
# warning in `_warn_on_implausible_size`.
MOBILESAM_ENCODER_EXPECTED_BYTES: Final = 26_800_000
MOBILESAM_DECODER_EXPECTED_BYTES: Final = 16_500_000
MOBILESAM_EXPECTED_BYTES: Final = {
    "encoder": MOBILESAM_ENCODER_EXPECTED_BYTES,
    "decoder": MOBILESAM_DECODER_EXPECTED_BYTES,
}

# Ceiling separating a mobile-scale artifact from a desktop-scale one. The
# smallest full SAM export (ViT-B) is comfortably above this bound, so any
# artifact under it cannot be a full-size SAM checkpoint.
MOBILE_SCALE_MAX_BYTES: Final = 64 * 1024 * 1024

# How far an artifact may drift from its pinned size before the loader warns.
MODEL_SIZE_TOLERANCE_FRACTION: Final = 0.25

# Streaming chunk size. Large enough that hashing dominates syscall overhead,
# small enough that a 27 MB download never materialises in memory whole.
_CHUNK_BYTES: Final = 1 << 18  # 256 KiB

_PART_SUFFIX: Final = ".part"


def is_mobile_scale(nbytes: int) -> bool:
    """Return whether ``nbytes`` is small enough to be a mobile-scale artifact."""
    return 0 < nbytes <= MOBILE_SCALE_MAX_BYTES


def size_is_plausible(kind: str, nbytes: int) -> bool:
    """Return whether ``nbytes`` is within tolerance of the pin for ``kind``.

    ``kind`` is ``"encoder"`` or ``"decoder"``. An unknown kind is treated as
    plausible so a future third artifact does not spuriously warn.
    """
    expected = MOBILESAM_EXPECTED_BYTES.get(kind)
    if expected is None:
        return True
    slack = expected * MODEL_SIZE_TOLERANCE_FRACTION
    return expected - slack <= nbytes <= expected + slack


@dataclass(frozen=True, slots=True)
class ModelArtifacts:
    """Verified on-disk weights plus the provider the sessions will run on."""

    encoder_path: Path
    decoder_path: Path
    provider: str  # CPU_PROVIDER | CUDA_PROVIDER


class ModelUnavailable(RuntimeError):
    """Raised for every weight acquisition or session initialisation failure.

    ``str(exc)`` is exactly one of :data:`REASON_DOWNLOAD_FAILED`,
    :data:`REASON_CHECKSUM_MISMATCH`, or :data:`REASON_SESSION_INIT_FAILED`
    (Requirement 4.5). Human-readable context lives on :attr:`detail` so the
    reason stays machine-comparable.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class ModelLoader:
    """Locates, verifies, and opens the pinned MobileSAM ONNX pair."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._settings = settings
        self._logger = logger
        self._weights_dir = Path(settings.weights_dir).expanduser()
        # Requirement 4.2: the cache directory is configurable and created on
        # demand, so a fresh host needs no manual mkdir.
        self._weights_dir.mkdir(parents=True, exist_ok=True)

    @property
    def weights_dir(self) -> Path:
        """The resolved local cache directory for weights."""
        return self._weights_dir

    # ---------------------------------------------------------------- weights

    def ensure_weights(self) -> ModelArtifacts:
        """Locate or download and verify both ONNX files.

        A cached file is re-verified against its pin and discarded on mismatch,
        so a truncated earlier download self-heals rather than poisoning every
        later start. Raises :class:`ModelUnavailable` on any failure.
        """
        encoder_path = self._acquire(
            kind="encoder",
            url=self._settings.mobilesam_encoder_url,
            expected_sha256=self._settings.mobilesam_encoder_sha256,
        )
        decoder_path = self._acquire(
            kind="decoder",
            url=self._settings.mobilesam_decoder_url,
            expected_sha256=self._settings.mobilesam_decoder_sha256,
        )
        return ModelArtifacts(
            encoder_path=encoder_path,
            decoder_path=decoder_path,
            provider=self.select_provider(self._available_providers()),
        )

    def _acquire(self, kind: str, url: str, expected_sha256: str) -> Path:
        """Return the path of a digest-verified local copy of ``url``."""
        expected = expected_sha256.strip().lower()
        dest = self._weights_dir / self._filename_for(kind, url)

        if dest.is_file():
            actual = self._sha256_of_file(dest)
            if actual == expected:
                self._warn_on_implausible_size(kind, dest.stat().st_size)
                return dest
            # A cached file that no longer matches its pin is worthless; drop it
            # and fall through to a fresh download (Requirement 4.3).
            self._logger.warning(
                "cached weights discarded: %s kind=%s path=%s expected=%s actual=%s",
                REASON_CHECKSUM_MISMATCH,
                kind,
                dest,
                expected,
                actual,
            )
            self._discard(dest)

        self._download_verified(kind=kind, url=url, expected_sha256=expected, dest=dest)
        return dest

    def _download_verified(self, kind: str, url: str, expected_sha256: str, dest: Path) -> None:
        """Stream ``url`` into ``dest`` only if its SHA-256 matches the pin.

        The body lands in a sibling ``.part`` file whose digest is computed
        incrementally; the file is promoted with :func:`os.replace` only after
        the digest matches, so a reader never observes a partial or unverified
        artifact (Requirement 4.3).
        """
        if urlparse(url).scheme.lower() != "https":
            self._fail(REASON_DOWNLOAD_FAILED, f"weight URL is not https: {url!r}")

        part = dest.with_name(dest.name + _PART_SUFFIX)
        digest = hashlib.sha256()
        total = 0
        timeout = self._settings.model_download_timeout_s

        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with part.open("wb") as handle:
                        for chunk in response.iter_bytes(_CHUNK_BYTES):
                            handle.write(chunk)
                            digest.update(chunk)
                            total += len(chunk)
        except Exception as exc:  # noqa: BLE001 - every failure is a fallback
            # Deliberately broad: a transport error, an HTTP status error, a disk
            # error, or a test harness's offline guard all mean the same thing to
            # the caller, and none of them may leave a partial file behind.
            self._discard(part)
            self._fail(REASON_DOWNLOAD_FAILED, f"kind={kind} url={url} error={exc!r}", cause=exc)
        except BaseException:
            # Interpreter-level interruptions propagate untouched, but the partial
            # file still must not survive.
            self._discard(part)
            raise

        actual = digest.hexdigest()
        if actual != expected_sha256:
            self._discard(part)
            self._fail(
                REASON_CHECKSUM_MISMATCH,
                f"kind={kind} url={url} expected={expected_sha256} actual={actual} bytes={total}",
            )

        os.replace(part, dest)
        self._logger.info(
            "weights ready: kind=%s path=%s bytes=%d sha256=%s", kind, dest, total, actual
        )
        self._warn_on_implausible_size(kind, total)

    # --------------------------------------------------------------- sessions

    def select_provider(self, available: Sequence[str]) -> str:
        """Return the execution provider to use (Requirement 4.4).

        Pure function of ``available`` so it is property-testable over arbitrary
        provider lists: CUDA when present, CPU otherwise.
        """
        return CUDA_PROVIDER if CUDA_PROVIDER in tuple(available) else CPU_PROVIDER

    def create_sessions(self) -> tuple["ort.InferenceSession", "ort.InferenceSession", str]:
        """Build the encoder and decoder sessions on the selected provider.

        Raises :class:`ModelUnavailable` on any failure, including missing or
        unverifiable weights.
        """
        artifacts = self.ensure_weights()

        if ort is None:
            self._fail(REASON_SESSION_INIT_FAILED, "onnxruntime is not installed")

        try:
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # onnxruntime is chatty on stderr about provider fallbacks; the
            # loader reports what matters itself.
            options.log_severity_level = 3
            providers = [artifacts.provider]
            encoder = ort.InferenceSession(
                str(artifacts.encoder_path), sess_options=options, providers=providers
            )
            decoder = ort.InferenceSession(
                str(artifacts.decoder_path), sess_options=options, providers=providers
            )
        except ModelUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - any init failure is a fallback
            self._fail(
                REASON_SESSION_INIT_FAILED,
                f"provider={artifacts.provider} error={exc!r}",
                cause=exc,
            )

        self._logger.info(
            "onnx sessions ready: provider=%s encoder=%s decoder=%s",
            artifacts.provider,
            artifacts.encoder_path.name,
            artifacts.decoder_path.name,
        )
        return encoder, decoder, artifacts.provider

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _available_providers() -> tuple[str, ...]:
        """Providers onnxruntime reports, or CPU only when it is absent."""
        if ort is None:  # pragma: no cover - onnxruntime is a pinned dependency
            return (CPU_PROVIDER,)
        return tuple(ort.get_available_providers())

    @staticmethod
    def _filename_for(kind: str, url: str) -> str:
        """Derive a safe cache filename from ``url``.

        Only the basename of the URL path is used, and anything that could
        escape the cache directory falls back to a fixed name, so a hostile or
        malformed pin cannot write outside ``weights_dir``. The kind is folded
        into the name when the basename does not already carry it, so an encoder
        and a decoder mirrored under the same filename cannot collide in the
        cache.
        """
        name = Path(urlparse(url).path).name
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            return f"mobilesam_{kind}.onnx"
        if kind not in name.lower():
            return f"{kind}_{name}"
        return name

    @staticmethod
    def _sha256_of_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _discard(path: Path) -> None:
        """Remove ``path`` if present, ignoring a concurrent removal."""
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass

    def _warn_on_implausible_size(self, kind: str, nbytes: int) -> None:
        """Warn when an artifact is too large to be the pinned mobile-scale model.

        Only oversized artifacts warn. An undersized one is already excluded by
        the digest gate, and warning on it would make every fixture-sized test
        payload noisy; an oversized one is the case worth shouting about, since
        that is what a substituted full-size SAM checkpoint looks like
        (Requirement 12.2).
        """
        expected = MOBILESAM_EXPECTED_BYTES.get(kind, 0)
        oversized = expected and nbytes > expected * (1.0 + MODEL_SIZE_TOLERANCE_FRACTION)
        if oversized or not is_mobile_scale(nbytes):
            self._logger.warning(
                "unexpected weight size: kind=%s bytes=%d expected~%d; "
                "a full-size SAM checkpoint would not meet the mobile-scale bound of %d bytes",
                kind,
                nbytes,
                expected,
                MOBILE_SCALE_MAX_BYTES,
            )

    def _fail(self, reason: str, detail: str, cause: BaseException | None = None) -> NoReturn:
        """Log the fallback reason at WARNING and raise :class:`ModelUnavailable`.

        The reason string is in the message so an operator reading the log can
        tell which of the three fallback paths fired (Requirement 4.5).
        """
        self._logger.warning("neural backend unavailable: %s (%s)", reason, detail)
        raise ModelUnavailable(reason, detail) from cause
