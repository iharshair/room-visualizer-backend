"""Tests for `backend.utils.model_loader` (Requirements 4.2, 4.3, 4.4, 4.5, 12.2).

The Model_Loader is the one component that would otherwise reach the network, so
every test here proves a contract *without* one. Two mechanisms do that work:

* The autouse `no_network` guard from `tests/conftest.py` is left in place, so any
  code path that genuinely dials out fails loudly. Two tests rely on exactly
  that -- the offline download failure and the cached-weights short circuit --
  which is what makes them evidence for Requirement 4.7 rather than simulations
  of it.
* Where a *served* response is needed, `_serving` routes the `httpx.Client` the
  loader opens through an `httpx.MockTransport`, so a payload arrives through the
  real streaming, hashing, and `.part`-promotion code with no socket involved.

Four concerns share the module, in the order the loader traverses them:

* **Checksum gating.** Property 9 drives arbitrary payloads against matching and
  mismatched pins and holds the cache directory to the "nothing left behind"
  half of Requirement 4.3.
* **Provider selection.** Property 10 pins the pure `select_provider` decision
  over arbitrary provider lists (Requirement 4.4).
* **Fallback reasons and the WARNING contract.** Each of the three documented
  failure modes is driven end to end and checked twice: the raised
  `ModelUnavailable` carries the documented reason string, and a warning-level
  record naming that same reason reaches the log, so an operator can tell which
  fallback fired (Requirement 4.5).
* **Artifact scale.** The pinned byte sizes are held below the mobile-scale
  bound, so substituting a full-size SAM checkpoint is detectable
  (Requirement 12.2).
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest import mock

import httpx
import pytest
from hypothesis import HealthCheck, given, settings as hyp_settings, strategies as st

from backend.config import Settings
from backend.utils import model_loader
from backend.utils.model_loader import (
    CPU_PROVIDER,
    CUDA_PROVIDER,
    MOBILE_SCALE_MAX_BYTES,
    MOBILESAM_DECODER_EXPECTED_BYTES,
    MOBILESAM_ENCODER_EXPECTED_BYTES,
    MOBILESAM_EXPECTED_BYTES,
    MODEL_SIZE_TOLERANCE_FRACTION,
    REASON_CHECKSUM_MISMATCH,
    REASON_DOWNLOAD_FAILED,
    REASON_SESSION_INIT_FAILED,
    ModelArtifacts,
    ModelLoader,
    ModelUnavailable,
    is_mobile_scale,
    size_is_plausible,
)

# --------------------------------------------------------------------------- #
# Fixed test pins
# --------------------------------------------------------------------------- #
#
# A reserved-invalid host, so a bug that bypasses the mock transport cannot
# reach a real mirror even if the offline guard were somehow removed. The
# basenames carry "encoder"/"decoder", which is what makes the two artifacts
# land under distinct cache filenames.

_ENCODER_URL = "https://weights.invalid/mobile_sam.encoder.onnx"
_DECODER_URL = "https://weights.invalid/mobile_sam.decoder.onnx"

#: Cache filenames the loader derives from the URLs above. Spelled out rather
#: than recomputed, so a change to the naming rule shows up here as a failure.
_ENCODER_FILENAME = "mobile_sam.encoder.onnx"
_DECODER_FILENAME = "mobile_sam.decoder.onnx"

#: Distinct payloads, so a test that crosses the encoder and decoder wires
#: cannot pass by symmetry.
_ENCODER_PAYLOAD = b"onnx-encoder-bytes-" + bytes(range(256)) * 4
_DECODER_PAYLOAD = b"onnx-decoder-bytes-" + bytes(range(255, -1, -1)) * 4

#: The smallest full SAM export (ViT-B, ~375 MB) -- the substitution
#: Requirement 12.2 wants detectable.
_FULL_SIZE_SAM_BYTES = 375 * 1024 * 1024


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# Logger for tests that do *not* inspect the log. Detached from the root logger
# so the loader's expected warnings -- fixture payloads are never mobile-scale --
# do not spam a Hypothesis run of 100 examples.
_QUIET_LOGGER = logging.getLogger("tests.model_loader.quiet")
_QUIET_LOGGER.addHandler(logging.NullHandler())
_QUIET_LOGGER.propagate = False

#: Name used by the caplog tests; propagates normally so records reach caplog's
#: root handler.
_CAPTURED_LOGGER_NAME = "tests.model_loader.captured"


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


def _make_settings(
    weights_dir: Path,
    encoder_sha256: str,
    decoder_sha256: str,
    **overrides: Any,
) -> Settings:
    """A real `Settings` pointed at `weights_dir` with the given pins.

    Constructed with explicit keywords rather than environment variables so the
    pins are visible at the call site, and the real validators still run.
    """
    return Settings(
        weights_dir=weights_dir,
        mobilesam_encoder_url=_ENCODER_URL,
        mobilesam_decoder_url=_DECODER_URL,
        mobilesam_encoder_sha256=encoder_sha256,
        mobilesam_decoder_sha256=decoder_sha256,
        model_download_timeout_s=5.0,
        **overrides,
    )


def _make_loader(
    weights_dir: Path,
    encoder_sha256: str,
    decoder_sha256: str,
    logger: logging.Logger | None = None,
) -> ModelLoader:
    return ModelLoader(
        _make_settings(weights_dir, encoder_sha256, decoder_sha256),
        logger or _QUIET_LOGGER,
    )


Handler = Callable[[httpx.Request], httpx.Response]


def _payload_handler(
    payloads: dict[str, bytes],
    status_code: int = 200,
    seen: list[str] | None = None,
) -> Handler:
    """Serve `payloads` keyed by URL, recording each request into `seen`."""

    def handle(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(str(request.url))
        return httpx.Response(status_code, content=payloads[str(request.url)])

    return handle


@contextlib.contextmanager
def _serving(handler: Handler) -> Iterator[None]:
    """Route every `httpx.Client` the loader opens through a `MockTransport`.

    The loader builds its own client with no transport argument, so the client
    class itself is swapped for a factory that injects one. `MockTransport` is
    not a socket-backed transport, so the autouse offline guard lets these
    requests through while continuing to block real ones.
    """
    real_client = httpx.Client
    transport = httpx.MockTransport(handler)

    def _client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    with mock.patch.object(httpx, "Client", _client):
        yield


def _warning_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage() for record in caplog.records if record.levelno == logging.WARNING
    ]


def _part_files(weights_dir: Path) -> list[Path]:
    return sorted(weights_dir.glob("*.part"))


@pytest.fixture(scope="module")
def pure_loader(tmp_path_factory: pytest.TempPathFactory) -> ModelLoader:
    """A loader used only for the pure `select_provider` decision.

    Module-scoped so a `@given` test can request it without tripping
    Hypothesis's function-scoped-fixture health check.
    """
    weights_dir = tmp_path_factory.mktemp("provider-weights") / "weights"
    return _make_loader(weights_dir, _sha256(b"encoder"), _sha256(b"decoder"))


# --------------------------------------------------------------------------- #
# Property 9: checksum gating (Requirement 4.3)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 9: Model weights are accepted only
# on checksum match
@hyp_settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    payload=st.binary(min_size=0, max_size=2048),
    pin_matches=st.booleans(),
    corruption=st.binary(min_size=1, max_size=8),
)
def test_property_9_weights_are_retained_only_on_checksum_match(
    payload: bytes, pin_matches: bool, corruption: bytes
) -> None:
    """For any downloaded byte payload, the file is retained in the cache
    directory exactly when its SHA-256 digest equals the pinned digest, and
    every rejection leaves behind neither a `.part` file nor a promoted file.

    A fresh cache directory per example, rather than a `tmp_path` shared across
    the whole run, so "the rejection left nothing behind" is a statement about
    this payload and not about a neighbouring example's leftovers.

    **Validates: Requirements 4.3**
    """
    # When the pin is meant to mismatch it is the digest of a *different* byte
    # string, so it is still a structurally valid pin that `Settings` accepts --
    # the rejection has to come from the loader's comparison, not from config
    # validation.
    pin = _sha256(payload) if pin_matches else _sha256(payload + corruption)

    with tempfile.TemporaryDirectory() as tmp:
        weights_dir = Path(tmp) / "weights"
        loader = _make_loader(weights_dir, pin, pin)
        handler = _payload_handler({_ENCODER_URL: payload, _DECODER_URL: payload})

        with _serving(handler):
            if pin_matches:
                artifacts = loader.ensure_weights()

                assert artifacts.encoder_path.is_file()
                assert artifacts.decoder_path.is_file()
                assert artifacts.encoder_path.read_bytes() == payload
                assert artifacts.decoder_path.read_bytes() == payload
                assert sorted(path.name for path in weights_dir.iterdir()) == sorted(
                    (_ENCODER_FILENAME, _DECODER_FILENAME)
                )
            else:
                with pytest.raises(ModelUnavailable) as excinfo:
                    loader.ensure_weights()

                assert str(excinfo.value) == REASON_CHECKSUM_MISMATCH
                # Nothing promoted: a rejected payload must not be reachable by
                # any later start, or the digest gate would only delay it.
                assert list(weights_dir.iterdir()) == []

        # Holds on both branches: the `.part` staging file is an implementation
        # detail that must never outlive the call that created it.
        assert _part_files(weights_dir) == []


# --------------------------------------------------------------------------- #
# Property 10: provider selection (Requirement 4.4)
# --------------------------------------------------------------------------- #

#: Real onnxruntime provider names, so the strategy explores the actual decision
#: space rather than arbitrary strings.
_PROVIDER_POOL = (
    CUDA_PROVIDER,
    CPU_PROVIDER,
    "TensorrtExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "ROCMExecutionProvider",
)


# Feature: ai-room-tile-visualizer, Property 10: Provider selection prefers CUDA
# when present
@hyp_settings(max_examples=100, deadline=None)
@given(available=st.lists(st.sampled_from(_PROVIDER_POOL), min_size=0, max_size=6))
def test_property_10_provider_selection_prefers_cuda_when_present(
    pure_loader: ModelLoader, available: list[str]
) -> None:
    """For any list of available onnxruntime providers, `select_provider`
    returns `CUDAExecutionProvider` exactly when that provider appears in the
    list, and `CPUExecutionProvider` otherwise.

    **Validates: Requirements 4.4**
    """
    chosen = pure_loader.select_provider(available)

    assert chosen == (CUDA_PROVIDER if CUDA_PROVIDER in available else CPU_PROVIDER)
    # Never a third provider: the selection is a two-way decision, so an
    # accelerator in the list must not leak through as the chosen one.
    assert chosen in {CUDA_PROVIDER, CPU_PROVIDER}


def test_select_provider_accepts_any_sequence(pure_loader: ModelLoader) -> None:
    """The decision is over a sequence, not a list, and an empty one is CPU.

    `create_sessions` passes whatever `onnxruntime.get_available_providers`
    returns, which is a list today and a tuple in the loader's own fallback.
    """
    assert pure_loader.select_provider(()) == CPU_PROVIDER
    assert pure_loader.select_provider([]) == CPU_PROVIDER
    assert pure_loader.select_provider((CPU_PROVIDER, CUDA_PROVIDER)) == CUDA_PROVIDER
    assert pure_loader.select_provider(iter([CUDA_PROVIDER])) == CUDA_PROVIDER


# --------------------------------------------------------------------------- #
# Fallback reasons and the WARNING contract (Requirement 4.5)
# --------------------------------------------------------------------------- #


def test_download_transport_error_raises_download_failed_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Failure mode 1 of 3: the mirror is unreachable.

    Both halves of Requirement 4.5 are asserted together -- the machine-readable
    reason on the exception and the human-readable reason in the log -- because
    they describe one event and a test that checked only one could pass while
    the operator was left guessing.
    """
    loader = _make_loader(
        tmp_path / "weights",
        _sha256(_ENCODER_PAYLOAD),
        _sha256(_DECODER_PAYLOAD),
        logger=logging.getLogger(_CAPTURED_LOGGER_NAME),
    )

    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("mirror unreachable", request=request)

    with caplog.at_level(logging.WARNING, logger=_CAPTURED_LOGGER_NAME):
        with _serving(handle), pytest.raises(ModelUnavailable) as excinfo:
            loader.ensure_weights()

    exc = excinfo.value
    assert str(exc) == REASON_DOWNLOAD_FAILED
    assert exc.reason == REASON_DOWNLOAD_FAILED
    # Context lives on `detail`, keeping `str(exc)` machine-comparable.
    assert exc.detail is not None and _ENCODER_URL in exc.detail

    assert any(REASON_DOWNLOAD_FAILED in message for message in _warning_messages(caplog))
    assert _part_files(loader.weights_dir) == []
    assert list(loader.weights_dir.iterdir()) == []


def test_download_http_error_status_raises_download_failed(tmp_path: Path) -> None:
    """A reachable mirror serving 500 is still a download failure, not a
    checksum failure -- the body must never be hashed and offered to the pin."""
    loader = _make_loader(
        tmp_path / "weights", _sha256(_ENCODER_PAYLOAD), _sha256(_DECODER_PAYLOAD)
    )
    handler = _payload_handler(
        {_ENCODER_URL: b"<html>mirror error</html>", _DECODER_URL: b""},
        status_code=500,
    )

    with _serving(handler), pytest.raises(ModelUnavailable) as excinfo:
        loader.ensure_weights()

    assert str(excinfo.value) == REASON_DOWNLOAD_FAILED
    assert list(loader.weights_dir.iterdir()) == []


def test_download_without_a_reachable_network_raises_download_failed(tmp_path: Path) -> None:
    """No mock transport at all: the autouse offline guard is what fires.

    This is the Requirement 4.7 precondition exercised through the real client,
    so the offline path is evidenced rather than simulated -- an absent network
    degrades to a documented fallback reason instead of an unhandled error.
    """
    loader = _make_loader(
        tmp_path / "weights", _sha256(_ENCODER_PAYLOAD), _sha256(_DECODER_PAYLOAD)
    )

    with pytest.raises(ModelUnavailable) as excinfo:
        loader.ensure_weights()

    assert str(excinfo.value) == REASON_DOWNLOAD_FAILED
    assert _part_files(loader.weights_dir) == []


def test_non_https_weight_url_raises_download_failed_without_a_request(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The loader re-checks the scheme instead of trusting its configuration.

    `Settings` already rejects a non-https pin, so reaching the loader's own
    guard requires bypassing validation with `model_construct`. That is the
    point: the check is defence in depth for a settings object assembled some
    other way, and no request may leave the process.
    """
    weights_dir = tmp_path / "weights"
    unvalidated = Settings.model_construct(
        weights_dir=weights_dir,
        mobilesam_encoder_url="http://weights.invalid/mobile_sam.encoder.onnx",
        mobilesam_decoder_url=_DECODER_URL,
        mobilesam_encoder_sha256=_sha256(_ENCODER_PAYLOAD),
        mobilesam_decoder_sha256=_sha256(_DECODER_PAYLOAD),
        model_download_timeout_s=5.0,
    )
    loader = ModelLoader(unvalidated, logging.getLogger(_CAPTURED_LOGGER_NAME))

    seen: list[str] = []
    handler = _payload_handler({}, seen=seen)

    with caplog.at_level(logging.WARNING, logger=_CAPTURED_LOGGER_NAME):
        with _serving(handler), pytest.raises(ModelUnavailable) as excinfo:
            loader.ensure_weights()

    assert str(excinfo.value) == REASON_DOWNLOAD_FAILED
    assert seen == []
    assert any(REASON_DOWNLOAD_FAILED in message for message in _warning_messages(caplog))


def test_checksum_mismatch_raises_checksum_mismatch_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Failure mode 2 of 3: the mirror served something other than the pin.

    Property 9 already covers the retention rule across arbitrary payloads; this
    pins the reason string and the operator-facing warning for the concrete
    case.
    """
    loader = _make_loader(
        tmp_path / "weights",
        _sha256(b"a completely different encoder"),
        _sha256(b"a completely different decoder"),
        logger=logging.getLogger(_CAPTURED_LOGGER_NAME),
    )
    handler = _payload_handler({_ENCODER_URL: _ENCODER_PAYLOAD, _DECODER_URL: _DECODER_PAYLOAD})

    with caplog.at_level(logging.WARNING, logger=_CAPTURED_LOGGER_NAME):
        with _serving(handler), pytest.raises(ModelUnavailable) as excinfo:
            loader.ensure_weights()

    exc = excinfo.value
    assert str(exc) == REASON_CHECKSUM_MISMATCH
    assert exc.reason == REASON_CHECKSUM_MISMATCH

    assert any(REASON_CHECKSUM_MISMATCH in message for message in _warning_messages(caplog))
    assert list(loader.weights_dir.iterdir()) == []


def test_session_init_error_raises_session_init_failed_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure mode 3 of 3: verified weights that onnxruntime cannot open.

    The weights must verify first, so this isolates session construction from
    acquisition -- a loader that reported `weights_download_failed` here would
    send the operator to the wrong place.
    """
    loader = _make_loader(
        tmp_path / "weights",
        _sha256(_ENCODER_PAYLOAD),
        _sha256(_DECODER_PAYLOAD),
        logger=logging.getLogger(_CAPTURED_LOGGER_NAME),
    )
    handler = _payload_handler({_ENCODER_URL: _ENCODER_PAYLOAD, _DECODER_URL: _DECODER_PAYLOAD})

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("invalid onnx graph")

    monkeypatch.setattr(model_loader.ort, "InferenceSession", _boom)

    with caplog.at_level(logging.WARNING, logger=_CAPTURED_LOGGER_NAME):
        with _serving(handler), pytest.raises(ModelUnavailable) as excinfo:
            loader.create_sessions()

    exc = excinfo.value
    assert str(exc) == REASON_SESSION_INIT_FAILED
    assert exc.reason == REASON_SESSION_INIT_FAILED

    assert any(REASON_SESSION_INIT_FAILED in message for message in _warning_messages(caplog))
    # The weights themselves verified, so they stay cached for the next start.
    assert (loader.weights_dir / _ENCODER_FILENAME).is_file()


def test_the_three_fallback_reasons_are_distinct_constants() -> None:
    """Requirement 4.5's three reasons must be distinguishable at a glance.

    Guards against a copy-paste that collapses two fallback paths onto one
    string, which would make the log ambiguous exactly when it matters.
    """
    reasons = (REASON_DOWNLOAD_FAILED, REASON_CHECKSUM_MISMATCH, REASON_SESSION_INIT_FAILED)

    assert len(set(reasons)) == 3
    assert reasons == ("weights_download_failed", "checksum_mismatch", "onnx_session_init_failed")


# --------------------------------------------------------------------------- #
# Cached weights (Requirements 4.2, 4.3, 4.7)
# --------------------------------------------------------------------------- #


def test_cached_weights_matching_their_pins_are_reused_offline(tmp_path: Path) -> None:
    """A digest-matching cache short-circuits the download entirely.

    No mock transport is installed, so the offline guard would raise the moment
    the loader tried to dial out. `ensure_weights` succeeding is therefore proof
    that a cached pair boots an offline host (Requirements 4.2, 4.7).
    """
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / _ENCODER_FILENAME).write_bytes(_ENCODER_PAYLOAD)
    (weights_dir / _DECODER_FILENAME).write_bytes(_DECODER_PAYLOAD)
    loader = _make_loader(
        weights_dir, _sha256(_ENCODER_PAYLOAD), _sha256(_DECODER_PAYLOAD)
    )

    artifacts = loader.ensure_weights()

    assert isinstance(artifacts, ModelArtifacts)
    assert artifacts.encoder_path == weights_dir / _ENCODER_FILENAME
    assert artifacts.decoder_path == weights_dir / _DECODER_FILENAME
    assert artifacts.provider in {CPU_PROVIDER, CUDA_PROVIDER}


def test_cached_weights_failing_their_pins_are_discarded_and_refetched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A stale cache self-heals rather than poisoning every later start.

    Requirement 4.3's "discard any file whose checksum does not match" has to
    apply to files already on disk, not only to fresh downloads: a truncated
    earlier download would otherwise be permanent.
    """
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True)
    (weights_dir / _ENCODER_FILENAME).write_bytes(b"truncated from an earlier run")
    (weights_dir / _DECODER_FILENAME).write_bytes(b"also truncated")
    loader = _make_loader(
        weights_dir,
        _sha256(_ENCODER_PAYLOAD),
        _sha256(_DECODER_PAYLOAD),
        logger=logging.getLogger(_CAPTURED_LOGGER_NAME),
    )
    handler = _payload_handler({_ENCODER_URL: _ENCODER_PAYLOAD, _DECODER_URL: _DECODER_PAYLOAD})

    with caplog.at_level(logging.WARNING, logger=_CAPTURED_LOGGER_NAME):
        with _serving(handler):
            artifacts = loader.ensure_weights()

    assert artifacts.encoder_path.read_bytes() == _ENCODER_PAYLOAD
    assert artifacts.decoder_path.read_bytes() == _DECODER_PAYLOAD
    assert any(REASON_CHECKSUM_MISMATCH in message for message in _warning_messages(caplog))
    assert _part_files(weights_dir) == []


def test_create_sessions_builds_both_sessions_on_the_selected_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path threads `select_provider`'s answer into both sessions.

    Requirement 4.4 is about the provider onnxruntime is actually asked for, so
    the stub records the `providers` argument rather than only counting calls.
    """
    loader = _make_loader(
        tmp_path / "weights", _sha256(_ENCODER_PAYLOAD), _sha256(_DECODER_PAYLOAD)
    )
    handler = _payload_handler({_ENCODER_URL: _ENCODER_PAYLOAD, _DECODER_URL: _DECODER_PAYLOAD})

    calls: list[tuple[str, list[str]]] = []

    class _StubSession:
        def __init__(self, path: str, sess_options: Any = None, providers: Any = None) -> None:
            calls.append((path, list(providers or ())))

    monkeypatch.setattr(model_loader.ort, "InferenceSession", _StubSession)
    monkeypatch.setattr(
        model_loader.ort, "get_available_providers", lambda: [CUDA_PROVIDER, CPU_PROVIDER]
    )

    with _serving(handler):
        encoder, decoder, provider = loader.create_sessions()

    assert isinstance(encoder, _StubSession)
    assert isinstance(decoder, _StubSession)
    assert provider == CUDA_PROVIDER
    assert [providers for _, providers in calls] == [[CUDA_PROVIDER], [CUDA_PROVIDER]]
    assert [Path(path).name for path, _ in calls] == [_ENCODER_FILENAME, _DECODER_FILENAME]


# --------------------------------------------------------------------------- #
# Pinned artifact scale (Requirement 12.2)
# --------------------------------------------------------------------------- #


def test_pinned_artifact_sizes_stay_within_the_mobile_scale_bound() -> None:
    """The pins describe MobileSAM, and MobileSAM is a mobile-scale model.

    Asserted on the sum as well as the parts: the pair is downloaded together,
    so it is the combined footprint that has to fit a modest CPU host.
    """
    assert set(MOBILESAM_EXPECTED_BYTES) == {"encoder", "decoder"}
    assert MOBILESAM_EXPECTED_BYTES["encoder"] == MOBILESAM_ENCODER_EXPECTED_BYTES
    assert MOBILESAM_EXPECTED_BYTES["decoder"] == MOBILESAM_DECODER_EXPECTED_BYTES

    assert is_mobile_scale(MOBILESAM_ENCODER_EXPECTED_BYTES)
    assert is_mobile_scale(MOBILESAM_DECODER_EXPECTED_BYTES)
    assert is_mobile_scale(sum(MOBILESAM_EXPECTED_BYTES.values()))


def test_is_mobile_scale_brackets_the_documented_bound() -> None:
    assert is_mobile_scale(MOBILE_SCALE_MAX_BYTES)
    assert not is_mobile_scale(MOBILE_SCALE_MAX_BYTES + 1)
    # A zero-byte artifact is not a plausible model either.
    assert not is_mobile_scale(0)
    assert not is_mobile_scale(-1)
    assert not is_mobile_scale(_FULL_SIZE_SAM_BYTES)


def test_size_is_plausible_accepts_the_pins_and_rejects_a_full_size_checkpoint() -> None:
    """The scale check is what makes a swapped-in full SAM export detectable."""
    for kind, expected in MOBILESAM_EXPECTED_BYTES.items():
        slack = int(expected * MODEL_SIZE_TOLERANCE_FRACTION)

        assert size_is_plausible(kind, expected)
        assert size_is_plausible(kind, expected + slack)
        assert size_is_plausible(kind, expected - slack)
        assert not size_is_plausible(kind, expected + 2 * slack)
        assert not size_is_plausible(kind, _FULL_SIZE_SAM_BYTES)

    # Documented leniency: an unknown kind has no pin, so nothing to contradict.
    assert size_is_plausible("some-future-artifact", _FULL_SIZE_SAM_BYTES)
