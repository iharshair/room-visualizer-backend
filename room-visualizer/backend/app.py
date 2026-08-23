"""Visualizer_API -- the FastAPI application and its wiring.

This module is the only place in the service that knows about HTTP. The domain
modules take and return numpy arrays; the routes here adapt them to requests,
responses, and status codes.

Four structural decisions shape the file:

**Everything expensive is built once, in ``lifespan``.** The ``Settings``, the
``SceneCache``, the ``CatalogLoader``, and the segmenter all land on
``app.state`` at startup, and routes read them from ``request.app.state`` rather
than from module globals. That is what lets the test harness point the app at a
temporary asset tree by setting ``RV_ASSETS_DIR`` before the ``TestClient``
context is entered, even though this module may already be imported.

**Startup never raises.** Weight acquisition is the one startup step that can
fail, and it is expected to fail on an offline host. :func:`build_segmenter`
converts every :class:`ModelUnavailable` into a ``WARNING`` log naming the
concrete reason plus a classical segmenter, so an offline or GPU-less host boots
and serves (Requirements 4.5, 4.7).

**Every failure leaves through one envelope.** Four exception handlers -- for
:class:`ApiError`, ``HTTPException``, request validation, and anything
unhandled -- all emit ``{"error": {"code", "message"}}``. The frontend renders
``message`` verbatim and branches on ``code``, so no endpoint needs a bespoke
error shape (Requirements 1.6, 10.6).

**API routes are registered on a router, static files are mounted last.** A
Starlette ``Mount`` at ``/`` matches every path, so it has to be the final route
in the table. Collecting the API surface on :data:`api_router` means later
endpoints can be added anywhere in this file without accidentally landing behind
the catch-all mount.

**Upload validation is ordered, and the order is the contract.** Size, MIME
type, extension, and an actual decode are checked in that sequence and each one
short-circuits, so no segmentation, geometry, or lighting work is reachable by
input that has not passed all four (Requirements 2.1-2.3, 2.5).

**The two passes share nothing but the cache.** ``/api/segment`` writes a
Scene_State; ``/api/render`` reads one and calls no analysis code at all. The
Compositor takes the cached homographies and lighting maps as plain data, so
there is no path from the render route back into the Segmenter or the
Geometry_Engine (Requirement 9.2) -- which is what buys the render budget of
Requirement 9.3: 70 ms fixed plus 40 ms per tiled plane at a 1600 px longest
edge.

Requirements: 1.1-1.6, 2.1-2.7, 4.5, 4.6, 4.7, 6.3, 6.5, 8.4, 9.1-9.4, 12.4,
12.5.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import IO, AsyncIterator, Callable, Final, TypeVar
from urllib.parse import quote

import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send

from backend.cache import SceneCache
from backend.catalog import CatalogLoader
from backend.config import Settings, get_settings
from backend.core.compositor import compose, encode_render
from backend.core.geometry import Line, calibrate
from backend.core.lighting import decompose
from backend.core.segmenter import (
    ClassicalSegmenter,
    NeuralSegmenter,
    SegmentationResult,
    Segmenter,
)
from backend.schemas import (
    PLANE_NAMES,
    ErrorEnvelope,
    HealthResponse,
    HorizonResponse,
    PlaneMetadata,
    PlaneName,
    PlaneRenderSpec,
    PlaneResponse,
    RenderRequest,
    RenderResponse,
    SceneState,
    SegmentResponse,
    TileDefinition,
    TileResponse,
    TilesResponse,
)
from backend.utils.imageio import DecodeError, clamp_longest_edge, decode_image
from backend.utils.model_loader import ModelLoader, ModelUnavailable
from backend.utils.texture_helper import SeamlessTexture

__all__ = [
    "app",
    "api_router",
    "ApiError",
    "build_segmenter",
    "NO_PROVIDER",
    "ASSETS_URL_PREFIX",
    "UPLOAD_CHUNK_BYTES",
    "VP_LABELS",
]

_T = TypeVar("_T")

logger = logging.getLogger("backend.app")

#: Reported as ``onnx_provider`` by ``/api/health`` when no onnxruntime session
#: is open, which is the case whenever the classical backend is active.
NO_PROVIDER = "n/a"

#: URL prefix the assets mount is served under. ``thumbnail_url`` values in
#: ``/api/tiles`` responses are built from it, so the two cannot drift.
ASSETS_URL_PREFIX = "/assets"

#: Chunk size for the streamed upload read. One mebibyte keeps the read count
#: low for a legitimate photograph while bounding how far past the configured cap
#: a hostile payload can push the accumulator before the check fires.
UPLOAD_CHUNK_BYTES: Final[int] = 1 << 20

#: The three vanishing point labels, reported in this order so the response shape
#: is stable whether or not a given label was recovered.
VP_LABELS: Final[tuple[str, str, str]] = ("VPx", "VPy", "VPz")

#: Requirement 6.5's message. Written for a shopper, because the frontend renders
#: it verbatim, and it names the concrete corrective action.
_NO_USABLE_PLANE_MESSAGE: Final[str] = (
    "No wall or floor large enough to tile was found in this photo. "
    "Try a photo showing more of the floor."
)

#: Requirement 9.4's message. One message covers both "never existed" and
#: "evicted", because the two are indistinguishable to the client and the
#: recovery action -- upload the photo again -- is the same either way.
_SCENE_EXPIRED_MESSAGE: Final[str] = (
    "This room photo is no longer loaded. Please upload it again to keep "
    "trying tiles."
)

#: Shopper-facing names for the Structural_Planes, for the ``unknown_plane``
#: message. The wire names are snake_case identifiers; "wall_left" in a sentence
#: the frontend shows verbatim would read as a leaked internal.
_PLANE_LABELS: Final[dict[PlaneName, str]] = {
    "floor": "floor",
    "wall_left": "left wall",
    "wall_right": "right wall",
    "wall_back": "back wall",
}

#: ``b`` below this magnitude means the horizon is (effectively) a vertical line,
#: which has no single row to report.
_HORIZON_EPS: Final[float] = 1e-9

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

#: HTTP status to error code, for failures raised as bare ``HTTPException``s --
#: chiefly the ones Starlette itself produces for unrouted paths and disallowed
#: methods. Routes that own their code raise :class:`ApiError` instead.
_STATUS_CODES: dict[int, str] = {
    400: "bad_request",
    404: "not_found",
    405: "method_not_allowed",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "unprocessable_entity",
    500: "internal_error",
}


class ApiError(Exception):
    """A failure with a chosen status, machine-readable code, and message.

    Routes raise this instead of ``HTTPException`` so the code carried to the
    client is the documented one (``no_usable_plane``, ``scene_expired``, and so
    on) rather than something inferred from the status. ``message`` is written
    for shoppers, because the frontend displays it verbatim (Requirement 10.6).
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.status_code = int(status_code)
        self.code = code
        self.message = message


def _envelope(status_code: int, code: str, message: str) -> JSONResponse:
    """Render the shared ``{"error": {...}}`` body (Requirement 1.6)."""
    return JSONResponse(
        status_code=status_code,
        content=ErrorEnvelope.of(code, message).model_dump(),
    )


# --------------------------------------------------------------------------- #
# Segmentation backend selection
# --------------------------------------------------------------------------- #


def build_segmenter(settings: Settings, log: logging.Logger) -> tuple[Segmenter, str]:
    """Return the segmenter to serve with, plus its onnxruntime provider.

    The single decision point for Requirement 4.5. Three outcomes:

    * ``enable_neural_backend`` is false -- the classical backend is returned
      without consulting the Model_Loader at all, so nothing touches the network
      or the weights directory. This is the test harness's configuration.
    * the sessions open -- the neural backend is returned with the provider that
      was selected (CUDA when available, CPU otherwise).
    * :class:`ModelUnavailable` -- the concrete reason
      (``weights_download_failed``, ``checksum_mismatch``, or
      ``onnx_session_init_failed``) is logged at ``WARNING`` and the classical
      backend is returned.

    This function does not raise. An unexpected exception from the loader is
    treated exactly like ``ModelUnavailable``, because a startup crash would take
    down a service that is perfectly capable of serving without weights
    (Requirement 4.7).

    Returns:
        ``(segmenter, provider)``, where ``provider`` is :data:`NO_PROVIDER`
        whenever the classical backend is active.
    """
    if not settings.enable_neural_backend:
        log.info("neural backend disabled by configuration; using classical backend")
        return ClassicalSegmenter(settings), NO_PROVIDER

    try:
        encoder, decoder, provider = ModelLoader(settings, log).create_sessions()
    except ModelUnavailable as exc:
        log.warning(
            "neural_backend_unavailable reason=%s detail=%s; using classical backend",
            exc.reason,
            exc.detail,
        )
        return ClassicalSegmenter(settings), NO_PROVIDER
    except Exception as exc:  # noqa: BLE001 - startup must never raise (R4.7)
        log.warning(
            "neural_backend_unavailable reason=%s detail=%r; using classical backend",
            "unexpected_loader_error",
            exc,
        )
        return ClassicalSegmenter(settings), NO_PROVIDER

    log.info("neural backend ready: provider=%s", provider)
    return NeuralSegmenter(encoder, decoder, settings, logger=log), provider


# --------------------------------------------------------------------------- #
# Static file serving
# --------------------------------------------------------------------------- #


class _SettingsStaticFiles:
    """ASGI app serving a directory resolved from ``app.state`` per request.

    ``assets_dir`` is a setting, and the settings object is built in
    ``lifespan`` rather than at import, so a mount bound to a directory at
    import time would serve the wrong tree whenever the configuration changed
    after this module was first imported -- exactly what the test harness does
    when it points ``RV_ASSETS_DIR`` at a temporary catalog.

    Resolving per request keeps ``/assets`` and the ``thumbnail_url`` values in
    ``/api/tiles`` describing the same files. One ``StaticFiles`` instance is
    memoised per distinct directory; the memo is dropped once it grows past a
    handful of entries, which only happens across many reconfigurations.
    """

    _MAX_CACHED = 8

    def __init__(self, resolve: Callable[[], Path], *, html: bool = False) -> None:
        self._resolve = resolve
        self._html = html
        self._lock = threading.Lock()
        self._apps: dict[Path, StaticFiles] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        directory = Path(self._resolve())
        with self._lock:
            static = self._apps.get(directory)
            if static is None:
                if len(self._apps) >= self._MAX_CACHED:
                    self._apps.clear()
                # ``check_dir=False``: a missing assets directory must not break
                # startup, it just makes every asset request a 404.
                static = StaticFiles(directory=directory, html=self._html, check_dir=False)
                self._apps[directory] = static
        await static(scope, receive, send)


# --------------------------------------------------------------------------- #
# Lifespan
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Build the per-process components, then release them on shutdown.

    The catalog is loaded eagerly so an invalid manifest entry is reported at
    startup rather than on the first ``/api/tiles`` request (Requirement 8.8).
    The cache is cleared on the way out so a draining process does not hold tens
    of megabytes per cached scene (Requirement 12.3).
    """
    settings = get_settings()
    application.state.settings = settings
    application.state.cache = SceneCache(
        settings.scene_cache_max_entries,
        settings.scene_cache_ttl_seconds,
    )
    application.state.catalog = CatalogLoader(
        settings.assets_dir,
        logger,
        manifest_name=settings.tiles_manifest_name,
    )
    application.state.catalog.load()
    application.state.segmenter, application.state.onnx_provider = build_segmenter(
        settings, logger
    )
    logger.info(
        "visualizer api ready: backend=%s provider=%s assets=%s",
        application.state.segmenter.backend_name,
        application.state.onnx_provider,
        settings.assets_dir,
    )
    try:
        yield
    finally:
        application.state.cache.clear()


app = FastAPI(
    lifespan=lifespan,
    title="Room Visualizer API",
    summary="Preview physical tile products on a photograph of a room.",
    # The shared error envelope is not FastAPI's default shape, so it is declared
    # once here and inherited by every route.
    responses={"4XX": {"model": ErrorEnvelope}, "5XX": {"model": ErrorEnvelope}},
)

# Requirement 2.7: a configurable CORS allow-list defaulting to every origin for
# local development. Read from module-level settings because middleware is added
# at import time, before ``lifespan`` runs; the allow-list is deployment
# configuration, so it is set in the environment before the process starts.
_startup_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(_startup_settings.cors_allow_origins),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Exception handlers -- one envelope for every failure
# --------------------------------------------------------------------------- #


@app.exception_handler(ApiError)
async def _handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    """Routes' own failures, with the documented code preserved."""
    return _envelope(exc.status_code, exc.code, exc.message)


@app.exception_handler(StarletteHTTPException)
async def _handle_http_exception(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Framework-raised failures: unrouted paths, disallowed methods.

    A ``detail`` that is already a ``{"code", "message"}`` mapping is honoured,
    so raising ``HTTPException`` remains a usable escape hatch.
    """
    detail = exc.detail
    if isinstance(detail, dict) and "code" in detail:
        code = str(detail["code"])
        message = str(detail.get("message", ""))
    else:
        code = _STATUS_CODES.get(exc.status_code, "error")
        message = str(detail) if detail else code.replace("_", " ")
    return _envelope(exc.status_code, code, message)


@app.exception_handler(RequestValidationError)
async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Malformed request bodies and query parameters.

    422 rather than 400 to match FastAPI's own convention, with the field path
    and reason folded into one sentence so the frontend has something to show.
    """
    problems = "; ".join(
        f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg', '')}".strip(": ")
        for error in exc.errors()
    )
    return _envelope(
        422,
        "invalid_request",
        problems or "The request could not be understood.",
    )


@app.exception_handler(Exception)
async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    """Last resort, so no failure escapes without the shared envelope."""
    logger.exception("unhandled error serving %s %s", request.method, request.url.path)
    return _envelope(
        500,
        "internal_error",
        "Something went wrong preparing your preview. Please try again.",
    )


# --------------------------------------------------------------------------- #
# Upload validation -- everything here runs before any pipeline stage
# --------------------------------------------------------------------------- #


def _normalise_mime(content_type: str | None) -> str:
    """Lowercase media type with any parameters (``; charset=...``) stripped."""
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def _unsupported_media_message(settings: Settings) -> str:
    """Shopper-facing 415 message naming the formats this deployment accepts."""
    formats = sorted({ext.lstrip(".").upper() for ext in settings.allowed_extensions})
    return (
        "That file is not an image we can read. Please choose a photo saved as "
        f"{', '.join(formats)}."
    )


def _read_upload(stream: IO[bytes], max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` from ``stream``, else raise 413.

    The accumulated byte count is the authoritative size rather than
    ``Content-Length``: a client is free to understate the header, and the header
    covers the whole multipart body -- boundaries and all -- rather than the file
    part the cap is written about, so trusting it would reject a photograph that
    is actually inside the limit. Reading in bounded chunks and aborting the
    moment the count passes the cap means an oversized payload is never fully
    materialised, whatever it claimed (Requirements 2.4, 2.5).
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            limit_mb = max_bytes / (1024 * 1024)
            raise ApiError(
                413,
                "payload_too_large",
                f"That photo is larger than the {limit_mb:.0f} MB upload limit. "
                "Please choose a smaller file.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_upload(upload: UploadFile, settings: Settings) -> np.ndarray:
    """Return the decoded photograph, or raise the documented rejection.

    The order is fixed and every step short-circuits, so no segmentation,
    geometry, or lighting work can be reached by input that has not passed all
    four checks (Requirement 2.3):

    1. streamed size against ``max_upload_bytes`` -- 413 ``payload_too_large``
    2. declared MIME type against the allow-list -- 415
    3. filename extension against the allow-list -- 415
    4. the bytes actually decode as a raster image -- 415

    MIME type and extension are both checked because either alone is trivially
    forged, and neither proves the content is an image, which is what the decode
    check is for -- a renamed executable or a polyglot file dies at step 4
    (Requirements 2.1, 2.2).
    """
    # Starlette's multipart parser rewinds the spooled part after writing it, but
    # rewinding again costs nothing and makes this function safe to call on a
    # stream any caller handed us.
    try:
        upload.file.seek(0)
    except (OSError, ValueError):  # pragma: no cover - non-seekable stream
        pass

    data = _read_upload(upload.file, settings.max_upload_bytes)

    if _normalise_mime(upload.content_type) not in settings.allowed_mime_types:
        raise ApiError(
            415, "unsupported_media_type", _unsupported_media_message(settings)
        )

    if Path(upload.filename or "").suffix.lower() not in settings.allowed_extensions:
        raise ApiError(
            415, "unsupported_media_type", _unsupported_media_message(settings)
        )

    try:
        return decode_image(data)
    except DecodeError as exc:
        # INFO, not WARNING: a shopper picking the wrong file is routine traffic,
        # not an operational problem.
        logger.info("upload rejected at decode: %s", exc)
        raise ApiError(
            415, "unsupported_media_type", _unsupported_media_message(settings)
        ) from exc


# --------------------------------------------------------------------------- #
# Analysis pass helpers
# --------------------------------------------------------------------------- #


def _stage(name: str, run: Callable[[], _T]) -> _T:
    """Run one analysis stage, converting any failure to 422 ``analysis_failed``.

    Requirement 1.6 asks for a machine-readable code plus a human-readable
    message on *any* stage failure, and carrying the stage name in the message is
    what makes a support report actionable. :class:`ApiError` passes through
    untouched so a deliberate rejection raised inside a stage -- ``no_usable_plane``
    in particular -- keeps its own code and status.
    """
    try:
        return run()
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001 - every stage failure is one 422
        logger.exception("analysis stage %r failed", name)
        raise ApiError(
            422,
            "analysis_failed",
            f"We could not finish analysing this photo (stage: {name}). "
            "Please try a different photo.",
        ) from exc


def _has_usable_plane(segmentation: SegmentationResult, settings: Settings) -> bool:
    """Whether some Structural_Plane reaches ``min_plane_area_fraction`` (R6.5).

    The Segmenter already drops sub-threshold planes, so this is normally a
    non-empty check; it is written against the fractions anyway so lowering the
    threshold in settings cannot leave the two disagreeing.
    """
    threshold = settings.min_plane_area_fraction
    return any(
        fraction >= threshold for fraction in segmentation.area_fractions.values()
    )


def _plane_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Mask centroid in image pixels, via image moments.

    ``cv2.moments`` runs in C over the mask rather than materialising the
    coordinate arrays ``np.nonzero`` would allocate, which matters at 2048 px.
    """
    moments = cv2.moments(mask, binaryImage=True)
    area = float(moments["m00"])
    if area <= 0.0:  # pragma: no cover - reported planes always have area
        height, width = mask.shape[:2]
        return ((width - 1) / 2.0, (height - 1) / 2.0)
    return (float(moments["m10"] / area), float(moments["m01"] / area))


def _horizon_response(horizon: Line, width: int, height: int) -> HorizonResponse:
    """Package the homogeneous horizon plus its row at the image centre column.

    ``y_at_center`` is a convenience for the frontend, which draws the horizon as
    a screen-space guide and would otherwise have to solve the line itself.
    """
    a, b, c = (float(horizon[0]), float(horizon[1]), float(horizon[2]))
    centre_x = (width - 1) / 2.0
    if abs(b) > _HORIZON_EPS:
        y_at_center = -(a * centre_x + c) / b
    else:  # pragma: no cover - the Geometry_Engine never returns a vertical horizon
        # A vertical horizon has no single row; mid-height keeps the field finite
        # so the frontend never has to guard against an infinity.
        y_at_center = (height - 1) / 2.0
    return HorizonResponse(a=a, b=b, c=c, y_at_center=float(y_at_center))


# --------------------------------------------------------------------------- #
# API routes
# --------------------------------------------------------------------------- #

#: Every HTTP endpoint hangs off this router. It is included below, before the
#: static mounts, so the catch-all ``/`` mount can never shadow an API path.
api_router = APIRouter(prefix="/api", tags=["visualizer"])


@api_router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> HealthResponse:
    """Liveness plus the runtime facts an operator needs (Requirement 12.5).

    Reports the segmentation backend actually in use -- so a host that fell back
    to the classical backend says so here rather than only in the startup log
    (Requirement 4.6) -- the onnxruntime provider, and the live cache occupancy
    against its configured bounds.
    """
    state = request.app.state
    cache: SceneCache = state.cache
    segmenter: Segmenter = state.segmenter
    stats = cache.stats()
    return HealthResponse(
        segmentation_backend=segmenter.backend_name,
        onnx_provider=state.onnx_provider,
        scene_cache_entries=int(stats["scene_cache_entries"]),
        scene_cache_max_entries=int(stats["scene_cache_max_entries"]),
        scene_cache_ttl_seconds=int(stats["scene_cache_ttl_seconds"]),
    )


@api_router.get("/tiles", response_model=TilesResponse)
def get_tiles(request: Request) -> TilesResponse:
    """Every valid Tile_Definition in the Asset_Catalog (Requirement 8.4).

    The Catalog_Loader re-reads the manifest when its modification stamp
    changed, so a tile image plus a manifest entry dropped into
    ``assets/tiles/`` appears here with no Python edit and no restart
    (Requirement 8.5). Invalid entries are absent and were logged at ``WARNING``
    (Requirement 8.8).
    """
    catalog: CatalogLoader = request.app.state.catalog
    tiles_dir = catalog.tiles_dir
    published: list[TileResponse] = []
    for tile in catalog.all():
        # Manifest ``file`` values are confined to the tiles directory by the
        # Catalog_Loader and stored unresolved, so this relative path is always
        # well formed. It is percent-encoded because a merchandiser's filename
        # may legitimately contain a space or another character that would
        # otherwise not survive the round trip to the ``/assets`` mount; ``/`` is
        # left intact so a nested path stays a path.
        relative = quote(tile.image_path.relative_to(tiles_dir).as_posix())
        published.append(
            TileResponse(
                id=tile.id,
                name=tile.name,
                width_mm=tile.width_mm,
                height_mm=tile.height_mm,
                finish=tile.finish,
                gloss=tile.gloss,
                thumbnail_url=f"{ASSETS_URL_PREFIX}/tiles/{relative}",
            )
        )
    return TilesResponse(tiles=published)


@api_router.post("/segment", response_model=SegmentResponse)
def post_segment(request: Request, file: UploadFile = File(...)) -> SegmentResponse:
    """Analyse one room photograph, once, and cache the result (Requirement 1.1).

    Declared with ``def`` rather than ``async def`` on purpose: the body is
    seconds of CPU-bound numpy and OpenCV work, so Starlette dispatches it to a
    threadpool and the event loop stays free to accept other requests. An
    ``async def`` here would block every concurrent caller for the duration of
    one analysis.

    The order of operations is the contract. Validation comes first and rejects
    before any stage is entered (Requirement 2.3). Then the photograph is clamped
    to ``max_longest_edge``, which bounds every allocation downstream regardless
    of the dimensions the upload declared (Requirement 2.6). Then segmentation,
    the area check, calibration, and lighting decomposition each run exactly once
    (Requirement 1.2) -- the whole point of the two-pass split is that
    ``/api/render`` never repeats any of it.

    Nothing in the request carries corner points, plane annotations, or
    perspective hints, and none of the stages accept any: the signature is the
    enforcement of Requirement 1.5.

    Raises:
        ApiError: 413 for an oversized upload; 415 for a disallowed MIME type or
            extension, or bytes that do not decode; 422 ``no_usable_plane`` when
            no Structural_Plane reaches ``min_plane_area_fraction`` or none can be
            given geometry; 422 ``analysis_failed`` when a stage raises.
    """
    started = time.perf_counter()
    state = request.app.state
    settings: Settings = state.settings
    segmenter: Segmenter = state.segmenter
    cache: SceneCache = state.cache

    decoded = _validate_upload(file, settings)

    image = _stage(
        "downscale", lambda: clamp_longest_edge(decoded, settings.max_longest_edge)
    )
    height, width = int(image.shape[0]), int(image.shape[1])

    segmentation = _stage("segmentation", lambda: segmenter.segment(image))
    if not _has_usable_plane(segmentation, settings):
        raise ApiError(422, "no_usable_plane", _NO_USABLE_PLANE_MESSAGE)

    calibration = _stage(
        "calibration", lambda: calibrate(image, segmentation, settings=settings)
    )

    # A plane needs both a mask and a homography to be tileable, and the
    # Geometry_Engine omits planes whose geometry it could not establish by either
    # path. Reporting the intersection keeps `planes` and `plane_masks` on one key
    # set, so the render route can trust that a plane named in the response is
    # renderable.
    usable: tuple[PlaneName, ...] = tuple(
        name
        for name in PLANE_NAMES
        if name in segmentation.plane_masks and name in calibration.homographies
    )
    if not usable:
        raise ApiError(422, "no_usable_plane", _NO_USABLE_PLANE_MESSAGE)

    masks = {name: segmentation.plane_masks[name] for name in usable}
    lighting = _stage("lighting", lambda: decompose(image, masks, settings))

    # Every reported plane has a non-empty mask, so `plane_medians` covers all of
    # them; the frame median is a defensive default rather than an expected path.
    frame_median = float(np.median(lighting.shading))
    planes: dict[PlaneName, PlaneMetadata] = {
        name: PlaneMetadata(
            name=name,
            contour=segmentation.contours[name],
            bounding_points=segmentation.bounding_points[name],
            area_fraction=float(segmentation.area_fractions[name]),
            centroid=_plane_centroid(masks[name]),
            homography=calibration.homographies[name],
            homography_inv=calibration.homography_inverses[name],
            plane_extent_mm=calibration.plane_extents_mm[name],
            reprojection_rmse_px=float(calibration.reprojection_rmse_px[name]),
            geometry_mode=calibration.geometry_mode,
            luminance_median=float(lighting.plane_medians.get(name, frame_median)),
        )
        for name in usable
    }

    scene = SceneState(
        scene_id=uuid.uuid4().hex,
        # Wall clock, not `perf_counter`: the Scene_Cache measures TTL against
        # `time.time()`.
        created_at=time.time(),
        image=image,
        width=width,
        height=height,
        planes=planes,
        plane_masks=masks,
        foreground_mask=segmentation.foreground_mask,
        shading_map=lighting.shading,
        detail_map=lighting.detail,
        horizon=(
            float(calibration.horizon[0]),
            float(calibration.horizon[1]),
            float(calibration.horizon[2]),
        ),
        vanishing_points={
            label: (
                None
                if (vp := calibration.vanishing_points.get(label)) is None
                else (float(vp[0]), float(vp[1]))
            )
            for label in VP_LABELS
        },
        geometry_mode=calibration.geometry_mode,
        segmentation_backend=segmentation.backend_name,
    )
    # Requirement 1.4: the returned scene_id resolves to this state, so the very
    # next render finds it.
    cache.put(scene)

    analysis_ms = int(round((time.perf_counter() - started) * 1000.0))
    logger.info(
        "analysed scene=%s size=%dx%d backend=%s geometry=%s planes=%s in %d ms",
        scene.scene_id,
        width,
        height,
        scene.segmentation_backend,
        scene.geometry_mode,
        ",".join(usable),
        analysis_ms,
    )

    return SegmentResponse(
        scene_id=scene.scene_id,
        width=width,
        height=height,
        segmentation_backend=scene.segmentation_backend,
        geometry_mode=scene.geometry_mode,
        horizon=_horizon_response(scene.horizon, width, height),
        vanishing_points=scene.vanishing_points,
        planes=[
            PlaneResponse(
                name=plane.name,
                area_fraction=plane.area_fraction,
                contour=[[int(x), int(y)] for x, y in plane.contour],
                bounding_points=[[int(x), int(y)] for x, y in plane.bounding_points],
                centroid=plane.centroid,
                reprojection_rmse_px=plane.reprojection_rmse_px,
            )
            for plane in planes.values()
        ],
        analysis_ms=max(analysis_ms, 0),
    )


@api_router.post("/render", response_model=RenderResponse)
def post_render(
    request: Request,
    payload: RenderRequest,
    binary: bool = Query(
        False,
        description=(
            "Return the raw encoded image instead of a JSON body carrying it "
            "base64-encoded."
        ),
    ),
) -> Response:
    """Composite tile selections onto an already-analysed scene (R9.1).

    The cheap half of the two-pass split. Every expensive artifact -- masks,
    homographies, the shading and detail maps, the per-plane luminance medians --
    is read from the Scene_State exactly as analysis left it, so neither
    segmentation nor vanishing point estimation is re-entered (Requirement 9.2).
    Nothing this route calls can reach them: the Compositor takes the
    homographies as data and never asks the Geometry_Engine for more. What
    remains is a vectorised inverse warp and blend per requested plane plus one
    encode, which is what fits inside the budget of Requirement 9.3 -- 70 ms
    fixed, mostly the encode, plus 40 ms per tiled plane.

    ``def``, not ``async def``, for the same reason as :func:`post_segment`: the
    body is tens of milliseconds of numpy and OpenCV work, so Starlette runs it
    in a threadpool and the event loop stays free.

    Resolution runs to completion before any pixel is touched, in
    :data:`PLANE_NAMES` order so a request with more than one problem always
    reports the same one. Then compositing runs under a single guard, because a
    failure there is an internal fault rather than something the caller can fix.

    Args:
        payload: the Render_Request -- ``scene_id``, per-plane specs, optional
            output format.
        binary: when true, respond with the raw encoded bytes plus
            ``X-Render-Ms`` and ``X-Scene-Id`` headers, saving base64's 33
            percent overhead for callers that do not need the JSON fields.

    Returns:
        A JSON :class:`RenderResponse`, or the raw image when ``binary`` is set.

    Raises:
        ApiError: 404 ``scene_expired`` when the ``scene_id`` is absent or past
            its TTL; 422 ``unknown_plane`` for a plane this scene never detected;
            422 ``unknown_tile`` for a ``tile_id`` the catalog does not publish;
            500 ``render_failed`` when compositing or encoding fails.
    """
    started = time.perf_counter()
    state = request.app.state
    settings: Settings = state.settings
    cache: SceneCache = state.cache
    catalog: CatalogLoader = state.catalog

    # A miss covers both an id that never existed and one the LRU or TTL bound
    # dropped; the client cannot tell them apart and re-uploads either way
    # (Requirement 9.4).
    scene = cache.get(payload.scene_id)
    if scene is None:
        raise ApiError(404, "scene_expired", _SCENE_EXPIRED_MESSAGE)

    # Pydantic has already restricted the request's keys to the four
    # Structural_Plane names, so an arbitrary string is a 422 ``invalid_request``
    # before this route runs. What is left to check is whether *this photograph*
    # has the named plane, which is what ``unknown_plane`` reports.
    specs: dict[PlaneName, PlaneRenderSpec] = {}
    tiles: dict[PlaneName, TileDefinition] = {}
    for name in PLANE_NAMES:
        spec = payload.planes.get(name)
        if spec is None:
            continue
        if name not in scene.planes:
            detected = ", ".join(_PLANE_LABELS[found] for found in scene.planes)
            raise ApiError(
                422,
                "unknown_plane",
                f"This photo has no {_PLANE_LABELS[name]} to tile "
                f"(found: {detected or 'nothing tileable'}).",
            )
        tile = catalog.get(spec.tile_id)
        if tile is None:
            raise ApiError(
                422,
                "unknown_tile",
                f"The tile {spec.tile_id!r} is not in the catalog. "
                "Please pick another product.",
            )
        specs[name] = spec
        tiles[name] = tile

    warnings: list[str] = []
    try:
        # Seamless synthesis is memoised per tile in the Catalog_Loader, so only
        # the first render of a product pays for it (Requirement 9.3).
        textures: dict[PlaneName, SeamlessTexture] = {
            name: catalog.seamless(tile.id) for name, tile in tiles.items()
        }
        composited = compose(
            scene,
            specs,
            textures,
            settings,
            tiles=tiles,
            # Read from and written back to the Scene_State, so the distance
            # transform behind each plane's feathered alpha is paid once per
            # scene rather than once per tile swap.
            alpha_cache=scene.plane_alpha,
            warnings=warnings,
        )
        encoded, mime = encode_render(composited, settings, fmt=payload.format)
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001 - compositing faults are one 500
        logger.exception("render failed for scene=%s", payload.scene_id)
        raise ApiError(
            500,
            "render_failed",
            "Something went wrong drawing your preview. Please try again.",
        ) from exc

    # Measured here, before base64 framing, so ``render_ms`` reports exactly the
    # work the Requirement 9.3 budget is written about: everything from the cache
    # lookup through the encode, and no network transfer. ``tests/
    # test_performance.py`` asserts the budget against this number.
    render_ms = max(int(round((time.perf_counter() - started) * 1000.0)), 0)
    height, width = int(composited.shape[0]), int(composited.shape[1])

    logger.info(
        "rendered scene=%s planes=%s bytes=%d in %d ms",
        scene.scene_id,
        ",".join(specs) or "none",
        len(encoded),
        render_ms,
    )
    for note in warnings:
        logger.warning("render scene=%s: %s", scene.scene_id, note)

    if binary:
        return Response(
            content=encoded,
            media_type=mime,
            headers={
                "X-Render-Ms": str(render_ms),
                "X-Scene-Id": scene.scene_id,
            },
        )

    return JSONResponse(
        content=RenderResponse(
            scene_id=scene.scene_id,
            mime=mime,
            image=base64.b64encode(encoded).decode("ascii"),
            width=width,
            height=height,
            render_ms=render_ms,
            warnings=warnings,
        ).model_dump()
    )


app.include_router(api_router)


# --------------------------------------------------------------------------- #
# Static mounts -- last, because a Mount at "/" matches every path
# --------------------------------------------------------------------------- #

# Tile images and the sample room, served at the same URLs ``/api/tiles``
# advertises. The directory is resolved per request from ``app.state.settings``.
def _assets_dir() -> Path:
    """The live assets directory, falling back to settings if startup has not run."""
    settings = getattr(app.state, "settings", None) or get_settings()
    return settings.assets_dir


app.mount(ASSETS_URL_PREFIX, _SettingsStaticFiles(_assets_dir), name="assets")

# The zero-build frontend, so ``frontend/index.html`` is served by the same
# process during development. Fixed path relative to this file, hence a plain
# ``StaticFiles``.
app.mount(
    "/",
    StaticFiles(directory=_FRONTEND_DIR, html=True, check_dir=False),
    name="frontend",
)
