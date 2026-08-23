"""Tests for the Visualizer_API in `backend/app.py` (Requirement 13.8).

This module verifies the two-pass split through HTTP: `POST /api/segment`
analyses a photograph exactly once and caches the result, and `POST /api/render`
draws tiles onto that cached result without ever re-entering analysis.

Cost shapes the design of the file. One `/api/segment` call is 200-400 ms of
honest CPU work -- segmentation, line detection, vanishing point estimation,
homography solving, and a CIELAB decomposition -- so the properties are organised
by what they actually quantify over.

**Properties 1, 2, and 33 quantify over photographs**, and are asserted twice
over, at two different costs.

Each has a focused test over :data:`UPLOAD_SPECS`, a curated corpus of sixteen
uploads spanning three-wall, two-wall, single-wall, and floor-only rooms, five
frame sizes, both yaw signs, shallow and steep pitch, zero to two occluders, and
all three accepted upload formats. The corpus is analysed once per session by
:func:`analysed_corpus` and reduced to plain facts, so those tests are dictionary
reads. The corpus is a *finite* space, so Hypothesis reports "nothing left to do"
after sixteen examples rather than running the full hundred -- which makes those
runs a complete check of the space rather than a sample of it, and is why the
corpus is curated to hold the awkward cases instead of being large.

Then :func:`test_properties_1_2_and_33_hold_for_freshly_drawn_photographs`
asserts all three over a hundred *freshly analysed* photographs drawn from a
continuous pose space: the wide universal quantifier, at about 25 seconds, and the
only genuinely slow test here. The three properties share it rather than getting
one each because they are claims about the same response and the same
Scene_State, and re-analysing per property would triple that cost for nothing.
Their conditions live in the three `assert_*` helpers, so the focused tests and
the drawn test check exactly the same things.

**Properties 3 and 28 quantify over render requests.** Those are cheap (~20 ms),
so they are drawn per example against one live scene analysed once per test. Both
take :func:`live_scene`, which is built on the function-scoped `client` fixture,
with `HealthCheck.function_scoped_fixture` suppressed -- correct here rather than a
workaround, because the fixture is *deliberately* shared across examples:
re-analysing the photograph per example is the exact cost the two-pass split
exists to avoid, and the examples only read the scene. The one mutation a render
makes to shared state -- populating `SceneState.plane_alpha` -- is idempotent.

Both the corpus and the drawn test build their own `TestClient` through
:func:`configured_client` rather than reusing the `client` fixture, because a
session-scoped record cannot be built from a function-scoped one. That helper sets
`RV_*` in the process environment for the life of one client and restores it on
the way out, so nothing it touched is live while an assertion runs. Everything a
record holds is snapshotted while its client is still open: leaving the
`TestClient` context runs `lifespan`, which clears the Scene_Cache and calls
`SceneState.release()`, nulling every array.

A note on Property 33, which is stated over "every plane mask, the foreground
mask, the shading map, and the detail map". `SceneState.plane_alpha` also lives in
the cache and is `float32`. That is deliberate and documented on the field: it is
derived render state, not analysis output, and Requirement 12.4 is stated over
masks and lighting maps. This module asserts the 8-bit contract over the four
named families and separately asserts that `plane_alpha` is the `float32` its
docstring promises, so the exclusion is pinned rather than merely tolerated.

Layout: constants, then shared helpers and fixtures, then one banner-delimited
section per property. Task 13.6 appends the rejection-path section at the foot of
the file and reuses everything above the first banner.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import cv2
import numpy as np
import pytest
from hypothesis import (
    HealthCheck,
    assume,
    event,
    given,
    settings as hypothesis_settings,
    strategies as st,
)

import backend.app as app_module
import backend.core.geometry as geometry_module
from backend.config import Settings, get_settings
from backend.core.segmenter import ClassicalSegmenter, SegmentationResult
from backend.schemas import PLANE_NAMES
from tests.conftest import TINY_CATALOG_TILES
from tests.fixtures.synthetic import make_synthetic_room

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: Upload MIME type and filename extension per encode format. All three are in
#: the default `allowed_mime_types` and `allowed_extensions`, so the happy path is
#: exercised on every accepted format rather than on PNG alone (R2.1).
UPLOAD_FORMATS: dict[str, tuple[str, str]] = {
    "png": ("image/png", ".png"),
    "jpeg": ("image/jpeg", ".jpg"),
    "webp": ("image/webp", ".webp"),
}

#: Cached artifact families Requirement 12.4's 8-bit contract is stated over. A
#: key in an artifact snapshot is either exactly one of these or one of them
#: suffixed with `:<plane>`.
CONTRACTED_ARTIFACTS: tuple[str, ...] = (
    "plane_mask",
    "foreground_mask",
    "shading_map",
    "detail_map",
)

#: Cached arrays deliberately *outside* that contract, with the dtype each is
#: documented to hold. `plane_alpha` is derived render state -- see the module
#: docstring and the field's own docstring in `backend/schemas.py`.
DERIVED_ARTIFACT_DTYPES: dict[str, str] = {"plane_alpha": "float32"}

#: Accepted uploads the corpus must yield, so a change that made every room
#: unanalysable could not satisfy Properties 1, 2, and 33 by vacuity. The corpus
#: holds sixteen specs; twelve leaves room for a pose or two to stop producing a
#: usable plane without the properties going silently empty.
MIN_ACCEPTED_UPLOADS: int = 12

#: Share of freshly drawn photographs that must be accepted, for the same reason.
#: The drawn space reaches wall-free and near-level poses, so some rejection is
#: allowed for; in practice none was observed -- 100 examples at each of three
#: independent Hypothesis seeds accepted every draw, splitting 53-71 percent
#: `planar_fallback` against 29-47 percent `vanishing_points`, so both geometry
#: modes of Requirement 6.3 are exercised. The floor is set well below that
#: because the quantity is a heuristic segmenter's yield, and a bound sitting just
#: under the observed value would fail on an OpenCV point release rather than on a
#: regression.
MIN_DRAWN_ACCEPTANCE: float = 0.60

#: Tolerance on `area_fraction` against its own mask's pixel count. The server
#: computes it as a Python float division and this re-derives it the same way, so
#: only float64 round-off separates them.
AREA_FRACTION_TOLERANCE: float = 1e-9

#: The two `geometry_mode` values Requirement 6.3 documents.
GEOMETRY_MODES: frozenset[str] = frozenset({"vanishing_points", "planar_fallback"})

#: Every wall subset the room generator accepts, single-wall and wall-free
#: included. A floor-only frame is the "unusual room" Requirement 6.1 promises to
#: serve rather than reject, and it is where a plane omission bug would surface.
WALL_SETS: tuple[tuple[str, ...], ...] = (
    ("left", "right", "back"),
    ("left", "back"),
    ("right", "back"),
    ("left", "right"),
    ("back",),
    ("left",),
    (),
)


# --------------------------------------------------------------------------- #
# Upload specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class UploadSpec:
    """One synthetic room plus the format it is uploaded as.

    Every field is a generator argument or an encode choice, so a failing example
    reproduces from the spec alone. Frozen so it can key the analysis memo.
    """

    label: str
    width: int
    height: int
    yaw_deg: float
    pitch_deg: float
    walls: tuple[str, ...]
    n_occluders: int
    seed: int
    fmt: str

    @property
    def mime(self) -> str:
        return UPLOAD_FORMATS[self.fmt][0]

    @property
    def filename(self) -> str:
        return f"{self.label}{UPLOAD_FORMATS[self.fmt][1]}"

    def render(self) -> np.ndarray:
        """The room image this spec describes, at a focal-to-width ratio of 0.875.

        The ratio matches the documented fixture's 1400/1600, so every pose here
        stays in the same field-of-view regime as the full-size room.
        """
        return make_synthetic_room(
            width=self.width,
            height=self.height,
            focal_px=0.875 * self.width,
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            walls=self.walls,
            n_occluders=self.n_occluders,
            seed=self.seed,
            supersample=1,
        ).image


#: The curated corpus: wall sets from three down to none, frames from 320x240 to
#: 800x600, yaw of both signs from 0 to 20 degrees, pitch from -6 to -28,
#: occluder counts 0 to 2, and all three accepted upload formats.
UPLOAD_SPECS: tuple[UploadSpec, ...] = (
    UploadSpec("three_wall_png", 480, 360, 8.0, -12.0, ("left", "right", "back"), 1, 0, "png"),
    UploadSpec("three_wall_jpeg", 480, 360, -8.0, -12.0, ("left", "right", "back"), 2, 1, "jpeg"),
    UploadSpec("three_wall_webp", 480, 360, 16.0, -20.0, ("left", "right", "back"), 0, 2, "webp"),
    UploadSpec("left_back_png", 480, 360, 0.0, -8.0, ("left", "back"), 1, 3, "png"),
    UploadSpec("right_back_jpeg", 480, 360, -4.0, -10.0, ("right", "back"), 1, 4, "jpeg"),
    UploadSpec("left_right_png", 480, 360, 6.0, -14.0, ("left", "right"), 0, 5, "png"),
    UploadSpec("back_only_webp", 480, 360, 2.0, -10.0, ("back",), 1, 6, "webp"),
    UploadSpec("left_only_png", 480, 360, 12.0, -12.0, ("left",), 0, 7, "png"),
    UploadSpec("floor_only_png", 480, 360, 8.0, -16.0, (), 1, 8, "png"),
    UploadSpec("wide_three_wall_png", 640, 480, 12.0, -16.0, ("left", "right", "back"), 2, 9, "png"),
    UploadSpec("wide_left_back_jpeg", 640, 480, -20.0, -6.0, ("left", "back"), 1, 10, "jpeg"),
    UploadSpec("small_three_wall_png", 320, 240, 8.0, -12.0, ("left", "right", "back"), 1, 11, "png"),
    UploadSpec("large_three_wall_webp", 800, 600, -10.0, -14.0, ("left", "right", "back"), 1, 12, "webp"),
    UploadSpec("steep_three_wall_png", 480, 360, 8.0, -28.0, ("left", "right", "back"), 2, 13, "png"),
    UploadSpec("tall_back_jpeg", 560, 420, 4.0, -14.0, ("back",), 1, 14, "jpeg"),
    UploadSpec("wide_steep_webp", 640, 480, -14.0, -22.0, ("left", "right", "back"), 2, 15, "webp"),
)


# --------------------------------------------------------------------------- #
# Recorded facts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArtifactFacts:
    """What a cached array is, without holding the array itself.

    Records hold these rather than the buffers: a hundred scenes of masks and
    lighting maps would be hundreds of megabytes held for a whole session, and
    every assertion in this module is about a dtype, a shape, or a pixel count.
    """

    dtype: str
    shape: tuple[int, ...]
    set_pixels: int


@dataclass(frozen=True)
class RenderOutcome:
    """One recorded `/api/render` response, JSON or binary."""

    label: str
    requested: tuple[str, ...]
    binary: bool
    status: int
    code: str | None
    mime: str | None
    width: int | None
    height: int | None
    render_ms: int | None
    image_bytes: int
    warnings: tuple[str, ...]
    scene_id_header: str | None


@dataclass(frozen=True)
class AnalysedUpload:
    """Everything one upload produced, reduced to plain data."""

    spec: UploadSpec
    source_shape: tuple[int, int]  # (H, W) of the generated room
    status: int
    body: Mapping[str, Any]
    error_code: str | None
    error_body: Mapping[str, Any] | None
    resolved_in_cache: bool
    cache_entries: int
    artifacts: Mapping[str, ArtifactFacts]
    artifacts_after_render: Mapping[str, ArtifactFacts]
    renders: tuple[RenderOutcome, ...]
    #: What the cached Scene_State says about itself, so a response field can be
    #: checked against the analysis it claims to describe rather than only against
    #: its own declared type.
    cached_backend: str | None
    cached_geometry_mode: str | None
    plane_geometry_modes: Mapping[str, str]

    @property
    def accepted(self) -> bool:
        return self.status == 200

    @property
    def scene_id(self) -> str:
        return str(self.body["scene_id"])

    @property
    def plane_names(self) -> tuple[str, ...]:
        return tuple(str(plane["name"]) for plane in self.body["planes"])

    def __repr__(self) -> str:  # pragma: no cover - Hypothesis reporting only
        return f"<AnalysedUpload {self.spec.label} status={self.status}>"


# --------------------------------------------------------------------------- #
# Shared helpers -- also used by the task 13.6 section appended below
# --------------------------------------------------------------------------- #


def encode_image(image: np.ndarray, fmt: str) -> bytes:
    """Encode `image` as `fmt`, raising rather than returning empty bytes."""
    extension = UPLOAD_FORMATS[fmt][1]
    ok, buffer = cv2.imencode(extension, image)
    if not ok:  # pragma: no cover - the installed OpenCV encodes all three
        raise RuntimeError(f"OpenCV could not encode a {fmt} upload")
    return bytes(buffer.tobytes())


def upload_part(
    payload: bytes, filename: str, mime: str
) -> dict[str, tuple[str, bytes, str]]:
    """The multipart `files=` mapping `/api/segment` expects."""
    return {"file": (filename, payload, mime)}


def error_code(response: Any) -> str | None:
    """The `error.code` of a failure body, or `None` for a success or odd shape.

    Every failure leaves through one envelope (Requirement 1.6), so this is the
    single accessor for a rejection's machine-readable code.
    """
    if response.status_code < 400:
        return None
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):  # pragma: no cover - always JSON
        return None
    if not isinstance(body, dict):  # pragma: no cover
        return None
    detail = body.get("error")
    if not isinstance(detail, dict):
        return None
    code = detail.get("code")
    return None if code is None else str(code)


def assert_error_envelope_body(body: Any, expected_code: str, where: str = "") -> None:
    """A parsed failure body is the shared envelope carrying `expected_code`.

    Stated over a parsed body rather than a response so a recorded rejection can
    be checked long after its response object is gone, which is what the corpus
    needs (Requirement 1.6).
    """
    prefix = f"{where}: " if where else ""
    assert isinstance(body, dict) and set(body) == {"error"}, (
        f"{prefix}failure body is {body!r}, not the shared {{'error': ...}} envelope"
    )
    detail = body["error"]
    assert isinstance(detail, dict) and set(detail) == {"code", "message"}, (
        f"{prefix}error detail is {detail!r}, expected a code and a message"
    )
    assert detail["code"] == expected_code, (
        f"{prefix}error code is {detail['code']!r}, expected {expected_code!r}"
    )
    assert isinstance(detail["message"], str) and detail["message"].strip(), (
        f"{prefix}the envelope carries no human-readable message"
    )


def assert_points_in_bounds(
    points: Sequence[Sequence[int]], width: int, height: int, what: str
) -> None:
    """Every `(x, y)` in `points` lies inside `[0, W-1] x [0, H-1]`."""
    for point in points:
        assert len(point) == 2, f"{what} holds {point!r}, which is not an (x, y) pair"
        x, y = int(point[0]), int(point[1])
        assert 0 <= x <= width - 1, f"{what} has x={x} outside [0, {width - 1}]"
        assert 0 <= y <= height - 1, f"{what} has y={y} outside [0, {height - 1}]"


def snapshot_artifacts(scene: Any) -> dict[str, ArtifactFacts]:
    """Reduce a live `SceneState` to per-array facts.

    Keys are `plane_mask:<plane>`, `foreground_mask`, `shading_map`, `detail_map`,
    `image`, and `plane_alpha:<plane>` for whatever alphas a render has populated
    so far.
    """

    def facts(array: np.ndarray) -> ArtifactFacts:
        return ArtifactFacts(
            dtype=str(array.dtype),
            shape=tuple(int(dim) for dim in array.shape),
            set_pixels=int(np.count_nonzero(array)),
        )

    out: dict[str, ArtifactFacts] = {
        "image": facts(scene.image),
        "foreground_mask": facts(scene.foreground_mask),
        "shading_map": facts(scene.shading_map),
        "detail_map": facts(scene.detail_map),
    }
    for plane, mask in scene.plane_masks.items():
        out[f"plane_mask:{plane}"] = facts(mask)
    for plane, alpha in scene.plane_alpha.items():
        out[f"plane_alpha:{plane}"] = facts(alpha)
    return out


def artifact_family(key: str) -> str:
    """The family a snapshot key belongs to (`plane_mask:floor` -> `plane_mask`)."""
    return key.split(":", 1)[0]


def default_settings() -> Settings:
    """Settings independent of the ambient environment.

    `get_settings` is cached and reads `RV_`-prefixed variables, so a fresh
    `Settings` keeps `min_plane_area_fraction` -- which decides which planes the
    service reports -- pinned to the documented default whatever the surrounding
    test configuration holds.
    """
    return Settings()


@dataclass
class StageCounters:
    """How many times each analysis stage was entered."""

    segment: int = 0
    vanishing_point: int = 0
    calibrate: int = 0
    decompose: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "segment": self.segment,
            "vanishing_point": self.vanishing_point,
            "calibrate": self.calibrate,
            "decompose": self.decompose,
        }


@contextmanager
def counted_analysis_stages() -> Iterator[StageCounters]:
    """Count entries into every analysis stage for the duration of the block.

    Four counters on the four stages `/api/segment` runs and `/api/render` must
    not:

    * `ClassicalSegmenter.segment` -- the class rather than the instance, because
      the segmenter uses `__slots__`. Every user of this asserts the active
      backend is the classical one, so the patch is provably on the live path.
    * `backend.core.geometry.estimate_vanishing_point` -- patched on the module,
      which is where `calibrate` resolves it from.
    * `backend.app.calibrate` and `backend.app.decompose` -- patched on the app
      module, which is where the route resolves them from.

    Every attribute is restored on the way out, so a failing example cannot leave
    a wrapper installed for the next one.
    """
    counters = StageCounters()

    real_segment = ClassicalSegmenter.segment
    real_vanishing_point = geometry_module.estimate_vanishing_point
    real_calibrate = app_module.calibrate
    real_decompose = app_module.decompose

    def counting_segment(self: ClassicalSegmenter, image_bgr: np.ndarray) -> Any:
        counters.segment += 1
        return real_segment(self, image_bgr)

    def counting_vanishing_point(*args: Any, **kwargs: Any) -> Any:
        counters.vanishing_point += 1
        return real_vanishing_point(*args, **kwargs)

    def counting_calibrate(*args: Any, **kwargs: Any) -> Any:
        counters.calibrate += 1
        return real_calibrate(*args, **kwargs)

    def counting_decompose(*args: Any, **kwargs: Any) -> Any:
        counters.decompose += 1
        return real_decompose(*args, **kwargs)

    ClassicalSegmenter.segment = counting_segment  # type: ignore[method-assign]
    geometry_module.estimate_vanishing_point = counting_vanishing_point  # type: ignore[assignment]
    app_module.calibrate = counting_calibrate  # type: ignore[assignment]
    app_module.decompose = counting_decompose  # type: ignore[assignment]
    try:
        yield counters
    finally:
        ClassicalSegmenter.segment = real_segment  # type: ignore[method-assign]
        geometry_module.estimate_vanishing_point = real_vanishing_point  # type: ignore[assignment]
        app_module.calibrate = real_calibrate  # type: ignore[assignment]
        app_module.decompose = real_decompose  # type: ignore[assignment]


# --------------------------------------------------------------------------- #
# Analysing an upload
# --------------------------------------------------------------------------- #

#: Tile ids every render here draws with, in catalog order: a 1:1, a 1:2, and a
#: plank, so a whole-room render exercises all three metric formats.
CATALOG_TILE_IDS: tuple[str, ...] = tuple(str(spec["id"]) for spec in TINY_CATALOG_TILES)


def _write_catalog(assets_dir: Path) -> Path:
    """Write `TINY_CATALOG_TILES` into `assets_dir/tiles/` with a manifest.

    A local generator rather than the `tiny_catalog` fixture, which is
    function-scoped and so unreachable from a session-scoped record. The declared
    millimetre dimensions are the fixture's, so the tiles rendered here are the
    same three metric formats every other module uses.
    """
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for index, spec in enumerate(TINY_CATALOG_TILES):
        width_px, height_px = spec["size_px"]
        rng = np.random.default_rng(index)
        image = np.full((height_px, width_px, 3), spec["base_bgr"], dtype=np.uint8)
        cell = max(8, min(width_px, height_px) // 4)
        ys, xs = np.mgrid[0:height_px, 0:width_px]
        image[((ys // cell) + (xs // cell)) % 2 == 1] = np.clip(
            np.asarray(spec["base_bgr"], dtype=np.int16) - 24, 0, 255
        ).astype(np.uint8)
        image = np.clip(
            image.astype(np.int16) + rng.integers(-3, 4, image.shape), 0, 255
        ).astype(np.uint8)
        if not cv2.imwrite(str(tiles_dir / spec["file"]), image):  # pragma: no cover
            raise RuntimeError(f"failed to write catalog tile {spec['file']!r}")
        entries.append(
            {
                key: spec[key]
                for key in ("id", "name", "file", "width_mm", "height_mm", "finish", "gloss")
            }
        )

    (tiles_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "tiles": entries}, indent=2), encoding="utf-8"
    )
    return assets_dir


def _render_outcome(
    response: Any, label: str, requested: Sequence[str], *, binary: bool
) -> RenderOutcome:
    """Reduce one `/api/render` response to a :class:`RenderOutcome`."""
    if binary:
        return RenderOutcome(
            label=label,
            requested=tuple(requested),
            binary=True,
            status=response.status_code,
            code=error_code(response),
            mime=response.headers.get("content-type"),
            width=None,
            height=None,
            render_ms=(
                int(response.headers["X-Render-Ms"])
                if "X-Render-Ms" in response.headers
                else None
            ),
            image_bytes=len(response.content),
            warnings=(),
            scene_id_header=response.headers.get("X-Scene-Id"),
        )

    body = response.json() if response.status_code == 200 else {}
    return RenderOutcome(
        label=label,
        requested=tuple(requested),
        binary=False,
        status=response.status_code,
        code=error_code(response),
        mime=body.get("mime"),
        width=body.get("width"),
        height=body.get("height"),
        render_ms=body.get("render_ms"),
        image_bytes=len(str(body.get("image", ""))),
        warnings=tuple(body.get("warnings", ())),
        scene_id_header=None,
    )


def _analyse_upload(client: Any, spec: UploadSpec) -> AnalysedUpload:
    """Upload one spec, render from it, and reduce everything to plain facts.

    Four renders per accepted upload, chosen to cover the shapes Requirement 9.1
    describes: every detected plane at once, one plane with every optional
    setting populated and a non-default output format, one through `?binary=1`,
    and one naming no plane at all.
    """
    image = spec.render()
    payload = encode_image(image, spec.fmt)
    response = client.post(
        "/api/segment", files=upload_part(payload, spec.filename, spec.mime)
    )

    cache = client.app.state.cache
    source_shape = (int(image.shape[0]), int(image.shape[1]))
    if response.status_code != 200:
        return AnalysedUpload(
            spec=spec,
            source_shape=source_shape,
            status=response.status_code,
            body={},
            error_code=error_code(response),
            error_body=response.json(),
            resolved_in_cache=False,
            cache_entries=len(cache),
            artifacts={},
            artifacts_after_render={},
            renders=(),
            cached_backend=None,
            cached_geometry_mode=None,
            plane_geometry_modes={},
        )

    body = response.json()
    scene_id = str(body["scene_id"])
    scene = cache.get(scene_id)
    assert scene is not None, (
        f"{spec.label}: /api/segment returned scene_id {scene_id!r} that the cache "
        "does not resolve"
    )
    artifacts = snapshot_artifacts(scene)
    cached_backend = str(scene.segmentation_backend)
    cached_geometry_mode = str(scene.geometry_mode)
    plane_geometry_modes = {
        str(name): str(plane.geometry_mode) for name, plane in scene.planes.items()
    }
    planes = [str(plane["name"]) for plane in body["planes"]]

    renders: list[RenderOutcome] = []

    all_planes = {
        name: {"tile_id": CATALOG_TILE_IDS[index % len(CATALOG_TILE_IDS)]}
        for index, name in enumerate(planes)
    }
    renders.append(
        _render_outcome(
            client.post("/api/render", json={"scene_id": scene_id, "planes": all_planes}),
            "all_planes",
            planes,
            binary=False,
        )
    )

    if planes:
        styled = {
            planes[0]: {
                "tile_id": CATALOG_TILE_IDS[-1],
                "rotation_deg": 37.5,
                "grout_mm": 5.0,
                "grout_rgb": [40, 42, 44],
                "offset_mm": [120.0, -80.0],
            }
        }
        renders.append(
            _render_outcome(
                client.post(
                    "/api/render",
                    json={"scene_id": scene_id, "planes": styled, "format": "jpeg"},
                ),
                "styled",
                planes[:1],
                binary=False,
            )
        )
        renders.append(
            _render_outcome(
                client.post(
                    "/api/render?binary=1",
                    json={
                        "scene_id": scene_id,
                        "planes": {planes[0]: {"tile_id": CATALOG_TILE_IDS[0]}},
                    },
                ),
                "binary",
                planes[:1],
                binary=True,
            )
        )

    renders.append(
        _render_outcome(
            client.post("/api/render", json={"scene_id": scene_id, "planes": {}}),
            "no_planes",
            (),
            binary=False,
        )
    )

    scene_after = cache.get(scene_id)
    assert scene_after is not None, f"{spec.label}: the scene vanished while rendering"

    return AnalysedUpload(
        spec=spec,
        source_shape=source_shape,
        status=response.status_code,
        body=body,
        error_code=None,
        error_body=None,
        resolved_in_cache=True,
        cache_entries=len(cache),
        artifacts=artifacts,
        artifacts_after_render=snapshot_artifacts(scene_after),
        renders=tuple(renders),
        cached_backend=cached_backend,
        cached_geometry_mode=cached_geometry_mode,
        plane_geometry_modes=plane_geometry_modes,
    )


@contextmanager
def configured_client(assets_dir: Path, weights_dir: Path) -> Iterator[Any]:
    """A `TestClient` over the app, configured and torn back down again.

    `RV_ENABLE_NEURAL_BACKEND=false` means `build_segmenter` never consults the
    Model_Loader, so nothing here needs weights on disk or a reachable mirror
    (Requirement 4.7). The environment is restored on the way out, so no override
    this made outlives the block.

    The app is imported inside the body for the same reason `conftest.client` does
    it: settings are read both at import and in `lifespan`, and both reads must
    observe the overrides.
    """
    overrides = {
        "RV_ENABLE_NEURAL_BACKEND": "false",
        "RV_ASSETS_DIR": str(assets_dir),
        "RV_WEIGHTS_DIR": str(weights_dir),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    get_settings.cache_clear()
    try:
        from fastapi.testclient import TestClient

        from backend.app import app

        with TestClient(app) as client:
            assert client.app.state.segmenter.backend_name == "classical", (
                "an offline harness must serve the classical backend"
            )
            yield client
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@pytest.fixture(scope="session")
def corpus_paths(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The assets tree and weights directory every own-client analysis uses."""
    assets_dir = _write_catalog(tmp_path_factory.mktemp("api-assets") / "assets")
    weights_dir = tmp_path_factory.mktemp("api-weights")
    return assets_dir, weights_dir


@pytest.fixture(scope="session")
def analysed_corpus(corpus_paths: tuple[Path, Path]) -> tuple[AnalysedUpload, ...]:
    """Analyse and render :data:`UPLOAD_SPECS` once, as immutable records.

    One client for all sixteen uploads, which is what lets the cross-upload tests
    -- distinct `scene_id`s, cache occupancy -- observe more than one scene at a
    time. Fact-only, so nothing downstream depends on the cache still holding
    arrays after the client closes.
    """
    assets_dir, weights_dir = corpus_paths
    with configured_client(assets_dir, weights_dir) as client:
        return tuple(_analyse_upload(client, spec) for spec in UPLOAD_SPECS)


@pytest.fixture(scope="session")
def analyse(corpus_paths: tuple[Path, Path]) -> Callable[[UploadSpec], AnalysedUpload]:
    """A memoised "analyse one upload spec" callable, for the drawn property.

    One client per spec rather than one for the whole session: a client held open
    across tests would have its `app.state` replaced out from under it by the next
    test to enter `lifespan` on the same application object, and entering
    `lifespan` costs milliseconds against a 250 ms analysis.

    The memo makes a repeated draw free. It holds facts rather than arrays, so a
    hundred entries is kilobytes.
    """
    assets_dir, weights_dir = corpus_paths
    memo: dict[UploadSpec, AnalysedUpload] = {}

    def _analyse(spec: UploadSpec) -> AnalysedUpload:
        cached = memo.get(spec)
        if cached is None:
            with configured_client(assets_dir, weights_dir) as client:
                cached = _analyse_upload(client, spec)
            memo[spec] = cached
        return cached

    return _analyse


@pytest.fixture(scope="session")
def accepted_uploads(
    analysed_corpus: tuple[AnalysedUpload, ...],
) -> tuple[AnalysedUpload, ...]:
    """The corpus records `/api/segment` accepted, with vacuity ruled out.

    Properties 1, 2, and 33 are all stated over *accepted* photographs, so this is
    the population the cross-upload tests read. The floor on the count stops a
    regression that rejected everything from satisfying them trivially, and every
    rejection is required to carry the documented envelope so a silent drop cannot
    hide here either.
    """
    accepted = tuple(record for record in analysed_corpus if record.accepted)

    for record in analysed_corpus:
        if record.accepted:
            continue
        # The only rejection this corpus can legitimately produce: every upload is
        # a real raster of an allowed type, well under the size cap, so anything
        # other than "nothing in this room is tileable" is a regression.
        assert record.status == 422, (
            f"{record.spec.label}: unexpected rejection status={record.status} "
            f"code={record.error_code!r}"
        )
        assert_error_envelope_body(
            record.error_body, "no_usable_plane", record.spec.label
        )

    assert len(accepted) >= MIN_ACCEPTED_UPLOADS, (
        f"only {len(accepted)} of {len(analysed_corpus)} corpus uploads were "
        f"accepted, under the {MIN_ACCEPTED_UPLOADS} floor; the properties over the "
        "corpus would be close to vacuous"
    )
    return accepted


# --------------------------------------------------------------------------- #
# The live scene
# --------------------------------------------------------------------------- #


@dataclass
class LiveScene:
    """One analysed scene plus the client that holds it.

    Handed to the render properties, which are drawn per example against a single
    analysis. `client` and `scene_id` are the whole interface; `planes` is carried
    so a test body does not have to re-read the response.
    """

    client: Any
    scene_id: str
    body: Mapping[str, Any]
    planes: tuple[str, ...]

    def render(self, payload: Mapping[str, Any], *, binary: bool = False) -> Any:
        url = "/api/render?binary=1" if binary else "/api/render"
        return self.client.post(url, json=dict(payload))


#: The room the live-scene fixture analyses: the first corpus spec, which yields
#: all four Structural_Planes at 480x360, so a drawn plane subset has something to
#: choose from.
LIVE_SCENE_SPEC: UploadSpec = UPLOAD_SPECS[0]


@pytest.fixture
def live_scene(client: Any) -> LiveScene:
    """Upload one photograph through the `client` fixture and return its scene.

    Function-scoped, and deliberately shared across the examples of the render
    properties that take it -- see the module docstring on why suppressing
    `HealthCheck.function_scoped_fixture` is right there rather than a workaround.
    """
    payload = encode_image(LIVE_SCENE_SPEC.render(), LIVE_SCENE_SPEC.fmt)
    response = client.post(
        "/api/segment",
        files=upload_part(payload, LIVE_SCENE_SPEC.filename, LIVE_SCENE_SPEC.mime),
    )
    assert response.status_code == 200, (
        f"the live-scene photograph was rejected: {response.status_code} "
        f"{error_code(response)!r}"
    )
    body = response.json()
    return LiveScene(
        client=client,
        scene_id=str(body["scene_id"]),
        body=body,
        planes=tuple(str(plane["name"]) for plane in body["planes"]),
    )


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

_finite = {"allow_nan": False, "allow_infinity": False}

#: Which corpus record a focused property looks at. A finite space, so Hypothesis
#: exhausts it and the run covers every curated upload exactly once.
_corpus_index = st.sampled_from(range(len(UPLOAD_SPECS)))

#: Frame sizes for the drawn property. Smaller than the corpus's, because these
#: are analysed per example: 320x240 costs ~230 ms against ~290 ms at 480x360, and
#: every claim asserted over them is resolution-independent.
_DRAWN_FRAME_SIZES: tuple[tuple[int, int], ...] = ((320, 240), (400, 300))


@st.composite
def drawn_upload_specs(draw: st.DrawFn) -> UploadSpec:
    """A freshly parameterised photograph: continuous pose, any wall set, any format.

    Deliberately wider than the corpus. Yaw runs both signs out to 24 degrees and
    pitch from nearly level to steeply downward, which is where vanishing point
    recovery gets hard and the planar fallback of Requirement 6.1 takes over --
    and the properties asserted over these draws are supposed to hold in both
    geometry modes.
    """
    width, height = draw(st.sampled_from(_DRAWN_FRAME_SIZES))
    fmt = draw(st.sampled_from(tuple(UPLOAD_FORMATS)))
    return UploadSpec(
        label=f"drawn_{width}x{height}",
        width=width,
        height=height,
        yaw_deg=draw(st.floats(min_value=-24.0, max_value=24.0, **_finite)),
        pitch_deg=draw(st.floats(min_value=-28.0, max_value=-4.0, **_finite)),
        walls=draw(st.sampled_from(WALL_SETS)),
        n_occluders=draw(st.integers(min_value=0, max_value=2)),
        seed=draw(st.integers(min_value=0, max_value=2**16)),
        fmt=fmt,
    )


#: One per-plane render spec, exercising every optional field of Requirement 9.1.
#: `None` for `grout_mm` and `grout_rgb` is the inherit path, which the Compositor
#: fills from the Tile_Definition and then from the configured defaults.
_plane_spec = st.fixed_dictionaries(
    {
        "tile_id": st.sampled_from(CATALOG_TILE_IDS),
        "rotation_deg": st.floats(min_value=-180.0, max_value=180.0, **_finite),
    },
    optional={
        "grout_mm": st.one_of(
            st.none(), st.floats(min_value=0.0, max_value=12.0, **_finite)
        ),
        "grout_rgb": st.one_of(
            st.none(),
            st.tuples(*(st.integers(min_value=0, max_value=255),) * 3).map(list),
        ),
        "offset_mm": st.tuples(
            st.floats(min_value=-3000.0, max_value=3000.0, **_finite),
            st.floats(min_value=-3000.0, max_value=3000.0, **_finite),
        ).map(list),
    },
)

#: Which planes a request names. Drawn from all four Structural_Plane names and
#: intersected with the scene's own planes in the test body, so the draw does not
#: depend on what this particular photograph happened to yield.
_plane_selection = st.lists(st.sampled_from(PLANE_NAMES), unique=True, max_size=4)

_render_format = st.sampled_from((None, "png", "jpeg"))

_CORPUS_PROPERTY_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    # The corpus is ~7 s of analysis, paid once by whichever test reaches it
    # first; the examples themselves are dictionary reads.
    suppress_health_check=[HealthCheck.too_slow],
)

_DRAWN_PROPERTY_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    # ~250 ms of genuine analysis per example. Slow on purpose, not by accident.
    suppress_health_check=[HealthCheck.too_slow],
)

_RENDER_PROPERTY_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        # `live_scene` is shared across examples on purpose: re-analysing the
        # photograph per example is the exact cost the two-pass split exists to
        # avoid, and the examples only read it.
        HealthCheck.function_scoped_fixture,
    ],
)


def render_payload(
    scene: LiveScene,
    planes: Sequence[str],
    spec: Mapping[str, Any],
    fmt: str | None,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build a Render_Request for `scene` over the planes it actually has.

    Returns the payload and the plane names it names, so a caller can assert
    against the selection rather than re-deriving it.
    """
    selected = tuple(name for name in planes if name in scene.planes)
    payload: dict[str, Any] = {
        "scene_id": scene.scene_id,
        "planes": {name: dict(spec) for name in selected},
    }
    if fmt is not None:
        payload["format"] = fmt
    return payload, selected


# =========================================================================== #
# Task 13.5 -- analysis and render happy paths
# (Requirements 1.1-1.5, 4.6, 6.3, 9.1, 9.2, 12.4, 13.8)
# =========================================================================== #

# --------------------------------------------------------------------------- #
# Property 1 -- the analysis response is wellformed
# (Requirements 1.1, 1.3, 4.6, 6.3)
# --------------------------------------------------------------------------- #


def assert_response_wellformed(record: AnalysedUpload) -> None:
    """Property 1's conditions over one accepted analysis.

    The whole documented shape of Requirement 1.3: a `scene_id`, width and height
    equal to the processed image's dimensions, a horizon, and a plane list whose
    entries carry a valid name, a contour of at least three in-bounds points,
    exactly four in-bounds bounding points, and an `area_fraction` in `(0, 1]`
    that equals its own mask's pixel count over the total.

    That last conjunct is why this reads the cached Scene_State and not only the
    response: `area_fraction` is the client's only handle on how much of the
    picture a plane covers, and a response reporting a plausible number unrelated
    to the mask it was computed from would satisfy every assertion made from the
    JSON alone.

    Width and height are checked against the uploaded room's own dimensions,
    which stay well under `max_longest_edge` throughout this module, so no
    clamping is in play; the clamp itself is Property 8's subject in the
    rejection-path section.

    The two reporting fields ride along, since they belong to the same response
    and Requirements 4.6 and 6.3 are each one field.
    """
    body = record.body
    height, width = record.source_shape

    assert isinstance(body["scene_id"], str) and body["scene_id"].strip(), (
        f"{record.spec.label}: scene_id is {body['scene_id']!r}"
    )
    assert (body["width"], body["height"]) == (width, height), (
        f"{record.spec.label}: reported {body['width']}x{body['height']} for a "
        f"{width}x{height} photograph"
    )
    assert body["analysis_ms"] >= 0

    # Requirement 4.6 and Requirement 6.3, each one field, each also checked
    # against what the cached scene says so the value cannot be decoration.
    assert body["segmentation_backend"] == "classical", (
        f"{record.spec.label}: backend reported as {body['segmentation_backend']!r} "
        "under an offline harness with the neural backend disabled"
    )
    assert body["segmentation_backend"] == record.cached_backend, (
        f"{record.spec.label}: response says {body['segmentation_backend']!r} but "
        f"the cached scene says {record.cached_backend!r}"
    )
    assert body["geometry_mode"] in GEOMETRY_MODES, (
        f"{record.spec.label}: geometry_mode {body['geometry_mode']!r} is not one of "
        f"{sorted(GEOMETRY_MODES)}"
    )

    # The horizon: a real line, and `y_at_center` genuinely on it. Without the
    # second half that field could be any number at all, and the frontend draws
    # its horizon guide from exactly that number.
    horizon = body["horizon"]
    a, b, c = float(horizon["a"]), float(horizon["b"]), float(horizon["c"])
    assert all(np.isfinite([a, b, c])), f"{record.spec.label}: horizon {horizon!r}"
    assert (a * a + b * b) > 0.0, (
        f"{record.spec.label}: the horizon has a zero direction, so it is not a line"
    )
    centre_x = (width - 1) / 2.0
    if abs(b) > 1e-9:
        assert float(horizon["y_at_center"]) == pytest.approx(
            -(a * centre_x + c) / b, rel=1e-6, abs=1e-6
        ), f"{record.spec.label}: y_at_center is not the horizon's row at the centre"

    # Vanishing points: the three documented labels always present, each either
    # `None` or a finite pair, so the frontend can read the field unconditionally.
    vanishing_points = body["vanishing_points"]
    assert set(vanishing_points) == {"VPx", "VPy", "VPz"}, (
        f"{record.spec.label}: vanishing point labels {sorted(vanishing_points)!r}"
    )
    for label, value in vanishing_points.items():
        if value is None:
            continue
        assert len(value) == 2 and all(np.isfinite(value)), (
            f"{record.spec.label}: {label} is {value!r}"
        )

    planes = body["planes"]
    assert planes, f"{record.spec.label}: accepted with no planes at all"
    names = [str(plane["name"]) for plane in planes]
    assert set(names) <= set(PLANE_NAMES), f"{record.spec.label}: names {names!r}"
    assert len(set(names)) == len(names), (
        f"{record.spec.label}: a plane name repeats in {names!r}"
    )

    total_pixels = float(width * height)
    for plane in planes:
        name = str(plane["name"])
        where = f"{record.spec.label}/{name}"

        contour = plane["contour"]
        assert len(contour) >= 3, f"{where}: contour has {len(contour)} points"
        assert_points_in_bounds(contour, width, height, f"{where} contour")

        quad = plane["bounding_points"]
        assert len(quad) == 4, f"{where}: bounding_points has {len(quad)} points"
        assert_points_in_bounds(quad, width, height, f"{where} bounding_points")

        centroid = plane["centroid"]
        assert 0.0 <= float(centroid[0]) <= width - 1, f"{where}: centroid {centroid!r}"
        assert 0.0 <= float(centroid[1]) <= height - 1, f"{where}: centroid {centroid!r}"

        rmse = float(plane["reprojection_rmse_px"])
        assert np.isfinite(rmse) and rmse >= 0.0, f"{where}: rmse {rmse!r}"

        fraction = float(plane["area_fraction"])
        assert 0.0 < fraction <= 1.0, f"{where}: area_fraction {fraction!r} not in (0, 1]"

        mask = record.artifacts[f"plane_mask:{name}"]
        assert mask.shape == (height, width), f"{where}: mask shape {mask.shape!r}"
        assert fraction == pytest.approx(
            mask.set_pixels / total_pixels, abs=AREA_FRACTION_TOLERANCE
        ), (
            f"{where}: area_fraction {fraction!r} against {mask.set_pixels} px of "
            f"{int(total_pixels)}"
        )


# Feature: ai-room-tile-visualizer, Property 1: Analysis response is wellformed
# for every accepted photograph
@given(index=_corpus_index)
@_CORPUS_PROPERTY_SETTINGS
def test_property_1_analysis_response_is_wellformed(
    analysed_corpus: tuple[AnalysedUpload, ...], index: int
) -> None:
    """Property 1 over the curated corpus, at every awkward pose in it.

    The corpus is finite, so this covers all sixteen uploads rather than sampling
    a hundred; the wide quantifier over freshly drawn photographs is
    `test_properties_1_2_and_33_hold_for_freshly_drawn_photographs`.

    **Validates: Requirements 1.1, 1.3, 3.1, 3.6, 4.6, 6.3**
    """
    record = analysed_corpus[index]
    event(f"{record.spec.label}: status={record.status}")
    if not record.accepted:
        # Rejections belong to the rejection-path section; `accepted_uploads` pins
        # that they stay rare and documented.
        assert record.error_code == "no_usable_plane"
        return
    event(f"geometry_mode={record.body['geometry_mode']}")
    assert_response_wellformed(record)


def test_scene_ids_are_distinct_across_uploads(
    accepted_uploads: tuple[AnalysedUpload, ...],
) -> None:
    """Each analysis gets its own `scene_id`.

    Property 1 asserts the field is a non-empty string per response, which a
    constant would satisfy. Two photographs sharing an id would have the second
    analysis silently overwrite the first in the cache, so a shopper's earlier
    room would start rendering with the later room's geometry.
    """
    ids = [record.scene_id for record in accepted_uploads]
    assert len(set(ids)) == len(ids), f"scene_id repeated across uploads: {ids!r}"


def test_segment_takes_only_the_photograph(client: Any) -> None:
    """Requirement 1.5, read off the request contract itself.

    The requirement is a claim about what the endpoint *cannot* accept: no corner
    points, no plane annotations, no perspective input. Sending such fields and
    watching them be ignored would only show that those particular names are
    unused. The published schema is the stronger statement -- the multipart body
    has exactly one property, the file -- and it is what any client integrating
    against the service reads.
    """
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/api/segment"]["post"]["requestBody"]
    reference = body["content"]["multipart/form-data"]["schema"]["$ref"]
    model = schema["components"]["schemas"][reference.rsplit("/", 1)[-1]]

    assert set(model["properties"]) == {"file"}, (
        f"/api/segment accepts {sorted(model['properties'])!r}; Requirement 1.5 "
        "allows the photograph alone"
    )
    assert model["required"] == ["file"]


# --------------------------------------------------------------------------- #
# Property 2 -- every returned plane is non-empty and above the minimum area
# (Requirements 3.5, 6.5)
# --------------------------------------------------------------------------- #


def assert_planes_are_usable(record: AnalysedUpload) -> None:
    """Property 2's conditions over one accepted analysis.

    Both halves, because they fail differently. A plane reported with an empty
    mask would give the frontend a selectable region that renders nothing --
    Requirement 3.5 says such a plane must be *omitted* instead. A plane under
    `min_plane_area_fraction` would be a sliver nobody can meaningfully tile, and
    it is the same threshold Requirement 6.5 rejects a whole photograph for
    missing, so the response and the rejection rule have to agree on it.

    The response's plane set and the cached mask set are compared in both
    directions: a plane in the response with no cached mask is unrenderable, and a
    cached mask with no response entry is a plane the shopper can never select.
    """
    minimum = default_settings().min_plane_area_fraction
    reported = {str(plane["name"]): plane for plane in record.body["planes"]}
    cached = {
        key.split(":", 1)[1]
        for key in record.artifacts
        if artifact_family(key) == "plane_mask"
    }

    assert set(reported) == cached, (
        f"{record.spec.label}: response planes {sorted(reported)!r} disagree with "
        f"cached masks {sorted(cached)!r}"
    )

    for name, plane in reported.items():
        mask = record.artifacts[f"plane_mask:{name}"]
        assert mask.set_pixels > 0, (
            f"{record.spec.label}/{name}: reported with an empty mask; Requirement "
            "3.5 requires it be omitted instead"
        )
        fraction = float(plane["area_fraction"])
        assert fraction >= minimum, (
            f"{record.spec.label}/{name}: area_fraction {fraction:.6f} is under the "
            f"{minimum} floor"
        )


# Feature: ai-room-tile-visualizer, Property 2: Every returned plane is non-empty
# and above the minimum area
@given(index=_corpus_index)
@_CORPUS_PROPERTY_SETTINGS
def test_property_2_returned_planes_are_non_empty_and_above_the_minimum(
    analysed_corpus: tuple[AnalysedUpload, ...], index: int
) -> None:
    """Property 2 over the curated corpus, including its single-plane rooms.

    **Validates: Requirements 3.5, 6.5**
    """
    record = analysed_corpus[index]
    if not record.accepted:
        assert record.error_code == "no_usable_plane"
        return
    event(f"planes={len(record.plane_names)}")
    assert_planes_are_usable(record)


def test_no_plane_name_appears_without_a_mask_anywhere_in_the_corpus(
    accepted_uploads: tuple[AnalysedUpload, ...],
) -> None:
    """Property 2's omission half, stated over the corpus as a whole.

    The per-record test visits one photograph at a time, so a backend that always
    reported all four planes would fail it only on the records where a plane is
    genuinely absent. This pins that absence actually occurs in the corpus, which
    is what makes the omission claim non-vacuous.
    """
    counts = {record.spec.label: len(record.plane_names) for record in accepted_uploads}
    assert any(count < len(PLANE_NAMES) for count in counts.values()), (
        "every corpus upload reported all four Structural_Planes, so the omission "
        f"rule of Requirement 3.5 was never exercised: {counts!r}"
    )


# --------------------------------------------------------------------------- #
# Property 3 -- the analysis result is retrievable by its returned scene_id
# (Requirements 1.4, 9.1)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 3: Analysis result is retrievable by
# its returned scene_id
@given(index=_corpus_index)
@_CORPUS_PROPERTY_SETTINGS
def test_property_3_every_analysed_scene_resolves_and_renders(
    analysed_corpus: tuple[AnalysedUpload, ...], index: int
) -> None:
    """Property 3 across photographs: the id comes back usable.

    Two claims, and the second is the one that matters to a shopper. The
    `scene_id` resolves to a Scene_State in the Scene_Cache (Requirement 1.4), and
    a render issued against it does not come back `scene_expired` -- which is what
    a shopper would read as "upload your photo again" immediately after uploading
    it.

    Every render the corpus recorded is checked: the one naming every detected
    plane, the one carrying every optional setting of Requirement 9.1, the
    `?binary=1` one, and the one naming no planes at all. A successful response is
    additionally required to describe the same image the analysis did, since a
    render whose dimensions disagreed with the scene would be a different
    photograph.

    **Validates: Requirements 1.4, 9.1**
    """
    record = analysed_corpus[index]
    if not record.accepted:
        assert record.error_code == "no_usable_plane"
        return

    assert record.resolved_in_cache, (
        f"{record.spec.label}: the returned scene_id did not resolve in the cache"
    )
    assert record.renders, f"{record.spec.label}: no render was recorded"

    height, width = record.source_shape
    for outcome in record.renders:
        where = f"{record.spec.label}/{outcome.label}"
        assert outcome.code != "scene_expired", (
            f"{where}: a render immediately after analysis reported scene_expired"
        )
        assert outcome.status == 200, (
            f"{where}: render returned {outcome.status} code={outcome.code!r}"
        )
        assert outcome.image_bytes > 0, f"{where}: render returned no image data"
        assert outcome.render_ms is not None and outcome.render_ms >= 0, (
            f"{where}: render_ms is {outcome.render_ms!r}"
        )
        if outcome.binary:
            assert outcome.mime is not None and outcome.mime.startswith("image/"), (
                f"{where}: binary render served {outcome.mime!r}"
            )
            assert outcome.scene_id_header == record.scene_id, (
                f"{where}: X-Scene-Id is {outcome.scene_id_header!r}"
            )
        else:
            assert (outcome.width, outcome.height) == (width, height), (
                f"{where}: render is {outcome.width}x{outcome.height} for a "
                f"{width}x{height} scene"
            )
            assert outcome.mime in {"image/png", "image/jpeg"}, (
                f"{where}: render mime {outcome.mime!r}"
            )


def test_every_accepted_analysis_stays_in_the_cache(
    accepted_uploads: tuple[AnalysedUpload, ...],
) -> None:
    """Requirement 1.4 across a session: one upload does not displace the last.

    Property 3 asserts each `scene_id` resolves at the moment it is issued, which
    a one-entry cache would also satisfy. The corpus uploads sixteen photographs
    against a 32-entry LRU bound, so occupancy must grow by one per accepted
    upload -- which is what makes the eviction bound of Requirement 9.5 a bound
    rather than a behaviour.
    """
    entries = [record.cache_entries for record in accepted_uploads]
    assert entries == list(range(1, len(accepted_uploads) + 1)), (
        f"cache occupancy did not grow one entry per accepted upload: {entries!r}"
    )


# Feature: ai-room-tile-visualizer, Property 3: Analysis result is retrievable by
# its returned scene_id
@given(
    planes=_plane_selection,
    spec=_plane_spec,
    fmt=_render_format,
    binary=st.booleans(),
)
@_RENDER_PROPERTY_SETTINGS
def test_property_3_a_live_scene_never_reports_expiry(
    live_scene: LiveScene,
    planes: list[str],
    spec: dict[str, Any],
    fmt: str | None,
    binary: bool,
) -> None:
    """Property 3 across render requests: retrievability does not depend on them.

    The companion to the corpus test above. There, one render shape was issued
    against many photographs; here, a hundred render shapes are issued against one
    live scene -- every plane subset, all three tile formats, rotations across a
    full turn, inherited and explicit grout, metric offsets, both output formats,
    and both response encodings.

    A `scene_id` that resolved only for the plain request would still pass the
    corpus test. What this rules out is any request shape that loses the scene,
    which a shopper meets as their room disappearing the moment they pick an
    unusual combination of settings.

    **Validates: Requirements 1.4, 9.1**
    """
    payload, selected = render_payload(live_scene, planes, spec, fmt)
    response = live_scene.render(payload, binary=binary)

    assert error_code(response) != "scene_expired", (
        f"render naming {selected!r} reported scene_expired against a live scene"
    )
    assert response.status_code == 200, (
        f"render naming {selected!r} returned {response.status_code} "
        f"code={error_code(response)!r}"
    )
    event(f"planes={len(selected)}, binary={binary}, format={fmt}")

    if binary:
        assert response.headers["X-Scene-Id"] == live_scene.scene_id
        assert response.headers["content-type"].startswith("image/")
        assert len(response.content) > 0
    else:
        body = response.json()
        assert body["scene_id"] == live_scene.scene_id
        assert (body["width"], body["height"]) == (
            live_scene.body["width"],
            live_scene.body["height"],
        )
        assert body["image"], "render returned an empty image field"


# --------------------------------------------------------------------------- #
# Property 28 -- rendering never repeats analysis
# (Requirements 1.2, 9.2)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 28: Rendering never repeats analysis
@given(specs=st.lists(_plane_spec, min_size=1, max_size=4), planes=_plane_selection)
@_RENDER_PROPERTY_SETTINGS
def test_property_28_rendering_never_repeats_analysis(
    live_scene: LiveScene, specs: list[dict[str, Any]], planes: list[str]
) -> None:
    """Property 28: no sequence of renders re-enters an analysis stage.

    This is the load-bearing claim of the whole two-pass split, and the only
    reason the render budget of Requirement 9.3 is reachable: analysis is
    hundreds of milliseconds to seconds of work, so a render path that re-entered
    any part of it would miss the budget by an order of magnitude no matter how
    fast the compositing got.

    The counters are installed *after* the photograph has been analysed by the
    shared fixture, so the assertion is the property's "stay at their
    post-analysis values": every counter must still read zero once the drawn
    sequence of renders has run. All four stages are counted rather than the two
    the task names, because re-running the CIELAB decomposition would blow the
    same budget as re-running segmentation, and counting it costs one attribute
    swap.

    `test_the_stage_counters_are_wired_to_the_stages_they_name` is what keeps this
    honest: a patch on a symbol the route never reaches would make every assertion
    here pass while measuring nothing.

    **Validates: Requirements 1.2, 9.2**
    """
    with counted_analysis_stages() as counters:
        for spec in specs:
            payload, selected = render_payload(live_scene, planes, spec, None)
            response = live_scene.render(payload)
            assert response.status_code == 200, (
                f"render naming {selected!r} returned {response.status_code} "
                f"code={error_code(response)!r}"
            )

    assert counters.as_dict() == {
        "segment": 0,
        "vanishing_point": 0,
        "calibrate": 0,
        "decompose": 0,
    }, f"{len(specs)} renders re-entered analysis: {counters.as_dict()!r}"
    event(f"renders={len(specs)}")


def test_the_stage_counters_are_wired_to_the_stages_they_name(client: Any) -> None:
    """Guard for Property 28: the counters observe the live analysis path.

    Property 28 asserts four counters stay at zero. A patch on a symbol
    `/api/segment` never reaches would satisfy that trivially, so this drives one
    real analysis through the same counters and requires every one of them to
    move. It is also Requirement 1.2 stated directly: segmentation, calibration,
    and lighting decomposition each run *exactly once* per photograph.

    **Validates: Requirements 1.2, 9.2**
    """
    assert client.app.state.segmenter.backend_name == "classical", (
        "the counter on ClassicalSegmenter.segment is only on the live path while "
        "the classical backend is active"
    )
    payload = encode_image(LIVE_SCENE_SPEC.render(), LIVE_SCENE_SPEC.fmt)

    with counted_analysis_stages() as counters:
        response = client.post(
            "/api/segment",
            files=upload_part(payload, LIVE_SCENE_SPEC.filename, LIVE_SCENE_SPEC.mime),
        )

    assert response.status_code == 200, f"analysis failed: {error_code(response)!r}"
    assert counters.segment == 1, (
        f"segmentation ran {counters.segment} times for one photograph"
    )
    assert counters.calibrate == 1, (
        f"calibration ran {counters.calibrate} times for one photograph"
    )
    assert counters.decompose == 1, (
        f"lighting decomposition ran {counters.decompose} times for one photograph"
    )
    assert counters.vanishing_point > 0, (
        "vanishing point estimation was never entered, so the counter Property 28 "
        "relies on is not on the analysis path"
    )


def test_a_second_upload_is_analysed_again(client: Any) -> None:
    """Guard for Requirement 1.2 in the other direction.

    "Exactly once for that photograph" is a statement about analysing each
    photograph once, not about analysing once per process. A service that reused
    the first scene for every upload would satisfy Property 28 and the counter
    guard above, and would show every shopper somebody else's room.
    """
    first_spec, second_spec = UPLOAD_SPECS[0], UPLOAD_SPECS[9]
    first = client.post(
        "/api/segment",
        files=upload_part(
            encode_image(first_spec.render(), first_spec.fmt),
            first_spec.filename,
            first_spec.mime,
        ),
    )
    with counted_analysis_stages() as counters:
        second = client.post(
            "/api/segment",
            files=upload_part(
                encode_image(second_spec.render(), second_spec.fmt),
                second_spec.filename,
                second_spec.mime,
            ),
        )

    assert first.status_code == 200 and second.status_code == 200
    assert counters.segment == 1, "the second photograph was not segmented"
    assert first.json()["scene_id"] != second.json()["scene_id"]
    assert len(client.app.state.cache) == 2


# --------------------------------------------------------------------------- #
# Property 33 -- cached artifacts are 8-bit per channel
# (Requirement 12.4)
# --------------------------------------------------------------------------- #


def assert_cached_artifacts_are_8_bit(record: AnalysedUpload) -> None:
    """Property 33's conditions over one accepted analysis.

    The memory budget of Requirement 12.1 rests on this. At 8 bits a 2048 px scene
    is about 31 MB and a full 32-entry cache is roughly 1 GB, inside the 2 GB
    ceiling with headroom for a concurrent analysis; a single `float32` map would
    quadruple that map's share, and a `float64` one would put a full cache over the
    ceiling on its own.

    Dtypes are checked both immediately after analysis and again after the
    recorded renders, because the render path writes back into the Scene_State and
    the alpha cache is the one thing that could promote a cached array's dtype
    after the fact.

    `plane_alpha` is deliberately outside this contract -- derived render state
    rather than analysis output, with Requirement 12.4 stated over masks and
    lighting maps -- so it is asserted to be the `float32` its docstring in
    `backend/schemas.py` promises. That pins the exclusion instead of merely
    tolerating it.
    """
    height, width = record.source_shape
    for stage, artifacts in (
        ("after analysis", record.artifacts),
        ("after rendering", record.artifacts_after_render),
    ):
        contracted = {
            key: facts
            for key, facts in artifacts.items()
            if artifact_family(key) in CONTRACTED_ARTIFACTS
        }
        # All four families must be represented, or an artifact could pass this by
        # being absent from the cache altogether.
        families = {artifact_family(key) for key in contracted}
        assert families == set(CONTRACTED_ARTIFACTS), (
            f"{record.spec.label} {stage}: cached families {sorted(families)!r} do "
            f"not cover {sorted(CONTRACTED_ARTIFACTS)!r}"
        )

        for key, facts in contracted.items():
            assert facts.dtype == "uint8", (
                f"{record.spec.label} {stage}: {key} is {facts.dtype}, not uint8"
            )
            assert facts.shape == (height, width), (
                f"{record.spec.label} {stage}: {key} has shape {facts.shape!r}, "
                f"expected {(height, width)!r}"
            )

        for key, facts in artifacts.items():
            family = artifact_family(key)
            if family in DERIVED_ARTIFACT_DTYPES:
                assert facts.dtype == DERIVED_ARTIFACT_DTYPES[family], (
                    f"{record.spec.label} {stage}: {key} is {facts.dtype}, but "
                    f"{family} is documented as {DERIVED_ARTIFACT_DTYPES[family]}"
                )

    # The photograph itself is cached alongside the maps and is the largest single
    # array in a Scene_State, so its precision belongs to the same budget.
    image = record.artifacts["image"]
    assert image.dtype == "uint8", f"{record.spec.label}: cached image is {image.dtype}"
    assert image.shape == (height, width, 3), (
        f"{record.spec.label}: cached image has shape {image.shape!r}"
    )


# Feature: ai-room-tile-visualizer, Property 33: Cached artifacts are 8-bit per
# channel
@given(index=_corpus_index)
@_CORPUS_PROPERTY_SETTINGS
def test_property_33_cached_artifacts_are_8_bit(
    analysed_corpus: tuple[AnalysedUpload, ...], index: int
) -> None:
    """Property 33 over the curated corpus, before and after rendering.

    **Validates: Requirements 12.4**
    """
    record = analysed_corpus[index]
    if not record.accepted:
        assert record.error_code == "no_usable_plane"
        return
    assert_cached_artifacts_are_8_bit(record)


def test_the_render_alpha_cache_is_populated_and_stays_derived(
    accepted_uploads: tuple[AnalysedUpload, ...],
) -> None:
    """The `plane_alpha` exclusion in Property 33 is real, not hypothetical.

    Two things have to hold for that exclusion to be a considered decision rather
    than a gap. The alpha cache must actually be populated by rendering -- if it
    never were, the exclusion would describe nothing -- and it must be populated
    only for the planes a render named, since an alpha for an untouched plane
    would be memory spent on work nobody asked for.
    """
    populated = 0
    for record in accepted_uploads:
        before = {
            key for key in record.artifacts if artifact_family(key) == "plane_alpha"
        }
        after = {
            key
            for key in record.artifacts_after_render
            if artifact_family(key) == "plane_alpha"
        }
        assert not before, (
            f"{record.spec.label}: analysis populated {sorted(before)!r}; the alpha "
            "cache is render state and should start empty"
        )
        rendered = {
            f"plane_alpha:{name}"
            for outcome in record.renders
            for name in outcome.requested
        }
        assert after <= rendered, (
            f"{record.spec.label}: alphas {sorted(after - rendered)!r} were cached "
            "for planes no render named"
        )
        populated += len(after)

    assert populated > 0, (
        "no render populated a plane alpha anywhere in the corpus, so Property 33's "
        "exclusion of plane_alpha describes nothing"
    )


# --------------------------------------------------------------------------- #
# Properties 1, 2, and 33 over freshly drawn photographs
# (Requirements 1.1, 1.3, 3.5, 6.5, 12.4)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 1: Analysis response is wellformed
# for every accepted photograph
# Feature: ai-room-tile-visualizer, Property 2: Every returned plane is non-empty
# and above the minimum area
# Feature: ai-room-tile-visualizer, Property 33: Cached artifacts are 8-bit per
# channel
@pytest.mark.slow
@given(spec=drawn_upload_specs())
@_DRAWN_PROPERTY_SETTINGS
def test_properties_1_2_and_33_hold_for_freshly_drawn_photographs(
    analyse: Callable[[UploadSpec], AnalysedUpload], spec: UploadSpec
) -> None:
    """Properties 1, 2, and 33 over a hundred freshly analysed photographs.

    The wide universal quantifier the three corpus tests cannot give: continuous
    yaw and pitch, every wall subset including none, zero to two occluders, drawn
    occluder seeds, and all three accepted upload formats -- a genuinely new
    photograph analysed per example, at about 250 ms each.

    The three properties share one test because they are claims about the same
    response and the same Scene_State, and analysing three times to assert them
    separately would triple the only slow test in this module for no additional
    coverage. Each condition set lives in its own `assert_*` helper, and each of
    those helpers is also what the corresponding focused corpus test calls, so
    there is one definition of each property rather than two.

    A drawn pose is allowed to be rejected -- a wall-free, nearly level frame may
    genuinely have nothing tileable in it -- but only with the documented
    `no_usable_plane` envelope, and the run is required to accept most of what it
    draws so the assertions are not quietly skipped.

    **Validates: Requirements 1.1, 1.3, 3.5, 6.5, 12.4**
    """
    record = analyse(spec)
    event(f"accepted={record.accepted}")

    if not record.accepted:
        assert record.status == 422, (
            f"{spec!r}: unexpected rejection status={record.status} "
            f"code={record.error_code!r}"
        )
        assert_error_envelope_body(record.error_body, "no_usable_plane", spec.label)
        return

    event(f"geometry_mode={record.body['geometry_mode']}")
    assert_response_wellformed(record)
    assert_planes_are_usable(record)
    assert_cached_artifacts_are_8_bit(record)


def test_most_drawn_photographs_are_accepted(
    analyse: Callable[[UploadSpec], AnalysedUpload],
) -> None:
    """Guard against the drawn property passing by rejecting everything.

    The drawn test returns early on a rejection, so a regression that rejected
    every photograph would leave it green with nothing asserted. Hypothesis's
    `event` counts are not visible to an assertion, so the acceptance rate is
    measured here instead, over a fixed eight-pose sweep of the same space: a
    deterministic figure that cannot flake, at the cost of eight analyses.
    """
    specs = [
        UploadSpec(
            label=f"sweep_{index}",
            width=width,
            height=height,
            yaw_deg=yaw,
            pitch_deg=pitch,
            walls=walls,
            n_occluders=index % 3,
            seed=index,
            fmt=fmt,
        )
        for index, (width, height, yaw, pitch, walls, fmt) in enumerate(
            (
                (320, 240, 8.0, -12.0, ("left", "right", "back"), "png"),
                (320, 240, -16.0, -20.0, ("left", "back"), "jpeg"),
                (400, 300, 0.0, -6.0, ("back",), "webp"),
                (400, 300, 24.0, -28.0, ("left", "right"), "png"),
                (320, 240, -8.0, -16.0, (), "png"),
                (400, 300, 12.0, -10.0, ("right", "back"), "jpeg"),
                (320, 240, -24.0, -24.0, ("left", "right", "back"), "webp"),
                (400, 300, 4.0, -14.0, ("left",), "png"),
            )
        )
    ]
    records = [analyse(spec) for spec in specs]
    accepted = sum(1 for record in records if record.accepted)
    rate = accepted / len(records)

    assert rate >= MIN_DRAWN_ACCEPTANCE, (
        f"only {accepted} of {len(records)} drawn photographs were accepted "
        f"({rate:.2f}), under the {MIN_DRAWN_ACCEPTANCE} floor; the drawn property "
        "would be mostly skipping its assertions"
    )


# --------------------------------------------------------------------------- #
# Reported backend and geometry mode (Requirements 4.6, 6.3)
# --------------------------------------------------------------------------- #


def test_health_and_segment_agree_on_the_active_backend(client: Any) -> None:
    """Requirement 4.6 across both endpoints that report the backend.

    `/api/health` is what an operator reads and `/api/segment` is what the client
    reads, so the two disagreeing would mean one of them describes something other
    than the code that ran. Under this harness the neural backend is disabled, so
    both must say `classical` and the provider must be the documented placeholder
    rather than an onnxruntime provider name.
    """
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["segmentation_backend"] == "classical"
    assert health["onnx_provider"] == app_module.NO_PROVIDER

    response = client.post(
        "/api/segment",
        files=upload_part(
            encode_image(LIVE_SCENE_SPEC.render(), LIVE_SCENE_SPEC.fmt),
            LIVE_SCENE_SPEC.filename,
            LIVE_SCENE_SPEC.mime,
        ),
    )
    assert response.status_code == 200
    assert response.json()["segmentation_backend"] == health["segmentation_backend"]
    assert client.app.state.segmenter.backend_name == health["segmentation_backend"]


def test_reported_geometry_mode_is_the_one_every_plane_was_calibrated_under(
    accepted_uploads: tuple[AnalysedUpload, ...],
) -> None:
    """Requirement 6.3: the reported mode describes the calibration that ran.

    Property 1 pins that the response field holds one of the two documented
    values, which a hardcoded string would also satisfy. This adds that the value
    is not decoration: it equals the mode recorded on the cached Scene_State, and
    equals the mode every individual plane was given geometry by. A client that
    treats `planar_fallback` as a hint that the preview is less trustworthy -- the
    only reason to publish the field at all -- depends on exactly that.
    """
    for record in accepted_uploads:
        reported = str(record.body["geometry_mode"])
        assert reported in GEOMETRY_MODES
        assert reported == record.cached_geometry_mode, (
            f"{record.spec.label}: the response says {reported!r} but the cached "
            f"scene says {record.cached_geometry_mode!r}"
        )
        assert record.plane_geometry_modes, f"{record.spec.label}: no plane geometry"
        for name, mode in record.plane_geometry_modes.items():
            assert mode == reported, (
                f"{record.spec.label}/{name}: calibrated as {mode!r} while the "
                f"response reports {reported!r}"
            )


# =========================================================================== #
# Task 13.6 -- rejection paths
# (Requirements 1.6, 2.1-2.3, 2.5, 2.6, 6.5, 8.4, 9.4, 12.5)
# =========================================================================== #

# Every path below is a rejection, and rejections are cheap: validation
# short-circuits before any pipeline stage, a cache miss short-circuits before any
# pixel is touched, and a patched stage fails immediately. So unlike the happy
# paths above, these properties can afford to be drawn per example against a live
# client with no corpus, no memo, and no session-scoped record.
#
# Two of them need configuration the default settings do not give. Property 7
# needs a cap small enough to cross with kilobyte payloads instead of the 12 MB
# the default demands, and Property 8 needs a `max_longest_edge` an ordinary test
# image can exceed. :func:`overridden_env` layers those `RV_*` values underneath
# :func:`configured_client`, which reads settings inside the `TestClient` context
# and restores everything it touched on the way out.
#
# The 415 and 413 properties are stated as *exact* conditions -- 415 precisely
# when a check fails, 413 precisely when the payload is over the cap -- rather
# than as the one-directional "only when" the requirements are phrased as. Both
# directions hold here, and the reverse direction is the half that catches a
# validator which rejected legitimate photographs.


# --------------------------------------------------------------------------- #
# Rejection-path constants
# --------------------------------------------------------------------------- #

#: Upload cap the Property 7 harness runs under. 64 KiB, so the whole size space
#: either side of the threshold is reachable with kilobyte payloads: crossing the
#: 12 MB default would mean allocating tens of megabytes per example.
LOWERED_UPLOAD_CAP: int = 64 * 1024

#: Longest-edge limit the Property 8 harness runs under. Large enough that a
#: clamped image keeps a short edge of 154 px or more across the drawn aspect
#: range, where the worst case half-pixel rounding is 0.33 percent -- inside the
#: 0.5 percent bound of Requirement 2.6 -- and small enough that an ordinary test
#: image exceeds it.
CLAMP_LIMIT: int = 512

#: Frame size of the probe photograph the validation properties upload. Small on
#: purpose: these properties are about what happens *before* analysis, and the
#: accepted branch still has to pay for one, so the cheapest photograph the
#: service will accept is the right one.
PROBE_PHOTO_SIZE: tuple[int, int] = (128, 96)

#: Aspect-ratio tolerance of Requirement 2.6.
ASPECT_TOLERANCE: float = 0.005

#: The allow-lists :data:`MIME_CASES` and :data:`FILENAME_CASES` were written
#: against. Asserted against live settings by
#: `test_the_upload_variant_tables_match_the_configured_allow_lists`, so a
#: configuration change cannot leave the tables silently stale.
EXPECTED_ALLOWED_MIME_TYPES: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
EXPECTED_ALLOWED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

#: What every analysis stage counter reads when no pipeline work was done.
NO_STAGES_ENTERED: dict[str, int] = {
    "segment": 0,
    "vanishing_point": 0,
    "calibrate": 0,
    "decompose": 0,
}


@dataclass(frozen=True)
class UploadVariant:
    """One axis value of the Property 6 input space, with its verdict.

    `key` is a MIME header, a filename, or a :func:`probe_payloads` key depending
    on which table the variant sits in; `allowed` is whether Requirement 2.1 (or,
    for a payload, Requirement 2.2) lets it through. Frozen and named so a failing
    Hypothesis example prints the offending value rather than a tuple.
    """

    key: str
    allowed: bool

    def __repr__(self) -> str:  # pragma: no cover - Hypothesis reporting only
        return f"<{self.key!r} {'allowed' if self.allowed else 'rejected'}>"


#: Declared MIME headers. The accepted three, plus the two normalisations the
#: service performs -- case folding and parameter stripping -- and six types that
#: are plausible for a file a shopper might pick and are not in the allow-list.
#: `image/svg+xml` is there deliberately: it is an image type, and it is markup.
MIME_CASES: tuple[UploadVariant, ...] = (
    UploadVariant("image/png", True),
    UploadVariant("image/jpeg", True),
    UploadVariant("image/webp", True),
    UploadVariant("image/PNG", True),
    UploadVariant("image/jpeg; charset=binary", True),
    UploadVariant("image/gif", False),
    UploadVariant("image/svg+xml", False),
    UploadVariant("image/tiff", False),
    UploadVariant("application/octet-stream", False),
    UploadVariant("text/plain", False),
    UploadVariant("application/pdf", False),
)

#: Declared filenames. All four accepted extensions, an uppercase spelling, a
#: name with no extension at all, and the double-extension case -- `room.png.txt`
#: is a `.txt`, and a validator reading the *first* dot rather than the last would
#: wave it through.
FILENAME_CASES: tuple[UploadVariant, ...] = (
    UploadVariant("room.png", True),
    UploadVariant("room.jpg", True),
    UploadVariant("room.jpeg", True),
    UploadVariant("room.webp", True),
    UploadVariant("room.PNG", True),
    UploadVariant("room.gif", False),
    UploadVariant("room.txt", False),
    UploadVariant("room.exe", False),
    UploadVariant("room", False),
    UploadVariant("room.png.txt", False),
    UploadVariant("room.tiff", False),
)

#: Payload bodies. Three genuine rasters, and six things that are not: nothing at
#: all, prose, a block of zeros, two truncations, and eight valid PNG magic bytes
#: followed by rubbish -- the case a magic-number sniff would accept and an actual
#: decode will not.
PAYLOAD_CASES: tuple[UploadVariant, ...] = (
    UploadVariant("png", True),
    UploadVariant("jpeg", True),
    UploadVariant("webp", True),
    UploadVariant("empty", False),
    UploadVariant("text", False),
    UploadVariant("zeros", False),
    UploadVariant("truncated_png", False),
    UploadVariant("truncated_jpeg", False),
    UploadVariant("png_magic_only", False),
)


# --------------------------------------------------------------------------- #
# Rejection-path helpers
# --------------------------------------------------------------------------- #


def probe_photo(width: int, height: int) -> np.ndarray:
    """A cheap, deterministic photograph the Segmenter reliably finds a plane in.

    Not :meth:`UploadSpec.render`. The properties in this section are about upload
    validation, the size cap, and the downscale clamp -- none of which look at
    what the picture depicts -- and every accepted example still pays for one
    analysis. A vertical luminance ramp crossed with a coarse checkerboard gives
    the classical backend the gradient and the straight edges it needs while
    costing a fraction of a projected room to build.

    `test_the_probe_photograph_is_accepted` pins the acceptance, so a segmenter
    change surfaces as one clear failure rather than as drift in three properties.
    """
    ys, xs = np.mgrid[0:height, 0:width]
    ramp = (ys * 255 // max(height - 1, 1)).astype(np.uint8)
    image = np.dstack(
        [ramp, (ramp // 2 + 40).astype(np.uint8), (255 - ramp).astype(np.uint8)]
    )
    cell = max(4, min(width, height) // 6)
    image[((ys // cell) + (xs // cell)) % 2 == 1] //= 2
    return np.ascontiguousarray(image.astype(np.uint8))


@lru_cache(maxsize=1)
def probe_payloads() -> Mapping[str, bytes]:
    """The nine payload bodies :data:`PAYLOAD_CASES` names, built once.

    Memoised rather than computed at import so collecting this module stays free.
    """
    width, height = PROBE_PHOTO_SIZE
    image = probe_photo(width, height)
    png = encode_image(image, "png")
    jpeg = encode_image(image, "jpeg")
    return {
        "png": png,
        "jpeg": jpeg,
        "webp": encode_image(image, "webp"),
        "empty": b"",
        "text": b"this file is plainly not a photograph of a room",
        "zeros": bytes(512),
        "truncated_png": png[: len(png) // 3],
        "truncated_jpeg": jpeg[: len(jpeg) // 3],
        # PNG's eight-byte signature and nothing else that is a PNG.
        "png_magic_only": png[:8] + bytes(64),
    }


def decodes_as_raster(payload: bytes) -> bool:
    """Whether OpenCV turns `payload` into an image, asked of OpenCV directly.

    Deliberately not `backend.utils.imageio.decode_image`: Property 6's expected
    verdicts have to come from somewhere other than the function under test, or
    the decode conjunct is an identity.
    """
    if not payload:
        return False
    return (
        cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        is not None
    )


@contextmanager
def overridden_env(**values: str) -> Iterator[None]:
    """Set `RV_*` values for the duration of the block and restore them after.

    The settings cache is cleared on both edges, so the `Settings` built inside
    :func:`configured_client` observes these values and the next test does not.
    """
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    get_settings.cache_clear()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@contextmanager
def spied_downscale() -> Iterator[list[tuple[int, int]]]:
    """Record the `(height, width)` of every array the downscale stage returned.

    The clamp is what Requirement 2.6 is about, and its output is the *processed*
    photograph every later stage and every cached artifact is sized from. Watching
    it directly means Property 8 can quantify over input geometries that analysis
    might reject, where reading the response's `width`/`height` could only cover
    the accepted ones.

    `backend.app` resolves `clamp_longest_edge` from its own module globals at
    call time, so patching the attribute puts this on the live route. The spy
    asserting it was called exactly once per upload is what keeps that honest.
    """
    real = app_module.clamp_longest_edge
    seen: list[tuple[int, int]] = []

    def spy(img: np.ndarray, limit: int) -> np.ndarray:
        out = real(img, limit)
        seen.append((int(out.shape[0]), int(out.shape[1])))
        return out

    app_module.clamp_longest_edge = spy  # type: ignore[assignment]
    try:
        yield seen
    finally:
        app_module.clamp_longest_edge = real  # type: ignore[assignment]


class _StageStopped(RuntimeError):
    """Sentinel raised by a deliberately short-circuited analysis stage."""


@contextmanager
def stopped_after_downscale() -> Iterator[None]:
    """Make segmentation raise, so the route stops right after the clamp.

    Property 8 draws a hundred distinct input geometries. Analysing each one is
    the better part of a second at :data:`CLAMP_LIMIT`, and none of it bears on
    the claim: the clamp has already run and been observed by
    :func:`spied_downscale` by the time segmentation is entered. Stopping there
    turns a twenty-second property into a fast one without moving the observation
    point.

    The route converts the failure into 422 ``analysis_failed``, which each
    example asserts -- so a run where this patch missed the live path would fail
    rather than quietly analyse a hundred photographs.
    """
    real = ClassicalSegmenter.segment

    def stop(self: ClassicalSegmenter, image_bgr: np.ndarray) -> SegmentationResult:
        raise _StageStopped("segmentation short-circuited for the downscale property")

    ClassicalSegmenter.segment = stop  # type: ignore[method-assign]
    try:
        yield
    finally:
        ClassicalSegmenter.segment = real  # type: ignore[method-assign]


@contextmanager
def segmentation_yielding(
    build: Callable[[int, int], SegmentationResult],
) -> Iterator[None]:
    """Replace the Segmenter's output with `build(height, width)`.

    How the `no_usable_plane` paths are reached. A photograph cannot be relied on
    to produce them -- the classical backend finds a tileable plane in very little
    -- so the condition Requirement 6.5 rejects on is supplied directly while the
    route, the validation, and the clamp all stay real.
    """
    real = ClassicalSegmenter.segment

    def substitute(
        self: ClassicalSegmenter, image_bgr: np.ndarray
    ) -> SegmentationResult:
        height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
        return build(height, width)

    ClassicalSegmenter.segment = substitute  # type: ignore[method-assign]
    try:
        yield
    finally:
        ClassicalSegmenter.segment = real  # type: ignore[method-assign]


def no_planes_at_all(height: int, width: int) -> SegmentationResult:
    """A result with no Structural_Plane in it, as a wall-free close-up gives."""
    return SegmentationResult(
        plane_masks={},
        foreground_mask=np.zeros((height, width), np.uint8),
        contours={},
        bounding_points={},
        area_fractions={},
        backend_name="classical",
    )


def only_a_sub_threshold_plane(height: int, width: int) -> SegmentationResult:
    """A result whose single plane is a sliver under `min_plane_area_fraction`.

    The other half of Requirement 6.5. `no_planes_at_all` exercises the emptiness
    check; this exercises the *threshold*, which is the number the requirement
    actually names.
    """
    minimum = default_settings().min_plane_area_fraction
    mask = np.zeros((height, width), np.uint8)
    band = max(1, int(height * minimum / 4))
    mask[:band, :] = 255
    fraction = float(np.count_nonzero(mask)) / float(height * width)
    assert fraction < minimum, (
        f"the sliver covers {fraction:.4f} of the frame, which is not under the "
        f"{minimum} floor this helper exists to sit below"
    )
    contour = np.array(
        [[0, 0], [width - 1, 0], [width - 1, band - 1], [0, band - 1]], np.int32
    )
    return SegmentationResult(
        plane_masks={"floor": mask},
        foreground_mask=np.zeros((height, width), np.uint8),
        contours={"floor": contour},
        bounding_points={"floor": contour},
        area_fractions={"floor": fraction},
        backend_name="classical",
    )


def upload_probe(client: Any, width: int = 0, height: int = 0) -> Any:
    """POST a probe photograph to `/api/segment` as an accepted PNG."""
    probe_width, probe_height = PROBE_PHOTO_SIZE
    image = probe_photo(width or probe_width, height or probe_height)
    return client.post(
        "/api/segment",
        files=upload_part(encode_image(image, "png"), "room.png", "image/png"),
    )


# --------------------------------------------------------------------------- #
# Rejection-path fixtures and strategies
# --------------------------------------------------------------------------- #


@pytest.fixture
def capped_client(corpus_paths: tuple[Path, Path]) -> Iterator[Any]:
    """A client whose upload cap is :data:`LOWERED_UPLOAD_CAP`.

    Function-scoped and shared across the examples of the property that takes it,
    for the same reason `live_scene` is: the examples only issue requests, and
    rebuilding the app per example would cost more than every example combined.
    """
    assets_dir, weights_dir = corpus_paths
    with overridden_env(RV_MAX_UPLOAD_BYTES=str(LOWERED_UPLOAD_CAP)):
        with configured_client(assets_dir, weights_dir) as client:
            assert client.app.state.settings.max_upload_bytes == LOWERED_UPLOAD_CAP, (
                "the lowered upload cap did not reach the running application"
            )
            yield client


@pytest.fixture
def clamp_client(corpus_paths: tuple[Path, Path]) -> Iterator[Any]:
    """A client whose longest-edge limit is :data:`CLAMP_LIMIT`."""
    assets_dir, weights_dir = corpus_paths
    with overridden_env(RV_MAX_LONGEST_EDGE=str(CLAMP_LIMIT)):
        with configured_client(assets_dir, weights_dir) as client:
            assert client.app.state.settings.max_longest_edge == CLAMP_LIMIT, (
                "the lowered longest-edge limit did not reach the running application"
            )
            yield client


#: Payload sizes for Property 7. Three overlapping ranges rather than one: the
#: boundary itself, where an off-by-one lives; the small end, where an empty or
#: near-empty body could take a different path; and a wide sweep out to three
#: times the cap, so the property is not only a boundary test.
_capped_sizes = st.one_of(
    st.integers(min_value=LOWERED_UPLOAD_CAP - 16, max_value=LOWERED_UPLOAD_CAP + 16),
    st.integers(min_value=0, max_value=16),
    st.integers(min_value=0, max_value=3 * LOWERED_UPLOAD_CAP),
)

#: Scene identifiers no cache holds. Well-formed uuid4 hexes -- the shape a real
#: `scene_id` has, so the rejection cannot be an input-format quibble -- arbitrary
#: printable text, and a handful of values that probe for path traversal or a
#: JSON/None confusion.
_absent_scene_ids = st.one_of(
    st.uuids().map(lambda value: value.hex),
    st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=1,
        max_size=48,
    ),
    st.sampled_from(
        (
            "0",
            "-",
            " ",
            "null",
            "undefined",
            "None",
            "0" * 32,
            "../../etc/passwd",
            "%00",
            "' OR 1=1 --",
        )
    ),
)

#: Input geometries for Property 8. The long edge straddles :data:`CLAMP_LIMIT`
#: in both directions, so the identity case and the downscale case are both drawn,
#: and `short_frac` bottoms out at 0.30 -- see :data:`CLAMP_LIMIT` on why that
#: floor is where the 0.5 percent bound stops being achievable at integer sizes.
_clamp_long_edges = st.integers(min_value=240, max_value=900)
_clamp_short_fracs = st.floats(min_value=0.30, max_value=1.0, **_finite)

_REJECTION_PROPERTY_SETTINGS = hypothesis_settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        # The client is shared across examples on purpose; the examples only issue
        # requests against it.
        HealthCheck.function_scoped_fixture,
    ],
)


# --------------------------------------------------------------------------- #
# Property 6 -- upload validation gates all pipeline work
# (Requirements 2.1, 2.2, 2.3)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 6: Upload validation gates all
# pipeline work
@given(
    mime=st.sampled_from(MIME_CASES),
    name=st.sampled_from(FILENAME_CASES),
    payload=st.sampled_from(PAYLOAD_CASES),
)
@_REJECTION_PROPERTY_SETTINGS
def test_property_6_upload_validation_gates_all_pipeline_work(
    client: Any, mime: UploadVariant, name: UploadVariant, payload: UploadVariant
) -> None:
    """Property 6: 415 exactly when a check fails, and no stage runs when it does.

    Eleven MIME headers against eleven filenames against nine bodies -- 1089
    combinations, sampled a hundred times -- and the expected verdict is the
    conjunction of the three declared allow-list facts. Both directions are
    asserted, because they fail differently: a validator that let a renamed
    executable through is a security hole, and one that rejected a legitimate
    JPEG is a shopper who cannot use the product at all.

    The second half of the property is the part a status code cannot show. Being
    told "415" says nothing about whether the service segmented the file first and
    threw the result away, and the whole point of Requirement 2.3 is that hostile
    input never reaches a pipeline stage. So every request runs inside
    :func:`counted_analysis_stages`, and a 415 is required to leave all four
    counters at zero.

    The accepted branch asserts `segment == 1` rather than only a non-415 status,
    which is what stops "all four counters are zero" from being vacuously true of
    a build where the counters were wired to nothing.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    data = probe_payloads()[payload.key]
    acceptable = mime.allowed and name.allowed and payload.allowed
    where = f"mime={mime.key!r} file={name.key!r} body={payload.key!r}"

    with counted_analysis_stages() as counters:
        response = client.post("/api/segment", files=upload_part(data, name.key, mime.key))

    event(f"acceptable={acceptable}")

    if acceptable:
        assert response.status_code != 415, (
            f"{where}: rejected as unsupported although the MIME type and extension "
            f"are allowed and the bytes decode ({error_code(response)!r})"
        )
        assert counters.segment == 1, (
            f"{where}: validation passed but segmentation ran {counters.segment} "
            "times, so the counters this property reads are not on the live path"
        )
        return

    assert response.status_code == 415, (
        f"{where}: returned {response.status_code} {error_code(response)!r}, but one "
        "of the three upload checks should have rejected it"
    )
    assert_error_envelope_body(response.json(), "unsupported_media_type", where)
    assert counters.as_dict() == NO_STAGES_ENTERED, (
        f"{where}: rejected with 415 after entering {counters.as_dict()!r}; "
        "Requirement 2.3 allows no pipeline processing at all"
    )


def test_the_upload_variant_tables_match_the_configured_allow_lists() -> None:
    """Guard for Property 6: its expected verdicts describe the live allow-lists.

    :data:`MIME_CASES` and :data:`FILENAME_CASES` carry hand-written verdicts, so
    they are only meaningful while the allow-lists they were written against are
    the ones in force. Widening `allowed_mime_types` without revisiting the tables
    would turn a genuine regression into a property that quietly asserts the wrong
    thing, which is the one failure mode a table-driven property has.
    """
    settings = default_settings()
    assert settings.allowed_mime_types == EXPECTED_ALLOWED_MIME_TYPES, (
        f"allowed_mime_types is {settings.allowed_mime_types!r}; MIME_CASES was "
        f"written against {EXPECTED_ALLOWED_MIME_TYPES!r}"
    )
    assert settings.allowed_extensions == EXPECTED_ALLOWED_EXTENSIONS, (
        f"allowed_extensions is {settings.allowed_extensions!r}; FILENAME_CASES was "
        f"written against {EXPECTED_ALLOWED_EXTENSIONS!r}"
    )

    # Each declared verdict, re-derived from the allow-lists rather than restated.
    for case in MIME_CASES:
        normalised = case.key.split(";", 1)[0].strip().lower()
        assert case.allowed == (normalised in settings.allowed_mime_types), (
            f"MIME_CASES declares {case!r}, which the allow-list disagrees with"
        )
    for case in FILENAME_CASES:
        suffix = Path(case.key).suffix.lower()
        assert case.allowed == (suffix in settings.allowed_extensions), (
            f"FILENAME_CASES declares {case!r}, which the allow-list disagrees with"
        )


def test_the_payload_variant_table_matches_what_opencv_can_decode() -> None:
    """Guard for Property 6: its declared decodability is OpenCV's, not a guess.

    A truncation that a future OpenCV learned to salvage, or a "valid" encode that
    stopped round-tripping, would silently invert one axis of the property's
    expected verdict. Asking OpenCV directly -- not the service's own
    `decode_image` -- keeps the oracle independent of the code under test.
    """
    payloads = probe_payloads()
    assert set(payloads) == {case.key for case in PAYLOAD_CASES}, (
        "PAYLOAD_CASES and probe_payloads() name different bodies"
    )
    for case in PAYLOAD_CASES:
        assert decodes_as_raster(payloads[case.key]) == case.allowed, (
            f"PAYLOAD_CASES declares {case!r}, but OpenCV disagrees"
        )


def test_the_probe_photograph_is_accepted(client: Any) -> None:
    """Guard for the accepted branch of Properties 6 and 7.

    Both properties assert something about uploads that pass validation, and both
    reach that branch only because :func:`probe_photo` yields a tileable plane. If
    it stopped doing so they would still pass -- a 422 is a non-415 status and is
    not a 413 -- while covering nothing. Pinning the acceptance here means a
    segmenter change fails once, loudly, in the place that explains it.
    """
    response = upload_probe(client)

    assert response.status_code == 200, (
        f"the probe photograph was rejected: {response.status_code} "
        f"{error_code(response)!r}; the accepted branch of Properties 6 and 7 would "
        "no longer be exercised"
    )
    body = response.json()
    assert body["planes"], "the probe photograph yielded no Structural_Plane"
    assert (body["width"], body["height"]) == PROBE_PHOTO_SIZE


# --------------------------------------------------------------------------- #
# Property 7 -- oversized uploads are rejected at the configured threshold
# (Requirements 2.4, 2.5)
# --------------------------------------------------------------------------- #


def sized_payload(size: int) -> tuple[bytes, bool]:
    """A body of exactly `size` bytes, plus whether it decodes as an image.

    Above the probe PNG's length the body is that PNG followed by padding, which
    libpng stops reading at `IEND`, so it stays a genuine decodable photograph at
    an arbitrary size. Below it, no real raster fits and the body is zeros.

    Carrying both a real image and a non-image across the threshold is what makes
    the sub-cap half of the property say something: a decodable body under the cap
    must be *accepted*, not merely "not 413".
    """
    png = probe_payloads()["png"]
    if size < len(png):
        return bytes(size), False
    return png + bytes(size - len(png)), True


# Feature: ai-room-tile-visualizer, Property 7: Oversized uploads are rejected at
# the configured threshold
@given(size=_capped_sizes)
@_REJECTION_PROPERTY_SETTINGS
def test_property_7_oversized_uploads_are_rejected_at_the_threshold(
    capped_client: Any, size: int
) -> None:
    """Property 7: 413 `payload_too_large` exactly when the body exceeds the cap.

    Run against a client configured down to :data:`LOWERED_UPLOAD_CAP`, so the
    threshold is crossed with kilobytes. The cap is deployment configuration, and
    a limit that only held at its default would be no limit at all.

    Three claims, and the two beyond the status code are where the interesting
    failures are. An over-cap upload must enter no analysis stage -- the point of
    the streamed check is that an oversized body is never even fully
    materialised, let alone segmented. And an under-cap body must not be rejected
    for size, whatever else happens to it: a real photograph under the cap comes
    back 200, and a same-sized block of zeros comes back 415, which together show
    the size gate ran first and then got out of the way.

    **Validates: Requirements 2.4, 2.5**
    """
    payload, decodable = sized_payload(size)
    assert len(payload) == size
    oversized = size > LOWERED_UPLOAD_CAP
    where = f"size={size} cap={LOWERED_UPLOAD_CAP} decodable={decodable}"

    with counted_analysis_stages() as counters:
        response = capped_client.post(
            "/api/segment", files=upload_part(payload, "room.png", "image/png")
        )

    event(f"oversized={oversized}")

    if oversized:
        assert response.status_code == 413, (
            f"{where}: returned {response.status_code} {error_code(response)!r} for a "
            "payload over the configured cap"
        )
        assert_error_envelope_body(response.json(), "payload_too_large", where)
        assert counters.as_dict() == NO_STAGES_ENTERED, (
            f"{where}: an oversized upload entered {counters.as_dict()!r}"
        )
        return

    assert response.status_code != 413, (
        f"{where}: rejected as too large although it is within the cap"
    )
    assert error_code(response) != "payload_too_large", (
        f"{where}: reported payload_too_large at {response.status_code}"
    )
    if decodable:
        assert response.status_code == 200, (
            f"{where}: a decodable photograph inside the cap returned "
            f"{response.status_code} {error_code(response)!r}"
        )
    else:
        assert response.status_code == 415, (
            f"{where}: undecodable bytes inside the cap returned "
            f"{response.status_code} {error_code(response)!r}"
        )


def test_an_oversized_upload_is_aborted_across_chunk_boundaries(
    corpus_paths: tuple[Path, Path],
) -> None:
    """Requirement 2.5 for a payload spanning several reads of the upload stream.

    :data:`LOWERED_UPLOAD_CAP` is smaller than `UPLOAD_CHUNK_BYTES`, so every
    example of Property 7 crosses the threshold inside the very first chunk. The
    accumulate-and-compare loop is the part that has to keep holding once the body
    outruns a single read, and it is the part that stops a hostile client forcing a
    multi-gigabyte buffer. This drives a body several chunks long against a cap
    that sits partway through it.
    """
    cap = 2 * app_module.UPLOAD_CHUNK_BYTES
    assets_dir, weights_dir = corpus_paths
    with overridden_env(RV_MAX_UPLOAD_BYTES=str(cap)):
        with configured_client(assets_dir, weights_dir) as client:
            payload = bytes(cap + app_module.UPLOAD_CHUNK_BYTES + 7)
            response = client.post(
                "/api/segment", files=upload_part(payload, "room.png", "image/png")
            )

    assert response.status_code == 413
    assert_error_envelope_body(response.json(), "payload_too_large", "multi-chunk")


# --------------------------------------------------------------------------- #
# Property 8 -- downscaling clamps the longest edge and preserves aspect ratio
# (Requirement 2.6)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 8: Downscaling clamps the longest
# edge and preserves aspect ratio
@given(
    long_edge=_clamp_long_edges,
    short_frac=_clamp_short_fracs,
    landscape=st.booleans(),
)
@_REJECTION_PROPERTY_SETTINGS
def test_property_8_downscaling_clamps_the_longest_edge_and_keeps_aspect(
    clamp_client: Any, long_edge: int, short_frac: float, landscape: bool
) -> None:
    """Property 8: the processed longest edge is capped and the shape is preserved.

    This is the allocation bound the whole memory budget of Requirement 12.1 rests
    on. Every downstream array -- masks, lighting maps, sample maps, the cached
    photograph itself -- is sized from the image this stage returns, so a clamp
    that missed would put an upload's declared dimensions in charge of how much
    memory the process uses.

    Observed at the clamp rather than in the response, through
    :func:`spied_downscale`, for two reasons. It covers input geometries analysis
    might reject, where a response carries no dimensions at all. And it lets
    :func:`stopped_after_downscale` end each example the moment the measurement is
    taken, which is what makes a hundred distinct geometries affordable --
    analysing them would cost twenty seconds to re-measure something already
    recorded. `test_the_reported_dimensions_are_the_processed_ones` closes the
    loop by running the real pipeline and requiring the response to report exactly
    what the clamp produced.

    Both halves are asserted. The cap is `min(limit, input)`, so an image already
    inside the limit must come through untouched -- upscaling a small photograph
    to the limit would waste the budget this exists to protect and invent detail
    that was never photographed. And the aspect ratio holds to 0.5 percent, which
    is what stops a clamped room from being subtly stretched before anything has
    even looked at it.

    **Validates: Requirements 2.6**
    """
    short_edge = max(1, int(round(long_edge * short_frac)))
    width, height = (long_edge, short_edge) if landscape else (short_edge, long_edge)
    payload = encode_image(probe_photo(width, height), "png")
    where = f"input={width}x{height} limit={CLAMP_LIMIT}"

    with spied_downscale() as processed, stopped_after_downscale():
        response = clamp_client.post(
            "/api/segment", files=upload_part(payload, "room.png", "image/png")
        )

    assert len(processed) == 1, (
        f"{where}: the downscale stage ran {len(processed)} times; the spy Property 8 "
        "reads is not on the live path"
    )
    # The short circuit landed where it was aimed, which is what proves the clamp
    # ran *before* segmentation rather than the property having measured a
    # fully analysed image by accident.
    assert response.status_code == 422 and error_code(response) == "analysis_failed", (
        f"{where}: expected the short-circuited stage to surface as analysis_failed, "
        f"got {response.status_code} {error_code(response)!r}"
    )

    out_height, out_width = processed[0]
    event(f"downscaled={max(width, height) > CLAMP_LIMIT}")

    assert max(out_height, out_width) == min(CLAMP_LIMIT, max(width, height)), (
        f"{where}: processed to {out_width}x{out_height}, whose longest edge is not "
        f"min({CLAMP_LIMIT}, {max(width, height)})"
    )
    assert min(out_height, out_width) >= 1, (
        f"{where}: processed to {out_width}x{out_height}, which has a collapsed edge"
    )

    in_ratio = width / height
    out_ratio = out_width / out_height
    assert abs(out_ratio - in_ratio) / in_ratio <= ASPECT_TOLERANCE, (
        f"{where}: aspect ratio moved from {in_ratio:.6f} to {out_ratio:.6f}, past "
        f"the {ASPECT_TOLERANCE:.1%} bound"
    )


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (900, 360, (512, 205)),
        (360, 900, (205, 512)),
        (700, 700, (512, 512)),
        (512, 512, (512, 512)),
        (400, 300, (400, 300)),
    ],
    ids=["landscape", "portrait", "square_over", "square_at_limit", "under_limit"],
)
def test_the_reported_dimensions_are_the_processed_ones(
    clamp_client: Any, width: int, height: int, expected: tuple[int, int]
) -> None:
    """Requirement 2.6 end to end: the response describes the clamped photograph.

    Property 8 watches the clamp with segmentation short-circuited, which leaves
    one thing unchecked: whether the dimensions the client is told about are the
    ones the service actually worked on. The frontend scales every contour and
    every pointer coordinate by `canvasWidth / sceneWidth`, so a response
    reporting the *uploaded* size after a downscale would put every plane overlay
    in the wrong place.

    Five shapes through the real pipeline: a landscape and a portrait that both
    need clamping, a square over the limit, a square exactly at it, and one under
    it that must come back untouched.
    """
    payload = encode_image(probe_photo(width, height), "png")
    response = clamp_client.post(
        "/api/segment", files=upload_part(payload, "room.png", "image/png")
    )

    assert response.status_code == 200, (
        f"{width}x{height}: {response.status_code} {error_code(response)!r}"
    )
    body = response.json()
    assert (body["width"], body["height"]) == expected, (
        f"{width}x{height} was reported as {body['width']}x{body['height']}, "
        f"expected {expected[0]}x{expected[1]}"
    )
    assert max(body["width"], body["height"]) == min(CLAMP_LIMIT, max(width, height))

    in_ratio = width / height
    out_ratio = body["width"] / body["height"]
    assert abs(out_ratio - in_ratio) / in_ratio <= ASPECT_TOLERANCE

    # Every cached artifact is sized from the processed image, so the reported
    # dimensions and the arrays the render pass will read must agree.
    scene = clamp_client.app.state.cache.get(body["scene_id"])
    assert scene is not None
    assert scene.foreground_mask.shape == (body["height"], body["width"])


# --------------------------------------------------------------------------- #
# Property 29 -- unknown scene identifiers are rejected consistently
# (Requirement 9.4)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 29: Unknown scene identifiers are
# rejected consistently
@given(
    scene_id=_absent_scene_ids,
    planes=_plane_selection,
    spec=_plane_spec,
    bogus_tile=st.booleans(),
    binary=st.booleans(),
)
@_REJECTION_PROPERTY_SETTINGS
def test_property_29_unknown_scene_identifiers_are_rejected_consistently(
    live_scene: LiveScene,
    scene_id: str,
    planes: list[str],
    spec: dict[str, Any],
    bogus_tile: bool,
    binary: bool,
) -> None:
    """Property 29: any absent `scene_id` is 404 `scene_expired`, and only that.

    "Consistently" is the load-bearing word, and it is why the drawn space is
    wider than the identifier. The frontend's entire recovery from an evicted
    scene hangs off this one code: it reads `scene_expired` and asks the shopper to
    upload the photo again. A request that reported `unknown_plane` or
    `unknown_tile` instead -- because resolution happened in a different order --
    would leave the widget telling a shopper to pick a different product when
    their room is what went missing.

    So each example also draws plane selections, tile identifiers that may not
    exist, and both response encodings. The cache lookup has to win against all of
    them. The identifiers themselves run from well-formed uuid4 hexes, which look
    exactly like the real thing, to path-traversal and injection shapes, so the
    rejection cannot be an input-format quibble that a plausible id would slip
    past.

    The run is against a client holding one *live* scene, which rules out the
    degenerate reading where every render 404s because the cache is empty.

    **Validates: Requirements 9.4**
    """
    assume(scene_id != live_scene.scene_id)
    if bogus_tile:
        spec = {**spec, "tile_id": "definitely-not-a-tile-in-this-catalog"}

    payload = {
        "scene_id": scene_id,
        "planes": {name: dict(spec) for name in planes},
    }
    response = live_scene.render(payload, binary=binary)
    where = f"scene_id={scene_id!r} planes={planes!r} bogus_tile={bogus_tile}"

    assert response.status_code == 404, (
        f"{where}: returned {response.status_code} {error_code(response)!r} for a "
        "scene the cache does not hold"
    )
    # The envelope arrives even on the `?binary=1` path: the failure is raised
    # before any image is produced, so there is nothing to send as image bytes.
    assert_error_envelope_body(response.json(), "scene_expired", where)
    event(f"planes={len(planes)}, bogus_tile={bogus_tile}, binary={binary}")


def test_the_live_scene_still_renders_alongside_the_rejected_ids(
    live_scene: LiveScene,
) -> None:
    """Guard for Property 29: the 404s are about the id, not about the service.

    A render route that had stopped working entirely would satisfy Property 29 for
    every drawn identifier. This asserts the same client, in the same
    configuration, still serves the one `scene_id` it does hold.
    """
    response = live_scene.render(
        {
            "scene_id": live_scene.scene_id,
            "planes": {live_scene.planes[0]: {"tile_id": CATALOG_TILE_IDS[0]}},
        }
    )

    assert response.status_code == 200, (
        f"the live scene itself failed to render: {response.status_code} "
        f"{error_code(response)!r}"
    )


# --------------------------------------------------------------------------- #
# Documented rejection codes -- 422 no_usable_plane, 422 analysis_failed
# (Requirements 1.6, 6.5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "build",
    [no_planes_at_all, only_a_sub_threshold_plane],
    ids=["no_plane_detected", "only_a_sub_threshold_sliver"],
)
def test_a_photograph_with_nothing_tileable_is_422_no_usable_plane(
    client: Any, build: Callable[[int, int], SegmentationResult]
) -> None:
    """Requirement 6.5, at both conditions that trigger it.

    The condition is supplied through the Segmenter rather than through a
    photograph, and that is a considered choice: the classical backend finds a
    plane above the 2 percent floor in almost anything, including a uniformly flat
    frame, so there is no image that reliably reaches this path. Substituting the
    segmentation output keeps the route, the validation, the clamp, and the
    threshold comparison real while making the branch reachable at all.

    Two conditions, because they are different bugs. `no_plane_detected` is the
    emptiness check; `only_a_sub_threshold_sliver` is the 2 percent *threshold*
    the requirement actually names, and a service that only checked for emptiness
    would happily hand a shopper a sliver nobody can tile.

    Analysis must also stop at the rejection: calibration and lighting are the
    expensive stages, and running them for a photograph already known to have
    nothing tileable is work no one asked for.
    """
    with segmentation_yielding(build), counted_analysis_stages() as counters:
        response = upload_probe(client)

    assert response.status_code == 422, (
        f"expected 422, got {response.status_code} {error_code(response)!r}"
    )
    assert_error_envelope_body(response.json(), "no_usable_plane", build.__name__)
    assert counters.segment == 1, "segmentation should have run exactly once"
    assert (counters.calibrate, counters.decompose) == (0, 0), (
        "calibration and lighting ran for a photograph with no usable plane: "
        f"{counters.as_dict()!r}"
    )


def test_a_plane_without_geometry_is_422_no_usable_plane(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement 6.5's second gate: a mask alone does not make a plane tileable.

    A Structural_Plane needs both a mask and a Plane_Homography to be rendered, so
    the route reports only their intersection and rejects the photograph when that
    intersection is empty. Without this gate a plane the Geometry_Engine could not
    calibrate would still be advertised to the frontend as selectable, and picking
    it would fail at render time -- after the shopper had chosen a product.
    """
    real_calibrate = app_module.calibrate

    def calibrate_without_homographies(*args: Any, **kwargs: Any) -> Any:
        return replace(
            real_calibrate(*args, **kwargs),
            homographies={},
            homography_inverses={},
        )

    monkeypatch.setattr(app_module, "calibrate", calibrate_without_homographies)

    response = upload_probe(client)

    assert response.status_code == 422
    assert_error_envelope_body(response.json(), "no_usable_plane", "no_homography")


@pytest.mark.parametrize(
    "stage",
    ["downscale", "segmentation", "calibration", "lighting"],
)
def test_a_failing_analysis_stage_is_422_analysis_failed(
    client: Any, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """Requirement 1.6: any stage failure is one 422 naming the stage.

    All four stages, because the value of a single error code is that it covers
    every one of them: a stage that escaped the wrapper would surface as an
    unhandled 500 `internal_error`, and the frontend would tell a shopper
    something went wrong on the server rather than that this photograph could not
    be analysed.

    The stage name is asserted inside the message. It is the only thing that makes
    a support report actionable -- "analysis_failed" alone does not distinguish a
    photograph with no straight edges from a decode that produced a strange
    array -- and it is why the message is built from the stage rather than being a
    constant.
    """

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"deliberate {stage} failure")

    if stage == "segmentation":
        monkeypatch.setattr(ClassicalSegmenter, "segment", boom)
    else:
        target = {
            "downscale": "clamp_longest_edge",
            "calibration": "calibrate",
            "lighting": "decompose",
        }[stage]
        monkeypatch.setattr(app_module, target, boom)

    response = upload_probe(client)

    assert response.status_code == 422, (
        f"{stage}: returned {response.status_code} {error_code(response)!r}"
    )
    assert_error_envelope_body(response.json(), "analysis_failed", stage)
    message = response.json()["error"]["message"]
    assert stage in message, (
        f"{stage}: the message {message!r} does not name the stage that failed"
    )


# --------------------------------------------------------------------------- #
# Documented rejection codes -- 422 unknown_tile / unknown_plane, 500 render_failed
# (Requirements 1.6, 9.1)
# --------------------------------------------------------------------------- #


def test_an_unlisted_tile_is_422_unknown_tile(live_scene: LiveScene) -> None:
    """Requirement 9.1: a `tile_id` the Asset_Catalog does not publish is rejected.

    Not a 500, and not a silent fallback to some default product. A merchandiser
    who removes a discontinued tile while a shopper has its page open is the
    ordinary way this happens, and the distinct code is what lets the widget say
    "pick another product" rather than "something went wrong".
    """
    missing = "definitely-not-a-tile-in-this-catalog"
    response = live_scene.render(
        {
            "scene_id": live_scene.scene_id,
            "planes": {live_scene.planes[0]: {"tile_id": missing}},
        }
    )

    assert response.status_code == 422, (
        f"returned {response.status_code} {error_code(response)!r}"
    )
    assert_error_envelope_body(response.json(), "unknown_tile", "unlisted tile")
    assert missing in response.json()["error"]["message"], (
        "the message does not name the tile that could not be found"
    )


def test_a_plane_this_photograph_lacks_is_422_unknown_plane(client: Any) -> None:
    """Requirement 9.1: a valid plane name the *scene* does not have is rejected.

    The distinction is easy to lose. An arbitrary string is not a plane name at
    all and never reaches the route -- the request schema restricts the keys to
    the four Structural_Plane names, so `"ceiling"` is a schema violation. What
    `unknown_plane` reports is the other case: a perfectly valid name for a
    surface this particular photograph does not contain, which is what a frontend
    holding a stale plane list from an earlier upload would send.

    The scene here is a back-wall-only room, so three of the four names are
    genuinely absent, and each is required to produce the same code.
    """
    spec = UPLOAD_SPECS[6]  # back_only_webp -- one wall, no floor to speak of
    response = client.post(
        "/api/segment",
        files=upload_part(
            encode_image(spec.render(), spec.fmt), spec.filename, spec.mime
        ),
    )
    assert response.status_code == 200, (
        f"the single-wall room was rejected: {error_code(response)!r}"
    )
    body = response.json()
    scene_id = str(body["scene_id"])
    detected = {str(plane["name"]) for plane in body["planes"]}
    absent = [name for name in PLANE_NAMES if name not in detected]
    assert absent, (
        f"{spec.label} yielded every Structural_Plane ({sorted(detected)!r}), so this "
        "test has no absent plane to ask for"
    )

    for name in absent:
        rejected = client.post(
            "/api/render",
            json={
                "scene_id": scene_id,
                "planes": {name: {"tile_id": CATALOG_TILE_IDS[0]}},
            },
        )
        assert rejected.status_code == 422, (
            f"{name}: returned {rejected.status_code} {error_code(rejected)!r}"
        )
        assert_error_envelope_body(rejected.json(), "unknown_plane", name)

    # And the plane the photograph does have still renders, so the rejection is
    # about the named plane rather than about the scene.
    accepted = client.post(
        "/api/render",
        json={
            "scene_id": scene_id,
            "planes": {sorted(detected)[0]: {"tile_id": CATALOG_TILE_IDS[0]}},
        },
    )
    assert accepted.status_code == 200, (
        f"the detected plane failed to render: {error_code(accepted)!r}"
    )


def test_a_plane_name_outside_the_schema_is_422_invalid_request(
    live_scene: LiveScene,
) -> None:
    """The other side of that distinction, pinned so the two cannot merge.

    `"ceiling"` is not a Structural_Plane, so it is a malformed request rather
    than a plane the scene lacks, and it is rejected by the schema before
    `post_render` runs. Both codes are 422 and both come through the same
    envelope, so nothing but a test keeps them from quietly collapsing into one --
    and they mean different things to a client: one is a bug in the caller, the
    other is a stale plane list.
    """
    response = live_scene.render(
        {
            "scene_id": live_scene.scene_id,
            "planes": {"ceiling": {"tile_id": CATALOG_TILE_IDS[0]}},
        }
    )

    assert response.status_code == 422
    assert_error_envelope_body(response.json(), "invalid_request", "ceiling")


@pytest.mark.parametrize("target", ["compose", "encode_render"])
def test_a_compositing_failure_is_500_render_failed(
    live_scene: LiveScene, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """Requirement 1.6 on the render pass: a fault there is a 500, not a 422.

    The status carries real information. Everything the caller controls has
    already been resolved by this point -- the scene exists, the planes exist, the
    tiles exist -- so a failure in the warp or the encode is the service's
    problem, and telling the shopper to change their request would be wrong. Both
    steps inside the guard are covered, since either can raise and both must land
    on the same code.
    """

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"deliberate {target} failure")

    monkeypatch.setattr(app_module, target, boom)

    response = live_scene.render(
        {
            "scene_id": live_scene.scene_id,
            "planes": {live_scene.planes[0]: {"tile_id": CATALOG_TILE_IDS[0]}},
        }
    )

    assert response.status_code == 500, (
        f"{target}: returned {response.status_code} {error_code(response)!r}"
    )
    assert_error_envelope_body(response.json(), "render_failed", target)


# --------------------------------------------------------------------------- #
# GET /api/health and GET /api/tiles response shapes
# (Requirements 8.4, 12.5)
# --------------------------------------------------------------------------- #


def test_health_reports_the_documented_runtime_facts(client: Any) -> None:
    """Requirement 12.5: the three facts an operator needs, and the live values.

    Field presence is the easy half. The half that matters is that
    `scene_cache_entries` reports the *live* occupancy rather than a constant: it
    is the only external signal that scenes are accumulating toward the eviction
    bound, so an operator sizing `RV_SCENE_CACHE_MAX_ENTRIES` reads it directly.
    So the count is checked before and after an upload, and against the cache
    itself.
    """
    body = client.get("/api/health").json()

    assert set(body) == {
        "status",
        "segmentation_backend",
        "onnx_provider",
        "scene_cache_entries",
        "scene_cache_max_entries",
        "scene_cache_ttl_seconds",
    }, f"/api/health returned {sorted(body)!r}"
    assert body["status"] == "ok"
    assert body["segmentation_backend"] in {"classical", "mobilesam-onnx"}
    assert isinstance(body["onnx_provider"], str) and body["onnx_provider"]
    assert body["scene_cache_max_entries"] > 0
    assert body["scene_cache_ttl_seconds"] > 0

    settings = client.app.state.settings
    assert body["scene_cache_max_entries"] == settings.scene_cache_max_entries
    assert body["scene_cache_ttl_seconds"] == settings.scene_cache_ttl_seconds

    before = body["scene_cache_entries"]
    assert before == len(client.app.state.cache)

    assert upload_probe(client).status_code == 200
    after = client.get("/api/health").json()["scene_cache_entries"]

    assert after == before + 1, (
        f"scene_cache_entries went {before} -> {after} across one accepted upload"
    )
    assert after == len(client.app.state.cache)


def test_tiles_publishes_the_documented_catalog_shape(client: Any) -> None:
    """Requirement 8.4: every valid Tile_Definition, with the metadata R8.3 needs.

    The frontend renders a swatch per entry and labels it from these fields, so
    each one has to be present and usable: millimetre dimensions to state the
    size, a finish label to name it, a gloss value the Compositor scales
    highlights by, and a thumbnail URL that actually resolves. The last is worth
    fetching rather than pattern-matching -- a plausible-looking URL that 404s is
    a catalog of broken images, which is what a mismatch between the manifest's
    `file` and the `/assets` mount would produce.
    """
    response = client.get("/api/tiles")
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"tiles"}, f"/api/tiles returned {sorted(body)!r}"
    ids = [str(tile["id"]) for tile in body["tiles"]]
    assert len(ids) == len(set(ids)), f"a tile id repeats: {ids!r}"
    assert set(ids) == set(CATALOG_TILE_IDS), (
        f"published {sorted(ids)!r}, expected {sorted(CATALOG_TILE_IDS)!r}"
    )

    declared = {str(spec["id"]): spec for spec in TINY_CATALOG_TILES}
    for tile in body["tiles"]:
        where = f"tile {tile['id']!r}"
        assert set(tile) == {
            "id",
            "name",
            "width_mm",
            "height_mm",
            "finish",
            "gloss",
            "thumbnail_url",
        }, f"{where}: fields {sorted(tile)!r}"

        source = declared[str(tile["id"])]
        assert tile["name"] == source["name"], f"{where}: name"
        assert tile["width_mm"] == pytest.approx(source["width_mm"]), f"{where}: width"
        assert tile["height_mm"] == pytest.approx(source["height_mm"]), f"{where}: height"
        assert tile["finish"] == source["finish"], f"{where}: finish"
        assert tile["gloss"] == pytest.approx(source["gloss"]), f"{where}: gloss"
        assert tile["width_mm"] > 0 and tile["height_mm"] > 0, f"{where}: dimensions"
        assert str(tile["finish"]).strip(), f"{where}: empty finish label"
        assert 0.0 <= float(tile["gloss"]) <= 1.0, f"{where}: gloss out of range"

        url = str(tile["thumbnail_url"])
        assert url.startswith(f"{app_module.ASSETS_URL_PREFIX}/tiles/"), (
            f"{where}: thumbnail_url {url!r} is not under the assets mount"
        )
        thumbnail = client.get(url)
        assert thumbnail.status_code == 200, (
            f"{where}: thumbnail_url {url!r} returned {thumbnail.status_code}"
        )
        assert thumbnail.headers["content-type"].startswith("image/"), (
            f"{where}: thumbnail served as {thumbnail.headers['content-type']!r}"
        )
