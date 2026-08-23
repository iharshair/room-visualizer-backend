"""Shared pytest fixtures for the AI Room & Tile Visualizer test suite.

Everything the suite needs that is expensive, environment-touching, or shared
across modules lives here:

* :func:`no_network` -- autouse, so *every* test in the suite is provably
  offline rather than incidentally offline (Requirement 13.1).
* :func:`empty_weights_dir` -- points ``RV_WEIGHTS_DIR`` at an empty temp
  directory, which is the "weights absent" precondition of the Model_Loader
  fallback path (Requirement 4.5).
* :func:`tiny_catalog` -- a temp assets dir carrying one tile per metric format
  the Compositor must honour: 1:1, 1:2, and plank (Requirements 8.6, 8.7).
* :func:`client` -- a ``TestClient`` with the neural backend disabled and the
  catalog pointed at ``tiny_catalog``, so no HTTP test needs weights or a
  network (Requirement 4.7). The FastAPI app is imported *inside the fixture
  body*, which is what lets this module be collected before ``backend/app.py``
  exists.
* :func:`synthetic_room` / :func:`randomized_room` -- the analytic room fixture
  at fixed and Hypothesis-driven parameters (Requirement 13.2).
"""

from __future__ import annotations

import dataclasses
import json
import socket
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import cv2
import httpx
import numpy as np
import pytest

from backend.config import get_settings
from tests.fixtures.synthetic import SyntheticRoom, make_synthetic_room

__all__ = [
    "OfflineNetworkError",
    "TINY_CATALOG_TILES",
    "no_network",
    "empty_weights_dir",
    "tiny_catalog",
    "client",
    "synthetic_room",
    "randomized_room",
]


# --------------------------------------------------------------------------- #
# Offline guard
# --------------------------------------------------------------------------- #


class OfflineNetworkError(RuntimeError):
    """Raised when a test attempts to reach the network.

    Tests assert on this type when they need to prove a code path degrades
    instead of dialling out -- the Model_Loader fallback in particular.
    """


# Transports that actually open sockets. Starlette's ``TestClient`` is itself an
# ``httpx.Client`` subclass driving the ASGI app in-process, so its requests
# never leave the interpreter and must be let through; anything carrying a real
# transport is blocked.
_NETWORK_TRANSPORTS: tuple[type, ...] = (httpx.HTTPTransport, httpx.AsyncHTTPTransport)

_real_httpx_send = httpx.Client.send
_real_socket_connect = socket.socket.connect


def _uses_network_transport(client: httpx.Client) -> bool:
    """True when ``client`` would dispatch through a socket-backed transport."""
    candidates: list[Any] = [getattr(client, "_transport", None)]
    candidates.extend(getattr(client, "_mounts", {}).values())
    return any(isinstance(candidate, _NETWORK_TRANSPORTS) for candidate in candidates)


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``socket.socket.connect`` and ``httpx.Client.send`` to raise.

    Two layers rather than one: the socket patch catches every library that
    ultimately dials out (``urllib``, ``requests``, ``onnxruntime`` download
    helpers), while the httpx patch produces a readable failure naming the URL
    for the one client the service actually uses (``backend/utils/model_loader``
    streams weights through ``httpx.Client``).
    """

    def _blocked_connect(self: socket.socket, address: Any, *args: Any, **kwargs: Any):
        raise OfflineNetworkError(
            f"network access is disabled in tests; socket connect to {address!r} blocked"
        )

    def _blocked_send(self: httpx.Client, request: httpx.Request, **kwargs: Any):
        if _uses_network_transport(self):
            raise OfflineNetworkError(
                f"network access is disabled in tests; {request.method} {request.url} blocked"
            )
        return _real_httpx_send(self, request, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(httpx.Client, "send", _blocked_send)


# --------------------------------------------------------------------------- #
# Settings-touching fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def empty_weights_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An existing but empty weights directory, exported as ``RV_WEIGHTS_DIR``.

    The settings cache is cleared on both sides of the yield so neither this
    test nor the next one sees a stale ``weights_dir``.
    """
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    monkeypatch.setenv("RV_WEIGHTS_DIR", str(weights_dir))
    get_settings.cache_clear()
    try:
        yield weights_dir
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Tile catalog
# --------------------------------------------------------------------------- #

#: One entry per metric format the Compositor must reproduce: a square 600x600,
#: a 1:2 600x1200, and a 200x1200 plank. ``size_px`` deliberately mirrors each
#: entry's millimetre ratio so a test can tell a scaling bug from a source-image
#: aspect bug. ``base_bgr`` and ``seed`` only make the images distinguishable.
TINY_CATALOG_TILES: tuple[dict[str, Any], ...] = (
    {
        "id": "tiny-marble-600",
        "name": "Tiny Marble 600x600",
        "file": "tiny_marble_600x600.png",
        "width_mm": 600.0,
        "height_mm": 600.0,
        "finish": "polished",
        "gloss": 0.85,
        "size_px": (96, 96),
        "base_bgr": (238, 240, 242),
        "seed": 1,
    },
    {
        "id": "tiny-concrete-600x1200",
        "name": "Tiny Concrete 600x1200",
        "file": "tiny_concrete_600x1200.png",
        "width_mm": 600.0,
        "height_mm": 1200.0,
        "finish": "matte",
        "gloss": 0.10,
        "size_px": (96, 192),
        "base_bgr": (150, 152, 154),
        "seed": 2,
    },
    {
        "id": "tiny-wood-200x1200",
        "name": "Tiny Wood Plank 200x1200",
        "file": "tiny_wood_200x1200.png",
        "width_mm": 200.0,
        "height_mm": 1200.0,
        "finish": "satin",
        "gloss": 0.35,
        "size_px": (32, 192),
        "base_bgr": (92, 126, 168),
        "seed": 3,
    },
)


def _generate_tile_image(
    size_px: tuple[int, int], base_bgr: tuple[int, int, int], seed: int
) -> np.ndarray:
    """A cheap, deterministic, decodable tile raster.

    Deliberately *not* the Texture_Helper generators: this fixture must stay
    importable and fast regardless of that module's state, and the catalog only
    requires an image that decodes. A coarse two-tone grid plus fixed-seed noise
    gives every tile visible structure and a distinct mean, which is enough for
    the catalog and API tests that consume it.
    """
    width_px, height_px = size_px
    rng = np.random.default_rng(seed)
    image = np.empty((height_px, width_px, 3), dtype=np.uint8)
    image[:, :] = np.asarray(base_bgr, dtype=np.uint8)

    cell = max(8, min(width_px, height_px) // 4)
    ys, xs = np.mgrid[0:height_px, 0:width_px]
    darker = ((ys // cell) + (xs // cell)) % 2 == 1
    image[darker] = np.clip(np.asarray(base_bgr, dtype=np.int16) - 24, 0, 255).astype(np.uint8)

    noise = rng.normal(0.0, 3.0, size=image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def tiny_catalog(tmp_path: Path) -> Path:
    """Build a temp assets dir with three tiles plus ``tiles/manifest.json``.

    Returns the *assets* directory, which is what ``Settings.assets_dir``
    expects; the tiles and manifest live under ``<returned>/tiles/`` exactly as
    they do in the shipped tree. Function-scoped on purpose: the hot-reload and
    invalid-entry tests rewrite the manifest, and each must start from a clean
    copy.
    """
    assets_dir = tmp_path / "assets"
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir(parents=True)

    entries: list[dict[str, Any]] = []
    for spec in TINY_CATALOG_TILES:
        image = _generate_tile_image(spec["size_px"], spec["base_bgr"], spec["seed"])
        if not cv2.imwrite(str(tiles_dir / spec["file"]), image):  # pragma: no cover
            raise RuntimeError(f"failed to write fixture tile {spec['file']!r}")
        entries.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "file": spec["file"],
                "width_mm": spec["width_mm"],
                "height_mm": spec["height_mm"],
                "finish": spec["finish"],
                "gloss": spec["gloss"],
            }
        )

    manifest_path = tiles_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "tiles": entries}, indent=2), encoding="utf-8"
    )
    return assets_dir


# --------------------------------------------------------------------------- #
# HTTP client
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(
    tiny_catalog: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Any]:
    """A ``TestClient`` over the Visualizer_API with the neural backend off.

    ``enable_neural_backend=False`` means ``build_segmenter`` returns the
    classical backend without ever consulting the Model_Loader, so no HTTP test
    needs weights on disk or a reachable mirror (Requirement 4.7). The
    environment is set and the settings cache cleared *before* the app import so
    the module-level settings read and the ``lifespan`` startup both observe the
    test configuration.

    Entering the ``TestClient`` context manager is what runs ``lifespan``, so
    ``app.state.cache``, ``app.state.catalog``, and ``app.state.segmenter`` are
    populated for the duration of the test and the cache is cleared afterwards.
    """
    monkeypatch.setenv("RV_ENABLE_NEURAL_BACKEND", "false")
    monkeypatch.setenv("RV_ASSETS_DIR", str(tiny_catalog))
    monkeypatch.setenv("RV_WEIGHTS_DIR", str(tmp_path / "weights"))
    get_settings.cache_clear()

    # Imported here, not at module scope: the harness must stay importable
    # before backend/app.py exists so the early test tasks can run.
    from fastapi.testclient import TestClient

    from backend.app import app

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Synthetic room
# --------------------------------------------------------------------------- #


def _copy_room(room: SyntheticRoom) -> SyntheticRoom:
    """Return a room whose arrays are independent of ``room``'s.

    ``SyntheticRoom`` is frozen but its arrays are not, so handing the same
    instance to two tests would let one test's in-place edit corrupt the other.
    Copying a 1600x1200 room costs microseconds; regenerating it costs ~320 ms.
    """
    return dataclasses.replace(
        room,
        image=room.image.copy(),
        occluder_mask=room.occluder_mask.copy(),
        plane_polygons={name: poly.copy() for name, poly in room.plane_polygons.items()},
        truth_homographies={name: H.copy() for name, H in room.truth_homographies.items()},
    )


@pytest.fixture(scope="session")
def _synthetic_room_master() -> SyntheticRoom:
    """Generated once per session; never handed to a test directly."""
    return make_synthetic_room()


@pytest.fixture
def synthetic_room(_synthetic_room_master: SyntheticRoom) -> SyntheticRoom:
    """The room at the documented default parameters, with analytic truth.

    1600x1200, focal 1400 px, yaw 8 deg, pitch -12 deg, 600 mm checkerboard,
    three walls, two occluders, seed 0 -- so a failure is reproducible from the
    call in this fixture alone.
    """
    return _copy_room(_synthetic_room_master)


@pytest.fixture(scope="session")
def randomized_room() -> Callable[..., SyntheticRoom]:
    """Factory turning Hypothesis-drawn camera parameters into a room.

    Session-scoped because the factory itself is stateless: a ``@given`` test can
    request it without tripping Hypothesis's ``function_scoped_fixture`` health
    check, and each call still returns a fresh room.

    Call it with any subset of ``focal_px``, ``yaw_deg``, ``pitch_deg``, and
    ``walls``; anything else :func:`make_synthetic_room` accepts can be passed
    through as a keyword.

    The defaults are tuned for property tests rather than for fidelity: 640x480
    at ``supersample=1`` with one occluder generates in roughly 20 ms, so 100
    examples cost about two seconds, where the 1600x1200 defaults would cost
    over thirty. ``focal_px`` defaults to ``0.875 * width``, the same
    focal-to-width ratio as the fixed fixture, so the drawn poses stay in the
    same field-of-view regime as the full-size room.
    """

    def _make(
        focal_px: float | None = None,
        yaw_deg: float = 8.0,
        pitch_deg: float = -12.0,
        walls: Iterable[str] = ("left", "right", "back"),
        *,
        width: int = 640,
        height: int = 480,
        n_occluders: int = 1,
        seed: int = 0,
        supersample: int = 1,
        **overrides: Any,
    ) -> SyntheticRoom:
        return make_synthetic_room(
            width=width,
            height=height,
            focal_px=0.875 * width if focal_px is None else float(focal_px),
            yaw_deg=float(yaw_deg),
            pitch_deg=float(pitch_deg),
            walls=tuple(walls),
            n_occluders=n_occluders,
            seed=seed,
            supersample=supersample,
            **overrides,
        )

    return _make
