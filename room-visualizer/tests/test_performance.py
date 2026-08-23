"""Timing and memory budgets for the Visualizer_API (Requirements 9.3, 12.1).

Both budgets in this module are excluded from the default ``pytest`` selection by
``addopts`` in ``pytest.ini``, and selected explicitly:

    pytest -m perf        # the render budget of Requirement 9.3
    pytest -m resource    # the memory bound of Requirement 12.1

They are excluded because a wall-clock assertion and a resident-set assertion are
the two things in this suite a busy host can turn red with no code having
changed. They are *not* skipped: this module carries the only verification of
either bound, so both are real assertions.

Every bound is a module constant sitting beside the value measured on the
development host, so a reader can see the headroom instead of guessing at it.
The measurements are from three consecutive runs on an 8-core CPU-only macOS
host with the neural backend disabled; medians varied by at most 1 ms between
runs, so the margins below are margin against *code* regression, not against
noise.

What is asserted, and what is only recorded:

* **Asserted** -- the per-plane-count endpoint median against
  :func:`render_budget_ms`, and peak RSS against
  :data:`PEAK_RSS_GUARD_BYTES`.
* **Recorded** -- the compose/encode split per plane count, and the per-render
  maximum. The split is recorded so a future budget failure localises to the
  warp-and-blend pass or to the encoder rather than to "render got slower". The
  maximum is recorded rather than asserted because it is the one genuinely noisy
  number here: an earlier round of runs showed a 266 ms outlier at three planes
  against a 141 ms median, which is the OS scheduling the process out, not the
  Compositor. Across the four runs the current figures come from, the per-plane
  maximum ran 2 to 8 ms above its own median.
"""

from __future__ import annotations

import json
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import psutil
import pytest

from backend.config import get_settings
from backend.core.compositor import compose, encode_render
from backend.schemas import PLANE_NAMES, PlaneName, PlaneRenderSpec
from backend.utils.imageio import encode_image
from backend.utils.texture_helper import (
    generate_concrete,
    generate_marble,
    generate_terrazzo,
    generate_wood_plank,
    make_seamless,
)
from tests.fixtures.synthetic import make_synthetic_room

__all__ = [
    "RENDER_BUDGET_FIXED_MS",
    "RENDER_BUDGET_PER_PLANE_MS",
    "MEASURED_ENDPOINT_MEDIAN_MS",
    "MEASURED_ENDPOINT_MEDIAN_JPEG_MS",
    "MEASURED_COMPOSE_MS",
    "MEASURED_ENCODE_MS",
    "PEAK_RSS_GUARD_BYTES",
    "REQUIREMENT_12_1_LIMIT_BYTES",
    "MEASURED_PEAK_RSS_BYTES",
    "render_budget_ms",
]


# --------------------------------------------------------------------------- #
# Requirement 9.3 -- the amended two-term render budget
# --------------------------------------------------------------------------- #
#
# The budget is two-term because the render path is a fixed cost -- cache lookup,
# memoised texture lookup, encode, response framing -- plus a per-plane cost,
# since each tiled plane is an independent inverse warp and lighting blend over
# its own mask bounding box. A flat bound would either fail a four-plane
# whole-room render or be vacuous for a one-plane render.

#: Fixed term. Dominated by PNG encode, which measures 47-56 ms at 1600x1200 and
#: is nearly flat in plane count. Unchanged by the Compositor optimisation below,
#: because that optimisation did not touch the encoder -- and at one tiled plane
#: this term now carries about two thirds of the render.
RENDER_BUDGET_FIXED_MS: float = 70.0

#: Per-plane term. Compose grows by about 18 ms per additional plane
#: (27 -> 38 -> 60 -> 82 ms for one through four planes), down from 27 ms per
#: plane (41 -> 56 -> 89 -> 123 ms) before the Compositor was optimised. The
#: budget is retightened from 40 ms to 26 ms rather than banking the headroom, so
#: the four-plane bound falls from 230 ms to 174 ms and every plane count keeps
#: roughly the 20 percent floor the original margins had.
RENDER_BUDGET_PER_PLANE_MS: float = 26.0


def render_budget_ms(plane_count: int) -> float:
    """The Requirement 9.3 budget for a render of ``plane_count`` tiled planes."""
    if plane_count < 1:
        raise ValueError(f"plane_count must be at least 1, got {plane_count}")
    return RENDER_BUDGET_FIXED_MS + RENDER_BUDGET_PER_PLANE_MS * plane_count


#: Measured endpoint medians, plane count -> ms, from the runs described above.
#: Against budgets of 96 / 122 / 148 / 174 ms, the margins are 22 / 30 / 24 / 21
#: percent. A change that costs more than a fifth of the render path trips this.
MEASURED_ENDPOINT_MEDIAN_MS: dict[int, float] = {1: 75.0, 2: 85.0, 3: 112.0, 4: 137.0}

#: Measured `compose` medians, plane count -> ms. Recorded so a budget failure
#: localises: if compose moved and encode did not, the regression is in the
#: Compositor.
MEASURED_COMPOSE_MS: dict[int, float] = {1: 27.0, 2: 38.0, 3: 60.0, 4: 82.0}

#: Measured `encode_render` medians, plane count -> ms. Roughly flat, and roughly
#: *twice* the 15-25 ms the design originally estimated: a 1600x1200 composite is
#: about 3 MB of PNG and zlib on that is the largest single fixed cost in the
#: render path. `RV_RENDER_FORMAT=jpeg` is the configured escape hatch.
MEASURED_ENCODE_MS: dict[int, float] = {1: 47.0, 2: 47.0, 3: 51.0, 4: 56.0}

#: The same endpoint medians with `RV_RENDER_FORMAT=jpeg`, recorded rather than
#: asserted -- Requirement 9.3's budget is stated over PNG, so this is here to
#: quantify the escape hatch rather than to bound it. JPEG encode of the same
#: composite is 4-5 ms against PNG's 47-56, which is what puts every plane count,
#: including a four-plane whole-room render, under 100 ms:
#: 31 / 42 / 64 / 86 ms. PNG cannot reach that at four planes -- its encoder alone
#: is 56 ms, so a sub-100 ms PNG render would need compose under about 40 ms,
#: another 2x below where it now sits.
MEASURED_ENDPOINT_MEDIAN_JPEG_MS: dict[int, float] = {1: 31.0, 2: 42.0, 3: 64.0, 4: 86.0}


# --------------------------------------------------------------------------- #
# Requirement 12.1 -- the analysis memory bound
# --------------------------------------------------------------------------- #

#: The Requirement 12.1 ceiling: 2 gigabytes of resident process memory for a
#: 2048 px longest-edge analysis pass on a CPU-only host.
REQUIREMENT_12_1_LIMIT_BYTES: int = 2 * 1024 * 1024 * 1024

#: Peak RSS observed across three runs: 704, 729, and 851 MiB. The spread is the
#: allocator's, not the pipeline's -- the same pass on the same input.
MEASURED_PEAK_RSS_BYTES: int = 892_518_400  # 851 MiB, the worst of the three

#: The operative bound: 1.25 GiB. Strictly tighter than Requirement 12.1, so
#: passing it proves the requirement, while leaving about 50 percent headroom
#: over the worst measured peak -- enough for allocator and interpreter variance,
#: tight enough that a stage which stopped releasing its intermediates, or a
#: float32 artifact leaking into the cache, fails here instead of passing at
#: 1.9 GiB.
PEAK_RSS_GUARD_BYTES: int = 1280 * 1024 * 1024


# --------------------------------------------------------------------------- #
# The scenes the budgets are stated over
# --------------------------------------------------------------------------- #

#: Requirement 9.3's reference scene: 1600x1200, the documented fixture pose, all
#: three walls plus the floor so a four-plane whole-room render is reachable.
RENDER_SCENE_SIZE: tuple[int, int] = (1600, 1200)

#: Requirement 12.1's reference photograph: the 2048 px longest-edge cap of
#: Requirement 2.6, which is the largest input any downstream stage can see.
ANALYSIS_LONGEST_EDGE: int = 2048

#: Repeats per plane count. The first render of a scene pays for the distance
#: transform behind each plane's feathered alpha, and the first render of a tile
#: pays for seamless synthesis; Requirement 9.3 places both outside the budget,
#: so both are warmed before the timed repeats.
RENDER_WARMUPS: int = 2
RENDER_REPEATS: int = 7

#: Texture authoring resolution for the perf catalog. `to_metric_texture` caps
#: the long edge at `MAX_TEXTURE_EDGE`, so this only needs to be large enough
#: that the sampled pattern is representative of a shipped tile.
PERF_TEXTURE_LONG_EDGE: int = 512


@dataclass(frozen=True)
class PerfTile:
    """One entry the perf catalog publishes."""

    id: str
    name: str
    file: str
    width_mm: float
    height_mm: float
    finish: str
    gloss: float
    generator: str


#: The four finishes and both metric formats the Setup_Tool ships, so the timed
#: render samples textures of the shape the product actually serves. Gloss varies
#: across the four because the highlight term is gloss-scaled, so a render that
#: only ever saw gloss 0 would skip real work.
PERF_TILES: tuple[PerfTile, ...] = (
    PerfTile("perf-marble-600", "Perf Marble 600x600", "perf_marble_600x600.png",
             600.0, 600.0, "polished", 0.85, "marble"),
    PerfTile("perf-concrete-600", "Perf Concrete 600x600", "perf_concrete_600x600.png",
             600.0, 600.0, "matte", 0.10, "concrete"),
    PerfTile("perf-wood-600x1200", "Perf Wood 600x1200", "perf_wood_600x1200.png",
             600.0, 1200.0, "satin", 0.35, "wood"),
    PerfTile("perf-terrazzo-600", "Perf Terrazzo 600x600", "perf_terrazzo_600x600.png",
             600.0, 600.0, "honed", 0.45, "terrazzo"),
)

_GENERATORS = {
    "marble": generate_marble,
    "concrete": generate_concrete,
    "wood": generate_wood_plank,
    "terrazzo": generate_terrazzo,
}


def _authoring_size_px(tile: PerfTile) -> tuple[int, int]:
    """``(width_px, height_px)`` at the tile's metric ratio and the long edge."""
    long_mm = max(tile.width_mm, tile.height_mm)
    short_px = max(
        16, int(round(PERF_TEXTURE_LONG_EDGE * min(tile.width_mm, tile.height_mm) / long_mm))
    )
    if tile.height_mm >= tile.width_mm:
        return short_px, PERF_TEXTURE_LONG_EDGE
    return PERF_TEXTURE_LONG_EDGE, short_px


@pytest.fixture(scope="module")
def perf_assets(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A generated assets dir carrying the four perf tiles plus a manifest.

    Generated rather than read from ``assets/``: ``.gitignore`` excludes the
    Setup_Tool's PNG output, so a checkout that has not run setup must still be
    able to run these budgets.
    """
    assets_dir = tmp_path_factory.mktemp("perf-assets")
    tiles_dir = assets_dir / "tiles"
    tiles_dir.mkdir()

    entries: list[dict[str, Any]] = []
    for index, tile in enumerate(PERF_TILES):
        pattern = make_seamless(_GENERATORS[tile.generator](_authoring_size_px(tile), seed=index))
        if not cv2.imwrite(str(tiles_dir / tile.file), pattern):  # pragma: no cover
            raise RuntimeError(f"failed to write perf tile {tile.file!r}")
        entries.append(
            {
                "id": tile.id,
                "name": tile.name,
                "file": tile.file,
                "width_mm": tile.width_mm,
                "height_mm": tile.height_mm,
                "finish": tile.finish,
                "gloss": tile.gloss,
            }
        )

    (tiles_dir / "manifest.json").write_text(
        json.dumps({"version": 1, "tiles": entries}, indent=2), encoding="utf-8"
    )
    return assets_dir


@pytest.fixture(scope="module")
def perf_client(perf_assets: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Any]:
    """A module-scoped ``TestClient`` with the neural backend disabled.

    Module-scoped deliberately: one 1600x1200 analysis pass is well over a second
    of honest CPU work, and every timing here is stated over renders against a
    single cached scene. ``os.environ`` is written directly and restored in a
    ``finally`` rather than through ``monkeypatch``, which is function-scoped and
    cannot be requested from a module-scoped fixture.
    """
    overrides = {
        "RV_ENABLE_NEURAL_BACKEND": "false",
        "RV_ASSETS_DIR": str(perf_assets),
        "RV_WEIGHTS_DIR": str(tmp_path_factory.mktemp("perf-weights")),
        # The budget is stated for PNG, so it is pinned here rather than inherited
        # from whatever the developer has in their environment.
        "RV_RENDER_FORMAT": "png",
    }
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    get_settings.cache_clear()

    # Imported inside the body for the same reason conftest does it: the module
    # must stay collectable without the app having been constructed.
    from fastapi.testclient import TestClient

    from backend.app import app

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_settings.cache_clear()


@dataclass(frozen=True)
class PerfScene:
    """The analysed 1600x1200 scene every render timing is stated over."""

    client: Any
    scene_id: str
    width: int
    height: int
    planes: tuple[PlaneName, ...]
    analysis_ms: int


@pytest.fixture(scope="module")
def perf_scene(perf_client: Any) -> PerfScene:
    """Analyse the 1600x1200 reference room once, through the real endpoint.

    Through HTTP rather than by hand-building a ``SceneState``, so the cached
    artifacts the render path reads are exactly the ones analysis produces.
    """
    width, height = RENDER_SCENE_SIZE
    room = make_synthetic_room(
        width=width,
        height=height,
        focal_px=0.875 * width,
        yaw_deg=8.0,
        pitch_deg=-12.0,
        walls=("left", "right", "back"),
        n_occluders=2,
        seed=0,
    )
    payload, _ = encode_image(room.image, "png")
    response = perf_client.post(
        "/api/segment", files={"file": ("perf_room.png", payload, "image/png")}
    )
    assert response.status_code == 200, (
        f"perf scene analysis failed with {response.status_code}: {response.text[:400]}"
    )
    body = response.json()
    assert (body["width"], body["height"]) == (width, height), (
        f"perf scene came back {body['width']}x{body['height']}, expected "
        f"{width}x{height}; the render budget is stated over the full-size frame"
    )
    return PerfScene(
        client=perf_client,
        scene_id=body["scene_id"],
        width=int(body["width"]),
        height=int(body["height"]),
        planes=tuple(plane["name"] for plane in body["planes"]),
        analysis_ms=int(body["analysis_ms"]),
    )


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def _tile_id_for(index: int) -> str:
    """A different tile per plane, cycling if there are more planes than tiles."""
    return PERF_TILES[index % len(PERF_TILES)].id


@dataclass(frozen=True)
class RenderSample:
    """Everything one plane count measured, in milliseconds."""

    plane_count: int
    planes: tuple[PlaneName, ...]
    endpoint_median_ms: float
    endpoint_max_ms: float
    compose_median_ms: float
    encode_median_ms: float
    image_bytes: int

    @property
    def budget_ms(self) -> float:
        return render_budget_ms(self.plane_count)

    @property
    def margin_fraction(self) -> float:
        return 1.0 - self.endpoint_median_ms / self.budget_ms

    def describe(self) -> str:
        return (
            f"n={self.plane_count} ({', '.join(self.planes)}): "
            f"endpoint median {self.endpoint_median_ms:.1f} ms "
            f"(max {self.endpoint_max_ms:.1f}) vs budget {self.budget_ms:.0f} ms, "
            f"margin {self.margin_fraction * 100:.0f}% | "
            f"compose {self.compose_median_ms:.1f} ms, "
            f"encode {self.encode_median_ms:.1f} ms, "
            f"{self.image_bytes} bytes"
        )


def _measure_endpoint(scene: PerfScene, planes: Sequence[PlaneName]) -> tuple[float, float, int]:
    """``(median_ms, max_ms, image_bytes)`` for ``/api/render`` over ``planes``.

    Reads ``X-Render-Ms`` from the binary response, which is the server-side
    duration the route itself measures -- cache lookup through encode, before any
    base64 framing -- so this is the number Requirement 9.3 is written about and
    carries no client-side or transport cost.
    """
    payload = {
        "scene_id": scene.scene_id,
        "planes": {
            name: {"tile_id": _tile_id_for(index), "rotation_deg": 0.0}
            for index, name in enumerate(planes)
        },
        "format": "png",
    }

    for _ in range(RENDER_WARMUPS):
        warm = scene.client.post("/api/render?binary=1", json=payload)
        assert warm.status_code == 200, (
            f"warmup render of {list(planes)} failed with {warm.status_code}: "
            f"{warm.text[:400]}"
        )

    durations: list[float] = []
    image_bytes = 0
    for _ in range(RENDER_REPEATS):
        response = scene.client.post("/api/render?binary=1", json=payload)
        assert response.status_code == 200, (
            f"render of {list(planes)} failed with {response.status_code}: "
            f"{response.text[:400]}"
        )
        durations.append(float(response.headers["X-Render-Ms"]))
        image_bytes = len(response.content)

    return statistics.median(durations), max(durations), image_bytes


def _measure_compose_encode(
    scene: PerfScene, planes: Sequence[PlaneName]
) -> tuple[float, float]:
    """``(compose_median_ms, encode_median_ms)`` against the cached scene.

    Calls the same two functions the route calls, with the same cached
    Scene_State and the same memoised textures, so the pair sums to essentially
    all of the route's work. Recorded rather than asserted: the budget is stated
    over the endpoint, and these two exist to say *where* a budget failure came
    from.
    """
    state = scene.client.app.state
    cached = state.cache.get(scene.scene_id)
    assert cached is not None, "perf scene fell out of the cache mid-measurement"

    specs: dict[PlaneName, PlaneRenderSpec] = {}
    tiles = {}
    textures = {}
    for index, name in enumerate(planes):
        tile_id = _tile_id_for(index)
        tile = state.catalog.get(tile_id)
        assert tile is not None, f"perf catalog is missing {tile_id!r}"
        specs[name] = PlaneRenderSpec(tile_id=tile.id, rotation_deg=0.0)
        tiles[name] = tile
        textures[name] = state.catalog.seamless(tile.id)

    compose_ms: list[float] = []
    encode_ms: list[float] = []
    for repeat in range(RENDER_WARMUPS + RENDER_REPEATS):
        start = time.perf_counter()
        composited = compose(
            cached,
            specs,
            textures,
            state.settings,
            tiles=tiles,
            alpha_cache=cached.plane_alpha,
        )
        mid = time.perf_counter()
        encode_render(composited, state.settings, fmt="png")
        end = time.perf_counter()
        if repeat >= RENDER_WARMUPS:
            compose_ms.append((mid - start) * 1000.0)
            encode_ms.append((end - mid) * 1000.0)

    return statistics.median(compose_ms), statistics.median(encode_ms)


@pytest.fixture(scope="module")
def render_samples(perf_scene: PerfScene) -> tuple[RenderSample, ...]:
    """One :class:`RenderSample` per plane count, from 1 up to every plane found.

    Module-scoped so the four plane counts are measured once and both the budget
    assertion and the recorded breakdown read the same numbers.
    """
    ordered = [name for name in PLANE_NAMES if name in perf_scene.planes]
    assert len(ordered) >= 2, (
        f"the perf scene detected only {ordered!r}; a per-plane budget needs at "
        "least two plane counts to be a per-plane statement rather than a flat one"
    )

    samples: list[RenderSample] = []
    for count in range(1, len(ordered) + 1):
        planes = tuple(ordered[:count])
        median_ms, max_ms, image_bytes = _measure_endpoint(perf_scene, planes)
        compose_ms, encode_ms = _measure_compose_encode(perf_scene, planes)
        samples.append(
            RenderSample(
                plane_count=count,
                planes=planes,
                endpoint_median_ms=median_ms,
                endpoint_max_ms=max_ms,
                compose_median_ms=compose_ms,
                encode_median_ms=encode_ms,
                image_bytes=image_bytes,
            )
        )
    return tuple(samples)


# --------------------------------------------------------------------------- #
# Requirement 9.3
# --------------------------------------------------------------------------- #


@pytest.mark.perf
def test_render_stays_within_the_per_plane_budget(
    render_samples: tuple[RenderSample, ...], perf_scene: PerfScene
) -> None:
    """Requirement 9.3: median render time is within ``70 + 40n`` ms.

    Asserted on the median rather than the maximum. The median is stable to
    within a millisecond across runs; the maximum is not -- one run in three
    produced a 266 ms outlier at three planes against a 141 ms median, which is
    the OS descheduling the process. Asserting the maximum would make this test
    flaky without making it more sensitive to a real regression.
    """
    assert perf_scene.width == RENDER_SCENE_SIZE[0], perf_scene
    assert len(render_samples) >= 2, render_samples

    breakdown = "\n".join(f"  {sample.describe()}" for sample in render_samples)
    over = [sample for sample in render_samples if sample.endpoint_median_ms > sample.budget_ms]
    assert not over, (
        "Requirement 9.3 render budget exceeded for plane count(s) "
        f"{[sample.plane_count for sample in over]} on a "
        f"{perf_scene.width}x{perf_scene.height} scene with PNG output.\n"
        f"Budget is {RENDER_BUDGET_FIXED_MS:.0f} ms + "
        f"{RENDER_BUDGET_PER_PLANE_MS:.0f} ms per tiled plane.\n"
        f"Measured this run:\n{breakdown}\n"
        f"Reference medians: {MEASURED_ENDPOINT_MEDIAN_MS}\n"
        f"Reference compose: {MEASURED_COMPOSE_MS}\n"
        f"Reference encode:  {MEASURED_ENCODE_MS}\n"
        "Compare the compose and encode columns against the reference rows to "
        "localise the regression: compose scales with plane count, encode does "
        "not."
    )


@pytest.mark.perf
def test_render_cost_splits_into_a_flat_encode_and_a_per_plane_compose(
    render_samples: tuple[RenderSample, ...],
) -> None:
    """The two-term budget's premise, checked rather than assumed.

    Requirement 9.3 is per-plane because compose scales with plane count while
    encode does not. If that ever stopped being true the budget's shape would be
    wrong, and this is the test that says so -- separately from the budget
    assertion, so a shape change and an overrun do not present as one failure.

    Bounds are loose on purpose: this is a claim about which term grows, not
    about how fast either is.
    """
    by_count = {sample.plane_count: sample for sample in render_samples}
    first, last = min(by_count), max(by_count)

    compose_growth = by_count[last].compose_median_ms / by_count[first].compose_median_ms
    encode_growth = by_count[last].encode_median_ms / by_count[first].encode_median_ms

    # Measured over 1 -> 4 planes: compose 27 -> 82 ms (3.0x), encode 47 -> 56 ms
    # (1.2x). Thresholds sit either side of that gap with room to spare. The
    # Compositor optimisation cut both ends of the compose range by about a third
    # and left the ratio where it was, which is the point: the *shape* of the
    # budget is a claim about scaling, not about absolute speed.
    assert compose_growth > 1.5, (
        f"compose barely grew from {first} to {last} planes "
        f"({by_count[first].compose_median_ms:.1f} -> "
        f"{by_count[last].compose_median_ms:.1f} ms, {compose_growth:.2f}x); "
        "Requirement 9.3's per-plane term assumes it scales with plane count "
        "(measured 3.0x)"
    )
    assert encode_growth < 1.5, (
        f"encode grew {encode_growth:.2f}x from {first} to {last} planes "
        f"({by_count[first].encode_median_ms:.1f} -> "
        f"{by_count[last].encode_median_ms:.1f} ms); Requirement 9.3 treats encode "
        "as part of the fixed term (measured 1.2x)"
    )

    # Encode is the fixed term's dominant cost, and roughly twice the design's
    # original 15-25 ms estimate. Asserted so the amended figure is protected: if
    # encode ever became cheap, the 70 ms fixed term is over-generous and the
    # budget should be retightened rather than silently banked.
    encode_ms = by_count[first].encode_median_ms
    assert 25.0 <= encode_ms <= RENDER_BUDGET_FIXED_MS, (
        f"PNG encode of a {RENDER_SCENE_SIZE[0]}x{RENDER_SCENE_SIZE[1]} composite "
        f"measured {encode_ms:.1f} ms; the amended Requirement 9.3 documents ~50 ms "
        f"and sizes the {RENDER_BUDGET_FIXED_MS:.0f} ms fixed term around it. "
        "Outside this range the fixed term needs revisiting."
    )


@pytest.mark.perf
def test_recorded_render_breakdown(
    render_samples: tuple[RenderSample, ...], perf_scene: PerfScene, record_property: Any
) -> None:
    """Attach the per-plane-count breakdown to the test report.

    Recorded data, not a budget: this is what makes a future failure diagnosable
    without re-instrumenting anything. ``record_property`` puts it in the JUnit
    XML when one is produced, and the printed block puts it in ``-s`` output.
    """
    record_property("scene", f"{perf_scene.width}x{perf_scene.height}")
    record_property("analysis_ms", perf_scene.analysis_ms)
    record_property("budget_ms_formula", f"{RENDER_BUDGET_FIXED_MS:.0f}+{RENDER_BUDGET_PER_PLANE_MS:.0f}n")

    lines = [
        f"render breakdown -- {perf_scene.width}x{perf_scene.height} PNG, "
        f"planes detected: {', '.join(perf_scene.planes)}, "
        f"analysis {perf_scene.analysis_ms} ms"
    ]
    for sample in render_samples:
        record_property(f"n{sample.plane_count}_endpoint_median_ms", sample.endpoint_median_ms)
        record_property(f"n{sample.plane_count}_compose_median_ms", sample.compose_median_ms)
        record_property(f"n{sample.plane_count}_encode_median_ms", sample.encode_median_ms)
        lines.append(f"  {sample.describe()}")
    print("\n" + "\n".join(lines))

    # The recording itself is the point, but an empty recording is a silent
    # failure, so require every plane count to have produced real numbers.
    assert all(
        sample.endpoint_median_ms > 0.0
        and sample.compose_median_ms > 0.0
        and sample.encode_median_ms > 0.0
        for sample in render_samples
    ), lines


# --------------------------------------------------------------------------- #
# Requirement 12.1
# --------------------------------------------------------------------------- #


def _peak_rss_during(work: Any, *, interval_s: float = 0.005) -> tuple[int, Any]:
    """Run ``work()`` while sampling this process's RSS; return the peak and result.

    A sampling thread rather than ``resource.getrusage``: ``ru_maxrss`` is a
    high-water mark for the whole process lifetime, so it would report the peak
    of every test that ran before this one. Sampling bounds the answer to this
    call. The thread only reads ``memory_info()``, so it adds no measurable
    allocation of its own.
    """
    process = psutil.Process()
    peak = process.memory_info().rss
    stop = threading.Event()

    def _sample() -> None:
        nonlocal peak
        while not stop.wait(interval_s):
            peak = max(peak, process.memory_info().rss)

    sampler = threading.Thread(target=_sample, name="rss-sampler", daemon=True)
    sampler.start()
    try:
        result = work()
    finally:
        stop.set()
        sampler.join(timeout=5.0)

    return max(peak, process.memory_info().rss), result


@pytest.mark.resource
def test_analysis_peak_rss_stays_under_the_memory_bound(perf_client: Any) -> None:
    """Requirement 12.1: a 2048 px analysis pass stays under 2 GB resident.

    The bound actually asserted is :data:`PEAK_RSS_GUARD_BYTES` (1.25 GiB),
    strictly tighter than the requirement's 2 GiB, so passing it proves the
    requirement while still failing on a regression that a 2 GiB bound would
    wave through. Measured peaks were 704, 729, and 851 MiB across three runs.

    Sampled around the real ``/api/segment`` call at the Requirement 2.6 cap,
    which is the largest input any stage can see, so this is the worst case the
    service admits rather than a synthetic one.
    """
    assert PEAK_RSS_GUARD_BYTES < REQUIREMENT_12_1_LIMIT_BYTES, (
        f"the regression guard ({PEAK_RSS_GUARD_BYTES} bytes) must stay below the "
        f"Requirement 12.1 ceiling ({REQUIREMENT_12_1_LIMIT_BYTES} bytes), otherwise "
        "this test stops verifying the requirement"
    )

    width = ANALYSIS_LONGEST_EDGE
    height = int(round(width * 3 / 4))
    room = make_synthetic_room(
        width=width,
        height=height,
        focal_px=0.875 * width,
        yaw_deg=8.0,
        pitch_deg=-12.0,
        walls=("left", "right", "back"),
        n_occluders=2,
        seed=0,
    )
    payload, _ = encode_image(room.image, "png")

    # The generated room is the largest array in this test and is not part of
    # what the service holds, so it is released before the pass is measured.
    del room

    peak_bytes, response = _peak_rss_during(
        lambda: perf_client.post(
            "/api/segment", files={"file": ("perf_big.png", payload, "image/png")}
        )
    )

    assert response.status_code == 200, (
        f"2048 px analysis failed with {response.status_code}: {response.text[:400]}"
    )
    body = response.json()
    assert max(body["width"], body["height"]) == ANALYSIS_LONGEST_EDGE, (
        f"processed image longest edge is {max(body['width'], body['height'])}, "
        f"expected the Requirement 2.6 cap of {ANALYSIS_LONGEST_EDGE}; the memory "
        "bound must be measured at the cap"
    )

    peak_mib = peak_bytes / (1024 * 1024)
    print(
        f"\n2048 px analysis: peak RSS {peak_mib:.0f} MiB "
        f"(guard {PEAK_RSS_GUARD_BYTES / 1024 / 1024:.0f} MiB, "
        f"R12.1 ceiling {REQUIREMENT_12_1_LIMIT_BYTES / 1024 / 1024:.0f} MiB), "
        f"analysis {body['analysis_ms']} ms"
    )

    assert peak_bytes <= PEAK_RSS_GUARD_BYTES, (
        f"peak RSS during a {ANALYSIS_LONGEST_EDGE} px analysis pass was "
        f"{peak_mib:.0f} MiB, over the {PEAK_RSS_GUARD_BYTES / 1024 / 1024:.0f} MiB "
        f"regression guard (measured reference: "
        f"{MEASURED_PEAK_RSS_BYTES / 1024 / 1024:.0f} MiB; Requirement 12.1 ceiling: "
        f"{REQUIREMENT_12_1_LIMIT_BYTES / 1024 / 1024:.0f} MiB). "
        "Most likely causes: a stage holding its intermediates past its return, a "
        "float32 or float64 artifact reaching the Scene_Cache in place of the "
        "uint8 required by Requirement 12.4, or an analysis buffer sized off the "
        "pre-downscale image."
    )


@pytest.mark.resource
def test_cached_scene_state_is_small_enough_for_a_full_cache(perf_client: Any) -> None:
    """Requirement 12.1: 32 cached 2048 px scenes must fit alongside an analysis.

    The peak-RSS test above covers one analysis pass in isolation. This covers
    the other half of the design's memory argument: the Scene_Cache holds up to
    ``scene_cache_max_entries`` scenes, so if one cached scene were much larger
    than the design's ~31 MB estimate, a full cache would breach the 2 GB ceiling
    even though a single pass does not.

    Asserted through ``SceneState.nbytes()`` rather than by filling the cache,
    which would take 32 analysis passes and minutes of CPU to prove arithmetic.
    """
    width = ANALYSIS_LONGEST_EDGE
    height = int(round(width * 3 / 4))
    room = make_synthetic_room(
        width=width,
        height=height,
        focal_px=0.875 * width,
        yaw_deg=8.0,
        pitch_deg=-12.0,
        walls=("left", "right", "back"),
        n_occluders=2,
        seed=0,
    )
    payload, _ = encode_image(room.image, "png")
    del room

    response = perf_client.post(
        "/api/segment", files={"file": ("perf_big2.png", payload, "image/png")}
    )
    assert response.status_code == 200, response.text[:400]

    state = perf_client.app.state
    scene = state.cache.get(response.json()["scene_id"])
    assert scene is not None, "the analysed scene is not in the cache"

    per_scene = scene.nbytes()
    max_entries = state.settings.scene_cache_max_entries
    full_cache = per_scene * max_entries

    # Measured: 30.0 MiB per 2048x1536 scene, so a full 32-entry cache is 960 MiB
    # against a bound of 1024 MiB -- the design's own division of the 2 GB ceiling
    # between the cache and one live analysis pass. Six percent of margin is thin
    # for a timing bound but fine here: `nbytes()` is exact arithmetic over array
    # sizes with no sampling and no noise, so the only way this moves is an
    # artifact being added to the cache or widening past uint8, which is precisely
    # what it should catch.
    print(
        f"\ncached scene: {per_scene / 1024 / 1024:.1f} MiB, "
        f"{max_entries} entries = {full_cache / 1024 / 1024:.0f} MiB "
        f"(half the R12.1 ceiling = "
        f"{REQUIREMENT_12_1_LIMIT_BYTES / 2 / 1024 / 1024:.0f} MiB)"
    )
    assert full_cache <= REQUIREMENT_12_1_LIMIT_BYTES // 2, (
        f"a full Scene_Cache of {max_entries} scenes at "
        f"{per_scene / 1024 / 1024:.1f} MiB each is "
        f"{full_cache / 1024 / 1024:.0f} MiB, over half the Requirement 12.1 "
        "ceiling, which leaves no headroom for a concurrent analysis pass. Check "
        "that every cached artifact is still uint8 (Requirement 12.4)."
    )


@pytest.mark.perf
@pytest.mark.resource
def test_module_bounds_are_internally_consistent() -> None:
    """Guard the constants themselves against being quietly loosened.

    Carries *both* markers, so it runs under ``-m perf`` and under
    ``-m resource`` -- whichever budget a developer is looking at, the check that
    its bound still means something runs alongside it. It stays out of the default
    selection so the default run remains exactly the correctness suite, and it
    costs no measurement, so paying for it twice is free.
    """
    assert PEAK_RSS_GUARD_BYTES < REQUIREMENT_12_1_LIMIT_BYTES, (
        "the RSS guard must be strictly tighter than Requirement 12.1's ceiling"
    )
    assert MEASURED_PEAK_RSS_BYTES < PEAK_RSS_GUARD_BYTES, (
        "the recorded measurement must sit below the guard, or the guard is "
        "already failing on the host it was set from"
    )

    for count, measured in MEASURED_ENDPOINT_MEDIAN_MS.items():
        budget = render_budget_ms(count)
        assert measured < budget, (
            f"recorded median for {count} plane(s) ({measured} ms) is not below its "
            f"budget ({budget} ms); the budget and the measurements disagree"
        )
        parts = MEASURED_COMPOSE_MS[count] + MEASURED_ENCODE_MS[count]
        assert parts <= measured * 1.10, (
            f"recorded compose+encode for {count} plane(s) ({parts} ms) exceeds the "
            f"recorded endpoint median ({measured} ms) by more than rounding; the "
            "breakdown does not describe the whole it claims to"
        )

    # The JPEG figures are recorded, not asserted against a budget, so the only
    # thing that can rot about them is their relationship to the PNG ones: JPEG
    # shares the same compose and only swaps the encoder, so it must be cheaper at
    # every plane count, and the claim that it clears 100 ms where PNG does not is
    # the reason the dict is here at all.
    assert set(MEASURED_ENDPOINT_MEDIAN_JPEG_MS) == set(MEASURED_ENDPOINT_MEDIAN_MS)
    for count, jpeg_ms in MEASURED_ENDPOINT_MEDIAN_JPEG_MS.items():
        assert jpeg_ms < MEASURED_ENDPOINT_MEDIAN_MS[count], (
            f"recorded JPEG median for {count} plane(s) ({jpeg_ms} ms) is not below "
            f"the PNG one ({MEASURED_ENDPOINT_MEDIAN_MS[count]} ms); JPEG differs "
            "only in the encoder, so it cannot be slower"
        )
        assert jpeg_ms >= MEASURED_COMPOSE_MS[count], (
            f"recorded JPEG median for {count} plane(s) ({jpeg_ms} ms) is below the "
            f"recorded compose time ({MEASURED_COMPOSE_MS[count]} ms), which is "
            "impossible: compose runs in both paths"
        )
    assert max(MEASURED_ENDPOINT_MEDIAN_JPEG_MS.values()) < 100.0, (
        "the recorded JPEG figures no longer clear 100 ms at every plane count; "
        "the note about JPEG being the sub-100 ms path needs revising with them"
    )

    with pytest.raises(ValueError):
        render_budget_ms(0)
