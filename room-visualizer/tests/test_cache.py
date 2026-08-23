"""Tests for the Scene_Cache in `backend/cache.py` (Requirements 9.4, 9.5, 9.6, 12.3, 12.5).

Three properties, each stated over one bound the cache has to hold:

* **Property 30** drives an arbitrary put/get sequence against a reference
  least-recently-used model and holds the cache to the same retained keys and to
  the configured entry bound (Requirement 9.5).
* **Property 31** adds the clock: under an injected, hand-advanced `clock` an
  entry must be retrievable exactly when its age is inside the time-to-live
  *and* it has survived the LRU bound (Requirement 9.6).
* **Property 32** checks the memory consequence of eviction: every evicted
  Scene_State's arrays are unreachable and its buffers are gone, which is what
  makes the release-on-eviction contract observable (Requirement 12.3).

The reference model in `_ReferenceCache` is shared by Properties 30 and 31,
which is deliberate: the LRU property is the TTL property with a time-to-live
long enough never to fire, so one model driven two ways proves the two bounds do
not interfere. Ages are measured from `SceneState.created_at`, not from
insertion time, so every state built here carries a `created_at` in the same
epoch as the clock the cache was given.

Unit tests around the properties pin the pieces a property cannot see: the
constructor's rejection of nonsense bounds, the exclusive TTL boundary, the
deliberate decision that `__contains__` does not refresh recency, and the
`stats()` payload the health route splats into `HealthResponse`.

The final section drives the cache from many threads at once. The lock there is
load-bearing -- FastAPI runs sync path operations in a threadpool -- so those
tests assert invariants that must hold under every interleaving rather than any
particular one (Requirement 9.5).
"""

from __future__ import annotations

import gc
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from backend.cache import SceneCache
from backend.schemas import HealthResponse, PlaneMetadata, SceneState

# --------------------------------------------------------------------------- #
# Scene_State builders
# --------------------------------------------------------------------------- #
#
# Tiny arrays on purpose: every assertion in this module is about identity,
# ordering, and reachability, none about pixel content, so a 4x4 state exercises
# the same paths as a 2048 px one for a few hundred bytes.

_STATE_PX = 4

#: Epoch the injected clocks start from. Far from zero so a bug that reads
#: insertion time instead of ``created_at`` cannot pass by accident.
_EPOCH = 1_000_000.0


def _plane_metadata(name: str = "floor") -> PlaneMetadata:
    """One plane's worth of metadata, with its own arrays to weakref."""
    quad = np.array([[0, 0], [_STATE_PX, 0], [_STATE_PX, _STATE_PX], [0, _STATE_PX]], dtype=np.int32)
    return PlaneMetadata(
        name=name,
        contour=quad,
        bounding_points=quad.copy(),
        area_fraction=0.25,
        centroid=(2.0, 2.0),
        homography=np.eye(3, dtype=np.float64),
        homography_inv=np.eye(3, dtype=np.float64),
        plane_extent_mm=(0.0, 0.0, 1000.0, 1000.0),
        reprojection_rmse_px=0.4,
        geometry_mode="vanishing_points",
        luminance_median=120.0,
    )


def _scene_state(scene_id: str, created_at: float = _EPOCH) -> SceneState:
    """A minimal but complete Scene_State carrying freshly allocated arrays.

    Every array is allocated here rather than shared across states, so a
    weakref taken to one state's buffers says something about that state alone.
    """
    n = _STATE_PX
    return SceneState(
        scene_id=scene_id,
        created_at=created_at,
        image=np.zeros((n, n, 3), dtype=np.uint8),
        width=n,
        height=n,
        planes={"floor": _plane_metadata()},
        plane_masks={"floor": np.zeros((n, n), dtype=np.uint8)},
        foreground_mask=np.zeros((n, n), dtype=np.uint8),
        shading_map=np.zeros((n, n), dtype=np.uint8),
        detail_map=np.full((n, n), 128, dtype=np.uint8),
        horizon=(0.0, 1.0, -2.0),
        vanishing_points={"VPx": (-100.0, 2.0), "VPy": None, "VPz": (200.0, 2.0)},
        geometry_mode="vanishing_points",
        segmentation_backend="classical",
    )


class ManualClock:
    """A hand-advanced stand-in for `time.time`.

    TTL behaviour is otherwise only reachable by sleeping, which makes the
    difference between "expired" and "slow test host" unobservable.
    """

    def __init__(self, now: float = _EPOCH) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


# --------------------------------------------------------------------------- #
# Reference LRU + TTL model
# --------------------------------------------------------------------------- #


class _ReferenceCache:
    """An independent model of the two bounds, holding ids and ages only.

    Written from the requirements rather than from `backend/cache.py`: entries
    older than the time-to-live are dropped before every operation, an insertion
    lands at the most-recent end, a hit refreshes recency, and the
    least-recently-used entry goes once the count would exceed the maximum. It
    stores no arrays, so it cannot accidentally reproduce an implementation bug
    in how the real cache handles state objects.
    """

    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        # id -> created_at, ordered least-recently-used first.
        self._created: OrderedDict[str, float] = OrderedDict()

    def purge(self, now: float) -> None:
        """Drop every entry whose age exceeds the TTL. The bound is exclusive."""
        for scene_id in [
            scene_id
            for scene_id, created_at in self._created.items()
            if now - created_at > self.ttl_seconds
        ]:
            del self._created[scene_id]

    def put(self, scene_id: str, created_at: float, now: float) -> None:
        self.purge(now)
        self._created.pop(scene_id, None)
        self._created[scene_id] = created_at
        while len(self._created) > self.max_entries:
            self._created.popitem(last=False)

    def get(self, scene_id: str, now: float) -> bool:
        """Return whether the id is retrievable, refreshing recency on a hit."""
        self.purge(now)
        if scene_id not in self._created:
            return False
        self._created.move_to_end(scene_id)
        return True

    def keys(self, now: float) -> tuple[str, ...]:
        self.purge(now)
        return tuple(self._created)


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

#: A small id pool, so an arbitrary sequence produces real hits, real recency
#: refreshes, and real re-insertions rather than a stream of unique misses.
_SCENE_IDS = tuple(f"scene-{i:02d}" for i in range(6))

_scene_id = st.sampled_from(_SCENE_IDS)
_max_entries = st.integers(min_value=1, max_value=4)

#: ``(kind, scene_id, amount)``. ``amount`` is the age the state already has at
#: insertion for ``put`` and the elapsed seconds for ``advance``; ``get``
#: ignores it. Integers keep the age-exactly-equals-TTL boundary reachable.
_cache_op = st.tuples(
    st.sampled_from(("put", "get", "advance")),
    _scene_id,
    st.integers(min_value=0, max_value=6),
)

_PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


# --------------------------------------------------------------------------- #
# Property 30 -- the LRU bound (Requirement 9.5)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 30: Scene cache obeys the LRU bound
@_PROPERTY_SETTINGS
@given(
    max_entries=_max_entries,
    ops=st.lists(
        st.tuples(st.sampled_from(("put", "get")), _scene_id), min_size=1, max_size=40
    ),
)
def test_property_30_cache_matches_a_reference_lru_model(max_entries, ops):
    """For any sequence of insertions and retrievals, the entry count never
    exceeds the configured maximum and the retained keys equal those of a
    reference least-recently-used model driven by the same sequence.

    The clock is frozen and the time-to-live is far larger than anything the
    frozen clock can reach, so this example isolates the LRU bound: no entry can
    expire, and any divergence from the model is a recency-ordering bug.

    **Validates: Requirements 9.5**
    """
    clock = ManualClock()
    ttl_seconds = 3600.0
    cache = SceneCache(max_entries=max_entries, ttl_seconds=ttl_seconds, clock=clock)
    model = _ReferenceCache(max_entries, ttl_seconds)

    for kind, scene_id in ops:
        if kind == "put":
            cache.put(_scene_state(scene_id, created_at=clock.now))
            model.put(scene_id, created_at=clock.now, now=clock.now)
        else:
            state = cache.get(scene_id)
            hit = model.get(scene_id, now=clock.now)
            assert (state is not None) == hit, f"{scene_id!r} hit disagreed with the model"
            if state is not None:
                assert state.scene_id == scene_id, "get returned some other scene's state"

        # The bound itself, then the stronger claim: not just the same set of
        # retained keys but the same recency order, since an ordering bug is
        # exactly what makes a *later* set diverge.
        assert len(cache) <= max_entries
        assert cache.keys() == model.keys(clock.now)


def test_lru_evicts_the_least_recently_used_entry_not_the_oldest_insertion():
    """Guard for Property 30: recency must be reading refreshed, not insertion.

    A model that ignored `get` would agree with the implementation on any
    put-only sequence, so the property needs at least one example where the two
    orders genuinely differ.
    """
    clock = ManualClock()
    cache = SceneCache(max_entries=2, ttl_seconds=3600.0, clock=clock)
    for scene_id in ("a", "b"):
        cache.put(_scene_state(scene_id, created_at=clock.now))

    assert cache.get("a") is not None  # "a" is now the most recent
    cache.put(_scene_state("c", created_at=clock.now))

    assert cache.keys() == ("a", "c")
    assert cache.get("b") is None, "the refreshed entry was evicted instead of the stale one"


def test_reinserting_a_scene_id_replaces_the_entry_without_growing_the_cache():
    clock = ManualClock()
    cache = SceneCache(max_entries=2, ttl_seconds=3600.0, clock=clock)
    first = _scene_state("a", created_at=clock.now)
    cache.put(first)
    second = _scene_state("a", created_at=clock.now)
    cache.put(second)

    assert len(cache) == 1
    assert cache.get("a") is second
    assert first.nbytes() == 0, "the displaced state was not released"


def test_contains_does_not_rescue_an_entry_from_eviction():
    """`__contains__` reports liveness without refreshing recency, on purpose.

    Documented as deliberate in `backend/cache.py`, so it is pinned here: a
    probe must not let a caller keep a cold scene resident.
    """
    clock = ManualClock()
    cache = SceneCache(max_entries=2, ttl_seconds=3600.0, clock=clock)
    for scene_id in ("a", "b"):
        cache.put(_scene_state(scene_id, created_at=clock.now))

    assert "a" in cache
    cache.put(_scene_state("c", created_at=clock.now))

    assert "a" not in cache
    assert cache.keys() == ("b", "c")


# --------------------------------------------------------------------------- #
# Property 31 -- the time-to-live bound (Requirement 9.6)
# --------------------------------------------------------------------------- #


# Feature: ai-room-tile-visualizer, Property 31: Scene cache obeys the
# time-to-live bound
@_PROPERTY_SETTINGS
@given(
    max_entries=_max_entries,
    ttl_seconds=st.integers(min_value=1, max_value=5),
    ops=st.lists(_cache_op, min_size=1, max_size=40),
)
def test_property_31_state_is_retrievable_exactly_within_the_ttl(max_entries, ttl_seconds, ops):
    """For any sequence of insertion times and query times under an injected
    clock, a state is retrievable exactly when its age at query time is within
    the time-to-live and it has not been LRU-evicted.

    Insertions draw an age the state already carries, so `created_at` and
    insertion time come apart: a state inserted at its TTL boundary must be
    treated as old, which is only true if age is measured from creation.
    Integer clock steps and an integer TTL keep the exclusive boundary --
    age exactly equal to the TTL is still live -- reachable rather than
    something only a lucky float would hit.

    **Validates: Requirements 9.6**
    """
    clock = ManualClock()
    cache = SceneCache(max_entries=max_entries, ttl_seconds=float(ttl_seconds), clock=clock)
    model = _ReferenceCache(max_entries, float(ttl_seconds))

    for kind, scene_id, amount in ops:
        if kind == "advance":
            clock.advance(amount)
        elif kind == "put":
            created_at = clock.now - amount
            cache.put(_scene_state(scene_id, created_at=created_at))
            model.put(scene_id, created_at=created_at, now=clock.now)
        else:
            state = cache.get(scene_id)
            hit = model.get(scene_id, now=clock.now)
            assert (state is not None) == hit, (
                f"{scene_id!r} retrievability disagreed with the model at t={clock.now}"
            )
            if state is not None:
                assert clock.now - state.created_at <= ttl_seconds, (
                    "a state past its time-to-live was returned"
                )

        assert len(cache) <= max_entries
        assert cache.keys() == model.keys(clock.now)


@pytest.mark.parametrize(
    ("age", "expected_live"),
    [(0.0, True), (5.0, True), (9.999, True), (10.0, True), (10.001, False), (60.0, False)],
)
def test_ttl_bound_is_exclusive_at_exactly_the_configured_age(age, expected_live):
    """Requirement 9.6 evicts when age *exceeds* the TTL, so equality is live.

    The property can only find this boundary if the boundary is reachable; this
    test states it outright, including both sides of it.
    """
    clock = ManualClock()
    cache = SceneCache(max_entries=4, ttl_seconds=10.0, clock=clock)
    cache.put(_scene_state("a", created_at=clock.now))

    clock.advance(age)

    assert (cache.get("a") is not None) is expected_live
    assert ("a" in cache) is expected_live
    assert len(cache) == (1 if expected_live else 0)


def test_purge_expired_reports_how_many_entries_it_dropped():
    clock = ManualClock()
    cache = SceneCache(max_entries=4, ttl_seconds=10.0, clock=clock)
    cache.put(_scene_state("old-a", created_at=clock.now))
    cache.put(_scene_state("old-b", created_at=clock.now))
    clock.advance(11.0)
    cache.put(_scene_state("fresh", created_at=clock.now))

    # The insertion above already purges, so the explicit call has nothing left.
    assert cache.purge_expired() == 0
    assert cache.keys() == ("fresh",)

    clock.advance(11.0)
    assert cache.purge_expired() == 1
    assert len(cache) == 0


def test_expiry_is_by_creation_time_not_by_insertion_order():
    """An entry inserted late can be older than one inserted early.

    `created_at` travels with the state, so the purge scan cannot stop at the
    first live entry the way a pure insertion-ordered walk would.
    """
    clock = ManualClock()
    cache = SceneCache(max_entries=4, ttl_seconds=10.0, clock=clock)
    cache.put(_scene_state("fresh", created_at=clock.now))
    cache.put(_scene_state("already-stale", created_at=clock.now - 20.0))

    assert cache.keys() == ("fresh",)
    assert cache.get("already-stale") is None


# --------------------------------------------------------------------------- #
# Property 32 -- eviction releases memory (Requirement 12.3)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _StateRecord:
    """A state plus weak references to the arrays it held when inserted."""

    state: SceneState
    array_refs: tuple[weakref.ref, ...]

    @property
    def all_arrays_alive(self) -> bool:
        return all(ref() is not None for ref in self.array_refs)

    @property
    def any_array_alive(self) -> bool:
        return any(ref() is not None for ref in self.array_refs)


def _record(state: SceneState) -> _StateRecord:
    """Take weak references to every array the state holds.

    Weak, and never stored strongly anywhere in the test, so the cache's
    reference is the last one: if `release()` did not run, these stay alive and
    the assertion fails for the right reason.
    """
    plane = state.planes["floor"]
    return _StateRecord(
        state=state,
        array_refs=tuple(
            weakref.ref(array)
            for array in (
                state.image,
                state.foreground_mask,
                state.shading_map,
                state.detail_map,
                state.plane_masks["floor"],
                plane.contour,
                plane.bounding_points,
                plane.homography,
                plane.homography_inv,
            )
        ),
    )


def _survivors(cache: SceneCache) -> set[int]:
    """Identities of the states still reachable from `cache`."""
    return {id(cache.get(scene_id)) for scene_id in cache.keys()}


# Feature: ai-room-tile-visualizer, Property 32: Evicted scene state releases
# its memory
@_PROPERTY_SETTINGS
@given(
    max_entries=_max_entries,
    scene_ids=st.lists(_scene_id, min_size=2, max_size=24),
)
def test_property_32_evicted_state_releases_its_arrays(max_entries, scene_ids):
    """For any insertion sequence, every state no longer reachable from the
    cache has had its arrays released and holds no bytes, while every retained
    state still holds its arrays.

    Both halves matter: a cache that released eagerly on *every* put would pass
    the first assertion and break rendering, so the surviving states are checked
    for the opposite.

    **Validates: Requirements 12.3**
    """
    clock = ManualClock()
    cache = SceneCache(max_entries=max_entries, ttl_seconds=3600.0, clock=clock)

    records = []
    for scene_id in scene_ids:
        state = _scene_state(scene_id, created_at=clock.now)
        records.append(_record(state))
        cache.put(state)
        del state

    survivors = _survivors(cache)
    gc.collect()

    for record in records:
        if id(record.state) in survivors:
            assert record.all_arrays_alive, "a retained state was released while still cached"
            assert record.state.nbytes() > 0
        else:
            assert not record.any_array_alive, (
                f"arrays of the evicted state {record.state.scene_id!r} are still reachable"
            )
            assert record.state.nbytes() == 0
            assert record.state.image is None
            assert record.state.plane_masks == {}
            assert record.state.planes == {}


def test_ttl_eviction_releases_the_expired_state():
    clock = ManualClock()
    cache = SceneCache(max_entries=4, ttl_seconds=10.0, clock=clock)
    state = _scene_state("a", created_at=clock.now)
    record = _record(state)
    cache.put(state)
    del state

    clock.advance(11.0)
    assert cache.purge_expired() == 1
    gc.collect()

    assert not record.any_array_alive
    assert record.state.nbytes() == 0


def test_discard_releases_and_reports_whether_anything_went():
    clock = ManualClock()
    cache = SceneCache(max_entries=4, ttl_seconds=3600.0, clock=clock)
    state = _scene_state("a", created_at=clock.now)
    record = _record(state)
    cache.put(state)
    del state

    assert cache.discard("a") is True
    assert cache.discard("a") is False
    assert cache.discard("never-stored") is False
    gc.collect()

    assert not record.any_array_alive
    assert "a" not in cache
    assert len(cache) == 0


def test_clear_releases_every_entry():
    """`clear()` runs on shutdown, so it must not leave a scene's worth of
    arrays alive while the process drains."""
    clock = ManualClock()
    cache = SceneCache(max_entries=4, ttl_seconds=3600.0, clock=clock)
    records = []
    for scene_id in ("a", "b", "c"):
        state = _scene_state(scene_id, created_at=clock.now)
        records.append(_record(state))
        cache.put(state)
        del state

    cache.clear()
    gc.collect()

    assert len(cache) == 0
    assert cache.keys() == ()
    assert all(not record.any_array_alive for record in records)


# --------------------------------------------------------------------------- #
# Construction and introspection (Requirements 9.4, 12.5)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("max_entries", "ttl_seconds"),
    [(0, 10.0), (-1, 10.0), (4, 0.0), (4, -1.0)],
)
def test_constructor_rejects_bounds_that_cannot_work(max_entries, ttl_seconds):
    """Both bounds come from operator settings, so nonsense fails at startup
    rather than producing a cache that silently stores nothing."""
    with pytest.raises(ValueError):
        SceneCache(max_entries=max_entries, ttl_seconds=ttl_seconds)


def test_configured_bounds_are_readable():
    cache = SceneCache(max_entries=7, ttl_seconds=42.5)
    assert cache.max_entries == 7
    assert cache.ttl_seconds == 42.5
    assert len(cache) == 0
    assert cache.keys() == ()
    assert "anything" not in cache
    assert cache.get("anything") is None


def test_contains_rejects_non_string_keys():
    cache = SceneCache(max_entries=2, ttl_seconds=10.0)
    assert 1 not in cache
    assert None not in cache


def test_stats_keys_match_the_health_response_fields():
    """The health route splats `stats()` into `HealthResponse`, so a rename on
    either side has to break here rather than at runtime (Requirement 12.5)."""
    clock = ManualClock()
    cache = SceneCache(max_entries=3, ttl_seconds=1800.0, clock=clock)
    cache.put(_scene_state("a", created_at=clock.now))

    stats = cache.stats()

    assert set(stats) <= set(HealthResponse.model_fields)
    assert stats == {
        "scene_cache_entries": 1,
        "scene_cache_max_entries": 3,
        "scene_cache_ttl_seconds": 1800,
    }
    # Whole TTLs report as int, which is what HealthResponse declares.
    assert isinstance(stats["scene_cache_ttl_seconds"], int)


def test_stats_excludes_expired_entries_and_keeps_a_fractional_ttl_as_float():
    clock = ManualClock()
    cache = SceneCache(max_entries=3, ttl_seconds=2.5, clock=clock)
    cache.put(_scene_state("a", created_at=clock.now))
    clock.advance(3.0)

    stats = cache.stats()

    assert stats["scene_cache_entries"] == 0
    assert stats["scene_cache_ttl_seconds"] == 2.5


def test_repr_reports_the_live_count_and_both_bounds():
    cache = SceneCache(max_entries=3, ttl_seconds=1800.0)
    text = repr(cache)
    assert "SceneCache(" in text
    assert "entries=0" in text
    assert "max_entries=3" in text


# --------------------------------------------------------------------------- #
# Concurrency (task 11.5)
# --------------------------------------------------------------------------- #
#
# The lock in `backend/cache.py` is load-bearing rather than defensive: FastAPI
# runs sync path operations in a threadpool, so two worker threads really do
# call `put` and `get` at the same time, and `move_to_end`-on-hit is a
# read-modify-write that would corrupt a bare `OrderedDict`.
#
# These tests therefore drive genuine concurrent access and assert *invariants*,
# never a particular interleaving -- an assertion about which scene survives a
# race would be a flaky test rather than a stronger one. Three claims:
#
# * no exception escapes any thread,
# * the entry bound holds at every observation, from every thread, and
# * the cache is left consistent and still usable once the threads are done.
#
# Nothing here reads a state's arrays while other threads are running, on
# purpose. `SceneState.release()` clears the `planes` and `plane_masks` dicts,
# so calling `nbytes()` on a state a *different* thread is concurrently evicting
# can raise "dictionary changed size during iteration". That is a property of
# handing a caller a state the cache may evict underneath it, not a cache-lock
# bug, so array reachability is only asserted after every thread has joined.

_CONCURRENCY_THREADS = 8
_CONCURRENCY_OPS_PER_THREAD = 64

#: Every public method, so the workload contends on the whole locked surface
#: rather than only on `put`/`get`. `purge_expired` is a no-op under the frozen
#: clock below; it is here to exercise the lock, not the TTL.
_CONCURRENT_OP_KINDS = (
    "put",
    "get",
    "len",
    "keys",
    "stats",
    "contains",
    "put",
    "get",
    "discard",
    "purge_expired",
)


def test_concurrent_mixed_operations_hold_the_entry_bound_and_raise_nothing():
    """Under concurrent mixed access from many threads, the entry count never
    exceeds the configured maximum, no exception escapes any thread, and the
    cache is left consistent and usable.

    The clock is frozen and the time-to-live is far beyond anything the frozen
    clock can reach, so expiry cannot fire: any missing or extra entry is a
    locking bug rather than a TTL effect. The workload issues far more
    insertions than the bound allows, so eviction genuinely runs concurrently
    with reads instead of the test passing on an under-full cache.

    _Requirements: 9.5_
    """
    max_entries = 3
    clock = ManualClock()
    cache = SceneCache(max_entries=max_entries, ttl_seconds=3600.0, clock=clock)

    # Released together, so the threads overlap instead of running in sequence
    # while the pool is still spinning up its workers.
    start = threading.Barrier(_CONCURRENCY_THREADS)
    failures: list[str] = []
    failures_lock = threading.Lock()

    def worker(thread_index: int) -> tuple[int, int]:
        """Run one thread's share of the workload.

        Returns ``(puts_issued, max_length_observed)``. Invariant violations are
        recorded rather than raised so that one thread tripping cannot mask what
        the others were doing; escaping exceptions are surfaced by the futures.
        """
        puts = 0
        max_seen = 0
        start.wait(timeout=10)

        for step in range(_CONCURRENCY_OPS_PER_THREAD):
            kind = _CONCURRENT_OP_KINDS[(step + thread_index) % len(_CONCURRENT_OP_KINDS)]
            # Coprime stride so threads collide on ids at the same step, which
            # is where a read-modify-write race would actually show up.
            scene_id = _SCENE_IDS[(step * 5 + thread_index) % len(_SCENE_IDS)]
            observed: int | None = None

            if kind == "put":
                cache.put(_scene_state(scene_id, created_at=clock.now))
                puts += 1
            elif kind == "get":
                state = cache.get(scene_id)
                if state is not None and state.scene_id != scene_id:
                    with failures_lock:
                        failures.append(
                            f"get({scene_id!r}) returned state {state.scene_id!r}"
                        )
            elif kind == "len":
                observed = len(cache)
            elif kind == "keys":
                keys = cache.keys()
                observed = len(keys)
                if len(set(keys)) != len(keys):
                    with failures_lock:
                        failures.append(f"keys() returned duplicates: {keys!r}")
            elif kind == "stats":
                observed = int(cache.stats()["scene_cache_entries"])
            elif kind == "contains":
                scene_id in cache
            elif kind == "discard":
                cache.discard(scene_id)
            else:
                cache.purge_expired()

            if observed is not None:
                max_seen = max(max_seen, observed)
                if observed > max_entries:
                    with failures_lock:
                        failures.append(
                            f"observed {observed} entries, above the bound of {max_entries}"
                        )

        return puts, max_seen

    with ThreadPoolExecutor(max_workers=_CONCURRENCY_THREADS) as pool:
        futures = [pool.submit(worker, i) for i in range(_CONCURRENCY_THREADS)]
        # `result()` re-raises anything the thread raised, which is the "no
        # exception escapes the lock" half of the claim.
        results = [future.result(timeout=60) for future in futures]

    assert not failures, "concurrent invariant violations: " + "; ".join(failures)

    total_puts = sum(puts for puts, _ in results)
    assert total_puts > max_entries, (
        "the workload never overfilled the cache, so eviction was not exercised"
    )

    # Consistency, now single-threaded: one authoritative length, agreed on by
    # every accessor, with no duplicate or unretrievable key.
    keys = cache.keys()
    assert len(cache) == len(keys) <= max_entries
    assert len(set(keys)) == len(keys)
    assert cache.stats()["scene_cache_entries"] == len(keys)
    for scene_id in keys:
        assert scene_id in cache
        state = cache.get(scene_id)
        assert state is not None, f"{scene_id!r} was listed by keys() but did not resolve"
        assert state.scene_id == scene_id
        # A live entry whose arrays were released would mean a concurrent
        # eviction released a state it did not remove.
        assert state.nbytes() > 0, f"live entry {scene_id!r} had already been released"

    # Still usable: the bound, the ordering, and eviction all behave afterwards.
    for scene_id in _SCENE_IDS:
        cache.put(_scene_state(scene_id, created_at=clock.now))
    assert cache.keys() == _SCENE_IDS[-max_entries:]
    assert len(cache) == max_entries


def test_concurrent_puts_of_one_scene_id_leave_exactly_one_live_state():
    """Racing insertions under a single id must not lose or leak a state.

    Every displaced state is the cache's own, so `put` releases it; the survivor
    must not be released. Which thread wins is a race, so the assertions are on
    the counts: exactly one state alive, every other one released, and the
    survivor is the state the cache still hands out.

    _Requirements: 9.5_
    """
    clock = ManualClock()
    cache = SceneCache(max_entries=4, ttl_seconds=3600.0, clock=clock)
    start = threading.Barrier(_CONCURRENCY_THREADS)

    def worker(_: int) -> _StateRecord:
        state = _scene_state("hot", created_at=clock.now)
        record = _record(state)
        start.wait(timeout=10)
        cache.put(state)
        del state  # the cache holds the only strong reference from here
        return record

    with ThreadPoolExecutor(max_workers=_CONCURRENCY_THREADS) as pool:
        records = [
            future.result(timeout=60)
            for future in [pool.submit(worker, i) for i in range(_CONCURRENCY_THREADS)]
        ]

    assert len(cache) == 1
    assert cache.keys() == ("hot",)
    survivor = cache.get("hot")
    assert survivor is not None
    gc.collect()

    alive = [record for record in records if record.any_array_alive]
    assert len(alive) == 1, f"{len(alive)} states still hold arrays, expected exactly 1"
    assert alive[0].state is survivor, "the live state is not the one the cache returns"
    assert alive[0].all_arrays_alive
    assert survivor.nbytes() > 0
    for record in records:
        if record.state is not survivor:
            assert record.state.nbytes() == 0, "a displaced state was not released"
