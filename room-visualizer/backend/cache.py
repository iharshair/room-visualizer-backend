"""Scene_Cache -- the bounded handoff between the analysis and render passes.

The two-pass split means a Scene_State produced by ``/api/segment`` has to
survive between requests so that every subsequent ``/api/render`` can reuse the
masks, homographies, and lighting maps instead of recomputing them (Requirement
9.2). This module is that store, and it is deliberately the *only* place in the
service where a Scene_State outlives a request.

Two bounds keep it honest, because a Scene_State for a 2048 pixel photograph is
tens of megabytes and an unbounded dictionary of them is a memory leak with a
polite name:

* a **least-recently-used entry bound** (Requirement 9.5, default 32), and
* a **time-to-live bound** (Requirement 9.6, default 1800 seconds).

Three design details are worth calling out:

**Eviction releases eagerly.** Every path that drops an entry -- LRU overflow,
TTL expiry, an explicit :meth:`discard`, or :meth:`clear` -- calls
:meth:`~backend.schemas.SceneState.release` before dropping the reference. That
nulls the state's array attributes, so the cache's last strong reference to each
buffer goes away immediately and CPython frees it at once rather than whenever
the next garbage-collection cycle happens to run (Requirement 12.3). Releasing
before dropping also means a caller still holding the evicted state cannot keep
tens of megabytes alive by accident.

**The lock is load-bearing, not defensive.** FastAPI runs sync path operations
in a threadpool, so two worker threads really do call :meth:`put` and
:meth:`get` concurrently. A bare ``OrderedDict`` would be corrupted by the
read-modify-write in ``move_to_end``-on-hit, so every public method takes a
``threading.RLock`` (re-entrant because the eviction helpers are called from
already-locked code).

**The clock is injectable.** TTL behaviour is otherwise only testable by
sleeping, which is slow and flaky. ``clock`` defaults to :func:`time.time`; the
tests pass a callable they advance by hand.

Ages are measured from ``SceneState.created_at``, the wall clock captured when
the analysis pass built the state, so a state's age is its real age rather than
the age of its most recent cache insertion.

Requirements: 9.4, 9.5, 9.6, 12.3, 12.5.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Callable

from backend.schemas import SceneState

__all__ = ["SceneCache"]


class SceneCache:
    """A process-local, thread-safe, LRU- and TTL-bounded Scene_State store.

    Process-local is a real limitation, not an oversight: a Scene_State holds
    numpy arrays that cannot cross a process boundary cheaply, so the service is
    intended to run with a single uvicorn worker unless a shared cache is
    introduced (Requirement 9.7, documented in the README).

    A miss -- whether the ``scene_id`` was never stored, was evicted by the LRU
    bound, or aged past the TTL -- is reported the same way, as ``None``, which
    the render route turns into HTTP 404 ``scene_expired`` (Requirement 9.4).
    """

    def __init__(
        self,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Build an empty cache.

        Args:
            max_entries: Maximum live entries; the least recently used entry is
                evicted once an insertion would exceed it. Must be at least 1.
            ttl_seconds: Maximum age, in seconds, of a retrievable entry. Must
                be positive.
            clock: Monotonically non-decreasing source of "now", in the same
                epoch as ``SceneState.created_at``. Injected by tests.

        Raises:
            ValueError: If ``max_entries`` is below 1 or ``ttl_seconds`` is not
                positive. Both bounds come from operator-supplied settings, so
                rejecting nonsense here turns a silently broken cache into a
                startup failure.
        """
        if max_entries < 1:
            raise ValueError(f"max_entries must be at least 1, got {max_entries!r}")
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive, got {ttl_seconds!r}")

        self._max_entries = int(max_entries)
        self._ttl_seconds = float(ttl_seconds)
        self._clock = clock
        # Insertion-ordered; the front is the least recently used entry.
        self._entries: OrderedDict[str, SceneState] = OrderedDict()
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Configuration, read-only
    # ------------------------------------------------------------------ #

    @property
    def max_entries(self) -> int:
        """Configured LRU bound (Requirement 9.5)."""
        return self._max_entries

    @property
    def ttl_seconds(self) -> float:
        """Configured time-to-live in seconds (Requirement 9.6)."""
        return self._ttl_seconds

    # ------------------------------------------------------------------ #
    # Core operations
    # ------------------------------------------------------------------ #

    def put(self, scene: SceneState) -> None:
        """Store ``scene`` under its own ``scene_id`` as the most recent entry.

        Expired entries are purged first, so a burst of insertions cannot keep
        stale multi-megabyte states resident just because nobody read them.
        Re-inserting an id replaces the previous state and releases it, since
        the cache was that state's owner.

        Once the insertion is in place, entries are popped from the front while
        the count exceeds ``max_entries`` (Requirement 9.5). The just-inserted
        entry sits at the back, so it is never the one evicted.
        """
        scene_id = scene.scene_id
        with self._lock:
            self._purge_expired_locked()

            previous = self._entries.pop(scene_id, None)
            self._entries[scene_id] = scene
            if previous is not None and previous is not scene:
                previous.release()

            while len(self._entries) > self._max_entries:
                _, evicted = self._entries.popitem(last=False)
                evicted.release()

    def get(self, scene_id: str) -> SceneState | None:
        """Return the live state for ``scene_id``, or ``None`` on a miss.

        Expired entries are purged before the lookup, so an entry past its TTL
        is a miss even though it was physically still present (Requirement 9.6).
        A hit is moved to the back to mark it as most recently used, which is
        what makes repeated renders of one scene keep that scene resident.
        """
        with self._lock:
            self._purge_expired_locked()
            scene = self._entries.get(scene_id)
            if scene is None:
                return None
            self._entries.move_to_end(scene_id)
            return scene

    def discard(self, scene_id: str) -> bool:
        """Explicitly evict one entry, releasing its arrays.

        Returns ``True`` if an entry was removed, ``False`` if the id was not
        present. This is the explicit eviction path, and like every other one it
        calls ``release()`` before dropping the reference (Requirement 12.3).
        """
        with self._lock:
            scene = self._entries.pop(scene_id, None)
            if scene is None:
                return False
            scene.release()
            return True

    def clear(self) -> None:
        """Evict every entry, releasing each one's arrays.

        Called on application shutdown so the process does not hold tens of
        megabytes per cached scene while it drains.
        """
        with self._lock:
            while self._entries:
                _, scene = self._entries.popitem(last=False)
                scene.release()

    def purge_expired(self) -> int:
        """Evict every entry older than the TTL and return how many went.

        :meth:`get`, :meth:`put`, :meth:`__len__`, :meth:`__contains__`, and
        :meth:`stats` all purge on their own, so calling this is only needed to
        reclaim memory on an idle service.
        """
        with self._lock:
            return self._purge_expired_locked()

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        """Live entry count, expired entries excluded (and purged)."""
        with self._lock:
            self._purge_expired_locked()
            return len(self._entries)

    def __contains__(self, scene_id: object) -> bool:
        """Report whether ``scene_id`` is live, without affecting recency.

        Deliberately not a substitute for :meth:`get`: membership does not mark
        the entry as recently used, so a probe cannot rescue an entry from LRU
        eviction.
        """
        if not isinstance(scene_id, str):
            return False
        with self._lock:
            self._purge_expired_locked()
            return scene_id in self._entries

    def keys(self) -> tuple[str, ...]:
        """Snapshot of live scene ids, least recently used first.

        A tuple rather than a view, so callers can iterate without holding the
        lock and without racing a concurrent eviction.
        """
        with self._lock:
            self._purge_expired_locked()
            return tuple(self._entries)

    def stats(self) -> dict[str, int | float]:
        """Cache facts for ``GET /api/health`` (Requirement 12.5).

        Keys match the ``HealthResponse`` field names exactly, so the route can
        splat this straight into the model:

        * ``scene_cache_entries`` -- live entry count
        * ``scene_cache_max_entries`` -- configured LRU bound
        * ``scene_cache_ttl_seconds`` -- configured TTL, as an ``int`` when the
          configured value is whole, which it is for every value that can come
          from ``Settings``
        """
        with self._lock:
            self._purge_expired_locked()
            ttl = self._ttl_seconds
            return {
                "scene_cache_entries": len(self._entries),
                "scene_cache_max_entries": self._max_entries,
                "scene_cache_ttl_seconds": int(ttl) if ttl.is_integer() else ttl,
            }

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        with self._lock:
            return (
                f"SceneCache(entries={len(self._entries)}, "
                f"max_entries={self._max_entries}, ttl_seconds={self._ttl_seconds})"
            )

    # ------------------------------------------------------------------ #
    # Internals -- callers must already hold the lock
    # ------------------------------------------------------------------ #

    def _purge_expired_locked(self) -> int:
        """Drop every entry whose age exceeds the TTL; return the count.

        Age is ``clock() - created_at``, and the bound is exclusive: an entry
        exactly at the TTL is still retrievable, matching Requirement 9.6's
        "age exceeds". Expiry is by creation time, not by position, so the scan
        cannot stop at the first live entry the way a pure insertion-ordered
        walk could.
        """
        now = self._clock()
        expired = [
            scene_id
            for scene_id, scene in self._entries.items()
            if now - scene.created_at > self._ttl_seconds
        ]
        for scene_id in expired:
            self._entries.pop(scene_id).release()
        return len(expired)
