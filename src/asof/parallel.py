"""Run one function over many units, serially or in forked worker processes, with a result that
never depends on the number of workers.

Rules a unit's function keeps so that it does not (the pool cannot check them): read the shared
context, never change it; return its own result, write no file; sort anything whose order is
not fixed (sets, dict views of unordered input, DuckDB results without ORDER BY). Each caller
keeps a test that ``jobs=1`` and ``jobs>1`` give equal results on data with non-empty results.
"""

import multiprocessing
import signal
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

# (fn, context) of the call in progress: set in the parent before the pool forks, so every
# worker inherits them (1 GB of bars, a closure) instead of receiving them through a pipe.
_CALL: tuple[Callable[[Any, Any], Any], Any] | None = None


def _run(unit: Any) -> Any:
    if _CALL is None:
        raise RuntimeError("_run runs only inside parallel_map, which sets _CALL")
    fn, context = _CALL
    return fn(context, unit)


def parallel_map[C, U, R](
    fn: Callable[[C, U], R],
    context: C,
    units: Sequence[U],
    *,
    jobs: int = 1,
    progress: Callable[[U, R], None] | None = None,
) -> list[R]:
    """``fn(context, unit)`` for each unit, in the order of ``units``. ``jobs=1`` runs in this
    process (the reference); ``jobs>1`` runs in min(jobs, len(units)) forked workers that
    inherit ``fn`` and ``context``, so only units and results are pickled. ``progress(unit,
    result)`` runs here, in unit order, once that unit and every earlier one are done. An error
    is raised once every earlier unit is done (a worker that dies fails every unit not done
    yet); queued units are cancelled and this call's workers terminated, never waited on.
    Workers ignore Ctrl-C: the parent handles it like any error. One call at a time, from one
    thread: another call made while one is in progress (from ``fn`` or ``progress``) raises
    RuntimeError."""
    global _CALL
    if jobs < 1:
        raise ValueError(f"jobs must be at least 1, got {jobs}")
    if _CALL is not None:
        raise RuntimeError("parallel_map is already running in this process")
    if not units:
        return []
    _CALL = (fn, context)
    try:
        out: list[R] = []
        if jobs == 1:
            for unit in units:
                out.append(fn(context, unit))
                if progress is not None:
                    progress(unit, out[-1])
            return out
        pool = ProcessPoolExecutor(
            max_workers=min(jobs, len(units)),
            mp_context=multiprocessing.get_context("fork"),
            initializer=_ignore_sigint,
        )
        try:
            # in the order of units, whatever finishes first
            for unit, result in zip(units, pool.map(_run, units), strict=True):
                out.append(result)
                if progress is not None:
                    progress(unit, result)
        except BaseException:
            _abandon(pool)
            raise
        pool.shutdown(wait=True)
        return out
    finally:
        _CALL = None


def _ignore_sigint() -> None:
    # Ctrl-C reaches the whole foreground group; a worker dying of it first would make the
    # parent report a dead worker instead of the interrupt
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _abandon(pool: ProcessPoolExecutor) -> None:
    """Stop ``pool`` without waiting on a stuck worker: cancel what is queued, end this pool's
    workers (no other child of this process), then wait for the pool's manager thread, which
    sees them gone, fails what is left and reaps them. A manager thread still running when a
    later call forks can hand that call's worker a lock no thread will release, and the worker
    never exits. ``shutdown`` drops its handle on the thread even with ``wait=False``, so it is
    taken first (both are private, but the pool's only handles)."""
    workers = list((pool._processes or {}).values())
    # typeshed types this attribute as the wakeup pipe; at run time it is the thread
    manager: threading.Thread | None = getattr(pool, "_executor_manager_thread", None)
    pool.shutdown(wait=False, cancel_futures=True)
    for worker in workers:
        worker.terminate()
    if manager is not None:
        manager.join(timeout=30)
    for worker in workers:  # only if the manager did not get to them
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=10)
    if manager is not None and manager.is_alive():
        raise RuntimeError("the pool's manager thread did not end after its workers were killed")
