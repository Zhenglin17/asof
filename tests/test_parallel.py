"""Tests for ``asof.parallel.parallel_map``: run ``fn(context, unit)`` over ``units`` either
serially in this process (``jobs=1``, the reference) or in a fork-context process pool, with
results that never depend on the number of workers.

API:

    asof.parallel.parallel_map(fn, context, units, *, jobs=1, progress=None) -> list[R]
        fn(context, unit) -> R; ``context`` is shared read-only input that forked workers
        inherit (never pickled; it may hold a lock or a lambda, and ``fn`` may be a closure);
        only units and results cross the pipe. Results in the order of ``units``.
        jobs == 1: in this process, serially. jobs > 1: a pool of min(jobs, len(units)) forked
        workers. jobs < 1: ValueError. No units: [] and no child process.
        progress(unit, result): once per unit, in unit order, in the parent, as soon as that
        unit and every earlier one are done.
        A call while another is in progress (from fn, from progress) raises RuntimeError. The
        module keeps no reference to fn / context once a call ends, normally or by an error.
        An error in fn for unit k reaches the caller with its type and message, after progress
        saw every unit before k; queued units are cancelled and this call's running workers
        terminated (a stuck worker is not waited on); other processes are left alone. A worker
        that dies surfaces as BrokenProcessPool.

No ``as_of`` here: the module moves no time-stamped data, so there is no look-ahead case to
plant. "Leak" below means state leaking across units or across calls (a worker's mutation of
the context, a previous call's context). Assumptions only real data can confirm:
- The fork-inherited inputs are the parent's at the moment of the call and are not mutated by
  any unit: check with ``asof market truncation --jobs 1`` versus ``--jobs N`` (N = 2 and the
  default) on the real store; every audit parquet must be byte-equal (same rows, same order).
- Forking a parent that holds the real store (~1 GB of bars) and has started DuckDB / pyarrow
  threads does not deadlock a worker: the real truncation run with ``--jobs N`` completes.
- Standard cases ride along with that run: BK / BNY at 2022-06-01 and BBBY at 2025-08-07 show no
  diff in both the serial and the parallel output.
"""

import gc
import multiprocessing
import os
import signal
import threading
import time
import weakref
from collections.abc import Iterator
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from typing import Any

import pyarrow as pa
import pytest

from asof.parallel import parallel_map

# =============================================================================================
# helpers
# =============================================================================================


@contextmanager
def within(seconds: float) -> Iterator[None]:
    """Fail instead of hanging: a pool that waits forever on a stuck worker raises here."""

    def expire(signum: int, frame: Any) -> None:
        raise TimeoutError(f"no answer within {seconds} s")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def children_left_besides(keep: set[int], seconds: float = 10.0) -> list[int]:
    """Pids of live multiprocessing children other than ``keep``, after waiting up to
    ``seconds`` for them to go."""
    deadline = time.monotonic() + seconds
    while True:
        left = [c.pid for c in multiprocessing.active_children() if c.pid not in keep]
        if not left or time.monotonic() > deadline:
            return [p for p in left if p is not None]
        time.sleep(0.05)


def no_children_left(seconds: float = 10.0) -> bool:
    return not children_left_besides(set(), seconds)


class PlantedError(Exception):
    """Module level so a worker can pickle it back to the parent."""


class Context:
    """A weak-referenceable context object."""

    def __init__(self, scale: int, names: dict[int, str]) -> None:
        self.scale = scale
        self.names = names
        self.seen: list[int] = []


def table_of(ctx: Context, unit: int) -> pa.Table:
    """A result that depends on both the context and the unit."""
    n = unit % 5 + 1
    return pa.table(
        {
            "unit": pa.array([unit] * n, pa.int64()),
            "value": pa.array([unit * ctx.scale + i for i in range(n)], pa.int64()),
            "name": pa.array([ctx.names.get(unit, "?")] * n, pa.string()),
        }
    )


def record_of(ctx: Context, unit: int) -> dict[str, Any]:
    return {"unit": unit, "scaled": unit * ctx.scale, "name": ctx.names.get(unit), "sq": [unit] * 3}


def make_context(scale: int = 7) -> Context:
    return Context(scale, {u: f"u{u}" for u in range(0, 40, 3)})


@pytest.fixture
def forks(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts os.fork calls made by this process (multiprocessing's fork popen calls it)."""
    calls: list[int] = []
    real = os.fork
    parent = os.getpid()

    def counting_fork() -> int:
        if os.getpid() == parent:
            calls.append(1)
        return real()

    monkeypatch.setattr(os, "fork", counting_fork)
    return calls


@pytest.fixture
def no_fork(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> int:
        raise AssertionError("os.fork called where no child process may be created")

    monkeypatch.setattr(os, "fork", refuse)


# =============================================================================================
# normal
# =============================================================================================


def test_serial_runs_in_this_process_in_unit_order(no_fork: None) -> None:
    pid = os.getpid()
    out = parallel_map(lambda ctx, u: (u, ctx.scale * u, os.getpid()), make_context(), [3, 1, 2])
    assert out == [(3, 21, pid), (1, 7, pid), (2, 14, pid)]


def test_default_jobs_is_serial(no_fork: None) -> None:
    out = parallel_map(lambda ctx, u: os.getpid(), make_context(), [0, 1, 2, 3])
    assert out == [os.getpid()] * 4


@pytest.mark.parametrize("jobs", [2, 4])
def test_pool_runs_in_other_processes(jobs: int) -> None:
    with within(30):
        pids = parallel_map(lambda ctx, u: os.getpid(), make_context(), list(range(6)), jobs=jobs)
    assert os.getpid() not in pids
    assert no_children_left()


@pytest.mark.parametrize("jobs", [2, 4])
def test_pool_tables_equal_serial_element_by_element(jobs: int) -> None:
    ctx = make_context()
    units = list(range(17))
    serial = parallel_map(table_of, ctx, units, jobs=1)
    with within(30):
        pooled = parallel_map(table_of, ctx, units, jobs=jobs)
    assert len(pooled) == len(serial) == len(units)
    assert all(t.num_rows > 0 for t in serial)
    for u, a, b in zip(units, serial, pooled, strict=True):
        assert a.schema.equals(b.schema), u
        assert a.equals(b), u


@pytest.mark.parametrize("jobs", [2, 4])
def test_pool_records_equal_serial(jobs: int) -> None:
    ctx = make_context(scale=11)
    units = [5, 0, 9, 3, 3, 12, 1]
    serial = parallel_map(record_of, ctx, units)
    with within(30):
        assert parallel_map(record_of, ctx, units, jobs=jobs) == serial
    assert serial[2] == {"unit": 9, "scaled": 99, "name": "u9", "sq": [9, 9, 9]}


def test_pool_results_come_in_unit_order_not_completion_order() -> None:
    # unit 0 sleeps longest, unit 3 finishes first
    units = [0, 1, 2, 3]
    finished: list[int] = []

    def slow_first(ctx: Context, u: int) -> int:
        time.sleep(0.1 * (len(units) - u))
        return u * ctx.scale

    with within(30):
        out = parallel_map(
            slow_first, make_context(), units, jobs=4, progress=lambda u, r: finished.append(u)
        )
    assert out == [0, 7, 14, 21]
    assert finished == units


@pytest.mark.parametrize("jobs", [1, 2, 4])
def test_progress_once_per_unit_in_order_in_the_parent(jobs: int) -> None:
    parent = os.getpid()
    seen: list[tuple[int, int, int]] = []
    units = [4, 2, 7, 2, 0]

    def note(u: int, r: int) -> None:
        seen.append((u, r, os.getpid()))

    with within(30):
        out = parallel_map(
            lambda ctx, u: u * ctx.scale, make_context(), units, jobs=jobs, progress=note
        )
    assert out == [28, 14, 49, 14, 0]
    assert seen == [(u, r, parent) for u, r in zip(units, out, strict=True)]


@pytest.mark.parametrize("jobs", [1, 2])
def test_fn_may_be_a_closure_and_context_unpicklable(jobs: int) -> None:
    # a lock and a lambda cannot be pickled: the pool must hand them to workers by fork
    offset = 100
    ctx = {"lock": threading.Lock(), "f": lambda u: u * 3}

    def fn(c: dict[str, Any], u: int) -> int:
        with c["lock"]:
            return c["f"](u) + offset

    with within(30):
        assert parallel_map(fn, ctx, [1, 2, 3], jobs=jobs) == [103, 106, 109]


# =============================================================================================
# edge
# =============================================================================================


@pytest.mark.parametrize("jobs", [0, -1])
@pytest.mark.parametrize("units", [[], [1, 2]], ids=["empty", "two"])
def test_refuses_fewer_than_one_job(jobs: int, units: list[int], no_fork: None) -> None:
    with pytest.raises(ValueError):
        parallel_map(lambda ctx, u: u, make_context(), units, jobs=jobs)


@pytest.mark.parametrize("jobs", [1, 2, 8])
def test_no_units_is_empty_and_creates_no_process(jobs: int, no_fork: None) -> None:
    called: list[Any] = []
    out = parallel_map(
        lambda ctx, u: called.append(u),
        make_context(),
        [],
        jobs=jobs,
        progress=lambda u, r: called.append(u),
    )
    assert out == []
    assert called == []
    assert multiprocessing.active_children() == []


def test_more_jobs_than_units_forks_one_worker_per_unit(forks: list[int]) -> None:
    with within(30):
        out = parallel_map(table_of, make_context(), [3, 6], jobs=8)
    assert [t.column("unit").to_pylist()[0] for t in out] == [3, 6]
    assert 1 <= len(forks) <= 2
    assert no_children_left()


def test_one_unit_with_many_jobs() -> None:
    with within(30):
        assert parallel_map(record_of, make_context(), [9], jobs=4) == [
            record_of(make_context(), 9)
        ]


@pytest.mark.parametrize("jobs", [1, 3])
def test_duplicate_units_are_kept_one_result_each(jobs: int) -> None:
    with within(30):
        assert parallel_map(lambda c, u: u + c.scale, make_context(), [2, 2, 5, 2], jobs=jobs) == [
            9,
            9,
            12,
            9,
        ]


@pytest.mark.parametrize("jobs", [1, 2])
def test_a_call_from_inside_fn_is_refused(jobs: int) -> None:
    def nested(ctx: Context, u: int) -> str:
        try:
            parallel_map(lambda c, v: v, ctx, [u])
        except RuntimeError:
            return "refused"
        return "ran"

    with within(30):
        assert parallel_map(nested, make_context(), [1, 2], jobs=jobs) == ["refused", "refused"]


@pytest.mark.parametrize("jobs", [1, 2])
def test_a_call_from_progress_is_refused(jobs: int) -> None:
    # progress runs in the parent while the outer call is still in progress
    outcomes: list[str] = []

    def nested(u: int, r: int) -> None:
        try:
            parallel_map(lambda c, v: v, make_context(), [u])
        except RuntimeError:
            outcomes.append("refused")
        else:
            outcomes.append("ran")

    with within(30):
        assert parallel_map(lambda c, u: u, make_context(), [1, 2], jobs=jobs, progress=nested) == [
            1,
            2,
        ]
    assert outcomes == ["refused", "refused"]


@pytest.mark.parametrize("jobs", [1, 2, 4])
def test_an_error_reaches_the_caller_after_progress_for_every_earlier_unit(jobs: int) -> None:
    # earlier units are slow, the failing unit 2 fails at once: the error must still wait for
    # units 0 and 1, and progress must have seen exactly them
    seen: list[int] = []

    def fn(ctx: Context, u: int) -> int:
        if u == 2:
            raise PlantedError(f"planted failure at unit {u}")
        if u < 2:
            time.sleep(0.3)
        return u

    with within(30), pytest.raises(PlantedError, match="planted failure at unit 2"):
        parallel_map(
            fn, make_context(), [0, 1, 2, 3, 4], jobs=jobs, progress=lambda u, r: seen.append(u)
        )
    assert seen == [0, 1]
    assert no_children_left()


@pytest.mark.parametrize("error", [ValueError, KeyError, PlantedError])
def test_an_error_keeps_its_type_in_a_pool(error: type[Exception]) -> None:
    def fn(ctx: Context, u: int) -> int:
        if u == 1:
            raise error("planted")
        return u

    with within(30), pytest.raises(error, match="planted"):
        parallel_map(fn, make_context(), [0, 1, 2], jobs=2)
    assert no_children_left()


def test_an_error_does_not_wait_on_a_stuck_worker() -> None:
    # unit 0 fails, unit 1 never returns: the error must arrive and the stuck worker must be
    # gone, not waited on
    def fn(ctx: Context, u: int) -> int:
        if u == 0:
            time.sleep(0.2)  # let the other worker pick up unit 1 first
            raise PlantedError("planted failure at unit 0")
        time.sleep(1e6)
        return u

    start = time.monotonic()
    with within(20), pytest.raises(PlantedError, match="planted failure"):
        parallel_map(fn, make_context(), [0, 1, 2], jobs=2)
    assert time.monotonic() - start < 5
    assert no_children_left()


def test_an_error_terminates_only_its_own_workers() -> None:
    bystander = multiprocessing.get_context("fork").Process(target=time.sleep, args=(60,))
    bystander.start()
    try:

        def fn(ctx: Context, u: int) -> int:
            if u == 0:
                time.sleep(0.2)
                raise PlantedError("planted failure at unit 0")
            time.sleep(1e6)
            return u

        with within(20), pytest.raises(PlantedError):
            parallel_map(fn, make_context(), [0, 1], jobs=2)
        assert bystander.pid is not None
        assert children_left_besides({bystander.pid}) == []
        assert bystander.is_alive()
    finally:
        bystander.terminate()
        bystander.join(timeout=10)
    assert no_children_left()


def test_a_dead_worker_is_a_broken_pool() -> None:
    def fn(ctx: Context, u: int) -> int:
        if u == 1:
            os._exit(3)
        return u

    with within(30), pytest.raises(BrokenProcessPool):
        parallel_map(fn, make_context(), [0, 1, 2, 3], jobs=2)
    assert no_children_left()


@pytest.mark.parametrize("jobs", [1, 2])
def test_a_call_after_an_error_works(jobs: int) -> None:
    def failing(ctx: Context, u: int) -> int:
        raise PlantedError("first call fails")

    # Several rounds: a pool abandoned with threads still running when the next call forks
    # can hand a worker a lock that no thread will release, and that worker never exits
    # (seen ~1 run in 25 with a single round when the error path left the pool's manager
    # thread alive).
    with within(30):
        for scale in range(1, 9):
            with pytest.raises(PlantedError):
                parallel_map(failing, make_context(), [0, 1], jobs=jobs)
            out = parallel_map(lambda c, u: u * c.scale, make_context(scale), [1, 2], jobs=jobs)
            assert out == [scale, 2 * scale]
    assert no_children_left()


def manager_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if type(t).__name__ == "_ExecutorManagerThread"]


def test_an_error_leaves_no_manager_thread_behind() -> None:
    # a pool's manager thread still running when the next call forks can leave a worker holding
    # a lock no thread releases: the error must not return before that thread has ended
    def fn(ctx: Context, u: int) -> int:
        if u == 0:
            raise PlantedError("planted")
        time.sleep(1e6)
        return u

    with within(30):
        for _ in range(10):
            with pytest.raises(PlantedError):
                parallel_map(fn, make_context(), [0, 1, 2], jobs=2)
            assert manager_threads() == []
    assert no_children_left()


def test_ctrl_c_ends_the_workers_and_the_next_call_works() -> None:
    def interrupt(u: int, r: int) -> None:
        raise KeyboardInterrupt

    def slow(ctx: Context, u: int) -> int:
        if u > 0:
            time.sleep(1e6)
        return u

    with within(30):
        with pytest.raises(KeyboardInterrupt):
            parallel_map(slow, make_context(), [0, 1, 2], jobs=2, progress=interrupt)
        assert no_children_left()
        assert manager_threads() == []
        assert parallel_map(lambda c, u: u * c.scale, make_context(), [1, 2], jobs=2) == [7, 14]


def test_workers_ignore_sigint_the_parent_alone_handles_it() -> None:
    # Ctrl-C reaches the whole foreground group: a worker that died of it first would make the
    # parent report a dead worker ("out of memory") instead of the interrupt
    with within(30):
        handlers = parallel_map(
            lambda c, u: signal.getsignal(signal.SIGINT) is signal.SIG_IGN,
            make_context(),
            [0, 1],
            jobs=2,
        )
    assert handlers == [True, True]
    assert signal.getsignal(signal.SIGINT) is not signal.SIG_IGN


# =============================================================================================
# leakage (across units and across calls; there is no as_of here)
# =============================================================================================


def test_a_pure_fn_over_many_units_is_the_same_whatever_the_worker_count() -> None:
    # 4 workers each run several units in turn: nothing a worker did for one unit may change
    # another unit's result
    ctx = make_context(scale=13)
    units = list(range(40))
    serial = parallel_map(table_of, ctx, units)
    for jobs in (2, 4):
        with within(30):
            pooled = parallel_map(table_of, ctx, units, jobs=jobs)
        for u, a, b in zip(units, serial, pooled, strict=True):
            assert a.equals(b), (jobs, u)


def test_a_workers_mutation_of_the_context_does_not_reach_the_parent() -> None:
    ctx = make_context()

    def mutate(c: Context, u: int) -> int:
        c.seen.append(u)
        c.scale = -1
        return u

    with within(30):
        assert parallel_map(mutate, ctx, [0, 1, 2, 3], jobs=2) == [0, 1, 2, 3]
    assert ctx.seen == []
    assert ctx.scale == 7


def test_a_second_call_sees_its_own_context_not_the_first() -> None:
    with within(30):
        first = parallel_map(lambda c, u: u * c.scale, make_context(2), [1, 2, 3], jobs=2)
        second = parallel_map(lambda c, u: u * c.scale, make_context(5), [1, 2, 3], jobs=2)
    assert (first, second) == ([2, 4, 6], [5, 10, 15])


@pytest.mark.parametrize("jobs", [1, 2])
@pytest.mark.parametrize("fails", [False, True], ids=["ok", "error"])
def test_the_module_keeps_no_reference_to_the_context(jobs: int, fails: bool) -> None:
    ctx = make_context()
    ref = weakref.ref(ctx)

    def fn(c: Context, u: int) -> int:
        if fails and u == 1:
            raise PlantedError("planted")
        return u

    with within(30):
        try:
            parallel_map(fn, ctx, [0, 1, 2], jobs=jobs)
        except PlantedError:  # not pytest.raises: its traceback would keep the frames alive
            assert fails
    del ctx
    gc.collect()
    assert ref() is None
