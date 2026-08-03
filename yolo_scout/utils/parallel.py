"""Bounded parallel iteration over worker processes."""

from collections import deque
from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any

# Items allowed in flight per worker. `Pool.imap` applies no backpressure: workers race
# through the whole input and every result sits in the parent until consumed, which on a
# large dataset means holding all of it in memory at once.
PENDING_ITEMS_PER_WORKER = 4


def _prefetched(items: Iterable[Any], size: int) -> Iterator[Any]:
    """Yield from `items` while a background thread keeps a bounded buffer filled.

    Pulling each payload from a database cursor costs real time. Fetched inline it lands on
    the consumer's critical path, so the workers and the GPU sit idle through every fetch;
    across a dataset-sized stream that adds up to minutes. The buffer is bounded, so
    overlapping the fetch does not reintroduce whole-dataset residency.
    """
    buffer: Queue = Queue(maxsize=size)
    stop = Event()
    end = object()

    def fill() -> None:
        try:
            for item in items:
                if stop.is_set():
                    return
                buffer.put(item)
        except BaseException as error:  # noqa: BLE001 - handed to the consumer to raise
            buffer.put(error)
        else:
            buffer.put(end)

    thread = Thread(target=fill, daemon=True)
    thread.start()

    try:
        while True:
            item = buffer.get()
            if item is end:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        # Retire the thread before returning: callers fork a fresh pool per split, and
        # forking while a thread holds a lock is how children deadlock.
        stop.set()
        while thread.is_alive():
            try:
                buffer.get_nowait()
            except Empty:
                thread.join(timeout=0.05)


def imap_workers(
    func: Callable[[Any], Any],
    items: Iterable[Any],
    initializer: Callable[[], None] | None = None,
) -> Iterator[Any]:
    """Map `func` over `items` across worker processes, yielding results in order.

    Only a bounded number of results may pile up unconsumed, so a consumer slower than the
    workers (model inference, database writes) cannot make the pool accumulate the entire
    input in the parent. `items` is pulled lazily on a background thread, so it can be a
    generator over a dataset rather than a materialized list, without its cost landing on
    the consumer's critical path.
    """
    workers = max(1, cpu_count() - 1)
    max_pending = workers * PENDING_ITEMS_PER_WORKER

    # Pool first: its workers are forked here, before the prefetch thread exists.
    with Pool(processes=workers, initializer=initializer) as pool:
        remaining = _prefetched(items, max_pending)
        pending = deque(pool.apply_async(func, (item,)) for item in islice(remaining, max_pending))

        while pending:
            result = pending.popleft().get()
            for item in islice(remaining, 1):
                pending.append(pool.apply_async(func, (item,)))
            yield result
