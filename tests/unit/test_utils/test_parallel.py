"""Unit tests for yolo_scout.utils.parallel."""

import time
from multiprocessing import cpu_count

import pytest

from yolo_scout.utils.parallel import PENDING_ITEMS_PER_WORKER, imap_workers

WORKERS = max(1, cpu_count() - 1)
MAX_PENDING = WORKERS * PENDING_ITEMS_PER_WORKER


def _record_and_double(item: tuple) -> tuple:
    """Append a byte to the marker file so the parent can count completed tasks."""
    index, marker_path = item
    with open(marker_path, "ab") as f:
        f.write(b"x")
    return index, index * 2


class TestImapWorkers:
    def test_yields_every_result_in_order(self, tmp_path):
        items = [(i, str(tmp_path / "marker")) for i in range(50)]

        results = list(imap_workers(_record_and_double, items))

        assert results == [(i, i * 2) for i in range(50)]

    def test_workers_do_not_run_ahead_of_the_consumer(self, tmp_path):
        """A stalled consumer must not let the pool drain the whole input into memory.

        This is the regression guard for the leak the streaming pipeline exists to fix:
        `pool.imap` would happily run all of these and cache every result in the parent.
        """
        marker = tmp_path / "marker"
        marker.touch()
        count = MAX_PENDING * 8
        items = [(i, str(marker)) for i in range(count)]

        stream = imap_workers(_record_and_double, items)
        next(stream)
        time.sleep(2)
        completed = marker.stat().st_size
        stream.close()

        # One consumed result frees exactly one slot, so at most the window plus the
        # in-flight replacements can have run.
        assert completed < count, "the pool drained the entire input while 1 result was consumed"
        assert completed <= MAX_PENDING + WORKERS, f"{completed} tasks ran while 1 result was consumed"

    def test_consumes_items_lazily(self, tmp_path):
        """The input may be a generator over a dataset, so it must not be drained up front.

        The prefetch thread reads ahead by at most one buffer, so the bound is looser than
        the in-flight window but must still be a small constant, not the whole input.
        """
        count = MAX_PENDING * 8
        pulled = []

        def items():
            for i in range(count):
                pulled.append(i)
                yield i, str(tmp_path / "marker")

        stream = imap_workers(_record_and_double, items())
        next(stream)
        time.sleep(2)
        pulled_count = len(pulled)
        stream.close()

        assert pulled_count < count, "the entire input was drained while 1 result was consumed"
        assert pulled_count <= 2 * MAX_PENDING + WORKERS + 1, f"{pulled_count} items pulled while 1 result was consumed"

    def test_source_exceptions_propagate(self):
        """A failure while reading the source must surface, not hang or vanish."""

        def items():
            yield 0, "/dev/null"
            raise RuntimeError("source blew up")

        with pytest.raises(RuntimeError, match="source blew up"):
            list(imap_workers(_record_and_double, items()))

    def test_worker_exceptions_propagate(self, tmp_path):
        items = [(i, str(tmp_path / "marker")) for i in range(10)]
        items[3] = (None, str(tmp_path / "marker"))  # None * 2 raises in the worker

        with pytest.raises(TypeError):
            list(imap_workers(_record_and_double, items))

    def test_empty_input_yields_nothing(self):
        assert list(imap_workers(_record_and_double, [])) == []
