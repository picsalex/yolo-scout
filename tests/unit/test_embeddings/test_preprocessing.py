"""Unit tests for yolo_scout.embeddings.preprocessing pool plumbing."""

import time
from multiprocessing.pool import Pool

import pytest

from yolo_scout.embeddings.preprocessing import _imap_bounded

WORKERS = 4
MAX_PENDING = 8


def _record_and_double(item: tuple) -> tuple:
    """Append a byte to the marker file so the parent can count completed tasks."""
    index, marker_path = item
    with open(marker_path, "ab") as f:
        f.write(b"x")
    return index, [index * 2]


class TestImapBounded:
    def test_yields_every_result_in_order(self, tmp_path):
        items = [(i, str(tmp_path / "marker")) for i in range(50)]

        with Pool(processes=WORKERS) as pool:
            results = list(_imap_bounded(pool, _record_and_double, items, MAX_PENDING))

        assert [index for index, _ in results] == list(range(50))
        assert [payload for _, payload in results] == [[i * 2] for i in range(50)]

    def test_workers_do_not_run_ahead_of_the_consumer(self, tmp_path):
        """A stalled consumer must not let the pool drain the whole input into memory.

        This is the regression guard for the leak the streaming rewrite exists to fix:
        `pool.imap` would happily run all 500 tasks here and cache every result.
        """
        marker = tmp_path / "marker"
        marker.touch()
        items = [(i, str(marker)) for i in range(500)]

        with Pool(processes=WORKERS) as pool:
            stream = _imap_bounded(pool, _record_and_double, items, MAX_PENDING)
            next(stream)
            time.sleep(2)
            completed = marker.stat().st_size

        # One consumed result frees exactly one slot, so at most the window plus the
        # in-flight replacements can have run.
        assert completed <= MAX_PENDING + WORKERS, f"{completed} tasks ran while 1 result was consumed"

    def test_worker_exceptions_propagate(self, tmp_path):
        items = [(i, str(tmp_path / "marker")) for i in range(10)]
        items[3] = (None, str(tmp_path / "marker"))  # None * 2 raises in the worker

        with Pool(processes=WORKERS) as pool, pytest.raises(TypeError):
            list(_imap_bounded(pool, _record_and_double, items, MAX_PENDING))
