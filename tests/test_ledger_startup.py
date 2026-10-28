"""Concurrent first runs must both initialize the shared usage database."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from nasdaq_agent.infra.usage import UsageLedger


def test_two_workers_can_initialize_new_usage_database(tmp_path):
    for trial in range(10):
        barrier = Barrier(2)
        root = tmp_path / str(trial)
        def start(worker):
            barrier.wait(timeout=5)
            ledger = UsageLedger(root)
            return ledger.start_run(f'worker_{worker}', 123.0)
        with ThreadPoolExecutor(2) as pool:
            assert list(pool.map(start, (1, 2))) == [123.0, 123.0]
