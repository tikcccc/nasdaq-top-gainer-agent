"""Actual OS-boundary tests: RUN_DOCKER_TESTS=1 requires the built analysis image."""

import os
import subprocess

import pandas as pd
import pytest

from nasdaq_agent.sandbox import runner

pytestmark = pytest.mark.skipif(os.getenv("RUN_DOCKER_TESTS") != "1", reason="set RUN_DOCKER_TESTS=1 for real container tests")


@pytest.fixture(scope="module", autouse=True)
def docker_ready():
    for command in (["docker", "version", "--format", "{{.Server.Version}}"],
                    ["docker", "image", "inspect", "nasdaq-analysis:local"]):
        result = subprocess.run(command, capture_output=True, timeout=15)
        assert result.returncode == 0, result.stderr.decode(errors="replace")


def test_container_analysis_and_read_only_inputs(tmp_path):
    source = tmp_path / "prices.csv"
    pd.DataFrame({"date": ["2026-10-23"], "close": [10]}).to_csv(source, index=False)
    code = """
import matplotlib.pyplot as plt
blocked = False
try:
    open('/input/data_0.csv', 'w').write('tampered')
except OSError:
    blocked = True
save_result({'close': float(load('ds_X').close.iloc[0]), 'input_read_only': blocked})
fig, ax = plt.subplots(); ax.plot([1,2]); save_chart(fig)
"""
    result = runner.run(code, {"ds_X": str(source)}, tmp_path / "sandbox", timeout_s=30, max_stdout=3000)
    assert result.ok, result.stderr_tail
    assert result.result == {"close": 10, "input_read_only": True}
    assert len(result.charts) == 1 and "tampered" not in source.read_text()


def test_container_cannot_read_host_files_or_access_network(tmp_path, monkeypatch):
    secret = tmp_path / "host_only.txt"
    secret.write_text("SYNTHETIC_HOST_MARKER")
    monkeypatch.setenv("SANDBOX_REVIEW_SECRET", "synthetic_secret")
    code = f"""
import socket, os
blocked_file = blocked_network = False
try:
    open({str(secret)!r}).read()
except OSError:
    blocked_file = True
try:
    socket.create_connection(('1.1.1.1', 443), timeout=1)
except OSError:
    blocked_network = True
save_result({{'host_file_blocked': blocked_file, 'network_blocked': blocked_network,
             'host_secret_absent': os.getenv('SANDBOX_REVIEW_SECRET') is None}})
"""
    result = runner.run(code, {}, tmp_path / "sandbox", timeout_s=20, max_stdout=3000)
    assert result.ok, result.stderr_tail
    assert all(result.result.values()), result.result


def test_container_resource_limits_are_effective(tmp_path):
    code = """
from pathlib import Path
save_result({'memory': Path('/sys/fs/cgroup/memory.max').read_text().strip(),
             'pids': Path('/sys/fs/cgroup/pids.max').read_text().strip(),
             'cpu': Path('/sys/fs/cgroup/cpu.max').read_text().strip()})
"""
    result = runner.run(code, {}, tmp_path / "sandbox", timeout_s=20, max_stdout=3000)
    assert result.ok, result.stderr_tail
    assert int(result.result["memory"]) == 512 * 1024 * 1024
    assert int(result.result["pids"]) == 32
    quota, period = map(int, result.result["cpu"].split())
    assert quota / period == 1


def test_container_timeout_removes_execution_process(tmp_path):
    result = runner.run("while True: pass", {}, tmp_path / "sandbox", timeout_s=4, max_stdout=3000)
    assert result.timed_out
    remaining = subprocess.run(["docker", "ps", "--filter", "name=nasdaq-sandbox-", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=10)
    assert remaining.returncode == 0 and remaining.stdout.strip() == ""


def test_output_storage_is_bounded_and_not_a_writable_host_mount(tmp_path):
    code = """
from pathlib import Path
blocked = False
try:
    for i in range(64):
        Path(f'/output/out/fill_{i}').write_bytes(b'x' * 1048576)
except OSError:
    blocked = True
for f in Path('/output/out').glob('fill_*'):
    f.unlink()
save_result({'output_storage_bounded': blocked})
"""
    result = runner.run(code, {}, tmp_path / "sandbox", timeout_s=20, max_stdout=3000)
    assert result.ok, result.stderr_tail
    assert result.result == {"output_storage_bounded": True}
    assert not list((tmp_path / "sandbox/out").glob("fill_*"))
