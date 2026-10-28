"""Execute generated code in Docker. trusted_local is only for reviewed test fixtures.

The child process gets three helpers - ``load(dataset_id)``, ``save_result(dict)``
and ``save_chart(fig)`` - so the code never needs file paths. It runs with
``python -I`` (isolated mode), a minimal environment and a private working dir.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

BOOTSTRAP = r'''
import json, pathlib, sys, re
import matplotlib
matplotlib.use("Agg")
import pandas as _pd

_work = pathlib.Path(sys.argv[1])
_code = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
_datasets = json.loads(sys.argv[3])
_out = _work / "out"
_out.mkdir(parents=True, exist_ok=True)

def _jsonable(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return float(o)

def load(dataset_id):
    """Return the cleaned daily bars for a dataset as a DataFrame sorted by date."""
    if dataset_id not in _datasets:
        raise KeyError(f"unknown dataset_id {dataset_id!r}; available: {sorted(_datasets)}")
    return _pd.read_csv(_datasets[dataset_id], parse_dates=["date"])

def save_result(result):
    """Persist the analysis result dict (JSON-serialisable values)."""
    (_out / "result.json").write_text(json.dumps(result, default=_jsonable, allow_nan=False), encoding="utf-8")

def save_chart(fig, name="chart"):
    """Persist a matplotlib figure as PNG."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise ValueError("chart name must be a simple filename without a path")
    fig.savefig(_out / f"{name}.png", dpi=110, bbox_inches="tight")

exec(compile(_code, "analysis.py", "exec"), {
    "__name__": "__main__", "load": load, "save_result": save_result,
    "save_chart": save_chart, "DATASETS": sorted(_datasets),
})
if len(sys.argv) > 4 and sys.argv[4] == "export":
    print("__NASDAQ_ARTIFACTS_READY__", flush=True)
    sys.stdin.buffer.read(1)  # keep tmpfs alive while the host retrieves bounded artefacts
'''


@dataclass
class SandboxResult:
    ok: bool
    duration_s: float
    stdout: str = ""
    stderr_tail: str = ""
    result: dict | None = None
    charts: list[Path] = field(default_factory=list)
    timed_out: bool = False
    output_limited: bool = False
    unavailable: str | None = None     # why nothing ran: cli_missing | daemon_down | image_missing | docker_error


UNAVAILABLE_LABELS = {
    "cli_missing": "the Docker CLI is not installed or not on PATH",
    "daemon_down": "Docker is installed but its daemon is not running",
    "image_missing": "the sandbox image has not been built",
    "docker_error": "docker could not start the container",
}


def classify_docker_failure(stderr: str) -> str:
    """`docker run` exit 125 means the container never started; say why."""
    text = stderr.lower()
    if "unable to find image" in text or "no such image" in text or "pull access denied" in text:
        return "image_missing"
    if ("cannot connect to the docker daemon" in text or "error during connect" in text
            or "is the docker daemon running" in text or "dockerdesktoplinuxengine" in text):
        return "daemon_down"
    return "docker_error"


def _minimal_env(work: Path) -> dict[str, str]:
    env = {"MPLBACKEND": "Agg", "MPLCONFIGDIR": str(work / ".mpl"), "PYTHONIOENCODING": "utf-8"}
    for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH"):  # required by the interpreter on Windows
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _docker_command(image: str, name: str, inputs: Path, output: Path, datasets: dict,
                    memory_mb: int, cpus: float) -> list[str]:
    return ["docker", "run", "--interactive", "--name", name, "--pull", "never", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
            "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m", "--cpus", str(cpus),
            "--pids-limit", "32", "--ulimit", "nofile=128:128", "--ulimit", "fsize=16777216:16777216",
            "--mount", f"type=bind,source={inputs.resolve()},target=/input,readonly",
            "--tmpfs", "/output:rw,noexec,nosuid,size=33554432,mode=1777",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=67108864,mode=1777", "--workdir", "/output",
            "--env", "MPLCONFIGDIR=/tmp/mpl", "--env", "MPLBACKEND=Agg",
            "--env", "OPENBLAS_NUM_THREADS=1", "--env", "OMP_NUM_THREADS=1", image,
            "python", "-I", "/input/bootstrap.py", "/output", "/input/analysis.py", json.dumps(datasets), "export"]


def _bounded_process(command: list[str], work: Path, timeout_s: float, limit: int, collect=None):
    chunks = [bytearray(), bytearray()]
    overflow = threading.Event()
    ready = threading.Event()
    # Docker's client needs its host context/TLS settings; only explicit --env
    # arguments enter the container. Reviewed local fixture code gets a minimal env.
    proc = subprocess.Popen(command, cwd=work, env=None if command[0] == "docker" else _minimal_env(work), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.PIPE if collect else subprocess.DEVNULL)

    def drain(pipe, i):
        total = 0
        try:
            while block := pipe.read1(4096):
                total += len(block)
                chunks[i].extend(block)
                if i == 0 and b"__NASDAQ_ARTIFACTS_READY__" in chunks[i]:
                    ready.set()
                del chunks[i][:-limit]
                if total > limit:
                    overflow.set()
                    proc.kill()
                    break
        finally:
            pipe.close()

    readers = [threading.Thread(target=drain, args=(pipe, i), daemon=True)
               for i, pipe in enumerate((proc.stdout, proc.stderr))]
    for reader in readers:
        reader.start()
    timed_out = False
    collection_error = None
    deadline = time.monotonic() + timeout_s
    try:
        while proc.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout_s)
            if collect and ready.is_set():
                try:
                    collect(remaining)
                    proc.stdin.write(b"\n")
                    proc.stdin.flush()
                except Exception as exc:
                    collection_error = f"ArtifactError: {exc}"
                    proc.kill()
                finally:
                    proc.stdin.close()
                    collect = None
            try:
                proc.wait(timeout=min(.05, remaining))
            except subprocess.TimeoutExpired:
                pass
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        proc.wait(timeout=5)
    for reader in readers:
        reader.join(timeout=2)
    output = [bytes(c).decode("utf-8", errors="replace") for c in chunks]
    output[0] = output[0].replace("__NASDAQ_ARTIFACTS_READY__\n", "")
    if collection_error:
        output[1] += "\n" + collection_error
    return proc.returncode or (1 if collection_error else 0), output, timed_out, overflow.is_set()


def run(code: str, datasets: dict[str, str], work: Path, *, timeout_s: float, max_stdout: int,
        backend: str = "docker", image: str = "nasdaq-analysis:local", memory_mb: int = 512,
        cpus: float = 1) -> SandboxResult:
    out = work / "out"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        return SandboxResult(False, 0, stderr_tail="execution directory must be fresh")
    code_path = work / "analysis.py"
    code_path.write_text(code, encoding="utf-8")

    started = time.monotonic()
    name = None
    collect = None
    if backend == "docker":
        if not shutil.which("docker"):
            return SandboxResult(False, 0, stderr_tail="SandboxUnavailable: docker CLI not found; local execution was not attempted",
                                 unavailable="cli_missing")
        inputs = work / "input"
        inputs.mkdir(exist_ok=True)
        (inputs / "analysis.py").write_text(code, encoding="utf-8")
        (inputs / "bootstrap.py").write_text(BOOTSTRAP, encoding="utf-8")
        mounted = {}
        for i, (ds, path) in enumerate(datasets.items()):
            filename = f"data_{i}.csv"
            shutil.copyfile(path, inputs / filename)
            mounted[ds] = f"/input/{filename}"
        name = f"nasdaq-sandbox-{uuid.uuid4().hex}"
        command = _docker_command(image, name, inputs, out, mounted, memory_mb, cpus)
        def collect(remaining):
            deadline = time.monotonic() + remaining
            listing = "import pathlib,re,json; p=pathlib.Path('/output/out'); print(json.dumps([f.name for f in sorted(p.iterdir()) if f.is_file() and not f.is_symlink() and re.fullmatch(r'(result\\.json|[A-Za-z0-9_-]{1,40}\\.png)',f.name) and f.stat().st_size<=16000000][:9]))"
            result = subprocess.run(["docker", "exec", name, "python", "-I", "-c", listing],
                                    capture_output=True, text=True, timeout=max(.1, deadline-time.monotonic()), check=True)
            budget = 32 * 1024 * 1024
            read_file = "import pathlib,sys; limit=int(sys.argv[2]); data=pathlib.Path(sys.argv[1]).open('rb').read(limit+1); assert len(data)<=limit, 'artefact limit exceeded'; sys.stdout.buffer.write(data)"
            for filename in json.loads(result.stdout):
                with (out / filename).open("xb") as fh:
                    subprocess.run(["docker", "exec", name, "python", "-I", "-c", read_file,
                                    f"/output/out/{filename}", str(min(16_000_000, budget))],
                                   stdout=fh, stderr=subprocess.PIPE,
                                   timeout=max(.1, deadline-time.monotonic()), check=True)
                budget -= (out / filename).stat().st_size
    elif backend == "trusted_local":
        command = [sys.executable, "-I", "-c", BOOTSTRAP, str(work.resolve()), str(code_path.resolve()), json.dumps(datasets)]
    else:
        return SandboxResult(False, 0, stderr_tail=f"unknown sandbox backend: {backend}")
    try:
        returncode, (stdout, stderr), timed_out, limited = _bounded_process(command, work, timeout_s, max(1024, max_stdout * 4), collect=collect)
    except OSError as exc:
        return SandboxResult(False, time.monotonic() - started, stderr_tail=f"SandboxUnavailable: {exc}",
                             unavailable="cli_missing" if name else "docker_error")
    finally:
        if name:
            try:
                subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
    duration = time.monotonic() - started
    if name and returncode == 125 and not timed_out:
        # `docker run` itself failed (daemon down, image missing): nothing was executed.
        return SandboxResult(False, duration, stderr_tail=f"SandboxUnavailable: {stderr.strip()[-300:]}",
                             unavailable=classify_docker_failure(stderr))
    if timed_out or limited:
        return SandboxResult(False, duration, stdout=stdout[-max_stdout:],
                             stderr_tail="OutputLimitError: execution output exceeded cap" if limited else f"TimeoutError: execution exceeded {timeout_s:.0f}s",
                             timed_out=timed_out, output_limited=limited)
    stderr_lines = stderr.strip().splitlines()
    result = None
    result_path = out / "result.json"
    if result_path.is_file() and not result_path.is_symlink() and result_path.stat().st_size <= 1_000_000:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except ValueError:
            result = None
    return SandboxResult(
        ok=returncode == 0,
        duration_s=duration,
        stdout=stdout[-max_stdout:],
        stderr_tail="\n".join(stderr_lines[-20:]),
        result=result,
        charts=[p for p in sorted(out.glob("*.png")) if p.is_file() and not p.is_symlink()
                and p.resolve().parent == out.resolve() and p.stat().st_size <= 16_000_000][:8],
    )
