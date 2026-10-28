import pandas as pd
import pytest

from nasdaq_agent.sandbox import policy, runner


@pytest.mark.parametrize("code, fragment", [
    ("import os", "import of 'os'"),
    ("import subprocess", "import of 'subprocess'"),
    ("from pathlib import Path", "from pathlib"),
    ("open('x').read()", "'open'"),
    ("import pandas as pd\npd.read_csv('http://x')", "read_csv"),
    ("df.to_csv('x.csv')", "to_csv"),
    ("().__class__.__bases__", "dunder"),
    ("eval('1')", "'eval'"),
    ("fig.savefig('x.png')", "savefig"),
])
def test_policy_rejects(code, fragment):
    problems = policy.check(code)
    assert problems and any(fragment in p for p in problems)


def test_policy_allows_analysis_code():
    code = "import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\ndf = load('ds_X')\nsave_result({'a': 1})"
    assert policy.check(code) == []


def test_policy_reports_syntax_errors():
    assert "SyntaxError" in policy.check("def (:")[0]


def test_runner_executes_with_helpers(tmp_path):
    data = tmp_path / "x.csv"
    pd.DataFrame({"date": ["2026-10-22", "2026-10-23"], "close": [1.0, 2.0]}).to_csv(data, index=False)
    code = (
        "import matplotlib.pyplot as plt\n"
        "df = load('ds_X')\n"
        "save_result({'growth': float(df.close.iloc[-1] / df.close.iloc[0] - 1)})\n"
        "fig, ax = plt.subplots(); ax.plot(df.close); save_chart(fig)\n"
        "print('ok')"
    )
    res = runner.run(code, {"ds_X": str(data)}, tmp_path / "work", timeout_s=60, max_stdout=1000, backend="trusted_local")
    assert res.ok, res.stderr_tail
    assert res.result == {"growth": 1.0}
    assert len(res.charts) == 1 and "ok" in res.stdout


def test_runner_reports_traceback(tmp_path):
    res = runner.run("x = load('ds_missing')", {}, tmp_path / "w", timeout_s=60, max_stdout=1000, backend="trusted_local")
    assert not res.ok and "unknown dataset_id" in res.stderr_tail


def test_runner_timeout(tmp_path):
    res = runner.run("while True:\n    pass", {}, tmp_path / "w", timeout_s=3, max_stdout=1000, backend="trusted_local")
    assert res.timed_out and not res.ok


@pytest.mark.parametrize("stderr, reason", [
    ("Unable to find image 'nasdaq-analysis:local' locally\ndocker: Error response from daemon: No such image: nasdaq-analysis:local", "image_missing"),
    ("docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?", "daemon_down"),
    ("docker: error during connect: Head \"http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping\": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.", "daemon_down"),
    ("docker: Error response from daemon: invalid mount config", "docker_error"),
])
def test_docker_start_failures_are_classified(stderr, reason):
    from nasdaq_agent.sandbox.runner import classify_docker_failure
    assert classify_docker_failure(stderr) == reason
