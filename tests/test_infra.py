from pathlib import Path

import httpx
import pytest
import respx

from nasdaq_agent.infra import breaker, http, sent_log
from nasdaq_agent.infra.errors import ProviderError


def test_first_success_falls_back_and_reports():
    breaker.reset(3)

    def broken():
        raise ProviderError("a", "down")

    res = breaker.first_success([("a", broken), ("b", lambda: 42)])
    assert res.value == 42 and res.source == "b" and "a failed" in res.warnings[0]


def test_circuit_opens_after_repeated_failures():
    breaker.reset(2)
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise RuntimeError("boom")

    for _ in range(3):
        breaker.first_success([("flaky", broken), ("ok", lambda: 1)])
    assert calls["n"] == 2 and breaker.is_open("flaky")


def test_all_providers_failing_raises():
    breaker.reset(3)
    with pytest.raises(ProviderError):
        breaker.first_success([("a", lambda: 1 / 0)])


@respx.mock
def test_http_retries_429_with_retry_after(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda s: None)
    route = respx.get("https://example.test/data").mock(side_effect=[
        httpx.Response(429, headers={"Retry-After": "0"}),
        httpx.Response(503),
        httpx.Response(200, json={"ok": True}),
    ])
    assert http.get_json("https://example.test/data") == {"ok": True}
    assert route.call_count == 3


@respx.mock
def test_http_does_not_retry_404():
    route = respx.get("https://example.test/missing").mock(return_value=httpx.Response(404))
    with pytest.raises(httpx.HTTPStatusError):
        http.get("https://example.test/missing")
    assert route.call_count == 1


def test_sent_log_idempotency(tmp_path):
    path = tmp_path / "sent.json"
    assert sent_log.already_sent(path, "2026-10-23", "AIOK", ["a@x.com"]) is None
    sent_log.record(path, "2026-10-23", "AIOK", ["a@x.com"], {"run_id": "r1"})
    assert sent_log.already_sent(path, "2026-10-23", "aiok", ["A@X.com"])["run_id"] == "r1"
    assert sent_log.already_sent(path, "2026-10-24", "AIOK", ["a@x.com"]) is None


def test_agent_imports_and_reports_status_without_the_jev_sdk():
    """typesafe-sdk is a recommended extra, not a hard dependency."""
    import subprocess
    import sys
    code = ("import sys; sys.modules['typesafe_sdk'] = None\n"
            "from nasdaq_agent.config import load_settings\n"
            "from nasdaq_agent.llm import jev, tracked\n"
            "import nasdaq_agent.agent.graph, nasdaq_agent.tools.sentiment\n"
            "s = load_settings(overrides={'jev': {'enabled': True, 'api_key': 'k'}})\n"
            "assert not jev.available(s) and jev.status(s)['state'] == 'sdk_missing', jev.status(s)\n"
            "assert not tracked.retryable(RuntimeError())\n"
            "print('ok')")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.stdout.strip() == "ok", result.stderr[-2000:]


SENTINELS = {"SMTP_USER": "leak-user@sentinel.invalid", "EMAIL_FROM": "leak-from@sentinel.invalid",
             "EMAIL_TO": "leak-to@sentinel.invalid", "SMTP_PASSWORD": "leak-password-sentinel",
             "MIMO_API_KEY": "leak-key-sentinel", "MIMO_BASE_URL": "https://mimo.invalid/v1", "MIMO_MODEL": "m"}


def test_settings_repr_hides_addresses_and_secrets(monkeypatch):
    from nasdaq_agent.config import load_settings
    for name, value in SENTINELS.items():
        monkeypatch.setenv(name, value)
    settings = load_settings()
    text = repr(settings) + str(settings) + repr(settings.email)
    assert settings.email.smtp.user == SENTINELS["SMTP_USER"]          # still usable by the mailer
    assert not [v for v in SENTINELS.values() if "sentinel" in v and v in text]


def test_failing_test_output_never_prints_real_configuration(tmp_path):
    """A failure with settings in scope (and -l locals) must not reveal addresses or keys."""
    import os
    import subprocess
    import sys
    probe = tmp_path / "test_probe.py"
    probe.write_text("from nasdaq_agent.config import load_settings\n"
                     "def test_fails():\n"
                     "    settings = load_settings()\n"
                     "    assert settings is None, settings\n", encoding="utf-8")
    env = {**os.environ, **SENTINELS, "NASDAQ_ENV_FILE": "none", "PYTHONPATH": str(Path(__file__).parents[1])}
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-l", "-p", "no:cacheprovider", str(probe)],
                            capture_output=True, text=True, env=env, cwd=tmp_path, timeout=120)
    assert "1 failed" in result.stdout, result.stdout[-2000:]
    leaked = [v for v in SENTINELS.values() if "sentinel" in v and v in result.stdout + result.stderr]
    assert not leaked, leaked


def test_settings_load_without_any_smtp_environment(monkeypatch):
    from nasdaq_agent.config import load_settings
    for name in ("SMTP_HOST", "SMTP_PORT"):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.email.smtp.host == "smtp.gmail.com" and settings.email.smtp.port == 587


def test_rejected_jev_key_is_reported_as_auth_failed(tmp_path):
    from nasdaq_agent.config import load_settings
    from nasdaq_agent.llm import jev
    settings = load_settings(overrides={"jev": {"enabled": True, "api_key": "k"}, "run": {"output_dir": str(tmp_path)}})
    rejected = {"outcome": "error", "error_type": "TypeSafeAuthenticationError", "http_status": 401, "started": 1.0}
    assert jev.status(settings, rejected)["state"] == "auth_failed"
    assert jev.status(settings, {**rejected, "error_type": "TimeoutError", "http_status": None})["state"] == "configured"
    assert jev.status(settings)["state"] == "configured" and jev.status(settings)["last_request"] is None
