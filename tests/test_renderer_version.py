from email import message_from_bytes
from email.policy import default
from random import Random

from fastapi.testclient import TestClient

from nasdaq_agent import api
from nasdaq_agent.infra.mailer import build_message
from nasdaq_agent.reporting.evidence_report import build
from nasdaq_agent.reporting.render import build_report
from nasdaq_agent.reporting.version import REPORT_VERSION, renderer_identity
from scripts.eval_blind_reports import load_state, make_case


def test_runtime_saved_report_html_and_mime_share_the_renderer_identity(tmp_path, settings, monkeypatch):
    state = load_state(tmp_path, make_case(4, Random(4), tmp_path))
    body = build(state, {}, settings)
    report = build_report(state, body.narrative, run_dir=tmp_path, run_info={})
    identity = renderer_identity()
    assert body.version == identity['version'] == REPORT_VERSION
    assert len(identity['fingerprint']) == 64
    assert report.renderer == report.facts['renderer'] == identity
    assert f'{identity["version"]}:{identity["fingerprint"]}' in report.html
    msg = message_from_bytes(bytes(build_message(report, 'from@example.invalid', ['to@example.invalid'])), policy=default)
    assert msg['X-Report-Renderer'] == identity['version']
    assert str(msg['X-Report-Fingerprint']).strip() == identity['fingerprint']
    monkeypatch.setattr(api, '_settings', lambda req=None: settings)
    monkeypatch.setattr(api, '_ensure_worker', lambda: None)
    with TestClient(api.app) as client:
        assert client.get('/api/config').json()['report_renderer'] == identity
