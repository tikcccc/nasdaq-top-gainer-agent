"""Turnover cannot validate a screener's undated market capitalisation."""
from random import Random

import pytest

from nasdaq_agent.reporting.placeholders import build_context
from nasdaq_agent.reporting.render import build_report
from scripts.eval_blind_reports import load_state, make_case


@pytest.mark.parametrize('cap', [2_200_000, 2_200_000_000])
def test_unverified_cap_is_hidden_regardless_of_turnover(tmp_path, cap):
    state = load_state(tmp_path, make_case(4, Random(4), tmp_path))
    state['selected'].update(market_cap=cap, price=5.41, volume=23_200_000)
    assert build_context(state)['market_cap'] == 'n/a'
    report = build_report(state, '**In brief.** Test.', run_dir=tmp_path, run_info={})
    assert 'mkt cap' not in report.html.lower()
    assert report.facts['screener_market_cap'] == cap
    assert report.facts['market_cap_status'] == 'unverified'
    assert 'stale' not in (state['selected'].get('flags') or [])
    assert any('unverified' in note.lower() for note in report.notes)
