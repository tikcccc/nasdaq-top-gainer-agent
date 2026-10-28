"""Compact reader notes and event-matched primary citations."""
import re
from random import Random

import pytest

from nasdaq_agent.reporting import announcements, evidence_report, references
from nasdaq_agent.reporting.facts import build_report_facts
from nasdaq_agent.reporting.render import _method, build_report
from scripts.eval_blind_reports import load_state, make_case


@pytest.fixture
def state(tmp_path):
    return load_state(tmp_path, make_case(4, Random(4), tmp_path))


def test_primary_announcement_matches_symbol_effective_date_and_ratio():
    state = {'selected': {'symbol': 'GCTK'}, 'checks': {
        'splits': [{'date': '2026-10-26', 'ratio': 0.5, 'status': 'no_gap'}]}}
    document = announcements.matched(state)['d8']
    assert document['url'] == 'https://www.sec.gov/Archives/edgar/data/1506983/000149315226048139/ex99-1.htm'
    refs = references.available(state)
    assert refs['d8'].url == document['url']
    sentence = evidence_report._split_sentence(state['checks']['splits'][0],
                                              announcement=announcements.for_split(state, state['checks']['splits'][0]))
    assert 'company announced' in sentence and '[d8]' in sentence
    assert 'price adjustment remains unconfirmed [d2]' in sentence
    assert len(sentence.split()) < 30


@pytest.mark.parametrize('symbol,date,ratio', [
    ('OTHER', '2026-10-26', 0.5), ('GCTK', '2025-10-26', 0.5), ('GCTK', '2026-10-26', 0.1),
])
def test_primary_document_is_not_reused_for_another_event(symbol, date, ratio):
    state = {'selected': {'symbol': symbol}, 'checks': {'splits': [{'date': date, 'ratio': ratio}]}}
    assert announcements.matched(state) == {}
    assert 'd8' not in references.available(state)


def test_method_is_two_lines_with_one_optional_split_line(state):
    state['selection']['filters'] = {'exclude_derivatives': True, 'min_price': 1, 'min_volume': 100_000}
    facts = build_report_facts(state)
    method = _method(facts, state)
    assert len(method) == 2
    assert 'price ≥$1' in method[0] and 'volume ≥100K shares' in method[0]
    assert '5-session price returns' in method[1]
    assert 'Snapshot' not in ' '.join(method)
    state['checks'] = {'splits': [{'date': state['session']['trade_date'], 'ratio': 0.5, 'status': 'no_gap'}]}
    method = _method(build_report_facts(state), state)
    assert len(method) == 3
    assert method[-1] == 'Volume comparison withheld across the split [d2].'
    assert len(' '.join(method).split()) < 50


def test_only_referenced_sources_are_listed_and_diagnostics_are_retained(state, settings, tmp_path):
    state['raw_top'] = {'symbol': 'HSCSW', 'pct_change': 154.9, 'flags': ['derivative']}
    state['datasets'][state['selected']['dataset_id']]['captured_at'] = '2026-10-27T10:15:50+00:00'
    body = evidence_report.build(state, {}, settings)
    report = build_report(state, body.narrative, run_dir=tmp_path, run_info={}, citations=body.citations)
    assert 'HSCSW' not in report.html and 'HSCSW' not in report.text
    assert 'company profile' not in report.html and 'company filings' not in report.html
    assert 'Snapshot saved:' not in report.html and 'Snapshot saved:' not in report.text
    assert report.facts['raw_top']['symbol'] == 'HSCSW'
    assert report.facts['captured_at'] == '2026-10-27T10:15:50+00:00'
    assert any('Unfiltered leader: HSCSW' in note for note in report.notes)
    main, bibliography = report.html.split('>Sources</div>', 1)
    listed = re.findall(r'font-weight:600;">\[(\d+)\]</td>', bibliography)
    assert len(listed) == 4
    for number in listed:
        assert re.search(r'\[' + number + r'\]</(?:a|span)>', main)


def test_short_method_names_both_vendors_for_a_proxy_benchmark(state):
    state['datasets'][state['selected']['dataset_id']]['source'] = 'yahoo'
    benchmark = next(m for m in state['datasets'].values() if m['symbol'] == '^GSPC')
    benchmark.update(symbol='SPY', proxy_for='^GSPC', source='nasdaq')
    method = _method(build_report_facts(state), state)
    assert len(method) == 2
    assert 'Yahoo Finance / Nasdaq.com' in method[1]
