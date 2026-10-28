"""Report contracts across quote conflicts, extra history, splits and partial sessions."""
import hashlib
from pathlib import Path
from random import Random

import pandas as pd
import pytest

from nasdaq_agent.reporting import evidence_report, placeholders, readable
from nasdaq_agent.reporting.chart import _series
from nasdaq_agent.reporting.facts import build_report_facts
from nasdaq_agent.reporting.render import build_report
from nasdaq_agent.data import metrics
from scripts.eval_blind_reports import load_state, make_case


@pytest.fixture
def state(tmp_path):
    return load_state(tmp_path, make_case(4, Random(4), tmp_path))


def render(state, settings, tmp_path, args=None):
    body = evidence_report.build(state, args or {}, settings)
    return body, build_report(state, body.narrative, run_dir=tmp_path, run_info={}, citations=body.citations + body.omitted_ids)


def test_cards_body_and_chart_use_the_history_snapshot_when_quotes_disagree(state, settings, tmp_path):
    stock = pd.read_csv(state['datasets'][state['selected']['dataset_id']]['path'])
    state['selected'].update(price=9876.54, volume=987_654_321, pct_change=9999.99, flags=['data_conflict'])
    state['datasets'][state['selected']['dataset_id']].update(source='yahoo', captured_at='2026-10-27T09:10:21+00:00')
    body, report = render(state, settings, tmp_path)
    expected_price = float(stock['close'].iloc[-1])
    expected_move = expected_price / float(stock['close'].iloc[-2]) - 1
    assert report.facts['price'] == expected_price
    assert report.facts['volume'] == 1_000_000
    assert report.facts['day_return'] == pytest.approx(expected_move)
    assert '9,876.54' not in report.html and '987.7M' not in report.html
    assert f'Close {placeholders.money(expected_price)}' in report.html
    assert placeholders.pct(expected_move) in report.subject
    assert 'quote and price history disagree' in body.narrative
    assert 'Snapshot saved:' not in report.html
    assert report.facts['captured_at'] == '2026-10-27T09:10:21+00:00'
    assert ' pp' in report.html and 'excess vs S&amp;P 500' in report.html
    facts = build_report_facts(state)
    plotted = _series(state['datasets'], facts.stock_id, facts.days, frames=facts.frames)[0][1]
    assert plotted['close'].tolist() == facts.stock['close'].tolist()
    assert plotted['idx'].iloc[-1] - 100 == pytest.approx(report.facts['window_return'] * 100)


def test_extra_history_cannot_change_the_volume_baseline(state, settings, tmp_path):
    meta = state['datasets'][state['selected']['dataset_id']]
    frame = pd.read_csv(meta['path'])
    extra = frame.iloc[[0]].copy()
    extra['date'] = (pd.Timestamp(frame['date'].iloc[0]) - pd.Timedelta(days=1)).date().isoformat()
    extra['volume'] = 999_000_000
    frame = pd.concat([extra, frame], ignore_index=True)
    frame.to_csv(meta['path'], index=False)
    meta['sha256'] = hashlib.sha256(Path(meta['path']).read_bytes()).hexdigest()
    body, report = render(state, settings, tmp_path, {'key_findings': ['volume_spike']})
    volume = report.facts['volume_comparison']
    assert volume['sessions'] == state['selected']['days']
    assert volume['ratio'] == 1.0
    assert '1.0×' in report.html
    assert 'in line with the previous' in body.narrative
    assert '999' not in body.narrative


@pytest.mark.parametrize('status', ['no_gap', 'unknown'])
def test_unconfirmed_split_hides_volume_multiple_and_preserves_uncertainty(state, settings, tmp_path, status):
    state['checks'] = {'splits': [{'date': state['session']['trade_date'], 'ratio': 0.5,
                                 'status': status, 'open': 0.84, 'previous_close': 0.806}]}
    body, report = render(state, settings, tmp_path)
    assert report.facts['volume_comparison'] is None
    assert 'Volume / prior' not in report.html
    assert 'Volume comparison withheld across the split' in report.html
    assert 'volume comparability across the split is unconfirmed' in report.facts['price_basis']
    assert 'came from trading' not in body.narrative
    assert 'split' in body.key_findings[-1]['text']
    if status == 'no_gap':
        assert 'price adjustment remains unconfirmed' in body.narrative
    with pytest.raises(evidence_report.ReportInputError, match='unavailable'):
        evidence_report.build(state, {'key_findings': ['volume_spike']}, settings)


def test_only_selected_secondary_metrics_appear_in_the_body(state, settings, tmp_path):
    body, report = render(state, settings, tmp_path, {'key_findings': ['final_session_share']})
    assert len(body.key_findings) == 1
    assert '**How it traded.**' not in body.narrative
    assert 'volatility' not in body.narrative and 'pullback' not in body.narrative
    assert 'Daily moves' in report.html and '<svg' in report.html
    assert 'Screen:' in report.html and 'Data:' in report.html
    assert 'Snapshot retrieval time' not in report.html
    assert report.facts['captured_at'] is None


def test_required_limits_take_a_finding_slot_without_exceeding_three(state, settings, tmp_path):
    state['checks'] = {'splits': [{'date': state['session']['trade_date'], 'ratio': 0.5,
                                 'status': 'no_gap', 'open': 0.84, 'previous_close': 0.806}]}
    body, _ = render(state, settings, tmp_path, {'key_findings': ['final_session_share', 'news:n1', 'drawdown']})
    assert [f['kind'] for f in body.key_findings] == ['final_session_share', 'news', 'data_limitation']
    assert 'price adjustment remains unconfirmed' in body.narrative
    assert 'pullback' not in body.narrative


def test_intraday_report_never_labels_the_price_as_a_final_close(state, settings, tmp_path):
    state['session']['is_intraday'] = True
    body, report = render(state, settings, tmp_path)
    assert 'intraday and not final' in body.narrative
    assert 'Last ' in report.html and 'intraday ·' in report.html
    assert 'last traded at' in report.html
    assert 'closed at' not in report.html


def test_intraday_missing_bar_uses_the_actual_price_date_in_prose(state, settings, tmp_path):
    actual = state['session']['trade_date']
    requested = (pd.Timestamp(actual) + pd.Timedelta(days=1)).date().isoformat()
    state['session'].update(trade_date=requested, is_intraday=True)
    state['selected']['trade_date'] = requested
    # No new quote or analysis is invented for the requested session.
    body, report = render(state, settings, tmp_path)
    assert f'latest price bar is dated {actual}' in body.narrative
    assert f'no price bar for the requested {requested}' in body.narrative
    assert f"on {readable.short_date(actual)}" in body.narrative
    assert report.facts['last_bar_date'] == actual


def test_benchmark_with_different_dates_is_not_compared_or_plotted(state):
    meta = next(m for m in state['datasets'].values() if m['symbol'] == '^GSPC')
    frame = pd.read_csv(meta['path'])
    frame.loc[0, 'date'] = (pd.Timestamp(frame['date'].iloc[0]) - pd.Timedelta(days=1)).date().isoformat()
    frame.to_csv(meta['path'], index=False)
    facts = build_report_facts(state)
    assert 'GSPC' not in facts.analysis['benchmark']
    assert facts.analysis['excess_return'] is None
    assert all(m[0] != 'S&P 500' for m in _series(state['datasets'], facts.stock_id, facts.days, frames=facts.frames))


def test_day_move_has_the_same_precision_in_subject_summary_findings_and_tooltip(state, settings, tmp_path):
    meta = state['datasets'][state['selected']['dataset_id']]
    stock = pd.read_csv(meta['path'])
    last = stock.index[-1]
    close = float(stock['close'].iloc[-2]) * 1.9975
    stock.loc[last, ['open', 'close']] = close
    stock.loc[last, 'high'] = close * 1.01
    stock.loc[last, 'low'] = close * 0.99
    stock.to_csv(meta['path'], index=False)
    meta['sha256'] = hashlib.sha256(Path(meta['path']).read_bytes()).hexdigest()
    benches = {m['symbol'].lstrip('^'): pd.read_csv(m['path']) for m in state['datasets'].values()
               if m['symbol'].startswith('^')}
    state['analysis'] = {**metrics.compute(stock, benches, state['selected']['days'], state['selected']['symbol']),
                         'analysis_key': state['selected']['analysis_key']}
    body, report = render(state, settings, tmp_path, {'key_findings': ['final_session_share']})
    assert '+99.75%' in report.subject
    assert 'jumped 99.75%' in body.narrative
    assert '(+99.75%)' in body.key_findings[0]['text']
    assert '>+99.75%</text>' in report.html
    assert '99.8%' not in report.html and '99.8%' not in report.text
    window_label = placeholders.pct(report.facts['window_return'])
    assert f'>{window_label}</text>' in report.html
