"""Earlier daily shocks survive model choices; UTC publication days are not US session days."""
import hashlib
from pathlib import Path
from random import Random

import pandas as pd
import pytest

from nasdaq_agent.data import metrics
from nasdaq_agent.reporting import evidence_report as reports
from nasdaq_agent.reporting.facts import build_report_facts
from nasdaq_agent.reporting.references import available
from nasdaq_agent.reporting.render import _news
from scripts.eval_blind_reports import load_state, make_case


def shock_state(tmp_path, returns):
    state = load_state(tmp_path, make_case(4, Random(4), tmp_path))
    dates = ['2026-10-20', '2026-10-21', '2026-10-22', '2026-10-23', '2026-10-26', '2026-10-27']
    closes = [10.0]
    for move in returns:
        closes.append(closes[-1] * (1 + move))
    for meta in state['datasets'].values():
        values = closes if not meta['symbol'].startswith('^') else [100.0] * 6
        frame = pd.DataFrame({'date': dates, 'close': values, 'volume': [1000] * 5 + [500_000]})
        frame.to_csv(meta['path'], index=False)
        meta['sha256'] = hashlib.sha256(Path(meta['path']).read_bytes()).hexdigest()
    selected = state['selected']
    selected.update(days=5, trade_date=dates[-1])
    state['session']['trade_date'] = dates[-1]
    for headline in state['headlines'].values():
        headline.update(trade_date=dates[-1], published='2026-10-27T12:00:00Z')
    stock = pd.read_csv(state['datasets'][selected['dataset_id']]['path'])
    benches = {m['symbol'].lstrip('^'): pd.read_csv(m['path']) for m in state['datasets'].values()
               if m['symbol'].startswith('^')}
    state['analysis'] = {**metrics.compute(stock, benches, 5, selected['symbol']),
                         'analysis_key': selected['analysis_key']}
    return state


@pytest.mark.parametrize('choices', [None, ['final_session_share', 'volume_spike', 'news:n1']])
def test_earlier_drop_is_retained_with_default_or_three_explicit_findings(tmp_path, settings, choices):
    state = shock_state(tmp_path, [-.05, -.065, -.34, .068, .747])
    result = reports.build(state, {'key_findings': choices}, settings)
    assert len(result.key_findings) <= 3
    assert 'prior_shock' in [f['kind'] for f in result.key_findings]
    assert 'fell 34.00% on Oct 23' in result.narrative
    assert 'the cause has not been verified' in result.narrative
    assert 'no news' not in result.narrative
    assert result.word_count <= 250


@pytest.mark.parametrize('returns', [[-.1, -.1, -.1, -.1, .75], [.01, .01, .01, .01, -.34]])
def test_cumulative_drop_and_final_session_alone_do_not_trigger_prior_shock(tmp_path, settings, returns):
    state = shock_state(tmp_path, returns)
    result = reports.build(state, {}, settings)
    assert 'prior_shock' not in [f['kind'] for f in result.key_findings]


def test_shock_and_material_limitation_cannot_displace_each_other(tmp_path, settings):
    state = shock_state(tmp_path, [0, 0, -.34, 0, .75])
    state['session']['is_intraday'] = True
    result = reports.build(state, {'key_findings': ['news:n1', 'volume_spike', 'data_limitation']}, settings)
    assert [f['kind'] for f in result.key_findings] == ['news', 'data_limitation', 'prior_shock']
    assert 'intraday and not final' in result.narrative


@pytest.mark.parametrize('move,expected', [(-.2, True), (.2, True), (-.1999, False), (.1999, False)])
def test_twenty_percent_threshold_is_inclusive_despite_float_rounding(tmp_path, move, expected):
    state = shock_state(tmp_path, [move, 0, 0, 0, .75])
    shock = reports._prior_shock(build_report_facts(state), reports.current_headlines(state))
    assert (shock is not None) == expected


@pytest.mark.parametrize('published,expected', [
    ('2026-10-24T00:30:00Z', 'n1'),      # Oct 23 in New York
    ('2026-10-23T01:00:00Z', None),     # Oct 22 in New York
    ('2026-10-23T20:30:00-04:00', 'n1'),
    ('2026-10-23', None),              # no time zone: do not invent one
    ('2026-10-23T20:30:00', None),
    ('not a timestamp', None),
    (None, None),
])
def test_headlines_match_the_shock_session_in_new_york_time(tmp_path, published, expected):
    state = shock_state(tmp_path, [0, 0, -.34, 0, .75])
    state['headlines']['n1']['published'] = published
    shock = reports._prior_shock(build_report_facts(state), reports.current_headlines(state))
    assert shock['date'] == '2026-10-23'
    assert shock['ref'] == expected


def test_new_york_conversion_observes_winter_offset():
    assert reports._news_date('2026-12-02T04:30:00Z') == '2026-12-01'


def test_cited_headline_displays_the_same_new_york_date_as_the_match(tmp_path, settings):
    state = shock_state(tmp_path, [0, 0, -.34, 0, .75])
    state['headlines']['n1']['published'] = '2026-10-24T00:30:00Z'
    result = reports.build(state, {'selected_news_ids': ['n1']}, settings)
    assert 'Synthetic source (2026-10-23)' in result.narrative
    assert available(state)['n1'].date == 'Fri, Oct 23, 2026'
    assert _news(state, ['n1'])[0]['when'] == 'Oct 23'
    assert 'a fetched headline is dated that day' in result.narrative


def test_matching_date_on_another_stock_is_not_used(tmp_path):
    state = shock_state(tmp_path, [0, 0, -.34, 0, .75])
    state['headlines']['n1'].update(symbol='OTHER', published='2026-10-23T12:00:00Z')
    assert reports._prior_shock(build_report_facts(state), reports.current_headlines(state))['ref'] is None
