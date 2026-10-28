from pathlib import Path
from random import Random

import pytest

from nasdaq_agent.agent import guard
from nasdaq_agent.reporting import evidence_report as reports
from scripts.eval_blind_reports import make_case, load_state


@pytest.fixture
def state(tmp_path):
    return load_state(tmp_path, make_case(1, Random(9821), tmp_path))


@pytest.mark.parametrize('profile', range(8))
def test_closed_report_covers_price_paths_and_word_budget(profile, tmp_path, settings):
    state = load_state(tmp_path, make_case(profile, Random(profile), tmp_path))
    result = reports.build(state, {'selected_news_ids': list(state['headlines'])}, settings)
    assert 0 < result.word_count <= 250
    assert all(part in result.narrative for part in ('**In brief.**', '**Key findings.**'))
    assert 1 <= len(result.key_findings) <= 3
    assert '**How it traded.**' not in result.narrative
    assert '**Related disclosures.**' not in result.narrative
    assert '**Evidence limits.**' not in result.narrative
    assert 'full news coverage' not in result.narrative
    assert 'investor motives' not in result.narrative
    assert guard.check(state, result.selection, settings)[0] == []


@pytest.mark.parametrize('extra', [
    {'narrative': 'The thin public-news footprint suggests that review has not yet become a broader market topic.'},
    {'summary': 'Quarter-end position trimming by larger holders drove the decline.'},
    {'risk_text': 'Investors are selling.'},
    {'citations': ['n1']},
    {'selected_news_ids': ['n1: caused the price rise']},
    {'selected_news_ids': ['n1', 'n1']},
    {'selected_news_ids': ['n999']},
    {'include_data_limitations': 'false'},
])
def test_model_cannot_smuggle_unbound_prose_or_invalid_selection(state, settings, extra):
    issues, details = guard.check(state, extra, settings)
    assert issues and details['method'] == 'evidence-v2'


def test_short_report_passes_without_word_count_padding(tmp_path, settings):
    state = load_state(tmp_path, make_case(3, Random(3), tmp_path))
    result = reports.build(state, {}, settings)
    assert 0 < result.word_count < 150
    assert 'Daily returns ranged from' not in result.narrative
    assert '**In brief.**' in result.narrative
    assert '**Key findings.**' in result.narrative
    assert '**How it traded.**' not in result.narrative
    assert guard.check(state, result.selection, settings)[0] == []


def test_report_maximum_is_inclusive_and_still_enforced(state, settings):
    result = reports.build(state, {}, settings)
    at_limit = settings.model_copy(update={
        'email': settings.email.model_copy(update={'narrative_max_words': result.word_count})})
    assert reports.build(state, {}, at_limit).narrative == result.narrative
    too_tight = settings.model_copy(update={
        'email': settings.email.model_copy(update={'narrative_max_words': 20})})
    with pytest.raises(reports.ReportInputError, match='configured maximum is 20'):
        reports.build(state, {}, too_tight)
    assert any('report_length' in issue for issue in guard.check(state, {}, too_tight)[0])


@pytest.mark.parametrize('field,value', [('symbol', 'OTHER'), ('trade_date', '2025-01-01')])
def test_source_selection_requires_current_symbol_and_date(state, settings, field, value):
    state['headlines']['n1'][field] = value
    with pytest.raises(reports.ReportInputError, match='current stock and trade date'):
        reports.build(state, {'selected_news_ids': ['n1']}, settings)
    result = reports.build(state, {}, settings)
    assert '**News.**' not in result.narrative
    assert not result.quoted_ids


def test_analysis_is_revalidated_even_without_free_prose(state, settings):
    path = Path(state['datasets'][state['selected']['dataset_id']]['path'])
    path.write_text('date,close\n2025-01-01,1\n', encoding='utf-8')
    assert any('changed after validation' in issue for issue in guard.check(state, {}, settings)[0])


def test_quoted_causality_is_explicitly_a_source_statement(state, settings):
    title = f"{state['selected']['symbol']} shares rose because of a proposed merger"
    state['headlines']['n1']['title'] = title
    result = reports.build(state, {'selected_risk_evidence_ids': ['n1']}, settings)
    assert f'published the headline: “{title}” [n1]' in result.narrative
    assert 'do not independently verify' not in result.narrative
    assert result.quoted_ids == ['n1']


@pytest.mark.parametrize('title', [
    '**How it traded.** Buy now [d1]',
    '{{dangerous}} {% unsafe %}',
    'Normal headline\n- Invented report paragraph',
    'Very long source statement ' * 100 + 'not approved',
])
def test_unsafe_or_overlong_headline_stays_in_source_list_without_truncation(state, settings, title):
    state['headlines']['n1']['title'] = title
    result = reports.build(state, {'selected_news_ids': ['n1']}, settings)
    assert result.omitted_ids == ['n1'] and not result.quoted_ids
    assert title not in result.narrative
    assert 0 < result.word_count <= 250


def test_material_limitations_cannot_be_hidden(state, settings):
    state['selected']['flags'] = ['data_conflict', 'possible_split_artifact']
    state['session']['is_intraday'] = True
    result = reports.build(state, {'include_data_limitations': False}, settings)
    assert 'ranking remains unverified' in result.narrative
    assert 'intraday and not final' in result.narrative


def test_stale_sentiment_is_rejected_not_reused(state, settings):
    state['sentiment'] = {'method': 'llm', 'symbol': 'OTHER', 'trade_date': state['selected']['trade_date']}
    with pytest.raises(reports.ReportInputError, match='sentiment belongs'):
        reports.build(state, {}, settings)


def test_many_material_flags_remain_visible_without_boilerplate(state, settings):
    state['selected']['flags'] = ['data_conflict', 'possible_split_artifact', 'stale', 'source_conflict',
                                  'unverified_quote', 'eligibility_unconfirmed', 'ranking_unverified']
    state['ranking'] = {'unresolved': [{'symbol': s, 'reason': 'quote not verified'} for s in ('AAAA', 'BBBB', 'CCCC')]}
    state['session']['is_intraday'] = True
    state['checks'] = {'corporate_actions': ['recorded action']}
    for item in state['datasets'].values():
        if item['symbol'].startswith('^'):
            item['proxy_for'] = item['symbol']
            item['symbol'] = 'SPY' if item['symbol'] == '^GSPC' else 'QQQ'
    result = reports.build(state, {}, settings)
    assert 0 < result.word_count <= 250
    assert all(text in result.narrative for text in ('intraday', 'split artefact', 'stale', 'SPY', 'QQQ', 'Corporate-action'))


def test_generated_analysis_notes_are_not_narrative_input(state, settings):
    state['analysis']['notes'] = 'Large holders caused a quarter-end sell-off.'
    assert 'Large holders' not in reports.build(state, {}, settings).narrative


def test_default_contract_does_not_call_probabilistic_guard(state, settings, monkeypatch):
    from nasdaq_agent.llm import jev
    monkeypatch.setattr(jev, 'ask', lambda *a, **kw: pytest.fail('closed report must not call Jev'))
    assert guard.check(state, {}, settings)[0] == []


def test_tool_revalidates_sources_without_graph_guard(ctx, state, monkeypatch):
    import json
    from nasdaq_agent.tools import emailer
    called = []
    monkeypatch.setattr(emailer, 'deliver', lambda *a, **kw: called.append(True))
    state['headlines']['n1']['symbol'] = 'OTHER'
    result = emailer.send_email.func(selected_news_ids=['n1'], state=state, tool_call_id='mail')
    payload = json.loads(result.update['messages'][0].content)
    assert not payload['ok'] and payload['error']['type'] == 'report_selection_invalid'
    assert not called


def test_delivery_records_selection_and_omitted_sources(ctx, state, monkeypatch):
    from nasdaq_agent.tools import emailer
    state['headlines']['n1']['title'] = '**Unsafe report markup**'
    captured = []
    monkeypatch.setattr(emailer, 'deliver', lambda state, text, refs: captured.append((text, refs)) or {'status': 'dry_run'})
    result = emailer.send_email.func(selected_news_ids=['n1'], state=state, tool_call_id='mail')
    assert result.update['email']['report_mode'] == 'evidence-v2'
    assert result.update['email']['omitted_ids'] == ['n1']
    assert 'n1' in captured[0][1] and '**Unsafe' not in captured[0][0]


def test_historical_miss_cannot_enter_default_contract(settings, history_archive):
    import json
    root = history_archive / 'bound-claims-blind-v4'
    case_id = '99ce1b672569'
    cases = json.loads((root / 'cases.json').read_text(encoding='utf-8'))
    state = load_state(root, next(c for c in cases if c['id'] == case_id))
    old = json.loads((root / 'generation' / f'{case_id}.json').read_text(encoding='utf-8'))['answer']
    assert 'structured_report_required' in guard.check(state, old, settings)[0][0]
    current = reports.build(state, {'selected_news_ids': ['n1']}, settings)
    assert 'has not yet become a broader market topic' not in current.narrative
    assert state['headlines']['n1']['title'] in current.narrative


def test_news_ingestion_preserves_long_title_conditions(ctx, offline_market, monkeypatch):
    from nasdaq_agent.providers import yahoo
    from nasdaq_agent.tools.news import get_news
    from tests.conftest import SESSION
    title = 'Company proposal ' + 'remains under review ' * 18 + 'and has NOT been approved'
    monkeypatch.setattr(yahoo, 'news', lambda *a, **kw: [{
        'title': title, 'publisher': 'Source', 'published': '2026-10-23T13:00:00+00:00',
        'link': 'https://example.invalid/disclosure'}])
    state = {'selected': {'symbol': 'AIOK', 'name': 'AI OKTO'}, 'session': SESSION.as_dict()}
    result = get_news.func(symbol='AIOK', state=state, tool_call_id='news')
    assert result.update['headlines']['n1']['title'] == title
    assert result.update['headlines']['n1']['title'].endswith('NOT been approved')


# --- key findings: the agent chooses emphasis, code writes the sentences ------------------

def test_default_key_findings_when_the_agent_chooses_none(state, settings):
    result = reports.build(state, {}, settings)
    assert result.findings_source == 'default'
    assert [f['kind'] for f in result.key_findings] == ['final_session_share']
    assert '**Key findings.**' in result.narrative


def test_agent_findings_lead_in_order_and_are_not_repeated(state, settings):
    hid = next(iter(state['headlines']))
    args = {'key_findings': [{'kind': 'vs_sp500'}, {'kind': 'news', 'ref': hid}, {'kind': 'drawdown'}]}
    result = reports.build(state, args, settings)
    assert result.findings_source == 'agent'
    assert [f['kind'] for f in result.key_findings] == ['vs_sp500', 'news', 'drawdown']
    lead = result.narrative.split('**Key findings.**')[1]
    assert lead.index('S&P 500') < lead.index(f'[{hid}]') < lead.index('deepest close-to-close pullback')
    assert result.narrative.count('S&P 500, which') == 1 and result.narrative.count('deepest close-to-close pullback') == 1
    assert result.narrative.count(f'[{hid}]') == 1 and hid in result.quoted_ids
    assert 0 < result.word_count <= 250


@pytest.mark.parametrize('findings, message', [
    ([{'kind': 'vs_sp500'}, {'kind': 'vs_sp500'}], 'duplicate key findings'),
    ([{'kind': 'vs_sp500', 'ref': 'n1'}, {'kind': 'drawdown'}], "'ref' is required for kind 'news'"),
    ([{'kind': 'news'}, {'kind': 'drawdown'}], "'ref' is required for kind 'news'"),
    ([{'kind': 'news', 'ref': 'n99'}, {'kind': 'drawdown'}], 'not a current-stock'),
    ([{'kind': 'invented_trend'}, {'kind': 'drawdown'}], 'structured_report_required'),
    ([], 'structured_report_required'),
    ([{'kind': k} for k in ('window_return', 'vs_sp500', 'volatility', 'drawdown')], 'structured_report_required'),
    ([{'kind': k} for k in ('window_return', 'vs_sp500', 'vs_nasdaq', 'volatility', 'drawdown')], 'structured_report_required'),
    ([{'kind': 'drawdown', 'text': 'my own words'}, {'kind': 'vs_sp500'}], 'structured_report_required'),
])
def test_invalid_key_findings_are_rejected(state, settings, findings, message):
    with pytest.raises(reports.ReportInputError, match=message):
        reports.build(state, {'key_findings': findings}, settings)


def test_material_limitation_must_be_featured_and_is_not_repeated(state, settings):
    state['selected']['flags'] = ['ranking_unverified']
    state['ranking'] = {'unresolved': [{'symbol': 'OLDQ', 'reason': 'quote dated 2026-10-21'}]}
    with pytest.raises(reports.ReportInputError, match="include a 'data_limitation' key finding"):
        reports.build(state, {'key_findings': [{'kind': 'vs_sp500'}, {'kind': 'drawdown'}]}, settings)
    result = reports.build(state, {'key_findings': [{'kind': 'data_limitation'}, {'kind': 'vs_sp500'}]}, settings)
    assert 'not a confirmed #1' in result.key_findings[0]['text']
    assert result.narrative.count('not a confirmed #1') == 1
    assert [f['kind'] for f in reports.build(state, {}, settings).key_findings][-1] == 'data_limitation'


def test_no_limitation_to_feature_is_rejected(state, settings):
    with pytest.raises(reports.ReportInputError, match='no material data limitation'):
        reports.build(state, {'key_findings': [{'kind': 'data_limitation'}, {'kind': 'vs_sp500'}]}, settings)


def test_volume_finding_is_computed_from_the_verified_dataset(state, settings):
    import pandas as pd
    meta = state['datasets'][state['selected']['dataset_id']]
    frame = pd.read_csv(meta['path'])
    earlier = frame['volume'][:-1]
    expected = frame['volume'].iloc[-1] / earlier[earlier > 0].mean()
    result = reports.build(state, {'key_findings': [{'kind': 'volume_spike'}, {'kind': 'window_return'}]}, settings)
    from nasdaq_agent.reporting.readable import NORMAL_VOLUME, times
    wording = 'in line with the previous' if NORMAL_VOLUME[0] <= expected <= NORMAL_VOLUME[1] else f'{times(expected)} times the previous'
    assert wording in result.key_findings[0]['text']


@pytest.mark.parametrize('profile', range(8))
def test_three_findings_with_news_and_limits_fit_the_word_budget(profile, tmp_path, settings):
    state = load_state(tmp_path, make_case(profile, Random(profile), tmp_path))
    state['selected']['flags'] = ['ranking_unverified', 'eligibility_unconfirmed']
    state['session']['is_intraday'] = True
    hid = next(iter(state['headlines']), None)   # two profiles have no fetched news
    second = {'kind': 'news', 'ref': hid} if hid else {'kind': 'drawdown'}
    findings = [{'kind': 'data_limitation'}, second, {'kind': 'volume_spike'}]
    result = reports.build(state, {'selected_news_ids': list(state['headlines']), 'key_findings': findings}, settings)
    assert 0 < result.word_count <= 250
    assert 'intraday' in result.narrative and 'not a confirmed #1' in result.narrative


@pytest.mark.parametrize('shape', [
    '[{"kind": "vs_sp500"}, {"kind": "drawdown"}]',      # JSON string (seen from MiMo)
    ['vs_sp500', 'drawdown'],                             # bare kinds
    'vs_sp500, drawdown',                                 # comma-separated string
])
def test_model_shapes_of_key_findings_are_normalised(state, settings, shape):
    result = reports.build(state, {'key_findings': shape}, settings)
    assert [f['kind'] for f in result.key_findings] == ['vs_sp500', 'drawdown'] and result.findings_source == 'agent'


def test_news_shorthand_and_strict_rules_still_apply(state, settings):
    hid = next(iter(state['headlines']))
    assert reports.build(state, {'key_findings': ['final_session_share', f'news:{hid}']}, settings).key_findings[1]['ref'] == hid
    with pytest.raises(reports.ReportInputError, match='key_findings must be a list'):
        reports.build(state, {'key_findings': '["made_up_kind", "drawdown"]'}, settings)


def test_send_tool_accepts_a_json_string_of_findings(ctx, state):
    from nasdaq_agent.tools.emailer import SendEmailInput
    parsed = SendEmailInput.model_validate({'key_findings': '[{"kind": "vs_sp500"}, {"kind": "drawdown"}]'})
    assert isinstance(parsed.key_findings, str)


def test_stock_specific_move_features_both_indexes_in_one_finding(tmp_path, settings):
    state = load_state(tmp_path, make_case(4, Random(4), tmp_path))      # stock +30%, both indexes within about 1%
    result = reports.build(state, {'key_findings': [{'kind': 'vs_sp500'}, {'kind': 'window_return'}]}, settings)
    finding = result.key_findings[0]['text']
    assert 'S&P 500' in finding and 'NASDAQ Composite' in finding and finding.endswith('[d2][d3][d4].')
    lead = result.narrative.split('**Key findings.**')[1]
    assert lead.count('NASDAQ Composite') == 1
    both = reports.build(state, {'key_findings': [{'kind': 'vs_sp500'}, {'kind': 'vs_nasdaq'}]}, settings)
    assert both.narrative.count('much larger than either index') == 1


def test_dividend_from_corporate_action_data_is_measured_against_the_prior_close(state, settings):
    import pandas as pd
    frame = pd.read_csv(state['datasets'][state['selected']['dataset_id']]['path'])
    ex_date, prior_close = str(frame['date'].iloc[3])[:10], float(frame['close'].iloc[2])
    state['checks'] = {**(state.get('checks') or {}), 'dividends': [{'ex_date': ex_date, 'amount': 0.5}],
                       'corporate_actions': [f'{ex_date}: dividend $0.50/share']}
    result = reports.build(state, {'key_findings': [{'kind': 'dividend'}, {'kind': 'window_return'}]}, settings)
    finding = result.key_findings[0]['text']
    assert finding.startswith('A $0.50 per-share dividend went ex on')
    assert f'close of ${prior_close:,.2f}' in finding and 'not dividend-adjusted' in finding
    assert 'Corporate-action data records' not in result.narrative       # the dividend sentence replaces the generic note
    assert 'dividend went ex' in reports.build(state, {}, settings).narrative


def test_dividend_finding_needs_corporate_action_data(state, settings):
    with pytest.raises(reports.ReportInputError, match="key finding 'dividend' is unavailable"):
        reports.build(state, {'key_findings': [{'kind': 'dividend'}, {'kind': 'window_return'}]}, settings)


def _dividend_headline(state, published='2025-02-25T14:00:00+00:00'):
    state['headlines']['n1'] = {**state['headlines']['n1'], 'published': published,
                                'title': 'ZB001 Declares Special Dividend of $3.50 Per Share'}
    return state


def test_headline_dividend_is_attributed_and_measured_against_the_close_before_publication(state, settings):
    args = {'selected_news_ids': ['n1'], 'key_findings': [{'kind': 'dividend'}, {'kind': 'window_return'}]}
    finding = reports.build(_dividend_headline(state), args, settings).key_findings[0]['text']
    assert finding.startswith('The Synthetic source headline [n1] states a $3.50 per-share dividend, equal to 9.37% of ZB001')
    assert "Feb 24 close of $37.37, the last close before it was published" in finding and finding.endswith('[d2].')
    # Published before the open: the same day's close came after it. After 16:00 ET: it came before.
    pre_open = reports.build(_dividend_headline(state, '2025-02-24T14:00:00+00:00'), args, settings).key_findings[0]['text']
    after_close = reports.build(_dividend_headline(state, '2025-02-24T22:00:00+00:00'), args, settings).key_findings[0]['text']
    assert 'Feb 21 close of $37.74' in pre_open and 'Feb 24 close of $37.37' in after_close


def test_headline_dividend_needs_a_cited_headline_and_yields_to_corporate_action_data(state, settings):
    _dividend_headline(state)
    with pytest.raises(reports.ReportInputError, match="key finding 'dividend' is unavailable"):
        reports.build(state, {'key_findings': [{'kind': 'dividend'}, {'kind': 'window_return'}]}, settings)
    state['checks'] = {**(state.get('checks') or {}), 'dividends': [{'ex_date': '2025-02-20', 'amount': 0.5}]}
    args = {'selected_news_ids': ['n1'], 'key_findings': [{'kind': 'dividend'}, {'kind': 'window_return'}]}
    assert reports.build(state, args, settings).key_findings[0]['text'].startswith('A $0.50 per-share dividend went ex on Feb 20')


@pytest.mark.parametrize('title, amount', [
    ('AI OKTO Corp. Declares Special Dividend of $3.50 Per Share', 3.5),
    ('Apple declares $0.27/share quarterly dividend', 0.27),
    ('Board approves dividend of $ 0.10 a share', 0.1),
    ('AI OKTO Corp. Declares Special Dividend Per Share', None),                         # no amount
    ('XYZ Declares $1.00 Per Share Dividend and $0.25 Per Share Special Dividend', None),  # ambiguous
    ('XYZ suspends $0.50 per share dividend', None),
    ('XYZ will not pay $1 per share dividend', None),
    ('XYZ to buy back shares at $2 per share', None),                                    # not a dividend
])
def test_headline_dividend_amount_is_read_only_when_unambiguous(title, amount):
    assert reports.headline_dividend(title) == amount


def test_split_on_the_trade_date_is_explained_in_plain_words(state, settings):
    trade = state['session']['trade_date']
    state['checks'] = {**(state.get('checks') or {}), 'splits': [{'date': trade, 'ratio': 0.5}],
                       'corporate_actions': [f'{trade}: reverse split 1-for-2']}
    narrative = reports.build(state, {}, settings).narrative
    assert 'Price data records a 1-for-2 reverse split on Feb 25' in narrative
    assert 'price adjustment remains unconfirmed' in narrative
    assert 'Corporate-action data records' not in narrative


def test_split_wording_follows_what_the_bars_showed(state, settings):
    trade = state['session']['trade_date']
    split = {'date': trade, 'ratio': 0.5, 'status': 'no_gap', 'open': 0.84, 'previous_close': 0.806}
    state['checks'] = {**(state.get('checks') or {}), 'splits': [split],
                       'corporate_actions': [f'{trade}: reverse split 1-for-2']}
    narrative = reports.build(state, {}, settings).narrative
    assert 'Price data records a 1-for-2 reverse split on Feb 25' in narrative
    assert 'price adjustment remains unconfirmed' in narrative
    assert '$0.8400' not in narrative and '$0.8060' not in narrative
    assert 'came from trading' not in narrative
    assert 'may reflect the split' not in narrative
    state['checks']['splits'] = [{**split, 'status': 'adjusted'}]
    state['checks']['split_move'] = {'date': trade, 'ratio': 0.5, 'screener_pct': 99.75, 'adjusted_pct': 2.0}
    assert 'earlier prices and volumes were restated [d2]' in reports.build(state, {}, settings).narrative
    assert 'screener\'s +99.75%' not in reports.build(state, {}, settings).narrative


def test_a_stock_excluded_as_a_split_artefact_is_retained_in_the_run_record(state, settings, tmp_path):
    from nasdaq_agent.reporting.render import build_report
    state['selection'] = {**(state.get('selection') or {}), 'split_excluded': [
        {'symbol': 'GCTK', 'date': state['session']['trade_date'], 'ratio': 0.5, 'screener_pct': 99.75, 'adjusted_pct': -0.12}]}
    report = reports.build(state, {}, settings)
    rendered = build_report(state, report.narrative, run_dir=tmp_path, run_info={}, citations=report.citations)
    assert 'GCTK' not in rendered.html
    assert rendered.facts['selection']['split_excluded'][0]['symbol'] == 'GCTK'
    assert any('Split-adjusted exclusion: GCTK' in note for note in rendered.notes)
    assert 'GCTK' not in report.narrative


def test_split_headline_triggers_a_check_against_split_data(state, settings):
    state['headlines']['n1'] = {**state['headlines']['n1'], 'title': 'ZB001 Announces Reverse Stock Split'}
    narrative = reports.build(state, {}, settings).narrative
    assert 'A fetched headline reports a stock split [n1], but the split data for this price window shows none' in narrative
    state['selected']['flags'] = [*(state['selected'].get('flags') or []), 'corporate_actions_unchecked']
    assert 'and split data could not be checked' in reports.build(state, {}, settings).narrative
    state['headlines']['n1']['title'] = 'ZB001 reported quarterly results.'
    assert 'Split and dividend data could not be checked' in reports.build(state, {}, settings).narrative


def test_report_text_uses_the_verified_day_move_not_the_screener(state, settings):
    import pandas as pd
    closes = pd.read_csv(state['datasets'][state['selected']['dataset_id']]['path'])['close']
    verified = closes.iloc[-1] / closes.iloc[-2] - 1
    state['selected']['pct_change'] = 4.0                     # a screener figure that disagrees
    narrative = reports.build(state, {}, settings).narrative
    brief = narrative.split('**Why we picked it.**')[0]
    assert f'fell {abs(verified) * 100:.2f}% on Feb 25' in brief and '4.0' not in narrative
