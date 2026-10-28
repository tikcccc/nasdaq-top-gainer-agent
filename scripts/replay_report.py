"""Render a saved API run/series snapshot with the current report code, without sending.

Usage: python -m scripts.replay_report --run run.json --series series.json --out outputs/review
The original archives remain unchanged. CSVs are reserialised from the API's
full-precision series JSON, and their new hashes are bound to fresh analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nasdaq_agent.agent import guard
from nasdaq_agent.config import load_settings
from nasdaq_agent.data import metrics
from nasdaq_agent.infra.mailer import build_message
from nasdaq_agent.reporting import evidence_report, placeholders, readable
from nasdaq_agent.reporting.facts import build_report_facts
from nasdaq_agent.reporting.render import build_report


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def replay(run_file: Path, series_file: Path, out: Path) -> dict:
    original = json.loads(run_file.read_text(encoding='utf-8'))
    summary = original['summary']
    series = json.loads(series_file.read_text(encoding='utf-8'))
    symbol, trade_date = summary['selected']['symbol'], summary['trade_date']
    if out.resolve() in {run_file.parent.resolve(), series_file.parent.resolve()}:
        raise ValueError('write the replay into a separate directory')
    (out / 'data').mkdir(parents=True, exist_ok=True)
    trace = original.get('trace') or []
    history = next((e for e in reversed(trace) if e.get('tool') == 'get_price_history'
                    and e.get('event') == 'tool_result' and e.get('ok')), {})
    metas = {m['dataset_id']: m for m in (history.get('data') or {}).get('datasets', [])}
    captured = datetime.fromtimestamp(history['ts'], timezone.utc).isoformat() if history.get('ts') else None
    datasets = {}
    for key, bars in series.items():
        ds = f'ds_{key}'
        frame = pd.DataFrame(bars)
        path = out / 'data' / f'{ds}.csv'
        frame.to_csv(path, index=False)
        datasets[ds] = {**metas.get(ds, {}), 'path': str(path.resolve()),
                        'symbol': symbol if key == symbol else '^' + key,
                        'first_date': frame['date'].iloc[0], 'last_date': frame['date'].iloc[-1],
                        'sha256': digest(path), 'captured_at': captured, 'trade_date': trade_date}
    screener = next((e.get('data') or {} for e in reversed(trace)
                     if e.get('tool') == 'get_top_nasdaq_gainers' and e.get('ok')), {})
    candidate = next((c for c in screener.get('candidates', []) if c['symbol'] == symbol), {})
    key = hashlib.sha256(json.dumps({ds: m['sha256'] for ds, m in datasets.items()}, sort_keys=True).encode()).hexdigest()
    selected = {**candidate, **summary['selected'], 'dataset_id': f'ds_{symbol}',
                'analysis_key': key, 'trade_date': trade_date}
    frames = {ds: pd.read_csv(m['path']) for ds, m in datasets.items()}
    computed = metrics.compute(frames[selected['dataset_id']],
                               {ds[3:]: f for ds, f in frames.items() if ds != selected['dataset_id']},
                               selected['days'], symbol)
    problems = metrics.compare(summary['analysis'], computed)
    if problems:
        raise ValueError('snapshot differs from archived analysis: ' + '; '.join(problems))
    state = {k: summary.get(k) for k in ('selection', 'ranking', 'raw_top', 'headlines', 'sentiment', 'sources')}
    state.update(selected=selected, datasets=datasets, session={'trade_date': trade_date, 'is_intraday': False},
                 analysis={**computed, 'analysis_key': key, 'origin': 'archived_series_replay'},
                 checks={k: v for k, v in (summary.get('checks') or {}).items() if k != 'guard'})
    news_ids = list(evidence_report.current_headlines(state))[:3]
    facts = build_report_facts(state)
    findings = ['final_session_share', *(['news:' + news_ids[0]] if news_ids else [])]
    if facts.volume:
        findings.append('volume_spike')
    if evidence_report._material(state):
        findings = findings[:2] + ['data_limitation']
    args = {'selected_news_ids': news_ids, 'key_findings': findings}
    settings = load_settings(overrides={'email': {'dry_run': True}})
    issues, checked = guard.check(state, args, settings)
    if issues:
        raise ValueError('; '.join(issues))
    body = evidence_report.build(state, args, settings)
    report = build_report(state, body.narrative, run_dir=out, run_info={'run_id': 'replay-' + original['run_id']},
                          citations=body.citations + body.omitted_ids)
    (out / 'email.html').write_text(report.html, encoding='utf-8')
    preview = report.html
    for cid, path in report.images.items():
        preview = preview.replace(f'cid:{cid}', path.name)
    (out / 'preview.html').write_text(preview, encoding='utf-8')
    (out / 'email.txt').write_text(report.text, encoding='utf-8')
    msg = build_message(report, 'review@example.invalid', ['review@example.invalid'])
    (out / 'email.eml').write_bytes(bytes(msg))
    write_json(out / 'state.json', state)
    source_record = {'run_id': original['run_id'], 'run_sha256': digest(run_file),
                     'series_sha256': digest(series_file), 'csv_basis': 'reserialised API series, same numerical values'}
    new_summary = {'status': 'preview', 'sent': False, 'trade_date': trade_date, 'source_snapshot': source_record,
                   'selected': selected, 'analysis': state['analysis'], 'guard': checked,
                   'email': {'report_mode': body.version, 'report_renderer': report.renderer,
                             'word_count': body.word_count, 'key_findings': body.key_findings,
                             'narrative': body.narrative, 'report_facts': report.facts, 'notes': report.notes}}
    write_json(out / 'summary.json', new_summary)
    checks = {
        'analysis_matches_original_snapshot': not problems,
        'at_most_three_findings': len(body.key_findings) <= 3,
        'word_limit_250': 0 < body.word_count <= 250,
        'unverified_market_cap_hidden': 'mkt cap' not in report.html.lower()
                                        and report.facts['screener_market_cap'] == candidate.get('market_cap'),
        'consistent_day_precision': placeholders.pct(report.facts['day_return']) in report.html
                                    and readable.pct(report.facts['day_return']) in body.narrative,
        'no_legacy_how_it_traded': '**How it traded.**' not in body.narrative,
        'no_unused_profile_filings_or_raw_leader': all(t not in report.html for t in ('company profile', 'company filings',
                                                                                     'unfiltered leader')),
        'renderer_in_html_and_mime': report.renderer['fingerprint'] in report.html
                                   and msg['X-Report-Fingerprint'] == report.renderer['fingerprint'],
        'no_delivery': True,
    }
    if facts.volume:
        multiple = readable.times(facts.volume['ratio'])
        checks['volume_uses_history_numerator_and_baseline'] = bool(
            facts.volume['shares'] == facts.shares and f'{multiple}×' in report.html
            and facts.volume['ratio'] == float(facts.stock['volume'].iloc[-1]) / facts.stock['volume'].iloc[:-1].mean())
    shock = evidence_report._prior_shock(facts, evidence_report.current_headlines(state))
    if shock:
        checks['earlier_shock_survives_explicit_model_choices'] = any(f['kind'] == 'prior_shock' for f in body.key_findings)
        checks['no_invented_shock_cause'] = 'the cause has not been verified' in body.narrative
    result = {'passed': all(checks.values()), 'checks': checks, 'renderer': report.renderer,
              'word_count': body.word_count, 'source_snapshot': source_record, 'prior_shock': shock}
    write_json(out / 'acceptance.json', result)
    if not result['passed']:
        raise ValueError('acceptance failed: ' + ', '.join(k for k, v in checks.items() if not v))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--series', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    options = parser.parse_args()
    print(json.dumps(replay(options.run, options.series, options.out), ensure_ascii=False, indent=2))
