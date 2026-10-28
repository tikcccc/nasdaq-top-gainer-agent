"""Package sealed evidence/reports for human review, excluding all model labels."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def package(root: Path) -> Path:
    folder = root / 'human-review'
    packets = [json.loads(line) for line in (folder / 'reports.jsonl').read_text(encoding='utf-8').splitlines()]
    lines = ['# Blind human review packet', '',
             'Synthetic evidence, not observed market records. Review factual support against the supplied evidence. '
             'Do not infer that a fetched news subset is exhaustive. Model annotations and detector decisions are excluded. '
             'Record supported / unsupported / uncertain in labels.csv; quote the issue and reference evidence. '
             'A missing generated report remains a missing case, not an accepted report.', '']
    for packet in packets:
        evidence = packet['evidence']
        lines.extend([f"## Case {packet['case_id']}", '', '### Source evidence', '',
                      f"Symbol: {evidence['symbol']}; company: {evidence['company']}; trade date: {evidence['trade_date']}", '',
                      f"Quote: {evidence['price']}; daily change: {evidence['daily_change_percent']:.8f}%; volume: {evidence['volume']}", '',
                      'Stock and benchmark closes:', '', '| Date | Stock | S&P 500 | NASDAQ Composite |', '|---|---:|---:|---:|'])
        benches = {k: {row['date']: row['close'] for row in rows} for k, rows in evidence['benchmarks'].items()}
        for row in evidence['stock_closes']:
            date = row['date']
            lines.append(f"| {date} | {row['close']} | {benches['GSPC'].get(date, '')} | {benches['IXIC'].get(date, '')} |")
        lines.extend(['', 'Market capitalization / comparative historical distribution: not supplied.', '',
                      'News scope: ' + evidence['news_scope'], ''])
        for hid, headline in evidence['headlines'].items():
            lines.extend([f"- [{hid}] {headline['title']} — {headline['publisher']}; {headline['published']}"])
        if not evidence['headlines']:
            lines.append('Fetched headline set: empty.')
        lines.extend(['', '### Report', '', packet['report'] or '[No generated report; generation error recorded]', '', '---', ''])
    (folder / 'reports.md').write_text('\n'.join(lines), encoding='utf-8')
    output = root / 'human-review.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ('reports.jsonl', 'reports.md', 'labels.csv', 'RUBRIC.txt'):
            archive.write(folder / name, arcname=name)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(package(parser.parse_args().root.resolve()))
