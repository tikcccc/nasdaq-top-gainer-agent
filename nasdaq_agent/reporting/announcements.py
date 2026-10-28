"""Match reviewed primary documents to a recorded corporate action, never by ticker alone.

The catalogue contains immutable source documents reviewed separately from headline
selection. An unmatched event retains attribution to price data; no URL is guessed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


CATALOGUE = Path(__file__).parents[1] / 'data' / 'primary_announcements.json'


def matched(state: dict) -> dict[str, dict]:
    symbol = (state.get('selected') or {}).get('symbol')
    splits = (state.get('checks') or {}).get('splits') or []
    if not symbol or not splits:
        return {}
    catalogue = json.loads(CATALOGUE.read_text(encoding='utf-8'))
    out = {}
    for item in catalogue:
        if item.get('symbol') != symbol or not item.get('reviewed_on'):
            continue
        if any(item['effective_date'] == split['date'] and
               math.isclose(item['ratio'], split['ratio'], rel_tol=1e-9) for split in splits):
            out[f'd{8 + len(out)}'] = item
    return out


def for_split(state: dict, split: dict) -> tuple[str, dict] | None:
    return next(((ref, item) for ref, item in matched(state).items()
                 if item['effective_date'] == split['date'] and
                 math.isclose(item['ratio'], split['ratio'], rel_tol=1e-9)), None)
