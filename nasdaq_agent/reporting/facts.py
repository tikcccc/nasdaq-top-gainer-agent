"""One snapshot and one calculation basis for report prose, cards, tables and charts."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from nasdaq_agent.data import metrics


def volume_facts(frame: pd.DataFrame | None) -> dict | None:
    if frame is None or len(frame) < 2:
        return None
    volumes = frame['volume'].astype(float)
    earlier = volumes.iloc[:-1]
    earlier = earlier[earlier > 0]
    if earlier.empty or volumes.iloc[-1] <= 0:
        return None
    average = float(earlier.mean())
    return {'shares': float(volumes.iloc[-1]), 'average': average,
            'ratio': float(volumes.iloc[-1]) / average, 'sessions': len(earlier)}


@dataclass(frozen=True)
class ReportFacts:
    symbol: str
    trade_date: str
    days: int
    price: float | None
    shares: float | None
    day_return: float | None
    analysis: dict
    source: str
    captured_at: str | None
    basis: str
    is_intraday: bool
    volume_comparable: bool
    frames: dict[str, pd.DataFrame] = field(repr=False)
    stock_id: str = ''
    snapshot_id: str | None = None

    @property
    def stock(self) -> pd.DataFrame | None:
        return self.frames.get(self.stock_id)

    @property
    def as_of_date(self) -> str:
        return str(self.stock['date'].iloc[-1]) if self.stock is not None else self.trade_date

    @property
    def volume(self) -> dict | None:
        return volume_facts(self.stock) if self.volume_comparable else None

    def record(self) -> dict:
        stock = self.stock
        return {'symbol': self.symbol, 'trade_date': self.trade_date, 'days': self.days,
                'dataset_id': self.stock_id, 'snapshot_id': self.snapshot_id,
                'price': self.price, 'volume': self.shares, 'day_return': self.day_return,
                'window_return': self.analysis.get('cumulative_return'),
                'excess_return_pp': (self.analysis['excess_return'] * 100
                                     if self.analysis.get('excess_return') is not None else None),
                'volume_comparison': self.volume, 'source': self.source, 'captured_at': self.captured_at,
                'price_basis': self.basis, 'is_intraday': self.is_intraday,
                'baseline_date': str(stock['date'].iloc[0]) if stock is not None else None,
                'last_bar_date': str(stock['date'].iloc[-1]) if stock is not None else None}


def build_report_facts(state: dict) -> ReportFacts:
    selected, session = state.get('selected') or {}, state.get('session') or {}
    datasets = state.get('datasets') or {}
    stock_id = selected.get('dataset_id', '')
    days = int(selected.get('days') or (state.get('analysis') or {}).get('days') or 5)
    trade_date = str(session.get('trade_date') or '')
    frames = {}
    for ds, meta in datasets.items():
        if ds != stock_id and not (meta.get('proxy_for') or meta.get('symbol', '').startswith('^')):
            continue
        frame = pd.read_csv(meta['path'])
        frame['date'] = frame['date'].astype(str).str[:10]
        if trade_date:
            frame = frame[frame['date'] <= trade_date]
        frame = frame.tail(days + 1).reset_index(drop=True)
        if not frame.empty:
            frames[ds] = frame
    stock = frames.get(stock_id)
    # A benchmark with different dates must never be plotted or compared by row position.
    if stock is not None:
        frames = {ds: f for ds, f in frames.items() if f['date'].tolist() == stock['date'].tolist()}
    analysis = state.get('analysis') or {}
    if stock is not None and len(stock) == days + 1 and analysis:
        benches = {(datasets[ds].get('proxy_for') or datasets[ds]['symbol']).lstrip('^'): frame
                   for ds, frame in frames.items() if ds != stock_id}
        analysis = metrics.compute(stock, benches, days, selected['symbol'])
    price, shares, move = selected.get('price'), selected.get('volume'), None
    if stock is not None:
        price, shares = float(stock['close'].iloc[-1]), float(stock['volume'].iloc[-1])
        if len(stock) >= 2:
            move = price / float(stock['close'].iloc[-2]) - 1
    meta = datasets.get(stock_id) or {}
    splits = (state.get('checks') or {}).get('splits') or []
    basis = 'Reported closes; price returns are not dividend-adjusted.'
    if any(s.get('status') != 'adjusted' for s in splits):
        basis += ' Split adjustment is not independently confirmed; volume comparability across the split is unconfirmed.'
    elif splits:
        basis += ' Earlier prices and volumes were locally restated for the recorded split.'
    return ReportFacts(selected.get('symbol', 'n/a'), trade_date, days, price, shares, move, analysis,
                       meta.get('source') or (state.get('sources') or {}).get('history') or selected.get('source') or 'unknown',
                       meta.get('captured_at'), basis, bool(session.get('is_intraday')),
                       not any(s.get('status') != 'adjusted' for s in splits), frames, stock_id,
                       selected.get('analysis_key'))
