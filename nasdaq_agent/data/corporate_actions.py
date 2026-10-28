"""Detect splits / reverse splits / special dividends that distort a price move."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass
class CorporateAction:
    date: str
    kind: str        # "split" | "reverse_split" | "dividend"
    value: float     # split ratio (new/old shares) or dividend per share

    def describe(self) -> str:
        if self.kind == "dividend":
            return f"{self.date}: dividend ${self.value:.2f}/share"
        if self.kind == "reverse_split":
            return f"{self.date}: reverse split 1-for-{1 / self.value:.0f}"
        return f"{self.date}: split {self.value:g}-for-1"


def from_events(splits: dict[date, float], dividends: dict[date, float]) -> list[CorporateAction]:
    actions = []
    for d, ratio in sorted(splits.items()):
        if ratio and ratio != 1:
            actions.append(CorporateAction(str(d), "reverse_split" if ratio < 1 else "split", float(ratio)))
    for d, amount in sorted(dividends.items()):
        if amount:
            actions.append(CorporateAction(str(d), "dividend", float(amount)))
    return actions


def split_name(ratio: float) -> str:
    return f"1-for-{1 / ratio:.0f} reverse split" if ratio < 1 else f"{ratio:g}-for-1 split"


@dataclass
class SplitCheck:
    date: str
    ratio: float                    # new shares per old share; 0.5 is a 1-for-2 reverse split
    status: str                     # "adjusted" | "no_gap" | "unknown"
    open: float | None = None       # split-day open before any adjustment
    previous_close: float | None = None


def adjust_for_splits(frame: pd.DataFrame, actions: list[CorporateAction]) -> tuple[pd.DataFrame, list[SplitCheck]]:
    """Put bars before a split on the post-split basis when the prices still jump by the split factor.

    The ratio comes from corporate-action data; the bars decide whether it applies. An unadjusted
    series gaps from the previous close to the split-day open by about 1/ratio. A vendor that
    already adjusted, or a split that had not reached trading yet, shows no such gap and is left
    as it is. The open, not the close, is the test: a close can move by the split factor in trading."""
    frame = frame.copy()
    checks = []
    days = [str(d) for d in frame["date"]]
    for action in sorted((a for a in actions if a.kind != "dividend"), key=lambda a: a.date, reverse=True):
        if action.date not in days or days.index(action.date) == 0:
            continue                # outside the bars, or no earlier bar to restate
        row = days.index(action.date)
        opened, previous = float(frame["open"].iloc[row]), float(frame["close"].iloc[row - 1])
        if not (opened > 0 and previous > 0):
            checks.append(SplitCheck(action.date, action.value, "unknown"))
            continue
        gap = math.log(opened / previous)
        if abs(gap + math.log(action.value)) < abs(gap):      # nearer 1/ratio than 1
            earlier = frame.index[:row]
            frame = frame.astype({c: float for c in ("open", "high", "low", "close", "volume")})
            frame.loc[earlier, ["open", "high", "low", "close"]] /= action.value
            frame.loc[earlier, "volume"] *= action.value
            status = "adjusted"
        else:
            status = "no_gap"
        checks.append(SplitCheck(action.date, action.value, status, opened, previous))
    return frame, sorted(checks, key=lambda c: c.date)


def possible_split_artifact(actions: list[CorporateAction], trade_date: date, screener_pct: float) -> bool:
    """A reverse split on the trade date with a gain close to the split factor is not a real gain.

    e.g. 1-for-10 reverse split -> an unadjusted feed shows roughly +900%.
    """
    for action in actions:
        if action.kind == "reverse_split" and action.date == str(trade_date):
            implied = (1 / action.value - 1) * 100
            if implied > 0 and abs(screener_pct / implied - 1) < 0.35:
                return True
    return False
