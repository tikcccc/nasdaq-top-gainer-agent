"""Deterministic binding of report numbers to the current immutable data snapshot."""

import hashlib
from pathlib import Path

import pandas as pd

from nasdaq_agent.data import metrics


def analysis_problems(state: dict) -> list[str]:
    selected, analysis = state.get("selected") or {}, state.get("analysis") or {}
    datasets = state.get("datasets") or {}
    problems = []
    if not selected.get("analysis_key"):
        return ["no versioned data snapshot; fetch history and generate analysis again"]
    if analysis.get("analysis_key") != selected["analysis_key"]:
        return ["analysis belongs to a different stock, date or dataset version"]
    try:
        for ds, meta in datasets.items():
            if hashlib.sha256(Path(meta["path"]).read_bytes()).hexdigest() != meta["sha256"]:
                problems.append(f"{ds}: data changed after validation")
        if problems:
            return problems
        stock = pd.read_csv(datasets[selected["dataset_id"]]["path"])
        benches = {ds[3:]: pd.read_csv(m["path"]) for ds, m in datasets.items()
                   if m.get("proxy_for") or m["symbol"].startswith("^")}
        return metrics.compare(analysis, metrics.compute(stock, benches, selected["days"], selected["symbol"]))
    except (KeyError, OSError, ValueError) as exc:
        return [f"cannot verify current data snapshot: {exc}"]
