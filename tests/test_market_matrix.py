import json
from pathlib import Path

import pytest

from scripts.eval_agent import CASES, evaluate

DATA = json.loads(CASES.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda case: case["id"] + "_" + case["regime"])
def test_market_scenario_contract(case, settings):
    result = evaluate(case, settings, "contract")
    assert result["contract_pass"], result


def test_held_out_dates_and_symbols_are_disjoint():
    calibration = [c for c in DATA["cases"] if c["split"] == "calibration"]
    test = [c for c in DATA["cases"] if c["split"] == "test"]
    assert len(calibration) == 16 and len(test) == 24
    for field in ("id", "symbol", "trade_date"):
        assert not ({c[field] for c in calibration} & {c[field] for c in test})
    assert len({c["regime"] for c in test}) == 8


def test_semantic_guard_labels_and_context_split():
    data = json.loads((Path(__file__).parent / "fixtures/guard_cases.json").read_text())
    assert len(data["cases"]) == 40
    calibration = {c["context_id"] for c in data["cases"] if c["split"] == "calibration"}
    test = {c["context_id"] for c in data["cases"] if c["split"] == "test"}
    assert not calibration & test
    assert {c["label"] for c in data["cases"]} == {"pass", "reject"}
