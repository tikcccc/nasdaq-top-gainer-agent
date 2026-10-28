import copy
import json

import pytest

from scripts.eval_historical import MANIFEST, evaluate, verify_recording
from scripts.eval_agent import build_report

DATA=json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", DATA["cases"], ids=lambda c:c["id"])
def test_recorded_historical_contract(case,settings):
    row=evaluate(case,settings,"contract")
    assert row["scenario_pass"],row
    assert row["ranking_accuracy"] is None


def test_tampered_recording_is_rejected():
    case=copy.deepcopy(DATA["cases"][0])
    case["series"][case["symbol"]]["sha256"]="0"*64
    with pytest.raises(ValueError,match="checksum"):
        verify_recording(case)


def test_corporate_events_and_holiday_window_are_present():
    split=next(c for c in DATA["cases"] if c["id"]=="nvda_split_dividend")
    events=split["series"]["NVDA"]["events"]
    assert next(iter(events["splits"].values()))["splitRatio"]=="10:1"
    assert next(iter(events["dividends"].values()))["amount"]==.01
    halfday=next(c for c in DATA["cases"] if c["id"]=="msft_half_day")
    assert halfday["calendar_close_utc"]=="2024-11-29T18:00:00+00:00"
    assert "2024-11-28" not in {r["date"] for r in halfday["series"]["MSFT"]["bars"]}


def test_failed_cases_stay_in_latency_and_completion_denominators():
    from types import SimpleNamespace
    args=SimpleNamespace(mode="live",split="test",allow_rule_guard=False)
    rows=[{"id":"a","regime":"up","elapsed_s":10,"agent_completed":True,"scenario_pass":True},
          {"id":"b","regime":"ipo","elapsed_s":20,"fallback":True,"scenario_pass":True},
          {"id":"c","regime":"down","elapsed_s":80,"fallback":True,"guard_rejections":3}]
    report=build_report({"version":1,"limitations":"test"},args,rows,{"case_ids":["a","b","c","d"]})
    assert report["count"]==3 and not report["complete"]
    assert report["agent_completed"]==1 and report["scenarios_passed"]==2
    assert report["fallback_count"]==2 and report["latency_s"]=={"p50":20,"p95":80,"max":80}


def test_new_semantic_partition_contexts_do_not_overlap():
    data=json.loads((MANIFEST.parents[1]/"guard_cases_v2.json").read_text(encoding="utf-8"))
    calibration={c["context_id"] for c in data["cases"] if c["split"]=="calibration"}
    test={c["context_id"] for c in data["cases"] if c["split"]=="test"}
    assert not calibration & test
    for split in ("calibration","test"):
        rows=[c for c in data["cases"] if c["split"]==split]
        assert sum(c["label"]=="pass" for c in rows)==sum(c["label"]=="reject" for c in rows)


def test_corrected_semantic_identities_preserve_text_and_labels():
    fixtures=MANIFEST.parents[1]
    original=json.loads((fixtures/"guard_cases_v3.json").read_text(encoding="utf-8"))
    corrected=json.loads((fixtures/"guard_cases_v4.json").read_text(encoding="utf-8"))
    old={c["id"]:c for c in original["cases"]}
    assert set(old)=={c["id"] for c in corrected["cases"]}
    for case in corrected["cases"]:
        assert all(case[key]==old[case["id"]][key] for key in ("text","label","split","context_id"))
        symbol=case["evidence"]["metrics_and_context"]["symbol"]
        assert all(h["title"].split()[0]==symbol for h in case["evidence"]["headlines"].values())
