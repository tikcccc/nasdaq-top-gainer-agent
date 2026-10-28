"""Score the semantic checker on explicit labels, including every false acceptance/rejection."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.legacy import narrative_guard as guard
from nasdaq_agent.config import load_settings
from nasdaq_agent.llm import jev

CASES = Path(__file__).resolve().parent.parent / "tests/fixtures/guard_cases.json"


def score(case, settings, method):
    if method == "jev":
        if not jev.available(settings):
            raise RuntimeError("Jev unavailable; refusing to substitute rules in a Jev evaluation")
        issues = guard._jev_consistency(settings, case["text"], case["evidence"])
        probability = guard.speculation_scores(settings, case["evidence"], [case["text"]])[0]
        if probability > settings.jev.speculation_threshold:
            issues.append("unsupported causal claim")
    else:
        issues = guard._rule_consistency(case["text"], case["evidence"]["metrics_and_context"])
        issues += guard._rule_speculation("**What's behind it.** " + case["text"])
    return "reject" if issues else "pass", issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["rules", "jev"], default="rules")
    parser.add_argument("--split", choices=["calibration", "test", "all"], default="test")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/semantic-guard.json"))
    parser.add_argument("--cases", type=Path, default=CASES)
    args = parser.parse_args()
    data = json.loads(args.cases.read_text(encoding="utf-8"))
    settings = load_settings()
    results=[]
    for case in data["cases"]:
        if args.split != "all" and case["split"] != args.split:
            continue
        try:
            predicted, issues = score(case, settings, args.method)
            row={"id":case["id"],"category":case["category"],"expected":case["label"],
                 "predicted":predicted,"correct":predicted==case["label"],"issues":issues}
        except Exception as exc:
            row={"id":case["id"],"expected":case["label"],"predicted":"error","correct":False,"error":str(exc)}
        results.append(row)
        print(json.dumps(row),flush=True)
    rejects=sum(r["expected"]=="reject" for r in results)
    passes=sum(r["expected"]=="pass" for r in results)
    false_accepts=sum(r["expected"]=="reject" and r["predicted"]=="pass" for r in results)
    false_rejects=sum(r["expected"]=="pass" and r["predicted"]=="reject" for r in results)
    report={"dataset_sha256":hashlib.sha256(args.cases.read_bytes()).hexdigest(),"version":data["version"],
            "guard_sha256":hashlib.sha256(Path(guard.__file__).read_bytes()).hexdigest(),
            "method":args.method,"model":settings.jev.model if args.method=="jev" else None,"split":args.split,
            "thresholds":{"consistency":settings.jev.consistency_threshold,"speculation":settings.jev.speculation_threshold,
                          "unsupported_fact":settings.jev.unsupported_fact_threshold},
            "count":len(results),"correct":sum(r["correct"] for r in results),"false_accepts":false_accepts,
            "false_rejects":false_rejects,"false_accept_rate":false_accepts/rejects if rejects else None,
            "false_reject_rate":false_rejects/passes if passes else None,
            "errors":sum(r["predicted"]=="error" for r in results),"limitations":data["limitations"],"results":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in {"results","limitations"}},indent=2))
    return 0 if results and all(r["correct"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
