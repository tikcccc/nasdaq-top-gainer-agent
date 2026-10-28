import json
import math
from pathlib import Path

import pytest

from nasdaq_agent.reporting import claims, placeholders


def analysis(returns, bench=None):
    import pandas as pd
    from nasdaq_agent.data.metrics import compute
    closes=[100.0]
    for r in returns:closes.append(closes[-1]*(1+r))
    stock=pd.DataFrame({"close":closes})
    benchmarks={key:pd.DataFrame({"close":[100.0]*(len(closes)-1)+[100*(1+value)]}) for key,value in (bench or {}).items()}
    return compute(stock,benchmarks,len(returns),"UNIT")


@pytest.mark.parametrize("returns,whole,majority", [
    ([.01,.01,.01,.01,.15],False,True),
    ([0,0,0,0,.2],True,False),
    ([.1,-1/11,0,0,.2],True,False),
    ([.1,.1,.1,.1,.01],False,False),
    ([-.02,-.02,-.02,-.02,-.2],False,True),
    ([-.1,-.1,0,0,.5],False,False),
    ([.1,-1/11,0,0,0],False,False),
])
def test_contribution_is_compounded_index_change(returns,whole,majority):
    a=analysis(returns)
    f=claims.path_facts(a)
    assert f["final_accounts_for_all_net_change"] is whole
    assert f["final_accounts_for_majority_net_change"] is majority
    assert math.isclose(f["pre_final_return"]+f["final_index_change"],f["window_return"],abs_tol=1e-12)
    assert "all of the net" in claims.text(a)["contribution"] if whole else "all of the net" not in claims.text(a)["contribution"]


def test_zero_denominator_flat_final_and_missing_benchmark():
    assert "not meaningful" in claims.text(analysis([.1,-1/11,0,0,0]))["contribution"]
    assert "added no material change" in claims.text(analysis([.1,0,0,0,0]))["contribution"]
    assert "No matching benchmark" in claims.text(analysis([.1,0,0,0,0]))["benchmarks"]
    assert claims.path_facts({"daily_returns":[float('nan')]})=={}


BOUND="**How it traded.** {{path.sequence}} {{path.contribution}} {{path.benchmarks}} [d2]\n\n**What's behind it.** {{news.coverage}}"


def test_bound_section_cannot_hide_extra_prose_or_omit_scope():
    state={"selected":{"analysis_key":"snapshot"}}
    assert claims.problems(state,BOUND)==[]
    assert any("bound_price_path_required" in p for p in claims.problems(state,BOUND.replace(" [d2]"," The gain was smooth. [d2]")))
    assert any("bound_news_scope_required" in p for p in claims.problems(state,BOUND.replace("{{news.coverage}}","")))
    duplicate=BOUND+"\n\n**How it traded.** A fictional price path."
    assert any("bound_price_path_required" in p for p in claims.problems(state,duplicate))
    assert "fictional price path" in claims.semantic_narrative(duplicate)


@pytest.mark.parametrize("text,kind", [
    ("The week's entire gain sits in the final session.","unbound_path_claim"),
    ("The bulk of the stock's advance came on the last day.","unbound_path_claim"),
    ("Every close exceeded the previous one.","unbound_path_claim"),
    ("The stock rose steadily throughout the window.","unbound_path_claim"),
    ("One of the sharpest single-session declines a mega-cap can deliver.","unsupported_comparison"),
    ("The stock outperformed the NASDAQ index.","unsupported_comparison"),
    ("Trading was earnings-driven, although no source proves a cause.","unbound_causal_claim"),
    ("The shares rose because of the merger.","unbound_causal_claim"),
    ("This is price action without a publicly reported driver.","unbounded_news_absence"),
])
def test_unbound_interpretations_rejected_in_any_section(text,kind):
    for heading in ("**Why this stock.** ","**What's behind it.** ","- "):
        assert any(kind in p for p in claims.problems({},heading+text))


def test_causal_quote_needs_attribution_exact_text_and_cited_source():
    title="UNIT shares rose on a contract announcement"
    state={"headlines":{"n1":{"title":title}}}
    valid=f'The headline reports "{title}" [n1].'
    assert not claims.problems(state,valid)
    assert claims.problems(state,valid.replace('[n1]','[n2]'))
    assert claims.problems(state,valid.replace('contract','merger'))
    assert claims.problems(state,valid.replace('The headline reports',''))
    assert not claims.problems({},"The evidence does not establish that the move was caused by earnings.")
    assert not claims.problems({},"If financing fails, costs could rise because of a delay.")
    assert claims.problems({},"No evidence is conclusive, but the move was driven by earnings.")


def test_old_accepted_audit_reports_are_now_rejected(history_archive):
    audit=json.loads((history_archive/"narrative-audit.json").read_text(encoding="utf-8"))
    rejected=[r for r in audit["results"] if r["review_label"]=="reject"]
    assert len(rejected)==2
    for r in rejected:
        problems=claims.problems({"selected":{"analysis_key":"snapshot"}},r["narrative"])
        assert any(p.startswith(('unbound_path_claim','unsupported_comparison','unbounded_news_absence')) for p in problems)


def test_news_scope_counts_only_current_symbol_and_date():
    state={"selected":{"symbol":"UNIT"},"session":{"trade_date":"2026-10-23"},"analysis":analysis([0,0,0,0,.1]),
           "headlines":{"n1":{"symbol":"OTHER","trade_date":"2026-10-23"},"n2":{"symbol":"UNIT","trade_date":"2026-10-22"}}}
    assert "returned no headlines" in placeholders.build_context(state)["news"]["coverage"]


def test_path_direction_uses_snapshot_even_when_agent_rounds_tiny_return(tmp_path):
    import pandas as pd
    path=tmp_path/'stock.csv'
    pd.DataFrame({'close':[100,100.000001,100.000002]}).to_csv(path,index=False)
    state={'selected':{'symbol':'UNIT','days':2,'analysis_key':'snapshot','dataset_id':'ds_UNIT'},
           'datasets':{'ds_UNIT':{'path':str(path),'symbol':'UNIT'}},'analysis':{'daily_returns':[0,0]}}
    exact=claims.analysis_from_state(state)
    assert claims.path_facts(exact)['session_directions']==['up','up']


def test_original_gain_counterexample_is_eighty_percent_not_all(tmp_path):
    import pandas as pd
    path=tmp_path/'stock.csv'
    pd.DataFrame({'close':[100,101,102,103,104,120]}).to_csv(path,index=False)
    state={'selected':{'symbol':'UNIT','days':5,'analysis_key':'snapshot','dataset_id':'ds_UNIT'},
           'datasets':{'ds_UNIT':{'path':str(path),'symbol':'UNIT'}}}
    exact=claims.analysis_from_state(state)
    facts=claims.path_facts(exact)
    assert facts['pre_final_return']==pytest.approx(.04)
    assert facts['final_share_of_net_change']==pytest.approx(.8)
    assert 'most (80.0%)' in claims.text(exact)['contribution']
    assert 'all of the net' not in claims.text(exact)['contribution']


def test_direct_email_tool_cannot_bypass_deterministic_claim_gate(settings,monkeypatch):
    from nasdaq_agent.context import start_run
    from nasdaq_agent.tools import emailer
    from pydantic import ValidationError
    start_run(settings)
    called=[]
    monkeypatch.setattr(emailer,'deliver',lambda *a,**k:called.append(True))
    state={'selected':{'symbol':'UNIT','analysis_key':'snapshot'},'analysis':analysis([0,0,0,0,.1])}
    with pytest.raises(ValidationError):
        emailer.send_email.invoke({'type':'tool_call','id':'mail','name':'send_email',
                                  'args':{'narrative':BOUND+'\nThe entire gain was in the final session.',
                                          'citations':['d2'],'state':state}})
    assert not called
