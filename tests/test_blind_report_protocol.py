import json
import random

import pytest

from nasdaq_agent.agent.evidence import analysis_problems
from scripts import eval_blind_reports as blind


def test_fresh_case_identity_and_snapshot_are_consistent(tmp_path):
    for i in range(8):
        case=blind.make_case(i,random.Random(i),tmp_path)
        state=blind.load_state(tmp_path,case)
        assert analysis_problems(state)==[]
        assert all(h['symbol']==state['selected']['symbol'] and h['title'].startswith(h['symbol']) for h in state['headlines'].values())


def test_blind_stage_boundary_and_uncertain_denominator(tmp_path,settings,monkeypatch):
    root=tmp_path/'blind'
    monkeypatch.setattr(blind,'load_settings',lambda *a,**k:settings)
    calls=[]
    def model(provider,schema,prompt,payload):
        if schema is blind.Draft:
            return {'answer':{'narrative':'{{company}} ({{symbol}}) {{trade_date}} {{pct_change}} [d1].\n\n**How it traded.** {{path.sequence}} {{path.contribution}} {{path.benchmarks}} [d2][d3][d4]\n\n**What\'s behind it.** {{news.coverage}}','citations':['d1','d2','d3','d4']}}
        assert set(payload)=={'evidence','report'}
        assert 'generation_mode' not in json.dumps(payload)
        calls.append(provider)
        return {'answer':{'factual_verdict':'supported' if provider=='mimo' else 'unsupported',
                          'confidence':.9,'issues':[] if provider=='mimo' else ['test disagreement']}}
    monkeypatch.setattr(blind,'model_call',model)
    blind.freeze(root,2)
    blind.generate(root,1)
    package=root/'human-review'
    assert {p.name for p in package.iterdir()}=={'reports.jsonl','labels.csv','RUBRIC.txt'}
    assert all(set(json.loads(line))=={'case_id','evidence','report','generation_error'} for line in (package/'reports.jsonl').read_text().splitlines())
    blind.label(root,2)
    assert sorted(calls)==['deepseek','deepseek','mimo','mimo']
    monkeypatch.setattr(blind.guard,'check_legacy_narrative',lambda *a,**k:([],{}))
    blind.predict(root,1)
    with pytest.raises(ValueError,match='predictions already exist'):blind.label(root,1)
    blind.summarize(root)
    report=blind.read(root/'results.json')
    assert report['count']==2 and report['unresolved']==2
    assert report['consensus_supported']==report['consensus_unsupported']==0
    with pytest.raises(ValueError,match='cannot regenerate'):blind.generate(root,1)
    path=next((root/'generation').glob('*.json'))
    path.write_text('{}')
    with pytest.raises(ValueError,match='sealed inputs'):blind.packets(root)


def test_annotation_render_does_not_hide_invalid_numeric_templates():
    actual=blind.review_render('{{symbol}} gained 12.5%; {{unknown}}. {% do not execute %}',{'symbol':'UNIT'})
    assert actual=='UNIT gained 12.5%; [UNRESOLVED: unknown]. {% do not execute %}'


def test_json_transport_uses_no_forced_tool_choice(settings,monkeypatch):
    from langchain_core.messages import AIMessage
    class Chat:
        def bind(self,**kwargs):
            assert kwargs=={'response_format':{'type':'json_object'}}
            return self
        def invoke(self,messages):
            assert 'not a schema' in messages[0].content
            assert 'or call tools' in messages[0].content
            return AIMessage(content='```json\n{"narrative":"A sufficiently long synthetic report for this parser test.","citations":[]}\n```')
    monkeypatch.setattr(blind,'load_settings',lambda *a,**k:settings)
    monkeypatch.setattr(blind,'_model',lambda *a,**k:Chat())
    assert 'answer' in blind.model_call('deepseek',blind.Draft,'task',{})
