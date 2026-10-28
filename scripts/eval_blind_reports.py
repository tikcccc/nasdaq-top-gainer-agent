"""Historical free-prose evaluation (not the production evidence-selection mode): freeze -> generate -> dual blind labels -> guard.

Reviewers receive rendered reports and source evidence, never detector code,
predictions, generation intent, or each other's labels. This is model annotation,
not independent human adjudication. Keep disagreements and failures in the totals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import secrets
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from scripts.legacy import narrative_guard as guard
from nasdaq_agent.config import load_settings
from nasdaq_agent.context import start_run
from nasdaq_agent.data.calendar import _cal
from nasdaq_agent.data import metrics
from nasdaq_agent.llm.factory import _model
from nasdaq_agent.reporting import placeholders
from nasdaq_agent.reporting.render import render_narrative


class Draft(BaseModel):
    narrative: str=Field(min_length=40,max_length=4000)
    citations: list[str]=Field(max_length=10,description='Source IDs only: d1, d2, d3, d4, n1. No brackets, descriptions or URLs.')


class Annotation(BaseModel):
    factual_verdict: str=Field(description="supported, unsupported, or uncertain")
    confidence: float=Field(ge=0,le=1)
    issues: list[str]=Field(description="Exact problematic quotation, explanation and evidence reference; empty for supported")


RUBRIC="""Judge factual support, not style or writing-contract compliance. Treat supplied source evidence as a closed set.
Check the full report, including risk bullets. Verify day-by-day directions from closes. Percentage returns compound;
the final session's share of NET price change is (last close - previous close)/(last close - first close), undefined
when the net change is zero. 'All net change' is different from 'only session with an increase'. News coincidence
does not establish cause or funding links. Exact, explicitly attributed headline quotations describe a source's
claim, not independently established physical causation. A fetched subset cannot prove absence of all public news.
Historical superlatives, business identities and size classes need supplied evidence. Clear hypothetical future
risks and acknowledgements of limited evidence do not assert actual events. Numbers displayed to two decimals
may differ through rounding. Return supported only if every factual assertion follows; otherwise unsupported.
Use uncertain when the evidence/wording genuinely prevents a decision. List exact problematic spans and evidence.
You are a stateless blind annotator: no source code, detector decision, author identity or expected label is supplied."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    pending.replace(path)


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sources():
    paths=sorted((ROOT/'nasdaq_agent').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'config.yaml']
    return {p.relative_to(ROOT).as_posix():digest(p.read_bytes()) for p in paths}


def verify(root):
    freeze=read(root/'freeze.json')
    if freeze['source_sha256']!=sources():raise ValueError('source changed after freeze; use a new experiment, not these labels')
    settings=load_settings()
    if freeze['models']!={p:settings.providers[p].model for p in ('mimo','deepseek')}:raise ValueError('configured models changed')
    return freeze


def freeze(root,count):
    if root.exists():raise ValueError('experiment exists; refusing to replace frozen evidence')
    root.mkdir(parents=True)
    settings=load_settings()
    write(root/'freeze.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),count=count,seed=secrets.randbits(64),
        source_sha256=sources(),models={p:settings.providers[p].model for p in ('mimo','deepseek')},
        limitations='Fresh synthetic full reports, not natural-distribution market reports. Generated challenge cases share error families with development. Dual stateless model annotation, not independent human labels; model families overlap generation/review. No detector feedback used to revise code or labels in this experiment.'))


def make_case(index,rng,root):
    symbol=f'ZB{index:03d}'
    count=rng.choice([5,7,10])
    calendar=_cal()
    dates=calendar.sessions_in_range(pd.Timestamp('2025-01-01'),pd.Timestamp('2025-12-31'))
    end=rng.randrange(count+1,len(dates))
    window=[d.date().isoformat() for d in dates[end-count:end+1]]
    profile=index%8
    base=round(rng.uniform(20,150),2)
    closes=[base]
    for i in range(count):
        r=(rng.uniform(.002,.03) if profile in (0,6) else -rng.uniform(.002,.03) if profile==1 else
           rng.uniform(.005,.035)*(1 if i%2==0 else -1) if profile in (2,7) else
           0 if profile in (3,5) else -rng.uniform(.005,.02))
        if i==count-1:
            if profile in (0,1):r=rng.uniform(.08,.25)*(1 if profile==0 else -1)
            if profile in (4,5):r=rng.uniform(.15,.40)
            if profile==6:r=0
        closes.append(round(closes[-1]*(1+r),2))
    if profile==7:closes[-1]=closes[0]
    case_id=secrets.token_hex(6)
    candidate=dict(symbol=symbol,name=f'{symbol} Common Stock',price=closes[-1],
        pct_change=(closes[-1]/closes[-2]-1)*100,volume=rng.randint(2,30)*100_000,days=count,flags=[])
    ds={}
    frames={}
    for key,values in [(symbol,closes)]+[(k,[round(100*(1+rng.uniform(-.03,.08)*i/count),5) for i in range(count+1)]) for k in ('GSPC','IXIC')]:
        frame=pd.DataFrame(dict(date=window,close=values,open=values,high=[v*1.01 for v in values],low=[v*.99 for v in values],volume=1_000_000))
        rel=Path('inputs')/case_id/f'ds_{key}.csv'
        (root/rel).parent.mkdir(parents=True,exist_ok=True)
        frame.to_csv(root/rel,index=False)
        frames[key]=frame
        ds['ds_'+key]=dict(path=rel.as_posix(),symbol=key if key==symbol else '^'+key,
            sha256=digest((root/rel).read_bytes()),first_date=window[0],last_date=window[-1],trade_date=window[-1])
    version=digest(json.dumps({k:v['sha256'] for k,v in ds.items()},sort_keys=True).encode())
    result=metrics.compute(frames[symbol],{k:frames[k] for k in ('GSPC','IXIC')},count,symbol)
    result['analysis_key']=version
    titles=[f'{symbol} announced a non-binding proposal; completion remains subject to a shareholder vote.',
            f'{symbol} reported quarterly results without revising its annual revenue guidance.',
            f'{symbol} reported that asset-sale proceeds funded its new laboratory.',
            f'{symbol} disclosed an ongoing operational review; no findings have been published.']
    headlines={} if index%6==0 else {'n1':dict(id='n1',title=titles[index%4],publisher='Synthetic source',
        published=window[-1]+'T14:00:00+00:00',link=f'https://example.invalid/{case_id}',symbol=symbol,trade_date=window[-1])}
    state=dict(selected={**candidate,'dataset_id':'ds_'+symbol,'analysis_key':version,'trade_date':window[-1]},
        analysis=result,datasets=ds,headlines=headlines,session=dict(trade_date=window[-1],is_intraday=False),
        selection=dict(primary_symbol=symbol,label='Supplied synthetic stock window (ranking not evaluated)',filtered=False,filters={}),sources={'history':'synthetic'})
    evidence=dict(symbol=symbol,company=candidate['name'],trade_date=window[-1],price=candidate['price'],
        daily_change_percent=candidate['pct_change'],volume=candidate['volume'],stock_closes=frames[symbol][['date','close']].to_dict('records'),
        benchmarks={k:frames[k][['date','close']].to_dict('records') for k in ('GSPC','IXIC')},headlines=headlines,
        news_scope='Only fetched items; not exhaustive public coverage.',market_cap=None)
    mode='faithful' if index%2==0 else ['contribution','daily path','historical comparison','causation','invented event','public news absence'][(index//2)%6]
    return dict(id=case_id,state=state,evidence=evidence,generation_mode=mode,generator=('mimo','deepseek')[(index//2)%2])


def load_state(root,case):
    state=json.loads(json.dumps(case['state']))
    for m in state['datasets'].values():m['path']=str((root/m['path']).resolve())
    return state


def model_call(provider,schema,prompt,payload):
    settings=load_settings(overrides={'llm':{'provider':provider,'fallback':[],'timeout_s':45,'max_retries':1}})
    example=({'factual_verdict':'supported','confidence':0.9,'issues':[]} if schema is Annotation else
             {'narrative':'REPORT TEXT HERE','citations':['d1','d2','d3','d4']})
    prompt+='\nReturn one flat JSON instance only, not a schema. Do not use properties/required/title wrappers or call tools. Output shape example:\n'+json.dumps(example)
    if schema is Annotation:
        prompt+='\nfactual_verdict must be supported, unsupported or uncertain; confidence is a number from zero to one; issues is a list of exact quotations and evidence explanations. Unsupported requires at least one issue.'
    raw=_model(settings,provider).bind(response_format={'type':'json_object'}).invoke([SystemMessage(prompt),HumanMessage(json.dumps(payload,ensure_ascii=False))])
    content=raw.content if isinstance(raw.content,str) else ''.join(x.get('text','') for x in raw.content if isinstance(x,dict))
    clean=content.strip()
    if clean.startswith('```') and clean.endswith('```'):
        clean=clean.split('\n',1)[1].rsplit('```',1)[0].strip()
    out=dict(reported_model=(raw.response_metadata or {}).get('model_name'),usage=raw.usage_metadata or {})
    try:out['answer']=schema.model_validate(json.loads(clean)).model_dump()
    except Exception as exc:out.update(error=type(exc).__name__,raw_response=content)
    return out


def review_render(narrative,ctx):
    """Render for annotation without filtering on the detector's format rules.

    Literal figures must remain available to judges even if production forbids
    them. Unknown placeholders remain visibly unresolved. Never execute templates.
    """
    flat=placeholders.flatten(placeholders.narrative_context(ctx))
    return placeholders.PLACEHOLDER.sub(lambda m:flat.get(m.group(1),f'[UNRESOLVED: {m.group(1)}]'),narrative)


def parallel(tasks,fn,workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(fn,*task) for task in tasks]
        for index,future in enumerate(as_completed(futures),1):
            future.result()
            print(f'completed {index}/{len(tasks)}',flush=True)


def generate(root,workers):
    frozen=verify(root)
    if (root/'labels-seal.json').exists() or (root/'predictions').exists():raise ValueError('cannot regenerate after labelling or predictions')
    path=root/'cases.json'
    if not path.exists():
        rng=random.Random(frozen['seed'])
        write(path,[make_case(i,rng,root) for i in range(frozen['count'])])
    cases=read(path)
    contract="""Write a financial report of at most 250 rendered words, with no minimum or padding, about the supplied synthetic evidence. Return JSON.
Use {{symbol}}, {{company}}, {{selection_label}}, {{trade_date}}, {{pct_change}}, {{price}}, {{volume}} for those facts.
Three paragraphs: **Why this stock.** ... [d1]; **How it traded.** {{path.sequence}} {{path.contribution}} {{path.benchmarks}} [d2][d3][d4];
**What's behind it.** {{news.coverage}} ... [n1 if available]. Optional zero to two cited risk bullets.
The three path tokens MUST be copied in that order with no extra prose in that paragraph. Each expands to sentences
supplied under computed_sentences. news.coverage also expands to supplied text. Count rendered words, not token words.
Other prose should add only supported company/news context. No literal numbers, investment advice, generic filler,
unbound path quantifiers/comparisons or inferred causes. Source attribution can quote an exact headline and cite it.
Copy this first sentence exactly, retaining the double-brace placeholders:
**Why this stock.** {{company}} ({{symbol}}) is the {{selection_label}} for {{trade_date}}, with a {{pct_change}} session move, a {{price}} close and {{volume}} shares traded [d1].
The citations JSON field contains only IDs such as ["d1","d2","d3","d4","n1"], never brackets or explanations.
Source IDs: d1 quote/selection metadata; d2 stock closes; d3 S&P 500 closes; d4 NASDAQ Composite closes; n1 fetched headline if present.
Do not replace the placeholder tokens with supplied text yourself; code renders them later. Double braces must survive in the JSON string.
Do not invent sentiment estimates; none were supplied. The report is a synthetic internal evaluation, never emailed."""
    def one(case):
        target=root/'generation'/f"{case['id']}.json"
        if target.exists():return
        state=load_state(root,case)
        ctx=placeholders.build_context(state)
        prompt=contract
        if case['generation_mode']!='faithful':
            prompt+='\nCHALLENGE: Keep the protected paragraph intact, but add exactly one subtle unsupported interpretation about '+case['generation_mode']+' elsewhere. Use fresh natural wording. Do not disclose that the claim is planted. The rest must follow the evidence and format.'
        try:
            out=model_call(case['generator'],Draft,prompt,dict(evidence=case['evidence'],computed_sentences={'path':ctx['path'],'news':ctx['news']}))
            if 'answer' in out:
                out.update(id=case['id'],rendered=review_render(out['answer']['narrative'],ctx))
            else:out['id']=case['id']
        except Exception as exc:out=dict(id=case['id'],error=type(exc).__name__)
        write(target,out)
    parallel([(case,) for case in cases],one,workers)
    packet=[]
    for c in cases:
        g=read(root/'generation'/f"{c['id']}.json")
        packet.append(dict(case_id=c['id'],evidence=c['evidence'],report=g.get('rendered'),generation_error=g.get('error')))
    rng=random.Random(frozen['seed']^0x5A5A)
    rng.shuffle(packet)
    package=root/'human-review'
    package.mkdir(exist_ok=True)
    (package/'reports.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in packet),encoding='utf-8')
    if not (package/'labels.csv').exists():
        with (package/'labels.csv').open('w',encoding='utf-8-sig',newline='') as fh:
            writer=csv.writer(fh);writer.writerow(['case_id','factual_verdict','confidence','issue_span','evidence_reference','reviewer','notes'])
            writer.writerows([[c['case_id'],'','','','','',''] for c in packet])
    (package/'RUBRIC.txt').write_text(RUBRIC+'\n\nNo model labels, detector decisions, author, or generation intent are included in this package.\n',encoding='utf-8')
    inputs=[root/'cases.json',*sorted((root/'inputs').rglob('*.csv')),*sorted((root/'generation').glob('*.json'))]
    write(root/'packet-seal.json',{'sha256':digest((package/'reports.jsonl').read_bytes()),'count':len(packet),
        'input_sha256':{p.relative_to(root).as_posix():digest(p.read_bytes()) for p in inputs},'source_sha256':frozen['source_sha256']})


def packets(root):
    verify(root)
    p=root/'human-review/reports.jsonl'
    seal=read(root/'packet-seal.json')
    if digest(p.read_bytes())!=seal['sha256']:raise ValueError('sealed report packet changed')
    if any(digest((root/name).read_bytes())!=sha for name,sha in seal['input_sha256'].items()):raise ValueError('sealed inputs/drafts changed')
    return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]


def label(root,workers):
    packet=packets(root)
    if (root/'predictions').exists():raise ValueError('predictions already exist; do not relabel after seeing detector feedback')
    def one(case,provider):
        target=root/'labels'/f"{case['case_id']}.{provider}.json"
        if target.exists():return
        try:
            if not case['report']:raise ValueError('no rendered report')
            out=model_call(provider,Annotation,RUBRIC,{'evidence':case['evidence'],'report':case['report']})
            if 'answer' in out:
                if out['answer']['factual_verdict'] not in {'supported','unsupported','uncertain'}:raise ValueError('invalid label')
                if out['answer']['factual_verdict']=='unsupported' and not out['answer']['issues']:raise ValueError('unsupported label lacks cited reason')
        except Exception as exc:out={'error':type(exc).__name__}
        write(target,out)
    parallel([(c,p) for c in packet for p in ('mimo','deepseek')],one,workers)
    write(root/'labels-seal.json',{'sealed_at_utc':datetime.now(timezone.utc).isoformat(),
        'sha256':{p.name:digest(p.read_bytes()) for p in sorted((root/'labels').glob('*.json'))}})


def predict(root,workers):
    packet=packets(root)
    sealed=read(root/'labels-seal.json')
    if any(digest((root/'labels'/name).read_bytes())!=sha for name,sha in sealed['sha256'].items()):raise ValueError('sealed labels changed')
    cases={c['id']:c for c in read(root/'cases.json')}
    def one(item):
        cid=item['case_id'];target=root/'predictions'/f'{cid}.json'
        if target.exists():return
        try:
            if not all((root/'labels'/f'{cid}.{p}.json').exists() for p in ('mimo','deepseek')):raise ValueError('missing labels')
            g=read(root/'generation'/f'{cid}.json')
            if 'error' in g:raise ValueError('generation failed')
            settings=load_settings(overrides={'jev':{'enabled':True,'allow_rule_fallback':False},
                'email':{'dry_run':True},'run':{'output_dir':str(root/'guard-runs')}})
            start_run(settings,run_id='blind_'+cid)
            issues,details=guard.check_legacy_narrative(load_state(root,cases[cid]),g['answer'],settings)
            out=dict(decision='reject' if issues else 'accept',issues=issues,details=details)
        except Exception as exc:out=dict(decision='error',error=type(exc).__name__)
        write(target,out)
    parallel([(item,) for item in packet],one,workers)


def summarize(root):
    packet=packets(root)
    sealed=read(root/'labels-seal.json')
    if any(digest((root/'labels'/name).read_bytes())!=sha for name,sha in sealed['sha256'].items()):raise ValueError('sealed labels changed')
    rows=[]
    for item in packet:
        cid=item['case_id']
        labels={p:read(root/'labels'/f'{cid}.{p}.json') for p in ('mimo','deepseek')}
        answers=[v.get('answer',{}) for v in labels.values()]
        agreed=(len({a.get('factual_verdict') for a in answers})==1 and all(a.get('confidence',0)>=.75 for a in answers))
        verdict=answers[0].get('factual_verdict') if agreed else 'unresolved'
        if verdict=='uncertain':verdict='unresolved'
        pred=read(root/'predictions'/f'{cid}.json')
        rows.append(dict(case_id=cid,consensus=verdict,labels=labels,prediction=pred,
                         generation_error=item.get('generation_error')))
    supported=[r for r in rows if r['consensus']=='supported'];unsupported=[r for r in rows if r['consensus']=='unsupported']
    fa=sum(r['prediction']['decision']=='accept' for r in unsupported)
    fr=sum(r['prediction']['decision']=='reject' for r in supported)
    out=dict(count=len(rows),consensus_supported=len(supported),consensus_unsupported=len(unsupported),
        unresolved=len(rows)-len(supported)-len(unsupported),false_accepts=fa,false_rejects=fr,
        false_accept_rate=fa/len(unsupported) if unsupported else None,false_reject_rate=fr/len(supported) if supported else None,
        detector_errors=sum(r['prediction']['decision']=='error' for r in rows),
        generation_errors=sum(bool(r['generation_error']) for r in rows),
        limitations=read(root/'freeze.json')['limitations']+' Primary labels assess factual support, not writing-policy compliance; factually supported reports rejected for formatting/binding policy remain counted as factual false rejections. Unresolved labels remain in the total and require human review.',
        human_review_package='human-review/',results=rows)
    write(root/'results.json',out)
    print(json.dumps({k:v for k,v in out.items() if k not in {'results','limitations'}},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['freeze','generate','label','predict','summarize'])
    # Recorded rounds are in docs/evaluation/history/history.zip; extract one there to re-run it.
    p.add_argument('--root',type=Path,default=ROOT/'outputs/eval/bound-claims-blind')
    p.add_argument('--count',type=int,default=24)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();root=a.root.resolve()
    if a.stage=='freeze':freeze(root,a.count)
    elif a.stage=='generate':generate(root,a.workers)
    elif a.stage=='label':label(root,a.workers)
    elif a.stage=='predict':predict(root,a.workers)
    else:summarize(root)


if __name__=='__main__':main()
