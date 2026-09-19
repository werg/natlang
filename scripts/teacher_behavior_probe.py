#!/usr/bin/env python3
"""Audit teacher execution against known results AND required call behavior.

Probe outputs are quarantined in runs; this script never admits training samples.
No student decoder, local validation feedback, self-review, or expanded state.
"""
import argparse
import json
import sys
import urllib.request
from types import SimpleNamespace
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang.decoder import LlamaServerDecoder
from natlang.corpus import file_digest
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import load_program, dump
from scripts.generate_agent_support import support_cases
from scripts.teacher_probe_trajectory_ir import convert_probe


CALL_FIELDS = ('function','to','inputs','over','init','until','max')


def call_signature(args, case=None):
    args = dict(args)
    if case and "init" in args:
        init = args["init"]
        if isinstance(init, str) and init.startswith("args/"):
            value = case.get("root", {}).get("$lambda", {}).get("args", {})
            for part in init[5:].split("/"):
                if not isinstance(value, dict) or part not in value:
                    break
                value = value[part]
            else:
                args["init"] = value
        # The fold fixture permits either a source reference or an equivalent
        # numeric literal, including the runtime's single JSON-text layer.
        fn = case.get("root", {}).get("$lambda", {}).get("codebase", {}).get(args.get("function"), {})
        if fn.get("args", {}).get("acc") == "Num" and isinstance(args["init"], str):
            try:
                parsed = json.loads(args["init"])
                if type(parsed) in (int, float):
                    args["init"] = parsed
            except ValueError:
                pass
    return {k:(args.get(k) or {} if k == 'inputs' else args.get(k)) for k in CALL_FIELDS}


def flow_cases(seed):
    values = [seed % 7 + 1, seed % 11 + 2, seed % 5 + 3]
    for fold in (False, True):
        fn = 'add' if fold else 'double'
        args = {'values':values, **({'start':5} if fold else {})}
        params = '{ values: Num[], start: Num }' if fold else '{ values: Num[] }'
        ret = 'Num' if fold else 'Num[]'
        function = {'args':{'acc':'Num','item':'Num'} if fold else {'item':'Num'},
                    'returns':'Num','code':'return args.acc + args.item' if fold else 'return args.item * 2'}
        body = ('Use one batched fold call to add over values, starting at args/start. Put the final accumulator directly into return.'
                if fold else 'Use one batched map call to double for every item of values. Put the resulting list directly into return.')
        call = {'function':fn,'to':'return','over':'args/values'}
        if fold: call['init']='args/start'
        call['done']=1
        yield {'name':'fold_calls' if fold else 'map_calls',
               'root':{'$lambda':{'type':f'Lambda<{params}, {ret}>','instructions':body,'args':args,'codebase':{fn:function}}},
               'steps':[('call',call)],'expected':sum(values)+5 if fold else [v*2 for v in values], 'failure':None,'effects':[]}


def cases(seed, group):
    yield from support_cases(seed,group)
    yield from flow_cases(seed+group)
    yield {'name':'evidence_empty', 'root':{'$lambda':{'type':'Lambda<{ document?: Text }, Text>',
           'instructions':'Return an exact copy of the supplied document. If none was supplied, do not invent its contents.',
           'args':{'document':''}}}, 'steps':[('write',{'path':'return','type':'Text','source':'args/document'})],
           'expected':'','failure':None,'effects':[]}

    for name, ty, value in (
        ('record_literal', '{ items: Num[], label: Text, ok: Bool }', {'items':[1,2], 'label':'a "quoted" word', 'ok':True}),
        ('list_literal', 'Text[]', ['alpha', 'He said "yes".'])):
        yield {'name':name, 'root':{'$lambda':{'type':f'Lambda<{{}}, {ty}>',
               'instructions':'Return this exact value: '+json.dumps(value), 'args':{}}},
               'steps':[('write',{'path':'return','type':ty,'value':value})],
               'expected':value,'failure':None,'effects':[]}


def assess(case, outcome, value, log, emitted):
    expected_calls = [a for n,a in case['steps'] if n == 'call']
    actual_calls = []
    premature_marks = []
    seen = set()
    required_lines = {a['done']: a for a in expected_calls if 'done' in a}
    for event in log:
        name, _, body = event['action'].partition(' ')
        args = json.loads(body)
        if name == 'call':
            actual_calls.append(call_signature(args, case))
            if event['kind'] == 'done':
                for line, expected in required_lines.items():
                    if call_signature(args, case) == call_signature(expected, case):
                        seen.add(line)
        if name == 'mark_done' and not args.get('skipped'):
            for line in range(args['start'],args.get('end',args['start'])+1):
                if line in required_lines and line not in seen:
                    premature_marks.append(line)
        if name != 'call' and 'done' in args:
            lines = args['done'] if isinstance(args['done'],list) else [args['done']]
            premature_marks.extend(line for line in range(lines[0],lines[-1]+1) if line in required_lines and line not in seen)
    expected = [call_signature(a, case) for a in expected_calls]
    allowed = case.get('allowed_failure_call')
    failed_calls = [event for event in log if event['action'].startswith('call ')]
    faithful_runtime_failure = bool(allowed and outcome.detail.startswith('validation failed:') and
        actual_calls == [call_signature(allowed, case)] and
        len(failed_calls) == 1 and failed_calls[0]['kind'] == 'rejected' and
        'type-does-not-fit-slot' in outcome.detail)
    allowed_write = case.get('allowed_failure_write')
    last_name, _, last_body = log[-1]['action'].partition(' ') if log else ('','','{}')
    last_args = json.loads(last_body)
    faithful_write_failure = bool(allowed_write and outcome.detail.startswith('validation failed:') and
        last_name == 'write' and log[-1]['kind'] == 'rejected' and actual_calls == expected and
        any(code in outcome.detail for code in ('type-mismatch', 'type-does-not-fit-slot')) and
        all(last_args.get(k) == allowed_write.get(k) for k in ('path','value','source')))
    faithful_runtime_failure = faithful_runtime_failure or faithful_write_failure
    calls_ok = actual_calls == expected or faithful_runtime_failure
    result_ok = faithful_runtime_failure or (outcome.kind == 'quiesced' and outcome.detail.startswith(case['failure']+':')) if case['failure'] else (
        outcome.kind == 'done' and dump(value) == case['expected'])
    return {'correct_result_or_explicit_failure': result_ok,
            'required_calls_exact': calls_ok, 'faithful_runtime_failure':faithful_runtime_failure,
            'premature_marks': premature_marks,
            'effects_correct': emitted == case['effects'],
            'safe_validation_failure': bool(case['failure']) and outcome.detail.startswith('validation failed:'),
            'pass': result_ok and calls_ok and not premature_marks and emitted == case['effects']}


def summary(rows):
    return {'cases':len(rows), 'passed':sum(r['pass'] for r in rows),
            'wrong_successes':sum(r['status']=='done' and not r['pass'] for r in rows)}


def rescore(doc):
    seed = int(doc['args']['seed'])
    for row in doc['rows']:
        case = next(c for c in cases(seed,row['group']) if c['name']==row['case'])
        assert case['root'] == row['program'], 'Fixture changed: cannot rescore this program'
        row['previous_pass'] = row['pass']
        emitted = row.get('emitted', case['effects'] if row['effects_correct'] else ['unrecorded effects'])
        row.update(assess(case,SimpleNamespace(kind=row['status'],detail=row['detail']),row['value'],row['log'],emitted))
    doc['summary'] = summary(doc['rows'])
    doc['audit_version'] = 3
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rescore', type=Path, help='rescore saved traces without model calls')
    ap.add_argument('--server', default='http://127.0.0.1:8081')
    ap.add_argument('--groups', type=int, default=1)
    ap.add_argument('--seed', type=int, default=881)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--thinking', type=int, default=256)
    ap.add_argument('--json-text-values', action='store_true', help='experimental explicit JSON-text value transport')
    ap.add_argument('--reasoning-effort', choices=['low','medium','xhigh'], default='xhigh')
    ap.add_argument('--system-file', type=Path, default=ROOT/'natlang/prompts/tools_small.md')
    ap.add_argument('--cases', nargs='+', help='optional subset of case names')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--trajectory-out', type=Path,
                    help='template-independent teacher trajectory IR (defaults beside --out)')
    a = ap.parse_args()
    if a.out.exists(): ap.error('refusing overwrite')
    trajectory_path = a.trajectory_out or a.out.with_suffix('.trajectory.ir.jsonl')
    if trajectory_path.resolve() == a.out.resolve() or trajectory_path.exists():
        ap.error('trajectory IR needs a separate fresh output path')
    if a.rescore:
        doc = rescore(json.loads(a.rescore.read_text()))
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(doc,indent=2)+'\n')
        print(json.dumps(doc['summary']))
        return
    if a.groups < 1:
        ap.error('groups must be positive')
    available = {c['name'] for c in cases(a.seed,0)}
    if a.cases and not set(a.cases) <= available:
        ap.error('unknown cases: ' + ', '.join(sorted(set(a.cases)-available)))
    with urllib.request.urlopen(a.server.rstrip('/') + '/v1/models',timeout=30) as response:
        model_metadata = json.load(response)
    prompt = a.system_file.read_text()
    source_hashes = {'surface_sha256':file_digest(ROOT/'natlang/surface.py'),
                     'decoder_sha256':file_digest(ROOT/'natlang/decoder.py'),
                     'runtime_sha256':file_digest(ROOT/'natlang/runtime.py'),
                     'agent_sha256':file_digest(ROOT/'natlang/tool_agent.py')}
    rows=[]
    for group in range(a.groups):
        for case in cases(a.seed,group):
            if a.cases and case['name'] not in a.cases:
                continue
            dec = LlamaServerDecoder(a.server, timeout=240, chat_extra={'thinking_budget_tokens':a.thinking,'top_p':.95,'top_k':20,'chat_template_kwargs':{'reasoning_effort':a.reasoning_effort}},tool_aliases={'call':'call_function'},json_text_values=a.json_text_values)
            log, transcript, teacher_turns = [], [], []
            rt = Runtime(lambda lam: ToolAgent(dec,system_prompt=prompt,temperature=a.temperature,log=log,transcript=transcript,
                teacher_turns=teacher_turns,validation_feedback='caller',max_turns=16,max_tokens=4000,max_seconds=240))
            out,value=rt.run_root(load_program(case['root']))
            row={'case':case['name'],'group':group,'program':case['root'],'expected':case['expected'],'expected_failure':case['failure'],
                 'status':out.kind,'detail':out.detail,'value':dump(value) if out.kind=='done' else None,
                 **assess(case,out,value,log,rt.emitted),'emitted':rt.emitted,'log':log,'transcript':transcript,
                 'teacher_turns':teacher_turns,'usage':dec.usage}
            rows.append(row)
            doc={'model':model_metadata['data'][0]['id'],'model_metadata':model_metadata,'args':{k:str(v) for k,v in vars(a).items()},'system_prompt':prompt,**source_hashes,
                 'audit_version':3,'summary':summary(rows),'rows':rows}
            a.out.parent.mkdir(parents=True,exist_ok=True)
            tmp=a.out.with_suffix('.tmp');tmp.write_text(json.dumps(doc,indent=2)+'\n');tmp.replace(a.out)
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory_tmp = trajectory_path.with_suffix(trajectory_path.suffix + '.tmp')
            with trajectory_tmp.open('w') as stream:
                for number, saved in enumerate(rows, 1):
                    stream.write(json.dumps(convert_probe(doc, saved, path=a.out, index=number),
                                            ensure_ascii=False) + '\n')
            trajectory_tmp.replace(trajectory_path)
            print(f'{len(rows)} {case["name"]}: pass={row["pass"]} {out.kind} {out.detail[:100]}',flush=True)

if __name__=='__main__': main()
