#!/usr/bin/env python3
"""Prepare/run a short, source-bound native feedback window; never admit samples."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'training/neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest,sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.runtime_inventory import snapshot_task_runtime,task_runtime_inventory
from natlang_neuralese.rl.graph_replay import save_bundle
from natlang_neuralese.rl.task_source import resolve_task_source
from natlang_neuralese.rl.task_rollouts import authored_root_row,student_root_row,build_queue
from natlang_neuralese.serve.native_protocol import validate_failure

IDENTITY_KEYS=('architecture','checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256')
FROZEN_PARENT='4ea5f59a371fee5d75302f3526ee5e2544531bee'
MARKER_COMMIT='a3bc714c7'
MARKER_MODULES={'rl/native_replay.py','serve/engine.py','serve/http.py',
                'serve/native_protocol.py','serve/native_recorded_objective.py'}


def binding(path):
    path=Path(path).resolve(strict=True)
    return {'path':str(path),'sha256':sha256_file_hex(path)}


def bound(reference):
    path=Path(reference['path']).resolve(strict=True)
    if sha256_file_hex(path)!=reference['sha256']:raise ValueError('feedback input changed: '+str(path))
    return path


def immutable(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('immutable feedback plan changed')
    else:write_json_durable(path,value)


def composition_inventory(root,baseline_runtime):
    """Authenticate exact published module bytes, not a dirty build's commit stamp."""
    root=Path(root).resolve(strict=True);prefix='training/neuralese/natlang_neuralese/'
    def tracked(commit,folder):
        return subprocess.check_output(['git','-C',str(root),'ls-tree','-r','--name-only',commit,'--',folder],text=True).splitlines()
    def committed(commit,name):
        return hashlib.sha256(subprocess.check_output(['git','-C',str(root),'show',commit+':'+name])).hexdigest()
    old={name[len(prefix):]:committed(FROZEN_PARENT,name) for name in tracked(FROZEN_PARENT,prefix) if name.endswith('.py')}
    expected=dict(old)
    for name in MARKER_MODULES:expected[name]=committed(MARKER_COMMIT,prefix+name)
    actual={str(path.relative_to(root/prefix)):sha256_file_hex(path) for path in (root/prefix).rglob('*.py')}
    if actual!=expected:raise ValueError('frozen marker package differs from its exact five-module composition')
    ts={name:committed(MARKER_COMMIT,name) for name in tracked(MARKER_COMMIT,'ts-host')}
    if any(not (root/name).is_file() or sha256_file_hex(root/name)!=value for name,value in ts.items()):
        raise ValueError('frozen marker TypeScript differs from complete published source')
    extra=subprocess.check_output(['git','-C',str(root),'ls-files','--others','--exclude-standard','ts-host'],text=True).splitlines()
    if extra:raise ValueError('frozen marker TypeScript has unbound additional source')
    return {'schema':'natlang.frozen-native-marker-composition/1','passed':True,
        'parent':FROZEN_PARENT,'marker_source':subprocess.check_output(['git','-C',str(root),'rev-parse',MARKER_COMMIT],text=True).strip(),
        'python_modules':actual,'changed_python_modules':sorted(MARKER_MODULES),
        'unchanged_numerical_modules':True,'typescript_source':ts,
        'candidate_runtime_inventory':task_runtime_inventory(root/'ts-host'),
        'baseline_runtime_inventory':task_runtime_inventory(baseline_runtime),
        'baseline_build_stamp_is_source_authority':False,'live_feedback_qualified':False}


def trained_parent(reference):
    receipt=json.loads(bound(reference).read_text())
    identity=receipt['identity']
    match=re.fullmatch(r'lfm/([0-9a-f]{64})/lfm-native-checkpoint-step-(\d+)\.pt',receipt['name'])
    if (not match or int(match[2])<1 or match[1]!=identity['sha256'] or receipt.get('private') is not True
            or type(identity.get('bytes')) is not int or identity['bytes']<=0
            or any((receipt.get('independent_verification') or {}).get(key)!=identity[key] for key in ('sha256','bytes'))):
        raise ValueError('feedback needs its actual independently verified trained LFM export receipt')
    return identity['sha256']


def select_rows(task_source,selection):
    source=resolve_task_source(task_source)
    names=json.loads(Path(selection).read_text())
    if not isinstance(names,list) or len(names)!=12 or len(set(names))!=12 or not all(isinstance(x,str) for x in names):
        raise ValueError('feedback policy must declare exactly twelve distinct current TRAIN task IDs')
    rows={}
    for index,line in enumerate(source['cases'].read_text().splitlines()):
        if not line.strip():continue
        row=json.loads(line)
        if row.get('id') in names:
            if row.get('split')!='train':raise ValueError('feedback refuses protected/nontrain tasks')
            rows[row['id']]=(index,row)
    if set(rows)!=set(names):raise ValueError('feedback selection is outside the admitted current task source')
    return source,[rows[name] for name in names]


def prepare(args):
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        source,selected=select_rows(args.task_source,args.selection)
        parent=binding(args.actor_receipt);checkpoint=trained_parent(parent)
        transform=authored_root_row if args.root_mode=='authored' else student_root_row
        records=[transform(row) for _,row in selected]
        packet=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records)
        runtimes={variant:snapshot_task_runtime(getattr(args,variant+'_runtime'),out/'runtimes')
                  for variant in ('baseline','candidate')}
        plan={'schema':'natlang.native-marker-feedback-window/1','diagnostic_only':True,'training_admitted':False,
            'source':binding(source['manifest']),'selection':binding(args.selection),'parent_receipt':parent,
            'checkpoint_sha256':checkpoint,'root_mode':args.root_mode,
            'cases':[{'id':row['id'],'source_index':index,'record_sha256':digest(row)} for index,row in selected],
            'sampling':{'group':4,'temperature':args.temperature,'seed_base':args.seed_base},
            'window':{'max_model_requests':args.max_model_requests,'max_turns':args.max_model_requests,
                'action_max_tokens':args.action_max_tokens,'case_seconds':args.case_seconds},
            'transport':'nonstreaming/exact-capture-with-legacy-error-projection',
            'runtimes':{variant:{'path':str(runtime),'inventory':task_runtime_inventory(runtime)}
                        for variant,runtime in runtimes.items()}}
        for variant in runtimes:
            directory=out/variant;directory.mkdir(exist_ok=True)
            target=directory/'packet.jsonl'
            if target.exists() and target.read_text()!=packet:raise ValueError('feedback packet changed')
            if not target.exists():
                from natlang_neuralese.common.jsonio import write_bytes_durable
                write_bytes_durable(target,packet.encode())
            queue=build_queue(target,12,4,directory,temperature=args.temperature,case_seconds=args.case_seconds,
                max_model_requests=args.max_model_requests,max_turns=args.max_model_requests,seed_base=args.seed_base)
            for entry in queue:
                entry.update(action_max_tokens=args.action_max_tokens,prompt_rendering='full',context_tokens=32768,
                    transport_retries=0,request_retries=0,neuralese_server=True,text_neuralese_emulation=False)
            from natlang_neuralese.common.jsonio import write_bytes_durable
            queue_path=directory/'queue.jsonl';text=''.join(json.dumps(entry)+'\n' for entry in queue)
            if queue_path.exists() and queue_path.read_text()!=text:raise ValueError('feedback queue changed')
            if not queue_path.exists():write_bytes_durable(queue_path,text.encode())
            plan.setdefault('queues',{})[variant]=binding(queue_path)
            plan.setdefault('packets',{})[variant]=binding(target)
        immutable(out/'plan.json',plan)
        return out/'plan.json'


def captures(response):
    if not isinstance(response,dict):return
    if response.get('x_natlang_rl_capture'):yield response['x_natlang_rl_capture']
    for turn in response.get('source_turns') or []:yield from captures(turn['response'])


def project_response(response,variant):
    """The old supported nonstreaming wire envelope, not another repair policy."""
    failure=response.get('x_natlang_native_failure') if isinstance(response,dict) else None
    if any(capture.get('protocol_failure') for capture in captures(response)) and not failure:
        raise ValueError('feedback failed capture lacks its typed failure envelope')
    root=response
    if failure and response.get('source_turns'):
        root=response['source_turns'][-1]['response']
    if failure:
        capture=root.get('x_natlang_rl_capture') or {}
        if failure!=capture.get('protocol_failure'):
            raise ValueError('feedback envelope differs from its actual failed capture')
        choice=(root.get('choices') or [{}])[0]
        if choice.get('finish_reason')!='error' or (choice.get('message') or {}).get('tool_calls'):
            raise ValueError('feedback failed envelope exposes an executable successful choice')
        validate_failure(capture,sum(len(segment['token_ids']) for segment in capture['prompt'] if segment['kind']=='tokens'))
    if variant=='baseline' and failure:
        return 400,{'error':{'code':failure['code'],'message':failure['message']}}
    return 200,response


def feedback_metrics(observations):
    """Correlate exact same-call runtime feedback; never infer successful task answers."""
    exposures=[];unmatched=0
    for index,current in enumerate(observations):
        messages=current['request'].get('messages') or []
        if len(messages)<2:continue
        for previous in reversed(observations[:index]):
            expected=previous.get('expected_runtime_feedback')
            prior=previous['request'].get('messages')
            if not expected or prior is None or len(messages)!=len(prior)+2 or messages[:-2]!=prior:continue
            if messages[-1]!={'role':'user','content':expected}:continue
            if messages[-2].get('role')!='assistant':continue
            if previous['variant']!='candidate':raise ValueError('legacy baseline unexpectedly received new runtime feedback')
            exposures.append({'failure_started_at':previous['started_at'],'repair_started_at':current['started_at'],
                'repair_elapsed_seconds':current['finished_at']-current['started_at'],
                'repeated_protocol_failure':bool(current['protocol_failures']),
                'protocol_accepted':not bool(current['protocol_failures'])})
            break
        else:
            if current['protocol_failures']:unmatched+=1
    return {'actual_feedback_exposures':len(exposures),'feedback_replies':exposures,
        'protocol_accepted_repairs':sum(x['protocol_accepted'] for x in exposures),
        'repeated_protocol_failures_after_feedback':sum(x['repeated_protocol_failure'] for x in exposures),
        'ordinary_protocol_valid_observations':sum(not x['protocol_failures'] for x in observations)-sum(x['protocol_accepted'] for x in exposures),
        'failed_observations_without_prior_feedback':unmatched}


class RecordingProxy:
    def __init__(self,endpoint,variant,directory,identity,feedback=None):
        self.endpoint=endpoint.rstrip('/');self.variant=variant;self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True);self.identity=identity;self.lock=threading.Lock()
        self.next=max([int(path.stem) for path in self.directory.glob('[0-9]*.json')]+[-1])+1
        self.feedback=feedback

    def fetch(self,path,*,method='GET',body=None,headers=None):
        request=urllib.request.Request(self.endpoint+path,data=body,method=method,headers=headers or {})
        with urllib.request.urlopen(request) as response:return response.read(),response.headers.get_content_type()

    def observe(self,path,request,response,started):
        projected=project_response(response,self.variant)
        entries=list(captures(response))
        if not entries:raise ValueError('feedback actor omitted the requested actual turn capture')
        for capture in entries:
            if any(capture['policy'].get(key)!=self.identity[key] for key in IDENTITY_KEYS):
                raise ValueError('feedback actor changed during an actual sampled turn')
            if capture.get('protocol_failure'):
                validate_failure(capture,sum(len(segment['token_ids']) for segment in capture['prompt'] if segment['kind']=='tokens'))
        with self.lock:
            ordinal=self.next;self.next+=1
            directory=self.directory/f'{ordinal:06d}'
            def fetch(ident):
                route='native-states' if ident.startswith('nzs1_') else 'native-tensor-artifacts' if ident.startswith('nnt1_') else 'blocks'
                return self.fetch('/v1/neuralese/'+route+'/'+ident)[0]
            bundle=save_bundle({'request':request,'response':response},directory,policy_id=self.identity['checkpoint_sha256'],fetch=fetch)
            record={'schema':'natlang.native-feedback-observation/1','variant':self.variant,'path':path,
                'started_at':started,'finished_at':time.time(),'request':request,'response':response,
                'bundle':binding(directory/'manifest.json'),'sampled_turns':len(entries),
                'protocol_failures':sum(bool(capture.get('protocol_failure')) for capture in entries),
                'diagnostic_only':True,'training_admitted':False}
            failure=response.get('x_natlang_native_failure')
            if failure and self.feedback is not None:
                record['expected_runtime_feedback']=self.feedback(failure)
            immutable(self.directory/f'{ordinal:06d}.json',record)
        return projected

    def handler(self):
        observer=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def send(self,status,raw,kind):
                self.send_response(status);self.send_header('content-type',kind)
                self.send_header('content-length',str(len(raw)));self.end_headers();self.wfile.write(raw)
            def relay(self):
                try:
                    started=time.time();body=self.rfile.read(int(self.headers.get('content-length','0')))
                    original=json.loads(body) if body and self.headers.get_content_type()=='application/json' else None
                    forwarded=original
                    observed=self.command=='POST' and self.path in ('/v1/chat/completions','/v1/neuralese/view')
                    if observed:forwarded={**original,'stream':False,'x_natlang_rl_capture':True,'x_natlang_native_protocol_recovery':True}
                    headers={key:value for key,value in self.headers.items() if key.lower() in ('content-type','x-natlang-owner')}
                    raw,kind=observer.fetch(self.path,method=self.command,
                        body=json.dumps(forwarded).encode() if original is not None else body or None,headers=headers)
                    status=200
                    if observed:
                        status,value=observer.observe(self.path,original,json.loads(raw),started);raw=json.dumps(value).encode();kind='application/json'
                    elif self.path=='/v1/neuralese/info':
                        value=json.loads(raw);value['stream']=False
                        if observer.variant=='baseline':value['capabilities']=[name for name in value.get('capabilities',[]) if name!='native.protocol-recovery']
                        raw=json.dumps(value).encode()
                    self.send(status,raw,kind)
                except urllib.error.HTTPError as error:self.send(error.code,error.read(),error.headers.get_content_type())
                except Exception as error:
                    # Persist exact local failure separately; never synthesize a model failure.
                    write_json_durable(observer.directory/'observer-error.json',{'type':type(error).__name__,'message':str(error)})
                    self.send(502,json.dumps({'error':{'code':'feedback-observer','message':type(error).__name__}}).encode(),'application/json')
            do_GET=relay;do_POST=relay;do_PUT=relay;do_DELETE=relay
        return Handler


def run_queue(command, *, directory, env, require_first_capture=False):
    """Measure the first actual draw before spending the rest of a live window."""
    directory=Path(directory)
    if require_first_capture:
        queue=Path(command[2]);lines=queue.read_bytes().splitlines(keepends=True)
        if not lines:raise ValueError('feedback queue is empty')
        prefix=directory/'first-sample-queue.jsonl'
        if prefix.exists():
            if prefix.read_bytes()!=lines[0]:raise ValueError('feedback first queue entry changed')
        else:
            from natlang_neuralese.common.jsonio import write_bytes_durable
            write_bytes_durable(prefix,lines[0])
        first_command=[*command[:2],str(prefix),*command[3:]]
        first=subprocess.run(first_command,env=env)
        observations=[json.loads(path.read_text()) for path in sorted((directory/'observations').glob('[0-9]*.json'))]
        sampled=sum(value['sampled_turns'] for value in observations)
        observer_error=(directory/'observations/observer-error.json').exists()
        passed=first.returncode==0 and sampled>0 and not observer_error
        write_json_durable(directory/'first-capture-check.json',{
            'schema':'natlang.native-feedback-first-capture/1','passed':passed,
            'queue':binding(queue),'first_queue':binding(prefix),'command':first_command,
            'returncode':first.returncode,'actual_sampled_turns':sampled,'observer_error':observer_error,
            'first_draw_preserved':True,'diagnostic_only':True,'training_admitted':False})
        if not passed:raise RuntimeError('feedback first case did not produce a valid actual sampled capture; remaining queue was not run')
    return subprocess.run(command,env=env)


def run(args):
    plan_path=Path(args.plan).resolve();plan=json.loads(plan_path.read_text());root=plan_path.parent
    if plan.get('schema')!='natlang.native-marker-feedback-window/1':raise ValueError('unsupported feedback plan')
    if trained_parent(plan['parent_receipt'])!=plan['checkpoint_sha256']:raise ValueError('feedback parent changed')
    _,selected=select_rows(bound(plan['source']),bound(plan['selection']))
    actual=[{'id':row['id'],'source_index':index,'record_sha256':digest(row)} for index,row in selected]
    if actual!=plan['cases']:raise ValueError('feedback task selection changed')
    bound(plan['packets'][args.variant])
    runtime=Path(plan['runtimes'][args.variant]['path']);inventory=plan['runtimes'][args.variant]['inventory']
    if task_runtime_inventory(runtime)!=inventory:raise ValueError('feedback runtime changed')
    with urllib.request.urlopen(args.endpoint.rstrip('/')+'/v1/neuralese/info') as response:info=json.load(response)
    identity={key:info['native_state'][key] for key in IDENTITY_KEYS}
    if (identity['architecture'] not in ('lfm2','lfm2_moe') or identity['checkpoint_sha256']!=plan['checkpoint_sha256']
            or 'native.protocol-recovery' not in info.get('capabilities',[])):
        raise ValueError('feedback endpoint must be the exact trained LFM marker-capable actor')
    directory=root/args.variant
    node=Path(args.node).resolve(strict=True)
    executor={'node':binding(node),'version':subprocess.check_output([str(node),'--version'],text=True).strip(),
              'model_id':args.model_id,'actor':identity}
    immutable(root/'executor.json',executor)
    feedback_module=Path(plan['runtimes']['candidate']['path'])/'dist/native/native-protocol.js'
    def feedback(failure):
        program='import {nativeProtocolFeedback} from '+json.dumps(feedback_module.as_uri())+'; console.log(nativeProtocolFeedback(JSON.parse(process.argv[1])));'
        return subprocess.check_output([str(node),'--input-type=module','-e',program,json.dumps(failure)],text=True).rstrip('\n')
    with (directory/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        proxy=RecordingProxy(args.endpoint,args.variant,directory/'observations',identity,feedback)
        proxy_path=directory/'proxy.json'
        old_proxy=json.loads(proxy_path.read_text()) if proxy_path.exists() else None
        if old_proxy and old_proxy['plan']!=binding(plan_path):raise ValueError('feedback proxy belongs to another plan')
        server=ThreadingHTTPServer(('127.0.0.1',old_proxy['port'] if old_proxy else 0),proxy.handler())
        server.daemon_threads=True
        try:
            immutable(proxy_path,{'plan':binding(plan_path),'port':server.server_address[1]})
            threading.Thread(target=server.serve_forever,daemon=True).start()
            endpoint=f'http://127.0.0.1:{server.server_address[1]}'
            from natlang_neuralese.eval.task_cadence import node_path_env
            env=node_path_env();env['PATH']=str(node.parent)+os.pathsep+env.get('PATH','')
            env['NATLANG_SOURCE_REVIEWS']=str(runtime.parent/'training/source-reviews')
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/'run_bonsai_queue.py'),
                str(bound(plan['queues'][args.variant])),str(directory/'journal.jsonl'),'--runtime',str(runtime),
                '--server',endpoint,'--model-id',args.model_id,'--chat-completions-url',endpoint+'/v1/chat/completions',
                '--neuralese-server','--model-concurrency','1','--max-batch-cases','1','--identical-turn-limit','0']
            immutable(directory/'launch.json',{'plan':binding(plan_path),'actor_info':info,'entrypoint':binding(__file__),
                'queue_runner':binding(command[1]),'command':command,'executor':executor,
                'upstream':args.endpoint,'diagnostic_only':True,'training_admitted':False})
            result=run_queue(command,directory=directory,env=env,
                require_first_capture=getattr(args,'require_first_capture',False))
        finally:
            # shutdown waits for serve_forever: do not call it before that thread started.
            if 'endpoint' in locals():server.shutdown()
            server.server_close()
        if task_runtime_inventory(runtime)!=inventory:raise ValueError('feedback runtime mutated during measurement')
        with urllib.request.urlopen(args.endpoint.rstrip('/')+'/v1/neuralese/info') as response:after=json.load(response)
        if any(after['native_state'][key]!=identity[key] for key in IDENTITY_KEYS):raise ValueError('feedback actor changed before window drain')
        observations=[json.loads(path.read_text()) for path in sorted((directory/'observations').glob('[0-9]*.json'))]
        journal=directory/'journal.jsonl'
        finished=[json.loads(line) for line in journal.read_text().splitlines() if line.strip() and json.loads(line).get('event')=='finish'] if journal.exists() else []
        finished_keys={entry['key'] for entry in finished}
        report={'schema':'natlang.native-marker-feedback-report/1','variant':args.variant,'returncode':result.returncode,
            'plan':binding(plan_path),'actor':identity,'runtime_inventory_sha256':digest(inventory),
            'planned_windows':48,'finished_windows':len(finished_keys),'finish_events':len(finished),
            'actual_sampled_turns':sum(x['sampled_turns'] for x in observations),
            'protocol_failures':sum(x['protocol_failures'] for x in observations),'observations':[binding(path) for path in sorted((directory/'observations').glob('[0-9]*.json'))],
            'observer_error':(directory/'observations/observer-error.json').exists(),
            'unfinished_workflows':'censored by declared diagnostic window; no model-negative rewards',
            'diagnostic_only':True,'training_admitted':False,'feedback_quality_passed':None}
        report.update(feedback_metrics(observations))
        write_json_durable(directory/'report.json',report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    prepare_parser=commands.add_parser('prepare')
    for name in ('task-source','selection','actor-receipt','baseline-runtime','candidate-runtime','out'):
        prepare_parser.add_argument('--'+name,type=Path,required=True)
    prepare_parser.add_argument('--root-mode',choices=('authored','student'),required=True)
    for name in ('max-model-requests','action-max-tokens','case-seconds','seed-base'):
        prepare_parser.add_argument('--'+name,type=int,required=True)
    prepare_parser.add_argument('--temperature',type=float,required=True)
    run_parser=commands.add_parser('run');run_parser.add_argument('--plan',type=Path,required=True)
    run_parser.add_argument('--variant',choices=('baseline','candidate'),required=True)
    run_parser.add_argument('--endpoint',required=True);run_parser.add_argument('--model-id',required=True)
    run_parser.add_argument('--node',type=Path,required=True)
    run_parser.add_argument('--require-first-capture',action='store_true')
    inventory_parser=commands.add_parser('inventory')
    inventory_parser.add_argument('--baseline-runtime',type=Path,required=True)
    inventory_parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='prepare' and (min(args.max_model_requests,args.action_max_tokens,args.case_seconds)<1 or args.temperature<=0):parser.error('diagnostic windows must be explicitly positive')
    if args.command=='inventory':
        value=composition_inventory(Path(__file__).resolve().parents[2],args.baseline_runtime)
        immutable(args.out,value);print(json.dumps({'path':str(args.out),'sha256':sha256_file_hex(args.out)}))
    else:print(prepare(args) if args.command=='prepare' else json.dumps(run(args)))


if __name__=='__main__':main()
