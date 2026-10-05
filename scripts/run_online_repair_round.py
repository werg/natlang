#!/usr/bin/env python3
"""Run one pinned repair batch after training releases the local GPU.

Config pins source/runtime/operator/server inputs. A new stable adapter snapshot is
captured from the completed checkpoint. Only hard rows enter teacher regeneration.
Outputs remain candidates until the normal source/native admission review and append.
"""
import argparse
import json
import shutil
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.prepare_online_repair_batch import prepare,sha


def snapshot_checkpoint(source,dest):
    source=Path(source);dest=Path(dest)
    paths=[source/'state.json',source/'adapter_config.json',source/'adapter_model.safetensors']
    before={p.name:sha(p) for p in paths}
    dest.mkdir(parents=True,exist_ok=False)
    for p in paths:shutil.copy2(p,dest/p.name)
    if before!={p.name:sha(p) for p in paths} or before!={p.name:sha(dest/p.name) for p in paths}:
        raise ValueError('checkpoint changed during snapshot; snapshot is not usable')
    return before


def wait_for_training(plan):
    expected=plan['completion']
    while True:
        state=json.loads(Path(plan['checkpoint_state']).read_text())
        if state['step']>=expected['step'] and state.get('trained_examples',0)>=expected['trained_examples']:
            if state['corpus'].get('phase_manifest_sha256')!=plan['checkpoint_identity']['phase_manifest_sha256']:
                raise ValueError('training checkpoint belongs to a different phase')
            # Wait through final held-out evaluation and supervisor completion.
            status_path=Path(plan['status_file'])
            if status_path.exists():
                status=json.loads(status_path.read_text())
                if status.get('state')=='complete':return
        time.sleep(30)


def run(config_path,expected_hash):
    if sha(config_path)!=expected_hash:raise ValueError('repair config changed')
    config=json.loads(Path(config_path).read_text())
    if config.get('schema')!='natlang.online_repair_round/1':raise ValueError('unsupported config')
    for path,digest in config['pins'].items():
        if sha(path)!=digest:raise ValueError('repair input changed: '+path)
    training=json.loads(Path(config['training_plan']).read_text())
    wait_for_training(training)
    for path,digest in config['pins'].items():
        if sha(path)!=digest:raise ValueError('repair input changed while waiting: '+path)
    root=Path(config['output'])
    root.mkdir(parents=True,exist_ok=False)
    checkpoint=Path(training['checkpoint_state']).parent
    weights=snapshot_checkpoint(checkpoint,root/'checkpoint')
    state=json.loads((root/'checkpoint/state.json').read_text())
    template=json.loads(Path(config['source_template']).read_text())
    template['student']['adapter']=str((root/'checkpoint').resolve())
    template['student']['weight_pins']={str((root/'checkpoint'/name).resolve()):h for name,h in weights.items()}
    template['online_repair_corpus']=state['args']['data']
    source_plan=root/'source-plan.json';source_plan.write_text(json.dumps(template,indent=2)+'\n')
    result=prepare(config['outbox'],root/'checkpoint',source_plan,root/'batch',config.get('batch_limit',32))
    plan_path=Path(result['plan']);plan=json.loads(plan_path.read_text())
    subprocess.run([config['node'],config['operator'],str(plan_path),'--preflight'],check=True)
    endpoint=plan['endpoint'];name=config['container_name'];adapter=plan['student']['adapter'];repo=config['repo']
    command=['docker','run','--rm','--gpus','all','--publish=127.0.0.1:18089:18089',
             '--name',name,'--memory=8g','--pids-limit=128','--user','1000:1000',
             '-v',config['server_dir']+':/server:ro','-v',adapter+':'+adapter+':ro',
             '-v',repo+'/models/hf:'+repo+'/models/hf:ro',
             '-e','HF_HOME='+repo+'/models/hf','-e','HF_HUB_OFFLINE=1',
             '-e','TOKENIZERS_PARALLELISM=false',config['image'],'python','/server/serve_improvement_student.py',
             '--revision',plan['student']['revision'],'--adapter',adapter,'--port','18089',
             '--host','0.0.0.0','--device','cuda','--max-context','16384','--max-output-tokens','1024']
    with (root/'server.log').open('xb') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        def stop(signum,frame):raise KeyboardInterrupt
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
        try:
            for _ in range(120):
                if child.poll() is not None:raise RuntimeError('student server exited')
                try:
                    with urllib.request.urlopen(endpoint+'/natlang/student-identity',timeout=2):pass
                    break
                except (OSError,TimeoutError):time.sleep(1)
            else:raise RuntimeError('student startup resource budget exhausted')
            subprocess.run([config['node'],config['operator'],str(plan_path),'--execute',result['sha256']],check=True)
        finally:
            subprocess.run(['docker','stop','--timeout','30',name],check=False)
            child.wait()
    base=['docker','run','--rm','--user','1000:1000','--cpus=2','--memory=4g','-v',repo+':'+repo,'-w',repo,
          '-e','HF_HOME='+repo+'/models/hf','-e','HF_HUB_OFFLINE=1',config['image'],'python']
    render_scripts=config.get('render_scripts',repo+'/scripts')
    common=['--model',plan['student']['base_model'],'--revision',plan['student']['revision'],'--streaming']
    rendered=root/'rendered.jsonl';ready=root/'ready.jsonl'
    subprocess.run(base+[render_scripts+'/render_training_corpus.py','--inputs',plan['output']+'/turns.jsonl',
                        '--output',str(rendered)]+common,check=True)
    subprocess.run(base+[render_scripts+'/audit_training_corpus.py','--input',str(rendered),
                        '--output',str(ready),'--max-len','16384']+common,check=True)
    receipt={**result,'schema':'natlang.online_repair_round_result/1','status':'candidates_token_audited',
             'ready':str(ready),'ready_sha256':sha(ready),'checkpoint_step':state['step'],
             'config_sha256':expected_hash,'automatic_training_publication':False}
    (root/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('config',type=Path);ap.add_argument('--execute',required=True)
    a=ap.parse_args();run(a.config,a.execute)


if __name__=='__main__':main()
