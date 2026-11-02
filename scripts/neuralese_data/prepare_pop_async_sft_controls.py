#!/usr/bin/env python3
"""Prepare reviewable 16-update SFT controls and separate protected-development monitoring.

The derived controls are not a running controller and confer no qualification.
The existing trainer retains its configured final/foundation task gate.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
from natlang_neuralese.eval.native_async import file_binding
from natlang_neuralese.eval.task_cadence import parse_task_eval,require_native_transport
from natlang_neuralese.train.native_execution_transition import DEV_PACKET_SHA256
from natlang_neuralese.common.jsonio import write_json_durable


def prepare(config_path,spec_path,baseline_sha,out):
    if not re.fullmatch('[0-9a-f]{64}',baseline_sha):raise ValueError('baseline requires an exact full-source SHA256')
    config_ref=file_binding(config_path);spec_ref=file_binding(spec_path)
    config=json.loads(Path(config_ref['path']).read_text());spec=require_native_transport(parse_task_eval(Path(spec_ref['path']).read_text()))
    packet=spec['packet']
    if (packet.get('sha256')!=DEV_PACKET_SHA256 or len(spec['cases'])!=16
            or spec.get('prompt_rendering')!='full'):
        raise ValueError('monitor requires the exact protected 16-case full-prompt development packet')
    if file_binding(packet['path'])['sha256']!=DEV_PACKET_SHA256:
        raise ValueError('protected development packet differs')
    rows=[json.loads(line) for line in Path(packet['path']).read_text().splitlines() if line.strip()]
    if (max(spec['cases'])>=len(rows) or len({rows[i].get('id') for i in spec['cases']})!=16
            or len({rows[i].get('family') for i in spec['cases']})!=16
            or any(not isinstance(rows[i].get('family'),str) or not rows[i]['family'] for i in spec['cases'])):
        raise ValueError('monitor cases do not cover 16 distinct declared families')
    trainer=config.get('trainer_parameters')
    if not isinstance(trainer,dict) or not config.get('native_state') or not trainer.get('task_eval'):
        raise ValueError('16-update controls require the actual native supervised controller config')
    derived=dict(config)
    for name in ('trainer_parameters','seed_trainer_parameters'):
        if name=='seed_trainer_parameters' and name not in config:continue
        values=config[name]
        if not isinstance(values,dict):raise ValueError('invalid supervised trainer controls')
        if name=='seed_trainer_parameters' and not values.get('task_eval'):
            raise ValueError('seed continuation lacks its existing final task gate')
        derived[name]={**values,'refresh_after_updates':16,'refresh_at_checkpoint':True,'task_eval_every':0}
    out=Path(out).absolute();out.mkdir(parents=True,exist_ok=False)
    write_json_durable(out/'controller-config.json',derived)
    write_json_durable(out/'monitor-spec.json',spec)
    receipt={'schema':'natlang.pop-async-sft-controls/1','input_controller':config_ref,'input_monitor_spec':spec_ref,
        'controller_config':file_binding(out/'controller-config.json'),'monitor_spec':file_binding(out/'monitor-spec.json'),
        'baseline_full_source_sha256':baseline_sha,'candidate_full_source_sha256':None,
        'window_updates':16,'task_cadence':'external monitoring; existing final/foundation gate retained',
        'monitor_status':'pending','qualifies_current_weights':False,'optimizer_updates':0,'training_credit':0}
    write_json_durable(out/'preparation.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','spec','baseline-sha256','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.config,a.spec,a.baseline_sha256,a.out)),flush=True)
if __name__=='__main__':main()
