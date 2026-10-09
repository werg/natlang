#!/usr/bin/env python3
"""Turn checkpoint-visible hard-row flags into a pinned, validated rewrite plan.

Only a reviewed source plan supplies executable tasks. Unmapped source rows remain
explicitly unresolved. Generated data still needs ordinary replay/render/admission.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha  # noqa: E402


def source_scope_reason(program, eligible, reviewed):
    if program in eligible:return None
    return 'not-selected-for-current-batch' if program in reviewed else 'reviewed-repair-source-missing'


def prepare(outbox, checkpoint, source_plan, output, limit=32):
    checkpoint=Path(checkpoint).resolve()
    state_sha=sha(checkpoint/'state.json')
    state=json.loads((checkpoint/'state.json').read_text())
    plan=json.loads(Path(source_plan).read_text())
    if plan.get('schema')!='natlang.student_chunk_rewrite_plan/1':
        raise ValueError('repair requires a reviewed chunk rewrite plan')
    db=sqlite3.connect(f'file:{Path(outbox).resolve()}?mode=ro',uri=True)
    try:
        identity=json.loads(db.execute('SELECT value FROM identity').fetchone()[0])
        observations=db.execute('SELECT key,step,row_offset,receipt FROM observations WHERE step<=? ORDER BY step DESC,row_offset',
                                (state['step'],)).fetchall()
    finally:db.close()
    if identity['data_sha256']!=state['corpus']['data_sha256']:
        raise ValueError('outbox corpus differs from checkpoint')
    if identity['policy']!=state['corpus'].get('online_repair'):
        raise ValueError('outbox policy differs from checkpoint')
    # The pinned student adapter must be an exact weight export of this checkpoint.
    adapter=Path(plan['student']['adapter'])
    student_weights=adapter/'adapter_model.safetensors'
    checkpoint_weights=(checkpoint/'weights' if (checkpoint/'weights').is_dir() else checkpoint)/'adapter_model.safetensors'
    if not student_weights.is_file() or not checkpoint_weights.is_file() or sha(student_weights)!=sha(checkpoint_weights):
        raise ValueError('repair student must match the committed checkpoint adapter exactly')
    pins=plan['student']['weight_pins']
    if pins.get(str(student_weights.resolve()))!=sha(student_weights):
        raise ValueError('repair student weights are not pinned')
    # Corpus path is supplied through an explicit plan extension, never guessed.
    corpus=Path(plan['online_repair_corpus'])
    if sha(corpus)!=identity['data_sha256']:
        raise ValueError('repair corpus hash mismatch')
    eligible=set(plan['selected_program_ids']);reviewed=set(plan.get('reviewed_program_ids',plan['selected_program_ids']))
    if not eligible.issubset(reviewed):raise ValueError('selected programs must belong to the reviewed source catalog')
    selected=[];unresolved=[];outside_batch=[];seen=set();flags=[];latest_offsets=set()
    with corpus.open('rb') as stream:
        for key,step,offset,text in observations:
            if offset in latest_offsets:continue
            latest_offsets.add(offset)
            receipt=json.loads(text)
            if receipt['cutoff'] is None:continue
            stream.seek(offset);row=json.loads(stream.readline())
            if row.get('split') in ('test','validation','heldout'):
                raise ValueError('held-out row in repair outbox')
            program=row.get('program_id')
            flags.append(dict(key=key,step=step,row_id=row.get('id'),program_id=program,**receipt))
            reason=source_scope_reason(program,eligible,reviewed)
            if reason:
                target=outside_batch if reason=='not-selected-for-current-batch' else unresolved
                target.append(dict(row_id=row.get('id'),program_id=program,reason=reason))
            elif program not in seen:
                seen.add(program)
                if len(selected)<limit:selected.append(program)
    if not selected:raise ValueError('no flagged samples have reviewed executable sources')
    chosen=set(selected)
    selected=[p for p in plan['selected_program_ids'] if p in chosen]
    artifacts=[]
    for artifact in plan['teacher_artifacts']:
        if sha(artifact['path']) != artifact['sha256']:
            raise ValueError('reviewed teacher reference changed')
        teacher=json.loads(Path(artifact['path']).read_text())
        if teacher.get('task',{}).get('program_ir',{}).get('id') in chosen:
            artifacts.append(artifact)
    if len(artifacts)!=len(selected):
        raise ValueError('flagged programs lack unique reviewed teacher references')
    plan['teacher_artifacts']=artifacts
    if sha(checkpoint/'state.json')!=state_sha or sha(checkpoint_weights)!=sha(student_weights):
        raise ValueError('checkpoint changed while preparing repair batch')
    output=Path(output).resolve()
    if output.exists():raise ValueError('repair batch output already exists')
    output.mkdir(parents=True)
    plan.pop('online_repair_corpus',None)
    plan['selected_program_ids']=selected
    plan['output']=str(output/'collection')
    plan['online_repair']={'schema':'natlang.online_repair_batch/1',
        'checkpoint_step':state['step'],'checkpoint_state_sha256':state_sha,
        'checkpoint_weights_sha256':sha(checkpoint_weights),'source_plan_sha256':sha(source_plan),
        'outbox_identity':identity,'flags':flags,'unresolved':unresolved,'outside_batch':outside_batch,
        'deferred_program_ids':sorted(seen-set(selected)),
        'admission':'fresh-native-task-oracle-plus-student-threshold; ordinary data audit required'}
    path=output/'plan.json';path.write_text(json.dumps(plan,indent=2)+'\n')
    return {'plan':str(path),'sha256':sha(path),'selected':len(selected),'unresolved':len(unresolved),'outside_batch':len(outside_batch)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outbox',required=True,type=Path)
    ap.add_argument('--checkpoint',required=True,type=Path)
    ap.add_argument('--source-plan',required=True,type=Path)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--limit',type=int,default=32)
    a=ap.parse_args()
    if a.limit<1:ap.error('limit must be positive')
    print(json.dumps(prepare(a.outbox,a.checkpoint,a.source_plan,a.output,a.limit)))


if __name__=='__main__':main()
