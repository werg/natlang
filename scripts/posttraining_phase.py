"""Authorize a new SFT phase without discarding optimizer/RNG or prior lineage."""
import json, math
from pathlib import Path
from scripts.corpus import file_digest, digest, index_pairs, records, program_id, split_programs


def checkpoint_hashes(root):
    return {str(p.relative_to(root)):file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()}


def verify_phase(manifest_path, data, checkpoint, *, seed, holdout, lr, accum):
    m=json.loads(Path(manifest_path).read_text())
    if m.get('schema')!='natlang.posttraining_phase/1' or m.get('status')!='approved':
        raise ValueError('post-training phase needs an exact approved manifest')
    for p,h in m['pins'].items():
        if file_digest(Path(p))!=h: raise ValueError('phase input changed: '+p)
    if str(Path(data).resolve())!=m['data']['path'] or file_digest(data)!=m['data']['sha256']:
        raise ValueError('phase corpus differs from manifest')
    if float(lr)!=m['learning_rate'] or accum!=m['accum'] or seed!=m['seed'] or holdout!=m['holdout']:
        raise ValueError('phase controls differ from manifest')
    if not math.isfinite(lr) or lr<=0 or not math.isfinite(m['epochs']) or m['epochs']<=0:
        raise ValueError('phase needs positive finite LR/exposure')
    state=json.loads((checkpoint/'state.json').read_text())
    first=state.get('posttraining_phase',{}).get('manifest_sha256')!=file_digest(manifest_path)
    if first:
        if checkpoint_hashes(checkpoint)!=m['parent']['files']:
            raise ValueError('phase must start from exact copied parent checkpoint')
        if state!=m['parent']['state']: raise ValueError('parent state differs')
        if state['trained_examples']<state['corpus']['target_examples']:
            raise ValueError('phase requires a completed parent example target')
    held,train,split=split_programs(index_pairs(data),holdout,seed)
    if digest(split)!=m['data']['split_sha256'] or len(train)!=m['data']['train_rows']:
        raise ValueError('phase split differs')
    # Check the actual rendered protected rows, not only program/group names.
    old_split=json.loads(Path(m['parent']['split_path']).read_text())
    held_programs=set(old_split['held_programs'])
    old_held={r['id']:digest(r) for r in records(Path(m['parent']['data_path'])) if program_id(r) in held_programs}
    new_held_ids={r['id'] for r in held}
    new_held={r['id']:digest(r) for r in records(data) if r['id'] in new_held_ids}
    if new_held!=old_held: raise ValueError('phase changes protected held-out bytes or membership')
    if m.get('gates',{}).get('native_admission') is not True or m['gates'].get('source_closure') is not True:
        raise ValueError('phase native/source gates missing')
    return m,first


def prepare_phase(*, data, parent, split_path, gate_path, output, lr=2e-5, epochs=1.0):
    """Prepare reviewable manifest; all ordinary trainer audits remain required."""
    parent=Path(parent).resolve();data=Path(data).resolve();split_path=Path(split_path).resolve();gate_path=Path(gate_path).resolve()
    state=json.loads((parent/'state.json').read_text());args=state['args'];gate=json.loads(gate_path.read_text())
    if gate.get('schema')!='natlang.posttraining_data_gates/1' or gate.get('data_sha256')!=file_digest(data):
        raise ValueError('missing exact post-training native/source gate receipt')
    if gate.get('native_admission') is not True or gate.get('source_closure') is not True:
        raise ValueError('unreviewed post-training data')
    seed=int(args['seed']);holdout=int(args['holdout']);accum=int(args['accum'])
    held,train,split=split_programs(index_pairs(data),holdout,seed)
    old_data=Path(args['data']).resolve()
    extra_examples=math.ceil(epochs*len(train));end_step=state['step']+math.ceil(extra_examples/accum)
    pins={str(p):file_digest(p) for p in [data,old_data,split_path,gate_path,parent/'state.json']}
    m={'schema':'natlang.posttraining_phase/1','status':'prepared_not_approved','pins':pins,
       'parent':{'files':checkpoint_hashes(parent),'state':state,'split_path':str(split_path),'data_path':str(old_data)},
       'data':{'path':str(data),'sha256':file_digest(data),'split_sha256':digest(split),'train_rows':len(train),'held_rows':len(held)},
       'learning_rate':lr,'epochs':epochs,'seed':seed,'holdout':holdout,'accum':accum,
       'additional_examples':extra_examples,'end_step':end_step,'target_examples':state['trained_examples']+extra_examples,
       'gates':{'native_admission':True,'source_closure':True,'receipt':str(gate_path),'sha256':file_digest(gate_path)},
       'preserves':['weights','optimizer buffers','RNG','protected held-out rows','parent lineage'],
       'new_phase_order':'new corpus split permutation; prior cursor preserved in lineage',
       'scheduler':'explicit positive LR, cosine from parent global step to phase end'}
    if not math.isfinite(lr) or lr<=0 or not math.isfinite(epochs) or epochs<=0: raise ValueError('invalid phase controls')
    with Path(output).open('x') as f:json.dump(m,f,indent=2);f.write('\n')
    return m

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','parent','split-path','gate-path','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--epochs',type=float,default=1)
    print(json.dumps(prepare_phase(**vars(p.parse_args())),indent=2))
