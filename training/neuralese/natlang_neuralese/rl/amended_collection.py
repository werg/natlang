"""Authenticate complete groups harvested from an interrupted collection."""
from pathlib import Path
import json

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import read_jsonl


def resume_amended_collection(reference, request, *, validate_actors):
    from ..train.native_readiness import _bound
    proof_path = _bound(reference)
    proof = json.loads(proof_path.read_text())
    out = Path(request['out']).resolve(strict=True)
    if proof.get('schema') != 'natlang.amended-collection-continuation/1' or proof.get('out') != str(out):
        raise ValueError('amended collection operation differs')
    files = proof.get('files') or {}
    required = {'collection-binding.json', 'score.json', 'rollouts.jsonl', 'packet.jsonl',
                'queue.jsonl', 'journal.jsonl', 'preparation.json', 'sampling-config.json'}
    if not required <= set(files):
        raise ValueError('amended collection omits original evidence')
    for name, ref in files.items():
        path = _bound(ref)
        if Path(name).is_absolute() or '..' in Path(name).parts or path != (out / name).resolve(strict=True) or path.stat().st_size != ref.get('bytes'):
            raise ValueError('amended collection evidence path or size changed')
    binding = json.loads((out/'collection-binding.json').read_text())
    report = json.loads((out/'score.json').read_text())
    if report.get('collection_complete') is not False or binding.get('schema') != 'natlang.rl-collection-binding/1':
        raise ValueError('amendment requires an interrupted bound collection')
    defaults = {'offset':0, 'group':4, 'root_mode':'student', 'seed_base':700000,
                'case_seconds':600, 'max_model_requests':128, 'max_turns':160, 'neuralese_server':True}
    expected = {key:request.get(key,default) for key,default in defaults.items()}
    expected.update(policy_id=request['policy_id'], rows=request['rows'], endpoints=request['endpoints'],
                    cases_sha256=sha256_file_hex_cached(Path(request['cases'])),temperature=1.,guided=False)
    if any(binding.get(key)!=value for key,value in expected.items()):
        raise ValueError('amended collection policy or sampling differs')
    original_native=binding.get('expected_native_actor')
    comparable=lambda value:{k:v for k,v in (value or {}).items() if k!='python_runtime'}
    if comparable(original_native)!=comparable(request.get('expected_native')):
        raise ValueError('amended collection actor identity differs')
    if binding.get('coefficient_policy') or request.get('coefficient_policy') is not None:
        raise ValueError('amended coefficient policy is not supported')
    validate_actors(report.get('actor_receipts'),policy_id=binding['policy_id'],
                    neuralese_server=binding['neuralese_server'],expected_native=original_native,expected_coefficient=None)
    chosen=proof.get('groups')
    if (not isinstance(chosen,list) or not chosen or any(type(i) is not int or i<0 or i>=binding['rows'] for i in chosen)
            or chosen!=sorted(set(chosen))):
        raise ValueError('amended groups must be unique ordered packet indices')
    entries=list(read_jsonl(out/'queue.jsonl'))
    if len(entries)!=binding['rows']*binding['group']:
        raise ValueError('original queue accounting differs')
    queues={(e['group'],e['sample']):e for e in entries}
    if len(queues)!=len(entries):raise ValueError('duplicate original queue sample')
    for group in chosen:
        for sample in range(binding['group']):
            entry=queues.get((group,sample))
            if not entry or entry.get('index')!=group:
                raise ValueError('amended group lacks a queue sample')
            path=Path(entry['output']).resolve(strict=True)
            name=str(path.relative_to(out))
            if name not in files or _bound(files[name])!=path:
                raise ValueError('amended group lacks bound completed result')
    filtered=Path(proof['filtered_rollouts']['path']).resolve(strict=True)
    if _bound(proof['filtered_rollouts'])!=filtered or filtered.stat().st_size!=proof['filtered_rollouts']['bytes']:
        raise ValueError('amended rollouts changed')
    # Verify the filtered bytes are exactly the ordered projection of the
    # authenticated original, not merely samples with matching identifiers.
    with (out/'rollouts.jsonl').open('rb') as original, filtered.open('rb') as projected:
        for line in original:
            row=json.loads(line)
            if row.get('group') in chosen and projected.readline()!=line:
                raise ValueError('amended rollout differs from original projection')
        if projected.read(1):raise ValueError('amended rollout has extra data')
    rows=list(read_jsonl(filtered))
    if len(rows)!=len(chosen)*binding['group']:
        raise ValueError('amended rollout count differs')
    actual={(r.get('group'),r.get('sample')) for r in rows}
    if actual!={(g,s) for g in chosen for s in range(binding['group'])}:
        raise ValueError('amended rollout sample set differs')
    if any(r.get('task_id') != queues[(r['group'],r['sample'])].get('task_id') for r in rows):
        raise ValueError('amended rollout task differs from original queue')
    if any((r.get('policy') or {}).get('checkpoint_sha256')!=binding['policy_id'] or r.get('reward') is None or not r.get('turns') for r in rows):
        raise ValueError('amended rollout lacks same-policy scored execution')
    from .graph_replay import block_ids, NativeBundleStore
    ids=set(block_ids(rows))
    if ids:
        manifest=Path(report.get('block_bundle','')).resolve(strict=True)
        if manifest!=(out/'blocks/manifest.json').resolve(strict=True) or 'blocks/manifest.json' not in files:
            raise ValueError('amended rollout lacks exact native bundle')
        store=NativeBundleStore(manifest,policy_id=binding['policy_id'])
        for ident in ids:
            record=store.get_native_state(ident) if ident.startswith('nzs1_') else store.get_native_tensor_artifact(ident)
            if record is None:raise ValueError('amended native artifact missing')
    groups=[report['groups'][i] for i in chosen]
    if any(g.get('scored')!=binding['group'] for g in groups):raise ValueError('amended group is not completely scored')
    return {**report,'groups':groups,'groups_with_spread':sum(bool(g.get('spread')) for g in groups),
            'scored':len(rows),'successes':sum(r['reward'].get('quality')==1 for r in rows),
            'rollouts':len(rows),'amended_complete_groups':chosen,'amended_rollouts':str(filtered),
            'amendment_proof':str(proof_path)}
