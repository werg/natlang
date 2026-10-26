"""Resume earned updates from an explicitly bound, already completed collection."""
from pathlib import Path
import json

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import read_jsonl


def resume_sealed_collection(reference, request, *, validate_actors):
    """Keep original sampling/runtime provenance; never reopen the collector."""
    from ..train.native_readiness import _bound
    proof_path = _bound(reference)
    proof = json.loads(proof_path.read_text())
    out = Path(request['out']).resolve(strict=True)
    if proof.get('schema') != 'natlang.sealed-collection-continuation/1' or proof.get('out') != str(out):
        raise ValueError('sealed collection continuation has a different operation')
    files = proof.get('files') or {}
    required = {'collection-binding.json', 'score.json', 'rollouts.jsonl', 'packet.jsonl',
                'queue.jsonl', 'journal.jsonl', 'preparation.json', 'sampling-config.json'}
    if not required <= set(files):
        raise ValueError('sealed collection continuation omits its source records')
    for name, ref in files.items():
        path = _bound(ref)
        if (not Path(name).is_absolute() and '..' not in Path(name).parts
                and path == (out / name).resolve(strict=True)
                and path.stat().st_size == ref.get('bytes')):
            continue
        raise ValueError('sealed collection source path/size changed')
    binding = json.loads((out / 'collection-binding.json').read_text())
    report = json.loads((out / 'score.json').read_text())
    if (binding.get('schema') != 'natlang.rl-collection-binding/1'
            or report.get('schema') != 'natlang.s7-task-rollout-score/1'
            or report.get('collection_complete') is not True):
        raise ValueError('only a completed collection may bypass recollection')
    defaults = {'offset': 0, 'group': 4, 'root_mode': 'student', 'seed_base': 700000,
                'case_seconds': 600, 'max_model_requests': 128, 'max_turns': 160,
                'neuralese_server': True}
    expected = {key: request.get(key, default) for key, default in defaults.items()}
    expected.update(policy_id=request['policy_id'], rows=request['rows'], endpoints=request['endpoints'],
                    cases_sha256=sha256_file_hex_cached(Path(request['cases'])), temperature=1., guided=False)
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError('sealed collection policy/cases/sampling contract changed')
    original_native = binding.get('expected_native_actor')
    requested_native = request.get('expected_native')
    # The old actor's Python source stays recorded in its original receipt. A
    # future runtime change cannot relabel it; checkpoint/tokenizer/state and
    # every other native selector must still match the actual sampled policy.
    comparable = lambda value: {k: v for k, v in (value or {}).items() if k != 'python_runtime'}
    if comparable(original_native) != comparable(requested_native):
        raise ValueError('sealed collection native actor identity changed')
    coefficient = binding.get('coefficient_policy')
    if coefficient:
        path = Path(request.get('coefficient_policy', '')).resolve(strict=True)
        if str(path) != coefficient['path'] or sha256_file_hex_cached(path) != coefficient['sha256']:
            raise ValueError('sealed collection coefficient policy changed')
    elif request.get('coefficient_policy') is not None:
        raise ValueError('sealed collection gained a coefficient policy')
    validate_actors(report.get('actor_receipts'), policy_id=binding['policy_id'],
                    neuralese_server=binding['neuralese_server'], expected_native=original_native,
                    expected_coefficient=coefficient.get('runtime') if coefficient else None)
    entries = list(read_jsonl(out / 'queue.jsonl'))
    if len(entries) != binding['rows'] * binding['group'] or report.get('root_mode') != binding['root_mode']:
        raise ValueError('sealed collection sample accounting changed')
    for entry in entries:
        result = Path(entry['output'])
        if result.exists() and str(result.relative_to(out)) not in files:
            raise ValueError('sealed collection omitted a completed result source')
    rollouts = list(read_jsonl(out / 'rollouts.jsonl'))
    if (report.get('emitted_rollouts') != str(out / 'rollouts.jsonl')
            or len(rollouts) != report.get('rollouts') or len(rollouts) != len(entries)):
        raise ValueError('sealed collection rollout accounting changed')
    if any((row.get('policy') or {}).get('checkpoint_sha256') != binding['policy_id'] for row in rollouts):
        raise ValueError('sealed collection contains another sampled policy')
    from .graph_replay import block_ids, NativeBundleStore
    ids = set(block_ids(rollouts))
    if ids:
        manifest_path = Path(report.get('block_bundle', '')).resolve(strict=True)
        if manifest_path != (out / 'blocks/manifest.json').resolve(strict=True) or 'blocks/manifest.json' not in files:
            raise ValueError('sealed collection lacks its exact native artifact manifest')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('collection_binding') != binding:
            raise ValueError('sealed native artifact provenance changed')
        store = NativeBundleStore(manifest_path, policy_id=binding['policy_id'])
        for ident in ids:
            record = store.get_native_state(ident) if ident.startswith('nzs1_') else store.get_native_tensor_artifact(ident)
            if record is None:
                raise ValueError('sealed collection lost a native artifact')
    return report
