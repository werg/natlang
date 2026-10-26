"""Restore an ancestor's exact finite foundation declaration, never earned credit."""
from __future__ import annotations

import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached

SCHEMA = 'natlang.native-foundation-declaration-recovery/1'


def _source_map_call(resolver, path, *arguments, maps):
    """Bind map custody around an unchanged historical routing helper.

    Maps in the resolver's ancestry are a conservative closure: an earlier
    match may stop its scan, but no map that it could read is omitted. Keep the
    actual routing implementation byte-identical for authenticated reader
    continuations; custody does not select or reinterpret a source path.
    """
    def candidates():
        return [parent / '.natlang-source-map.json'
                for parent in Path(path).resolve().parents
                if (parent / '.natlang-source-map.json').is_file()]
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    before = {}
    for index in candidates():
        stat = index.stat()
        digest = sha256_file_hex_cached(index)
        if tuple(getattr(stat, key) for key in fields) != tuple(getattr(index.stat(), key) for key in fields):
            raise ValueError('staged native source map changed while reading: ' + str(index))
        before[index] = ({'path': str(index), 'sha256': digest, 'bytes': stat.st_size},
                         tuple(getattr(stat, key) for key in fields))
    result = resolver(path, *arguments)
    if set(candidates()) != set(before):
        raise ValueError('staged native source map changed while reading')
    for index, (reference, signature) in before.items():
        if (tuple(getattr(index.stat(), key) for key in fields) != signature
                or sha256_file_hex_cached(index) != reference['sha256']):
            raise ValueError('staged native source map changed while reading: ' + str(index))
        maps.append(reference)
    return result


def native_graph_bindings(reference):
    """Enumerate a graph header, its payload closure, and maps used to resolve it."""
    header = _bound(reference)
    # Match the production resolver exactly, while retaining the custody map
    # bytes that determined where each declared graph member was loaded from.
    from .recipe import native_source_dependencies
    from ..data.native_source_tree import _staged_path, _source_identity
    manifests = [header]
    seen = set()
    maps = []
    consumed = []
    _source_map_call(_source_identity, header, maps=maps)
    while manifests:
        manifest_path = manifests.pop().resolve()
        if manifest_path in seen or manifest_path.suffix != '.json':
            continue
        seen.add(manifest_path)
        value = json.loads(manifest_path.read_text())
        if value.get('schema') != 'natlang.native-graph-sources/1':
            continue
        for source in value.get('sources') or []:
            target = _source_map_call(_staged_path, manifest_path, source['path'], maps=maps)
            if not target.is_absolute():
                target = manifest_path.parent / target
            target = target.resolve(strict=True)
            _source_map_call(_source_identity, target, maps=maps)
            consumed.append(target)
            if source.get('kind') == 'native-graphs':
                manifests.append(target)

    # native_source_dependencies performs the canonical recursive schema,
    # identity and payload hash checks. Include its results unchanged.
    direct = [header, *consumed]
    def route(manifest, target):
        return _source_map_call(_staged_path, manifest, target, maps=maps)
    for item in native_source_dependencies(header, source_route=route):
        direct.append(_bound({'path': item['path'], 'sha256': item['sha256']}))
    direct.extend(_bound(reference) for reference in maps)
    unique = {}
    for path in direct:
        path = path.resolve(strict=True)
        unique[str(path)] = {'path': str(path), 'sha256': sha256_file_hex_cached(path),
                             'bytes': path.stat().st_size}
    if unique[str(header)]['sha256'] != reference['sha256'] or (
            'bytes' in reference and unique[str(header)]['bytes'] != reference['bytes']):
        raise ValueError('foundation graph dependency changed during enumeration')
    return list(unique.values())


def dependency_bindings(reference):
    """Return files needed to revalidate a selected recovery.

    Historical recipe/metadata/verification files are authenticated as
    documentary evidence only. Paths mentioned inside those documents, such
    as an unavailable old checkpoint, are never followed.
    """
    proof_path = _bound(reference)
    proof = json.loads(proof_path.read_text())
    required = {'schema', 'stage_id', 'historical_recipe', 'historical_training_metadata',
                'historical_checkpoint_verification', 'foundation_sources', 'reason'}
    if (not isinstance(proof, dict) or set(proof) != required or proof.get('schema') != SCHEMA
            or not isinstance(proof.get('stage_id'), str) or not proof['stage_id']
            or not isinstance(proof.get('reason'), str) or not proof['reason'].strip()):
        raise ValueError('foundation recovery dependency proof has an invalid schema')
    refs = [reference, proof['historical_recipe'], proof['historical_training_metadata'],
            proof['historical_checkpoint_verification']]
    for ref in refs:
        _bound(ref)
    direct = [proof_path, *(_bound(proof[key]) for key in ('historical_recipe',
        'historical_training_metadata', 'historical_checkpoint_verification'))]
    graph_items = native_graph_bindings(proof['foundation_sources'])
    direct.extend(Path(item['path']) for item in graph_items)
    unique = {}
    for path in direct:
        path = path.resolve(strict=True)
        unique[str(path)] = {'path': str(path), 'sha256': sha256_file_hex_cached(path),
                             'bytes': path.stat().st_size}
    # Recheck references against bytes captured after traversal, catching any
    # mutation between proof validation and the final inventory.
    for ref in (reference, proof['historical_recipe'], proof['historical_training_metadata'],
                proof['historical_checkpoint_verification'], proof['foundation_sources']):
        path = Path(ref['path']).resolve(strict=True)
        if unique[str(path)]['sha256'] != ref['sha256'] or (
                'bytes' in ref and unique[str(path)]['bytes'] != ref['bytes']):
            raise ValueError('foundation recovery dependency changed during enumeration')
    return list(unique.values())


def _bound(reference):
    if (not isinstance(reference, dict) or not isinstance(reference.get('path'), str)
            or not isinstance(reference.get('sha256'), str) or len(reference['sha256']) != 64):
        raise ValueError('foundation recovery requires exact artifact bindings')
    path = Path(reference['path']).resolve(strict=True)
    if sha256_file_hex_cached(path) != reference['sha256']:
        raise ValueError('foundation recovery artifact changed: ' + str(path))
    if 'bytes' in reference and path.stat().st_size != reference['bytes']:
        raise ValueError('foundation recovery artifact size changed: ' + str(path))
    return path


def restore_declaration(path, *, model_config, stage_id, training_frame,
                        snapshot_ancestors, foundation, active_bindings, epochs):
    """Authenticate historical declarations using actual ancestry and graph bindings.

    Exported metadata and verification receipts are trusted operator-owned
    recovery artifacts, authenticated by the recipe's exact input binding.
    They restore a declaration only. Current per-row counters and all numerical
    and behavioral qualification remain the executing trainer's responsibility.
    No supplied exposure counter is copied into the checkpoint.
    """
    path = Path(path).resolve(strict=True)
    proof = json.loads(path.read_text())
    required = {'schema', 'stage_id', 'historical_recipe', 'historical_training_metadata',
                'historical_checkpoint_verification', 'foundation_sources', 'reason'}
    if (not isinstance(proof, dict) or set(proof) != required or proof.get('schema') != SCHEMA
            or proof.get('stage_id') != stage_id or not isinstance(proof.get('reason'), str)
            or not proof['reason'].strip()):
        raise ValueError('foundation recovery needs an explicit exact historical stage declaration')
    recipe = json.loads(_bound(proof['historical_recipe']).read_text())
    metadata = json.loads(_bound(proof['historical_training_metadata']).read_text())
    verification = json.loads(_bound(proof['historical_checkpoint_verification']).read_text())
    source_path = _bound(proof['foundation_sources'])
    if (source_path != foundation.path.resolve() or proof['foundation_sources']['sha256'] != foundation.file_sha256):
        raise ValueError('foundation recovery selected another source header')
    checkpoint = verification.get('checkpoint') or {}
    digest = checkpoint.get('sha256')
    if (verification.get('passed') is not True or not isinstance(digest, str) or len(digest) != 64
            or type(checkpoint.get('bytes')) is not int or checkpoint['bytes'] <= 0
            or digest not in snapshot_ancestors or verification.get('optimizer_present') is not True
            or verification.get('rng_present') is not True or verification.get('stage_id') != stage_id):
        raise ValueError('foundation recovery checkpoint is not a verified actual ancestor of this stage')
    historical_config = metadata.get('model_config') or {}
    if (any(historical_config.get(key) != model_config.get(key)
            for key in ('architecture', 'base_inventory_sha256', 'tokenizer_fingerprint', 'marker_ids'))
            or verification.get('architecture') != model_config.get('architecture')):
        raise ValueError('foundation recovery historical model/tokenizer identity differs')
    historical_training = metadata.get('training') or {}
    historical_options = historical_training.get('options') or {}
    old_step = historical_training.get('global_step')
    if (type(old_step) is not int or old_step < 0 or old_step != verification.get('global_step')
            or old_step > training_frame.get('global_step', -1)
            or historical_options.get('stage_id') != stage_id
            or historical_options.get('communication_foundation') is not True
            or historical_options.get('epochs', 1.) != epochs):
        raise ValueError('foundation recovery did not recover a previously declared stage')
    if metadata.get('checkpoint_identity') is not None:
        if any(metadata['checkpoint_identity'].get(key) != checkpoint[key] for key in ('sha256', 'bytes')):
            raise ValueError('foundation recovery metadata names another historical checkpoint')
    elif metadata.get('checkpoint_path') != verification.get('path'):
        raise ValueError('foundation recovery metadata and checkpoint verification paths differ')
    stages = [stage for stage in recipe.get('stages', []) if stage.get('id') == stage_id]
    if len(stages) != 1 or stages[0].get('kind') != 'native_state_sft':
        raise ValueError('foundation recovery historical recipe lacks this native stage')
    stage = stages[0]
    source_name = (stage.get('inputs') or {}).get('foundation_native_graphs')
    historical_source = (recipe.get('input_bindings') or {}).get(source_name) or {}
    if (historical_source.get('sha256') != foundation.file_sha256
            or historical_source.get('path') != historical_options.get('foundation_native_graphs')
            or (stage.get('parameters') or {}).get('communication_foundation') is not True
            or (stage.get('parameters') or {}).get('epochs', 1.) != epochs):
        raise ValueError('foundation recovery historical recipe declared another foundation')
    saved_bindings = training_frame.get('source_row_bindings') or {}
    if (not foundation.bindings or any(saved_bindings.get(key) != value
            or active_bindings.get(key) != value for key, value in foundation.bindings.items())):
        raise ValueError('foundation recovery lost or changed a previously admitted graph binding')
    from .native_continuation import communication_foundation
    actual = communication_foundation(foundation, training_frame.get('exposure_counts') or {},
        epochs=epochs, trained_bindings=saved_bindings)
    if not actual['graphs'] or actual['graphs'] != len(foundation):
        raise ValueError('foundation recovery must contain only genuine producer/consumer graphs')
    return {'schema': SCHEMA, 'recovery': {'path': str(path), 'sha256': sha256_file_hex_cached(path)},
            'historical_checkpoint': checkpoint, 'foundation_sources': proof['foundation_sources'],
            'cohort_sha256': actual['cohort_sha256'], 'stage_id': stage_id,
            'at_global_step': training_frame['global_step'],
            'new_exposure': 0, 'qualification_granted': False}
