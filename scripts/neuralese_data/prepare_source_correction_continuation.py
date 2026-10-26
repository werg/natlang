#!/usr/bin/env python3
"""Prepare exact, non-deployed source-correction continuations for LFM/Mellum.

This operator reads current source manifests and authenticated replacement
outputs. It writes new manifests and row ledgers in a fresh output directory;
it never edits an input or a learner configuration/checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from pathlib import Path

from natlang_neuralese.common.hashing import (canonical_json_sha256_hex, sha256_file_hex,
                                               sha256_file_hex_cached)

from natlang_neuralese.common.paths import root as machine_root

ROOT = machine_root('data_nvme')
REPAIRS = machine_root('data_hdd') / 'natlang-data-repair-20261026-v4b'
LUNA = ROOT / 'luna-generation-20261026'
PACKAGE = LUNA / 'registered-packages'
CORRECTION_SCHEMA = 'natlang.native-source-correction/1'
MELLUM_REMOTE_ROOT = Path('/work/resume/training/native-source-correction-continuation-20261026-v5/mellum')

BAD_TRAIN = {
    '68a1ec0e2721c7506d530952dedaad9c89e025d62c101ff739065b51c555bd76': 'rest27 source shard remains pending full corrected conversion',
    'b9e08e3c0bb2327e33edf227a7dc3dcdf7b48c33f9a45b270dda394871a3d58b': 'rest27 partial source remains pending full corrected conversion',
    '024878bf8433708002f69c34c39f29f203e3766a0a069f24a623758c68ebe6be': 'typed-read source remains pending full corrected conversion',
    'e33989dff24f38b618ff79894f79c1064e05658f725c5e4b2d2c80779c5fa668': 'rest27 repair-tail source remains pending full corrected conversion',
    '6a9b8b5398709be503502ce5256c9c78eef69e0b0e54d6fc9d281d1a1ccddc3a': 'Big2 first100 remains pending bounded source recovery and corrected conversion',
}
CORRECTED_OLD = {
    '74179f3cae476a91d8005fe8f93d9f3afe273a2a6fa4cf5d0782eb03fc70e079',
    'a52ef4a459b43ab0fed57b543355c06fac27ff44249b8808306260a9c7b2d73f',
    'd8681fade4df5715445c07ee54278775c02069b9714f2f3520968a4e738bf81e',
    'fbc69df0b7b8133ccbe1b4f7be587e76e97abbfaa020fdf0fed8fc5db43ef356',
}
HELD_REPLACE = {
    '81e8f7446681402acee5a2d376d6a835127a45c763c2508765739b1b830c59cb': 'big2-offload-1000',
    'd59ca214245abb0dd3d8faf9975cdf4a4e50f453f0f61aca5a33559c3ea5c62a': 'big2-offload-5000',
}
BAD_HELD = {
    'ddec24c6036b18db393ccd76f4dd772a3208a587449597eefd3f29c07f1afeea': 'rest27 partial held targets await full corrected conversion',
    'a7ba2fc0fcc5d3cedc5d76e3d1481f0b0de0720b0ea9ad10d87ba0ecf5e0c8d3': 'typed-read held targets await full corrected conversion',
    '52bc090977197b1a5edadbffab1417191c1beb3efe515099824861dcb89980ff': 'rest27 repair-tail held targets await full corrected conversion',
    '5c7a693d1ea8cb12a36bdb6f8dbc767ab866d9345a787902c6201967f443f706': 'Big2 first100 held targets await bounded source recovery and corrected conversion',
}


def jread(path):
    return json.loads(Path(path).read_text())


def exact_ref(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'sha256': sha256_file_hex_cached(path)}


def copy_evidence(source, evidence_dir, name):
    source = Path(source).resolve(strict=True)
    before = source.stat()
    digest = sha256_file_hex_cached(source)
    target = evidence_dir / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    after = source.stat()
    copied = sha256_file_hex_cached(target)
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or copied != digest):
        raise ValueError(f'provenance evidence changed while staging: {source}')
    return {'path': str(Path('evidence') / name), 'sha256': digest}


def _source_map(path):
    if not path:
        return {}
    doc = jread(path)
    if doc.get('schema') != 'natlang.staged-source-map/1' or not isinstance(doc.get('paths'), dict):
        raise ValueError(f'invalid source-map custody file: {path}')
    return dict(doc['paths'])


def _mapped(path, mapping):
    p = str(path)
    candidate = Path(mapping.get(p, p))
    if not candidate.is_file():
        raise FileNotFoundError(f'bound source is not locally available: {p} -> {candidate}')
    return candidate


def flatten(manifest, mapping, *, parents=(), logical_manifest=None):
    requested_manifest = Path(manifest)
    if logical_manifest is None:
        logical_manifest = str(requested_manifest)
    manifest = requested_manifest.resolve(strict=True)
    if manifest in parents:
        raise ValueError('source manifest cycle')
    doc = jread(manifest)
    if doc.get('schema') != 'natlang.native-graph-sources/1' or not isinstance(doc.get('sources'), list):
        raise ValueError(f'invalid native graph source manifest: {manifest}')
    output = []
    for source in doc['sources']:
        source_path = Path(source['path'])
        logical_path = (source_path if source_path.is_absolute()
                        else Path(logical_manifest).parent / source_path)
        logical_text = str(logical_path)
        path = (_mapped(source['path'], mapping) if source_path.is_absolute()
                else (manifest.parent / source_path).resolve(strict=True))
        if sha256_file_hex_cached(path) != source.get('sha256'):
            raise ValueError(f'source binding differs for {source["path"]}')
        if path.suffix == '.json':
            output.extend(flatten(path, mapping, parents=(*parents, manifest),
                                  logical_manifest=logical_text))
        else:
            # Preserve an original logical path only when the explicit staged
            # source map binds that exact name to these verified bytes. This
            # keeps cloud manifests portable while preventing relative paths
            # from being rebased under the new output directory.
            mapped = mapping.get(logical_text)
            output_path = (logical_text if mapped and Path(mapped).resolve() == path
                           else str(path))
            output.append({**source, 'path': output_path})
    return output


def _declared_leaf_sources(manifest, mapping, *, parents=()):
    """Walk only manifests named by the supplied source root, without loading graph rows."""
    manifest = Path(manifest).resolve(strict=True)
    if manifest in parents:
        raise ValueError('source manifest cycle')
    doc = jread(manifest)
    if doc.get('schema') != 'natlang.native-graph-sources/1' or not isinstance(doc.get('sources'), list):
        raise ValueError(f'invalid native graph source manifest: {manifest}')
    for source in doc['sources']:
        raw = Path(source['path'])
        if raw.is_absolute():
            candidate = Path(mapping.get(str(raw), str(raw)))
        else:
            candidate = manifest.parent / raw
        if candidate.suffix == '.json':
            # Existing staged maps must resolve nested manifests; no inferred
            # registry relocation is allowed for control documents.
            nested = candidate.resolve(strict=True)
            if sha256_file_hex_cached(nested) != source.get('sha256'):
                raise ValueError(f'nested source manifest binding differs: {source["path"]}')
            yield from _declared_leaf_sources(nested, mapping, parents=(*parents, manifest))
        else:
            yield source


def _relocate_declared_mellum_package_leaves(train_manifest, held_manifest, mapping, *, cfg,
                                             registry_root=None):
    """Resolve missing old cloud leaves only through their exact registry role binding."""
    registry = (Path(registry_root) if registry_root is not None else
        ROOT / 'natlang-internal-spool/teacher-qwen36-continuation-20261023/registered-packages')
    relocated = []
    pattern = re.compile(r'^/work/resume/(?:teacher-append-20261025|teacher-append-20261026-v13|training/teacher-append-20261026-v13)/artifacts/runs/'
        r'(?P<corpus>qwen36-verified-positive-native-graphs-20261023-internal-batch[0-9]{4}-tranche[0-9]{4}'
        r'(?:-[a-z0-9]+(?:-[a-z0-9]+)*)?)/'
        r'(?P<role>mellum21)/(?P<split>train|test)/native-graphs\.jsonl\.zst$')
    visited = set()
    for manifest in (train_manifest, held_manifest):
        for source in _declared_leaf_sources(manifest, mapping):
            original = source['path']
            candidate = Path(mapping.get(original, original)) if Path(original).is_absolute() else Path(original)
            if candidate.is_file() or original in visited:
                continue
            match = pattern.fullmatch(original)
            if match is None:
                continue
            corpus, role, split = match.group('corpus','role','split')
            entry_path = registry / f'{corpus}.entry.json'
            package_root = registry / corpus
            if not entry_path.is_file():
                raise FileNotFoundError(f'no local registered entry for declared Mellum source: {corpus}')
            entry = jread(entry_path)
            if entry.get('id') != corpus:
                raise ValueError('registered Mellum source entry ID differs from manifest path')
            selected = (entry.get('source_binding') or {}).get('models', {}).get(role)
            declared = selected.get(split) if isinstance(selected, dict) else None
            if (not isinstance(declared, dict) or selected.get('model_id') != cfg['model_id']
                    or selected.get('tokenizer_fingerprint') != cfg['fingerprint']):
                raise ValueError(f'{corpus} lacks matching registered Mellum role identity')
            resolved = Path(declared.get('path','')).resolve(strict=True)
            expected = {'path': str(package_root / role / split / 'native-graphs.jsonl.zst'),
                        'sha256': source.get('sha256'), 'bytes': declared.get('bytes')}
            if (resolved != (package_root / role / split / 'native-graphs.jsonl.zst').resolve(strict=True)
                    or declared.get('sha256') != expected['sha256']
                    or type(expected['bytes']) is not int or resolved.stat().st_size != expected['bytes']):
                raise ValueError(f'{corpus} registry member differs from exact declared Mellum source')
            _validate_registered_role(entry, role, split, resolved,
                model_id=cfg['model_id'], fingerprint=cfg['fingerprint'])
            mapping[original] = str(resolved)
            receipt = selected.get('receipt') or {}
            receipt_path = Path(receipt.get('path','')).resolve(strict=True)
            if (receipt_path.stat().st_size != receipt.get('bytes')
                    or sha256_file_hex_cached(receipt_path) != receipt.get('sha256')):
                raise ValueError(f'{corpus} conversion receipt differs from registry entry')
            relocated.append({'original_path': original, 'resolved_path': str(resolved),
                'source_sha256': source['sha256'], 'bytes': expected['bytes'],
                'registry_entry': str(entry_path.resolve()),
                'registry_entry_sha256': sha256_file_hex_cached(entry_path),
                'conversion_receipt': str(receipt_path),
                'conversion_receipt_sha256': receipt['sha256'],
                'model_id': cfg['model_id'], 'tokenizer_fingerprint': cfg['fingerprint'],
                'role': role, 'split': split})
            visited.add(original)
    return relocated


def add_source(path, *, kind, model=None, **extra):
    path = Path(path).resolve(strict=True)
    item = {'kind': kind, 'path': str(path), 'sha256': sha256_file_hex_cached(path), **extra}
    if model is not None:
        item['model'] = model
    return item


def _bundle_input(model, role):
    if model == 'lfm':
        if role == 'train':
            return (LUNA / 'append/lfm25-tranche000000-v3/native-training-sources.json',
                    LUNA / 'append/lfm25-tranche000000-v3/.natlang-source-map.json')
        return (ROOT / 'lfm-local-recovery-20261024/held-tree/work/native-production/control/lfm-next-cohort-capacity-20261016-v1/appended-held-sources.json',
                ROOT / 'lfm-local-recovery-20261024/held-tree/.natlang-source-map.json')
    if role == 'train':
        base = ROOT / 'mellum-cloud-continuation-20261025-r2/teacher-v13-control-backup'
        return (base / 'recovered-source/training/teacher-append-20261026-v13/native-training-sources.json',
                base / 'recovered-source-map.json')
    return (ROOT / 'mellum-cloud-resume-20261025/packet/held-recovered-available-sources.json',
            ROOT / 'mellum-cloud-resume-20261025/packet/.natlang-source-map.json')


def _merge_map(model, train_manifest, held_manifest, train_map_path, held_map_path, out, cfg):
    mapping = _source_map(train_map_path)
    mapping.update(_source_map(held_map_path))
    relocated = []
    if model == 'mellum':
        # The immutable cloud snapshot refers to /work/resume/inputs/payloads.
        # Route those exact content-addressed objects to the independently
        # staged local bytes; the original receipt paths remain unchanged.
        packet = ROOT / 'mellum-cloud-resume-20261025/packet'
        for original, target in list(mapping.items()):
            marker = '/work/resume/inputs/payloads/'
            if target.startswith(marker):
                sha = Path(target).name
                local = packet / 'payloads' / sha
                if local.is_file() and sha256_file_hex_cached(local) == sha:
                    mapping[original] = str(local)
        # Include the dedicated recovered teacher-v13 mappings.
        recovered = _source_map(ROOT / 'mellum-cloud-continuation-20261025-r2/teacher-v13-control-backup/recovered-source-map.json')
        mapping.update(recovered)
        relocated = _relocate_declared_mellum_package_leaves(
            train_manifest, held_manifest, mapping, cfg=cfg)
        # Keep the source receipt's future container identity stable while the
        # preparer validates bytes through its local staged path.
        mapping[str(MELLUM_REMOTE_ROOT / 'native-training-sources.json')] = str(out / 'native-training-sources.json')
        mapping[str(MELLUM_REMOTE_ROOT / 'held-native-graphs.json')] = str(out / 'held-native-graphs.json')
    doc = {'schema': 'natlang.staged-source-map/1', 'paths': mapping}
    target = out / '.natlang-source-map.json'
    target.write_text(json.dumps(doc, sort_keys=True, indent=2) + '\n')
    return target, mapping, relocated


def _stage_manifest_for_validation(source, output_path, mapping, map_path):
    """Copy a source manifest beside the combined custody map for exact leaf resolution."""
    source = Path(source).resolve(strict=True)
    output_path = Path(output_path)
    before = source.stat()
    digest = sha256_file_hex_cached(source)
    shutil.copyfile(source, output_path)
    after = source.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or sha256_file_hex_cached(output_path) != digest):
        raise ValueError(f'prior source manifest changed while staging: {source}')
    mapping[str(source)] = str(output_path.resolve())
    map_path.write_text(json.dumps({'schema': 'natlang.staged-source-map/1', 'paths': mapping},
                                   sort_keys=True, indent=2) + '\n')
    return output_path


def _replacement_sources(model_id):
    out = []
    for name in ('rest27-reviewed', 'big2-offload-1000', 'big2-offload-5000'):
        receipt = jread(REPAIRS / name / 'receipt.json')
        if receipt.get('schema') != 'natlang.native-semantic-e2e-build/2' or not receipt.get('whole_source_complete'):
            raise ValueError(f'repair receipt is not a complete whole-source result: {name}')
        binding = receipt['outputs']['pairs']['train']
        path = Path(binding['path']).resolve(strict=True)
        if path.stat().st_size != binding['bytes'] or sha256_file_hex_cached(path) != binding['sha256']:
            raise ValueError(f'corrected train source differs from its conversion receipt: {name}')
        out.append({'kind': 'semantic-pairs', 'model': model_id,
                    'path': str(path), 'sha256': binding['sha256']})
    return out


def _registered_sources(model, role):
    out = []
    cfg = _current_config(model)
    for tranche in ('000001', '000002'):
        corpus = f'luna-verified-positive-native-graphs-20261026-tranche{tranche}'
        package = PACKAGE / corpus
        private = ROOT / f'luna-generation-20261026/private-exports/{corpus}'
        ready = jread(private / 'ready.json')
        verified = jread(private / 'source-private-verified.json')
        entry_path = PACKAGE / f'{corpus}.entry.json'
        entry = jread(entry_path)
        if not ready.get('passed') or ready.get('training_exposure') is not False:
            raise ValueError(f'{corpus} lacks an explicit non-exposure private-ready receipt')
        if not verified.get('passed') or verified.get('corpus_id') != corpus:
            raise ValueError(f'{corpus} private source verification differs')
        if ready.get('entry_sha256') != canonical_json_sha256_hex(entry):
            raise ValueError(f'{corpus} registry entry differs from its ready receipt')
        if entry.get('id') != corpus:
            raise ValueError(f'{corpus} registry identity differs')
        private_receipt = jread(private / 'private-receipt.json')
        identity = ready.get('identity')
        if (not private_receipt.get('private')
                or private_receipt.get('identity') != identity
                or verified.get('identity') != identity
                or verified.get('archive') != ready.get('archive')
                or verified.get('corpus_id') != corpus):
            raise ValueError(f'{corpus} private archive evidence disagrees')
        archive=Path(ready['archive']).resolve(strict=True)
        if (archive.stat().st_size != identity.get('bytes')
                or sha256_file_hex_cached(archive) != identity.get('sha256')):
            raise ValueError(f'{corpus} actual private archive differs from its independent proof')
        package_role = 'lfm25' if model == 'lfm' else 'mellum21'
        graph = package / package_role / role / 'native-graphs.jsonl.zst'
        _validate_registered_role(entry, package_role, role, graph,
                                  model_id=cfg['model_id'], fingerprint=cfg['fingerprint'])
        out.append(add_source(graph, kind='native-graphs'))
    return out


def _validate_registered_role(entry, package_role, split, graph, *, model_id, fingerprint):
    """Bind the selected unpacked graph leaf to its exact admitted registry role."""
    models = (entry.get('source_binding') or {}).get('models') or {}
    selected = models.get(package_role)
    if not isinstance(selected, dict):
        raise ValueError(f'registry entry has no exact {package_role} source binding')
    declared = selected.get(split)
    if not isinstance(declared, dict):
        raise ValueError(f'registry entry has no exact {package_role}/{split} graph binding')
    graph = Path(graph).resolve(strict=True)
    declared_path = Path(declared.get('path', '')).resolve(strict=True)
    if (declared_path != graph or declared.get('bytes') != graph.stat().st_size
            or declared.get('sha256') != sha256_file_hex_cached(graph)
            or selected.get('model_id') != model_id
            or selected.get('tokenizer_fingerprint') != fingerprint):
        raise ValueError(f'unpacked {package_role}/{split} graph differs from its registered model binding')


def _current_config(model):
    role = 'lfm25' if model == 'lfm' else 'mellum21'
    model_id = ('lfm2:f40d7a5bf651a03618ae8c06bda24a180e22fa79de964e55862c2c4984249da6'
                if model == 'lfm' else
                'mellum:bc700045baee1c7ea1869675c38efedd59d9db09975ea0b38cff8db152d6148f')
    fingerprints = set()
    role_id_sets = set()
    for name in ('rest27-reviewed', 'big2-offload-1000', 'big2-offload-5000'):
        token = jread(REPAIRS / name / 'receipt.json')['tokenizers'][role]
        fingerprints.add(token['expanded'] if isinstance(token, dict) else token)
        if isinstance(token, dict):
            role_id_sets.add(tuple(token['role_ids']))
    if len(fingerprints) != 1 or len(role_id_sets) != 1:
        raise ValueError(f'{model} corrected corpus receipts disagree on tokenizer/role identity')
    return {'model_id': model_id, 'fingerprint': fingerprints.pop(), 'roles': role,
            'role_ids': list(role_id_sets.pop())}


def _role_ids(cfg):
    ids=tuple(cfg['role_ids'])
    if len(ids)!=4 or any(type(i)is not int for i in ids) or len(set(ids))!=4:
        raise ValueError(f'{cfg["roles"]} receipts lack exact native role tokens')
    return ids


def _flatten_train_inputs(model, source_manifest, mapping, model_id, out_dir):
    sources = flatten(source_manifest, mapping)
    discarded = []
    kept = []
    remove = set(BAD_TRAIN) | CORRECTED_OLD
    for src in sources:
        sha = src['sha256']
        if sha in remove:
            discarded.append(src)
        else:
            kept.append(src)
    if not set(CORRECTED_OLD).issubset({s['sha256'] for s in discarded}):
        raise ValueError(f'{model} source manifest omitted one exact fully corrected train leaf')
    if not set(BAD_TRAIN).issubset({s['sha256'] for s in discarded}):
        raise ValueError(f'{model} source manifest did not contain the exact reviewed faulty train leaves')
    current = list(kept)
    current += _replacement_sources(model_id)
    current += _registered_sources(model, 'train')
    manifest = {'schema': 'natlang.native-graph-sources/1', 'sources': current}
    path = out_dir / 'native-training-sources.json'
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n')
    return path, discarded


def _flatten_held(model, source_manifest, mapping, model_id, out_dir):
    sources = flatten(source_manifest, mapping)
    changed = []
    current = []
    for src in sources:
        replacement = HELD_REPLACE.get(src['sha256'])
        if replacement is None:
            current.append(src)
            continue
        receipt = jread(REPAIRS / replacement / 'receipt.json')
        binding = receipt['outputs']['pairs']['test']
        path = Path(binding['path']).resolve(strict=True)
        if path.stat().st_size != binding['bytes'] or sha256_file_hex_cached(path) != binding['sha256']:
            raise ValueError(f'{model} held replacement differs from receipt: {replacement}')
        new = {'kind': 'semantic-pairs', 'model': model_id,
               'path': str(path), 'sha256': binding['sha256'],
               'repair_receipt': str((REPAIRS / replacement / 'receipt.json').resolve()),
               'repair_receipt_sha256': sha256_file_hex_cached(REPAIRS / replacement / 'receipt.json')}
        current.append(new)
        changed.append({'old': src, 'new': new})
    if {x['old']['sha256'] for x in changed} != (set(HELD_REPLACE) if model == 'lfm' else set(HELD_REPLACE)):
        raise ValueError(f'{model} held source omitted one exact corrected Big2 test leaf')
    path = out_dir / 'held-native-graphs.json'
    path.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': current},
                               sort_keys=True, indent=2) + '\n')
    return path, changed, sources


def _origins(receipt, order):
    result, offset = {}, 0
    def visit(item):
        nonlocal offset
        if item.get('sources'):
            for child in item['sources']:
                visit(child)
            return
        count, digest = item.get('rows'), item.get('sha256')
        ids = order[offset:offset + count]
        if type(count) is not int or len(ids) != count or not isinstance(digest, str):
            raise ValueError('source receipt leaves do not exactly cover saved row order')
        for row_id in ids:
            if row_id in result:
                raise ValueError('source receipt repeats row ids')
            result[row_id] = digest
        offset += count
    visit(receipt)
    if offset != len(order):
        raise ValueError('source receipt row count differs from saved order')
    return result


def write_train_correction_rows(row_path, *, model, old_train, current_train, old_origins, new_origins, removed_ids, source_reasons, old_communication_ids):
    old_ids = set(old_train.bindings)
    current_ids = set(current_train.bindings)
    with row_path.open('w') as f:
        for row_id in old_train.ordered_ids():
            if row_id in removed_ids:
                digest = old_origins[row_id]
                if row_id in current_ids:
                    if old_train.bindings[row_id] == current_train.bindings[row_id]:
                        raise ValueError(f'{model}: corrected replacement left a faulty row binding unchanged: {row_id}')
                    f.write(json.dumps({'id': row_id, 'old_binding': old_train.bindings[row_id],
                        'old_source_sha256': digest, 'new_binding': current_train.bindings[row_id],
                        'new_source_sha256': new_origins[row_id], 'disposition': 'corrected',
                        'old_communication': row_id in old_communication_ids,
                        'reason': source_reasons[digest]}, sort_keys=True) + '\n')
                else:
                    if digest not in BAD_TRAIN:
                        raise ValueError(f'{model}: corrected replacement changed graph identity: {row_id}')
                    f.write(json.dumps({'id': row_id, 'old_binding': old_train.bindings[row_id],
                        'old_source_sha256': digest, 'new_binding': None,
                        'disposition': 'deferred', 'old_communication': row_id in old_communication_ids,
                        'reason': source_reasons[digest]}, sort_keys=True) + '\n')
        for row_id in current_train.ordered_ids():
            if row_id not in old_ids:
                f.write(json.dumps({'id': row_id, 'old_binding': None, 'old_source_sha256': None,
                    'new_binding': current_train.bindings[row_id], 'new_source_sha256': new_origins[row_id],
                    'disposition': 'added'}, sort_keys=True) + '\n')
            elif row_id not in removed_ids and current_train.bindings[row_id] != old_train.bindings[row_id]:
                if row_id not in old_train.bindings:
                    raise ValueError('replacement row identity is not in old source inventory')
                f.write(json.dumps({'id': row_id, 'old_binding': old_train.bindings[row_id],
                    'old_source_sha256': old_origins[row_id], 'new_binding': current_train.bindings[row_id],
                    'new_source_sha256': new_origins[row_id], 'disposition': 'corrected',
                    'old_communication': row_id in old_communication_ids}, sort_keys=True) + '\n')


def build(model, out_root):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.native_source_correction import SCHEMA
    train_old, train_map = _bundle_input(model, 'train')
    held_old, held_map = _bundle_input(model, 'held')
    out = out_root / model
    out.mkdir(parents=True, exist_ok=False)
    cfg = _current_config(model)
    map_path, mapping, relocated_sources = _merge_map(
        model, train_old, held_old, train_map, held_map, out, cfg)
    # The original Mellum held manifest lives under the recovery packet, whose
    # historical source map names remote /work/resume payload aliases. Stage the
    # exact manifest next to the merged map so validation uses the already
    # authenticated local payload copies without changing that global map.
    old_train_validation = _stage_manifest_for_validation(
        train_old, out / 'prior-native-training-sources.json', mapping, map_path)
    old_held_validation = _stage_manifest_for_validation(
        held_old, out / 'prior-held-native-graphs.json', mapping, map_path)
    train_path, removed = _flatten_train_inputs(model, train_old, mapping, cfg['model_id'], out)
    held_path, held_changed, old_held_leaves = _flatten_held(model, held_old, mapping, cfg['model_id'], out)
    roles = _role_ids(cfg)
    old_train = NativeGraphSource(old_train_validation, cfg['fingerprint'], required_split='train', role_ids=roles)
    current_train = NativeGraphSource(train_path, cfg['fingerprint'], required_split='train', role_ids=roles)
    old_held = NativeGraphSource(old_held_validation, cfg['fingerprint'], required_split='test', role_ids=roles)
    current_held = NativeGraphSource(held_path, cfg['fingerprint'], required_split='test', role_ids=roles)
    if old_held.ordered_ids() != current_held.ordered_ids():
        raise ValueError(f'{model}: protected held correction changed exact ID/order membership')
    if old_held.groups != current_held.groups:
        raise ValueError(f'{model}: protected held correction changed source groups')
    if current_train.groups & current_held.groups:
        raise ValueError(f'{model}: corrected training source overlaps protected held groups')

    old_receipt, current_receipt = old_train.receipt(), current_train.receipt()
    old_held_receipt, current_held_receipt = old_held.receipt(), current_held.receipt()
    old_origins = _origins(old_receipt, old_train.ordered_ids())
    new_origins = _origins(current_receipt, current_train.ordered_ids())
    old_communication_ids = {entry['id'] for entry in old_train.entries if entry.get('communication')}
    current_ids = set(current_train.bindings)
    old_ids = set(old_train.bindings)
    deferred = []
    for source in removed:
        # Source receipt order gives the exact IDs belonging to a removed leaf.
        pass
    # Derive removed-leaf row ownership by a fully parsed source with exact
    # original order, then compare with retained/current source IDs.
    removed_sha = {item['sha256'] for item in removed}
    removed_ids = {row_id for row_id, digest in old_origins.items() if digest in removed_sha}
    if not removed_ids:
        raise ValueError(f'{model}: no exact source-correction deferral rows were found')
    corrected_ids=removed_ids & current_ids
    deferred_ids=removed_ids - current_ids
    source_reasons = {s['sha256']: BAD_TRAIN[s['sha256']] for s in removed if s['sha256'] in BAD_TRAIN}
    source_reasons.update({s['sha256']: 'replaced by exact policy-v4 full producer target rebuild; corrected rows start fresh exposure'
                           for s in removed if s['sha256'] in CORRECTED_OLD})
    row_path = out / 'train-row-bindings.jsonl'
    write_train_correction_rows(row_path, model=model, old_train=old_train, current_train=current_train,
        old_origins=old_origins, new_origins=new_origins, removed_ids=removed_ids,
        source_reasons=source_reasons, old_communication_ids=old_communication_ids)
    # This subset is intentionally correction-only; it does not authorize
    # promotion of unresolved held targets or full readiness.
    old_held_origins = _origins(old_held_receipt, old_held.ordered_ids())
    new_held_origins = _origins(current_held_receipt, current_held.ordered_ids())
    pending_held=[]
    for source in old_held_leaves:
        if source['sha256'] in BAD_HELD:
            pending_held.append({'source':source,'rows':sum(value==source['sha256'] for value in old_held_origins.values()),
                                 'reason':BAD_HELD[source['sha256']]})
    old_groups_by_id = {e['id']: e['groups'] for e in old_held.entries}
    new_groups_by_id = {e['id']: e['groups'] for e in current_held.entries}
    held_changed_ids = [row_id for row_id in old_held.ordered_ids()
                        if old_held.bindings[row_id] != current_held.bindings[row_id]]
    held_row_path = out / 'protected-held-row-bindings.jsonl'
    with held_row_path.open('w') as f:
        for row_id in held_changed_ids:
            if old_groups_by_id[row_id] != new_groups_by_id[row_id]:
                raise ValueError('held correction changed a protected source group')
            f.write(json.dumps({'id': row_id, 'old_binding': old_held.bindings[row_id],
                'new_binding': current_held.bindings[row_id], 'old_source_sha256': old_held_origins[row_id],
                'new_source_sha256': new_held_origins[row_id], 'old_groups': old_groups_by_id[row_id],
                'new_groups': new_groups_by_id[row_id], 'disposition': 'corrected'}, sort_keys=True) + '\n')

    held_quality_rows = [row_id for row_id in old_held.ordered_ids()
        if old_held_origins[row_id] in BAD_HELD]
    held_quality_path = out / 'held-quality-hold-row-bindings.jsonl'
    with held_quality_path.open('w') as f:
        for row_id in held_quality_rows:
            source_sha = old_held_origins[row_id]
            if current_held.bindings[row_id] != old_held.bindings[row_id]:
                raise ValueError('pending held quality row unexpectedly changed target binding')
            f.write(json.dumps({'id':row_id, 'binding':current_held.bindings[row_id],
                'source_sha256':source_sha, 'reason':BAD_HELD[source_sha],
                'groups':new_groups_by_id[row_id]}, sort_keys=True) + '\n')
    if len(held_quality_rows) != sum(item['rows'] for item in pending_held):
        raise ValueError('held quality hold rows do not exactly cover pending faulty held sources')
    held_quality_manifest = out / 'held-quality-hold.json'
    held_quality_manifest.write_text(json.dumps({
        'schema':'natlang.native-held-quality-hold/1',
        'held_source':current_held_receipt,
        'row_bindings_path':held_quality_path.name,
        'row_bindings_sha256':sha256_file_hex(held_quality_path),
        'excluded_rows':len(held_quality_rows),
        'pending_sources':[{'sha256':item['source']['sha256'],'rows':item['rows'],
                            'reason':item['reason']} for item in pending_held],
        'purpose':'exclude known truncated targets from held loss/model selection while preserving protected membership; full-held qualification remains incomplete'},
        sort_keys=True, indent=2)+'\n')

    proof_path = out / 'source-correction.json'
    evidence_dir = out / 'evidence'
    evidence_dir.mkdir()
    evidence=[]
    for name in ('rest27-reviewed', 'big2-offload-1000', 'big2-offload-5000'):
        evidence.append(copy_evidence(REPAIRS / name / 'receipt.json', evidence_dir,
                                      f'repair-{name}-receipt.json'))
    for tranche in ('000001', '000002'):
        corpus=f'luna-verified-positive-native-graphs-20261026-tranche{tranche}'
        private=ROOT / f'luna-generation-20261026/private-exports/{corpus}'
        evidence.extend(copy_evidence(private / name, evidence_dir,
            f'{corpus}-{name}') for name in ('ready.json','private-receipt.json','source-private-verified.json'))
        evidence.append(copy_evidence(PACKAGE / f'{corpus}.entry.json', evidence_dir,
                                      f'{corpus}-entry.json'))
    evidence.extend((copy_evidence(map_path, evidence_dir, 'current-staged-source-map.json'),
        copy_evidence(train_old, evidence_dir, 'predecessor-training-manifest.json'),
        copy_evidence(held_old, evidence_dir, 'predecessor-held-manifest.json'),
        copy_evidence(train_map, evidence_dir, 'predecessor-training-source-map.json'),
        copy_evidence(held_map, evidence_dir, 'predecessor-held-source-map.json')))
    evidence.extend((copy_evidence(held_quality_path, evidence_dir, 'held-quality-hold-row-bindings.jsonl'),
                     copy_evidence(held_quality_manifest, evidence_dir, 'held-quality-hold.json')))
    for item in relocated_sources:
        entry_path = Path(item['registry_entry'])
        receipt_path = Path(item['conversion_receipt'])
        evidence.append(copy_evidence(entry_path, evidence_dir,
            'registered-' + entry_path.name))
        evidence.append(copy_evidence(receipt_path, evidence_dir,
            'conversion-' + receipt_path.parent.parent.name + '-' + receipt_path.name))
    proof = {'schema': SCHEMA, 'model_id': cfg['model_id'], 'stage_id': 'native_full_sft',
        'prior_source': old_receipt, 'current_source': current_receipt,
        'row_bindings_path': row_path.name, 'row_bindings_sha256': sha256_file_hex(row_path),
        'protected_held_correction': {'schema': 'natlang.native-protected-held-correction/1',
            'prior_source': old_held_receipt, 'current_source': current_held_receipt,
            'row_bindings_path': held_row_path.name,
            'row_bindings_sha256': sha256_file_hex(held_row_path)},
        'held_quality_hold': {'path':held_quality_manifest.name,
            'sha256':sha256_file_hex(held_quality_manifest)},
        'source_map': {'path': map_path.name, 'sha256': sha256_file_hex(map_path)},
        'registered_source_relocations': relocated_sources,
        'evidence': evidence,
        'pending_held_correction': {'remaining_original_sources': pending_held,
            'qualification_claim': False}}
    proof_path.write_text(json.dumps(proof, sort_keys=True, indent=2) + '\n')
    output = {'schema': 'natlang.prepared-source-correction-continuation/1', 'model': model,
        'model_id': cfg['model_id'], 'tokenizer_fingerprint': cfg['fingerprint'],
        'native_role_token_ids': cfg['role_ids'],
        'prior_train': exact_ref(train_old), 'current_train': exact_ref(train_path),
        'prior_held': exact_ref(held_old), 'current_held': exact_ref(held_path),
        'source_correction': exact_ref(proof_path), 'train_rows_before': len(old_train),
        'train_rows_after': len(current_train), 'deferred_train_rows': len(deferred_ids),
        'corrected_train_rows': len(corrected_ids),
        'added_train_rows': len(current_ids - old_ids), 'held_rows': len(old_held),
        'held_corrected_rows': len(held_changed_ids), 'held_membership_order_unchanged': True,
        'pending_held_correction_rows': sum(item['rows'] for item in pending_held),
        'pending_held_correction_sources': pending_held,
        'train_deferred_sha256s': sorted({sha for sha in removed_sha if sha in BAD_TRAIN}),
        'train_replaced_sha256s': sorted(CORRECTED_OLD),
        'train_communication_before': sum(bool(e.get('communication')) for e in old_train.entries),
        'train_communication_after': sum(bool(e.get('communication')) for e in current_train.entries),
        'held_communication': sum(bool(e.get('communication')) for e in old_held.entries),
        'new_train_manifest_rows': len(current_train), 'qualification_claim': False,
        'active_config_changed': False, 'optimizer_rng_exposure_credit': 0,
        'tranches': ['000001', '000002'], 'unpublished_tranche003_added': False,
        'target_runtime_paths': ({'train': str(MELLUM_REMOTE_ROOT / 'native-training-sources.json'),
            'held': str(MELLUM_REMOTE_ROOT / 'held-native-graphs.json'),
            'source_correction': str(MELLUM_REMOTE_ROOT / 'source-correction.json')} if model == 'mellum' else
            {'train': str(train_path.resolve()), 'held': str(held_path.resolve()),
             'source_correction': str(proof_path.resolve())}),
        'adoption_requirements': {
            'stage_id': 'native_full_sft',
            'continue_from': 'latest_actual_full_state_child_at_drained_boundary',
            'continued_native_graphs': old_train.source_identity_path.as_posix(),
            'continued_held_native_graphs': old_held.source_identity_path.as_posix(),
            'native_graphs': current_train.source_identity_path.as_posix(),
            'held_native_graphs': current_held.source_identity_path.as_posix(),
            'source_correction': (str(MELLUM_REMOTE_ROOT / 'source-correction.json') if model == 'mellum'
                                  else str(proof_path.resolve())),
            'preserve_weights_optimizer_rng_global_step': True,
            'new_optimizer_rng_exposure_credit_from_correction': 0,
            'corrected_train_rows_require_fresh_exposure': True,
            'protected_held_rows_train_credit': 0,
            'fresh_current_weight_qualification': 'required_after_corrected_source_exposure',
            'active_configuration_changed': False,
            'requires_exact_config_and_cursor_cas_at_activation': True,
        },
        'artifacts': {p.name: {'path': str(p.resolve()), 'bytes': p.stat().st_size,
                               'sha256': sha256_file_hex(p)} for p in
                      (train_path, held_path, row_path, held_row_path, held_quality_path,
                       held_quality_manifest, proof_path, map_path)}}
    (out / 'prepared.json').write_text(json.dumps(output, sort_keys=True, indent=2) + '\n')
    print(json.dumps(output, sort_keys=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', choices=('lfm','mellum','both'), default='both')
    args=parser.parse_args()
    models = ('lfm','mellum') if args.model=='both' else (args.model,)
    if args.output.exists() and not args.output.is_dir():
        raise SystemExit(f'output path is not a directory: {args.output}')
    if args.output.exists() and any((args.output / model).exists() for model in models):
        raise SystemExit(f'model output already exists; preserve evidence: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    for model in models:
        build(model,args.output)


if __name__=='__main__': main()
