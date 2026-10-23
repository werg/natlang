"""Convert a source-verified internal Qwen36 tranche with complete native dedup inputs."""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.hashing import sha256_file_hex, canonical_json_sha256_hex, sha256_reference
from natlang_neuralese.common.paths import resolve
from scripts.coreweave.teacher_tranche_admission import (
    LEGACY_TEACHER_NAMESPACE, Tranche, admitted_predecessors, file_binding,
    preceding_tranches, read_bound,
)
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import (
    _content_index_policy_sha256, tokenizer_fingerprint, provision,
)


FLINT_NAMESPACE = 'flint-verified-positive-native-graphs-20261021-recovery7853'
LFM_HISTORY_KEEP = 'qwen-verified-positive-native-graphs-20261015-v5-lfm-history-keep-v1-20261020'
MODEL_CONFIG_READER_SHA256 = 'ff2031ded1225ded550b8ff97a90145a401d798477601120e5cbaf0c88c13a88'


def _tokenizers(manifest_path: Path, root: Path) -> dict[str, Path]:
    reference = file_binding(manifest_path)
    manifest = read_bound(reference)
    if manifest.get('schema') != 'natlang.teacher-tokenizer-internal-relocation/1' or manifest.get('training_admission') is not False:
        raise ValueError('internal tokenizer relocation is not an exact non-admission transfer declaration')
    result = {}
    expected_contracts = {
        'lfm25': {'source_fingerprint': 'e52de346c5a9d6fe666b5a412be077525daee0ee40f81a91c36728cfb4cbf008',
                  'expanded_fingerprint': 'be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105',
                  'sandbox': '393f9b75-2666-456c-a850-1b46a8b7d147'},
        'mellum21': {'source_fingerprint': '1abe5e6333be0157f76f5484b389e5586b0e6dbb9bc776f54ca9fd62fce6cb56',
                     'expanded_fingerprint': 'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f',
                     'sandbox': 'ee341c7b-a425-4b4b-ad7e-f3cf7cbc2369'},
    }
    rows = manifest.get('models')
    if not isinstance(rows, list) or {row.get('model') for row in rows} != set(expected_contracts):
        raise ValueError('tokenizer relocation does not declare both exact model views')
    for row in rows:
        model = row['model']
        expected = expected_contracts[model]
        if (row.get('source_sandbox_id') != expected['sandbox']
                or row.get('tokenizer_contract') != {key: expected[key] for key in ('source_fingerprint', 'expanded_fingerprint')}):
            raise ValueError('tokenizer relocation changes its original model identity/fingerprints')
        directory = row.get('source_directory')
        if directory != '/work/native-evaluation/models/' + model:
            raise ValueError('tokenizer relocation source directory differs from its exact evaluation artifact root')
        local = Path(row['files'][0]['path']).parent
        for item in row['files']:
            path = Path(item['path'])
            actual = file_binding(path)
            if (actual['bytes'] != item['bytes'] or actual['sha256'] != item['sha256']
                    or not item['source_path'].startswith(str(directory) + '/')
                    or item['source_sandbox_id'] != expected['sandbox']):
                raise ValueError('tokenizer relocation member differs from its authenticated source')
        result[model] = local
    # Recreate the exact source and expanded fingerprints from the files, with no model weights loaded.
    from transformers import AutoTokenizer
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import tokenizer_fingerprint, provision
    for model, path in result.items():
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
        bind_history_reasoning(tokenizer, policy=None, require_declared=True, allow_transition=False)
        before = tokenizer_fingerprint(tokenizer)
        if before != expected_contracts[model]['source_fingerprint']:
            raise ValueError(f'{model} source tokenizer fingerprint differs from its original receipt')
        provision(tokenizer)
        if tokenizer_fingerprint(tokenizer) != expected_contracts[model]['expanded_fingerprint']:
            raise ValueError(f'{model} expanded tokenizer fingerprint differs from its original receipt')
    if file_binding(manifest_path) != reference:
        raise ValueError('tokenizer relocation declaration changed during verification')
    return result


def _add_file(path: Path, expected_sha256: str | None = None) -> Path:
    ref = file_binding(path)
    if expected_sha256 is not None and ref['sha256'] != expected_sha256:
        raise ValueError(f'deduplication source differs from its bound receipt: {path}')
    return path


def _legacy_inputs(root: Path, model: str) -> list[Path]:
    prior = root / 'runs/qwen-verified-positive-native-graphs-20261015-v21' / model
    receipt_ref = file_binding(prior / 'receipt.json')
    receipt = read_bound(receipt_ref)
    result = []
    for entry in receipt['deduplication']['inputs']:
        path = Path(entry['path'])
        if not path.is_absolute():
            path = root / path
        result.append(_add_file(path, entry['sha256']))
    train = prior / 'train/native-graphs.jsonl.zst'
    result.append(_add_file(train, receipt['outputs']['train']['sha256']))
    if file_binding(prior / 'receipt.json') != receipt_ref:
        raise ValueError('legacy predecessor receipt changed during dedup preparation')
    return result


def _same_actor_boundary_inputs(root: Path, ctx: Tranche, source_proof: dict,
                               inputs: dict[str, list[Path]]) -> list[dict]:
    """Include exact admitted copies of this boundary even across namespace/path changes.

    A corpus ID is an admission label, not a content identity. An identical
    actor/source boundary may already be registered under the original namespace
    while this continuation names its internal-spool copy differently. Bind the
    source bytes and actor/runtime first, then let the converter's native graph
    content fingerprints decide which model rows are duplicates under the current
    tokenizer and rendering policy.
    """
    registry_path = root / 'training/neuralese_corpora.json'
    registry_ref = file_binding(registry_path)
    registry = read_bound(registry_ref)
    source_sha = (source_proof.get('source') or {}).get('sha256')
    actor = source_proof.get('actor') or {}
    identity = source_proof.get('execution_identity') or {}
    if not all(isinstance(x, str) and len(x) == 64 for x in
               (source_sha, actor.get('sha256'), identity.get('sha256'))):
        raise ValueError('current source proof lacks exact source, actor, or runtime identity bindings')
    manifests_path = root / 'training/corpus-manifests'
    internal_spool = resolve('data_nvme','natlang-internal-spool','teacher-qwen36-continuation-20261023')
    matched = []
    already = {str(path.resolve()) for rows in inputs.values() for path in rows}
    for entry in registry.get('corpora', []):
        if entry.get('training_admission') is not True:
            continue
        binding = entry.get('source_binding') or {}
        raw = binding.get('raw_source') or binding.get('source') or {}
        declared_actor = sha256_reference(binding.get('actor'))
        declared_identity = sha256_reference(binding.get('execution_identity'))
        corpus_root = root / entry['path']
        if raw.get('sha256') != source_sha:
            # Internal-spool registrations use a compact package proof rather
            # than the older expanded source_binding shape. Resolve that proof
            # through the immutable registry manifest and source verification.
            proof_ref = binding.get('registration_proof') or {}
            if not proof_ref:
                continue
            proof_path = root / proof_ref.get('path', '')
            if not proof_path.is_file():
                # The private registry publisher may synchronize only the
                # registry/manifest into this checkout while the external run
                # link lives in another linked worktree. The registered entry
                # binds an internal package ID; resolve that exact immutable
                # package under the selected internal spool as a relocation.
                package_id = entry.get('id', '')
                proof_path = internal_spool / 'registered-packages' / package_id / 'evidence/registration-proof.json'
            try:
                proof_actual = file_binding(proof_path)
            except OSError:
                continue
            if (proof_actual.get('sha256') != proof_ref.get('sha256')
                    or ('bytes' in proof_ref and proof_actual.get('bytes') != proof_ref.get('bytes'))):
                raise ValueError(f'registered internal proof differs from entry: {entry["id"]}')
            internal_proof = read_bound(proof_actual)
            if (internal_proof.get('schema') != 'natlang.teacher-internal-tranche-registration-proof/1'
                    or internal_proof.get('status') != 'passed'
                    or internal_proof.get('training_admission') is not True
                    or internal_proof.get('corpus_id') != entry.get('id')):
                raise ValueError(f'internal registration proof identity differs from registry entry: {entry["id"]}')
            if Path(proof_actual['path']).is_relative_to(internal_spool):
                corpus_root = proof_path.parents[1]
            source_ref = internal_proof.get('source_verification') or {}
            source_verification = read_bound(source_ref)
            raw_sha = (source_verification.get('source') or {}).get('sha256')
            declared_actor = sha256_reference(source_verification.get('actor'))
            declared_identity = sha256_reference(source_verification.get('execution_identity'))
            if raw_sha != source_sha:
                continue
        # Preserve exact teacher provenance where the entry declares it. Older
        # manifests without these fields remain handled by the regular admitted
        # predecessor path; they cannot claim an exact same-actor continuation.
        if (declared_actor != actor['sha256'] or declared_identity != identity['sha256']):
            continue
        corpus_id = entry['id']
        manifest_ref = file_binding(manifests_path / f'{corpus_id}.json')
        manifest = read_bound(manifest_ref)
        if manifest.get('id') != corpus_id or manifest.get('path') != entry.get('path'):
            raise ValueError(f'exact actor/source boundary manifest differs from registry: {corpus_id}')
        resolved_views = {}
        policy_compatible = True
        for model in ('lfm25', 'mellum21'):
            rel = f'{model}/train/native-graphs.jsonl.zst'
            members = [item for item in manifest.get('files', []) if item.get('path') == rel]
            if len(members) != 1:
                raise ValueError(f'exact actor/source boundary lacks one {model} train graph: {corpus_id}')
            graph_path = corpus_root / rel
            try:
                graph_ref = file_binding(graph_path)
            except OSError as error:
                if error.errno not in (errno.ENOENT, errno.EIO):
                    raise
                raise ValueError(f'exact actor/source graph is unavailable without a bound content cache: {corpus_id}/{model}') from error
            if any(graph_ref.get(key) != members[0].get(key) for key in ('sha256', 'bytes')):
                raise ValueError(f'exact actor/source graph differs from immutable manifest: {corpus_id}/{model}')
            receipt_ref = file_binding(graph_path.parents[1] / 'receipt.json')
            receipt = read_bound(receipt_ref)
            if (receipt.get('status') != 'passed' or receipt.get('model_key') != model
                    or receipt.get('source', {}).get('sha256') != source_sha):
                raise ValueError(f'exact actor/source graph receipt differs from current boundary: {corpus_id}/{model}')
            expected_explicit = 'keep' if model == 'lfm25' else None
            if (receipt.get('explicit_history_reasoning_policy') != expected_explicit
                    or receipt.get('failed_assistant_history_policy') != 'mask-runtime-failures'):
                # A registered unexposed package with a declared policy defect
                # is correction provenance, not valid coverage. It must not
                # hide rows in the corrected complete package.
                policy_compatible = False
            resolved_views[model] = (graph_path, graph_ref, receipt_ref)
        if not policy_compatible:
            continue
        for model, (graph_path, graph_ref, receipt_ref) in resolved_views.items():
            if str(graph_path.resolve()) not in already:
                inputs[model].append(graph_path)
                already.add(str(graph_path.resolve()))
            matched.append({'corpus_id': corpus_id, 'model': model,
                            'manifest': manifest_ref, 'receipt': receipt_ref,
                            'graph': graph_ref})
    if file_binding(registry_path) != registry_ref:
        raise ValueError('registry changed while resolving exact actor/source boundary inputs')
    return matched


def _registered_inputs(root: Path, spool: Path, ctx: Tranche, relocation_manifest: Path | None,
                       tokenizers: dict[str, Path], source_proof: dict):
    contracts = {}
    for model, path in tokenizers.items():
        from transformers import AutoTokenizer
        from natlang_neuralese.serve.chat import bind_history_reasoning
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
        bind_history_reasoning(tokenizer, policy=None, require_declared=True, allow_transition=False)
        provision(tokenizer)
        contracts[model] = {'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer),
                            'policy_sha256': _content_index_policy_sha256()}
    admitted, rejected = preceding_tranches(root, spool, ctx,
        relocation_manifest=relocation_manifest, cache_contracts=contracts)
    namespaces = tuple(dict.fromkeys((LEGACY_TEACHER_NAMESPACE, FLINT_NAMESPACE, ctx.teacher_namespace)))
    additional = admitted_predecessors(root, ctx.corpus_id, namespaces=namespaces,
                                       relocation_manifest=relocation_manifest,
                                       cache_contracts=contracts)
    by_id = {row['id']: row for row in [*admitted, *additional]}
    inputs = {model: [] for model in ('lfm25', 'mellum21')}
    cached_ids = {model: [] for model in ('lfm25', 'mellum21')}
    for row in by_id.values():
        for model in inputs:
            reference = row['outputs'][model]
            if reference.get('cache_only_source') is True:
                cached_ids[model].append(row['id'])
            else:
                inputs[model].append(_add_file(Path(reference['path']), reference['sha256']))
    same_actor_boundary = _same_actor_boundary_inputs(root, ctx, source_proof, inputs)
    return inputs, cached_ids, list(by_id), rejected, same_actor_boundary


def history_policy_args(model: str) -> list[str]:
    """Return the established model history mode and independent failure mask."""
    if model not in ('lfm25', 'mellum21'):
        raise ValueError(f'unsupported native student view: {model}')
    args = ['--failed-assistant-history-policy', 'mask-runtime-failures']
    if model == 'lfm25':
        args.extend(['--history-reasoning-policy', 'keep'])
    return args


def _registered_model_identity(root: Path, model: str, direct_inputs: list[Path], output: Path) -> Path:
    registry_path = root / 'training/neuralese_corpora.json'
    registry_ref = file_binding(registry_path)
    registry = read_bound(registry_ref)
    sources, model_ids = [], set()
    for graph_path in direct_inputs:
        ident = graph_path.parents[2].name
        matches = [entry for entry in registry['corpora'] if entry.get('id') == ident]
        if len(matches) != 1 or matches[0].get('training_admission') is not True:
            raise ValueError('model identity receipt is not under one admitted registry entry')
        entry = matches[0]
        manifest_path = root / 'training/corpus-manifests' / f'{ident}.json'
        manifest_ref = file_binding(manifest_path)
        manifest = read_bound(manifest_ref)
        if manifest.get('id') != ident or manifest.get('path') != entry.get('path'):
            raise ValueError('model identity receipt manifest differs from admitted registry')
        receipt_path = graph_path.parents[1] / 'receipt.json'
        receipt_ref = file_binding(receipt_path)
        receipt_rel = f'{model}/receipt.json'
        members = [item for item in manifest.get('files', []) if item.get('path') == receipt_rel]
        graph_rel = f'{model}/train/native-graphs.jsonl.zst'
        graph_members = [item for item in manifest.get('files', []) if item.get('path') == graph_rel]
        if (len(members) != 1 or len(graph_members) != 1
                or any(members[0].get(key) != receipt_ref.get(key) for key in ('sha256', 'bytes'))
                or any(graph_members[0].get(key) != file_binding(graph_path).get(key) for key in ('sha256', 'bytes'))):
            raise ValueError('model identity source is not the exact registered receipt and graph pair')
        receipt = read_bound(receipt_ref)
        if receipt.get('status') != 'passed' or receipt.get('model_key') != model:
            raise ValueError('registered model identity receipt is not a passed matching model view')
        model_id = receipt.get('model_id')
        if not isinstance(model_id, str) or ':' not in model_id:
            raise ValueError('registered model receipt has no exact base model identity')
        model_ids.add(model_id)
        sources.append({'corpus_id': ident,
                        'registry_entry_sha256': canonical_json_sha256_hex(entry),
                        'manifest': manifest_ref, 'receipt': receipt_ref})
    if len(model_ids) != 1 or not sources:
        raise ValueError('available admitted predecessor receipts do not agree on one exact model identity')
    proof = {'schema': 'natlang.teacher-registered-model-identity/1', 'model_key': model,
        'model_id': next(iter(model_ids)), 'registry': registry_ref, 'sources': sources,
        'basis': 'all locally available admitted predecessor receipts agree; unavailable registered packages remain content-index-bound separately',
        'training_admission': False}
    path = output / f'model-identity-{model}.json'
    write_json_durable(path, proof)
    if file_binding(registry_path) != registry_ref:
        raise ValueError('registry changed during model identity declaration')
    return path


def _base_inventory_reference(model: str, model_dir: Path, output: Path,
                              tokenizer_manifest: Path, identity_path: Path) -> Path:
    metadata_path = model_dir / f'{model}-metadata-readback.json'
    config_path = model_dir / f'{model}-model-config.json'
    metadata_ref, config_ref = file_binding(metadata_path), file_binding(config_path)
    metadata, config = read_bound(metadata_ref), read_bound(config_ref)
    if metadata.get('model_config') != config:
        raise ValueError(f'{model} remotely read model config differs from its local metadata asset')
    inventory = config.get('base_inventory')
    if (not isinstance(inventory, dict) or not inventory
            or canonical_json_sha256_hex(inventory) != config.get('base_inventory_sha256')):
        raise ValueError(f'{model} model config base inventory digest is invalid')
    tokenizer_ref = file_binding(tokenizer_manifest)
    tokenizer_data = read_bound(tokenizer_ref)
    row = next((item for item in tokenizer_data['models'] if item.get('model') == model), None)
    if row is None or row.get('tokenizer_contract', {}).get('expanded_fingerprint') != config.get('tokenizer_fingerprint'):
        raise ValueError(f'{model} remote inventory tokenizer fingerprint differs from exact tokenizer relocation')
    token_dir = Path(row['files'][0]['path']).parent
    tokenizer_files = []
    for name, digest in inventory.items():
        if name.endswith('.safetensors') or name == 'model.safetensors.index.json':
            continue
        asset = token_dir / name
        binding = file_binding(asset)
        if binding['sha256'] != digest:
            raise ValueError(f'{model} local tokenizer/config asset differs from its remote base inventory: {name}')
        tokenizer_files.append({'name': name, 'path': str(asset), 'binding': binding})
    identity_ref = file_binding(identity_path)
    identity = read_bound(identity_ref)
    model_id = config['architecture'] + ':' + config['base_inventory_sha256']
    if identity.get('model_id') != model_id or identity.get('model_key') != model:
        raise ValueError(f'{model} remote base inventory differs from all available admitted receipt identities')
    if model == 'mellum21':
        actor_path = model_dir / 'mellum21-actor-view-manifest.json'
        actor_ref = file_binding(actor_path)
        actor = read_bound(actor_ref)
        if actor.get('model_config') != config or actor.get('source') != metadata.get('source_model'):
            raise ValueError('Mellum inventory readback differs from its exact actor-view manifest')
        source_evidence = {'kind': 'mellum-actor-view', 'actor_view_manifest': actor_ref,
                           'actor_checkpoint': actor['source']}
    else:
        verified_path = resolve('data_nvme','dgxops','coreweave-provider-artifacts',
            '7b79178c1d5275cc48c17f89d582f241e3912f4e84378c5f074085e09c219430-'
            '393f9b75-2666-456c-a850-1b46a8b7d147.verified.json')
        verified_ref = file_binding(verified_path)
        verified = read_bound(verified_ref)
        checkpoint = metadata['source']
        binding = verified.get('binding') or {}
        if (binding.get('path') != checkpoint.get('path') or binding.get('sha256') !=
                '7b79178c1d5275cc48c17f89d582f241e3912f4e84378c5f074085e09c219430'
                or binding.get('bytes') != checkpoint.get('bytes')
                or verified.get('target_id') != '393f9b75-2666-456c-a850-1b46a8b7d147'):
            raise ValueError('LFM metadata source differs from its independently verified exact checkpoint')
        source_evidence = {'kind': 'lfm-verified-checkpoint', 'checkpoint_verification': verified_ref,
                           'checkpoint_sha256': binding['sha256'], 'checkpoint_bytes': binding['bytes']}
    readback_declaration = file_binding(model_dir / 'model-config-readback-v2.json')
    proof = {'schema': 'natlang.teacher-base-inventory-reference/1', 'model_key': model,
        'model_id': model_id, 'architecture': config['architecture'],
        'base_inventory': inventory, 'base_inventory_sha256': config['base_inventory_sha256'],
        'model_config': config_ref, 'metadata_readback': metadata_ref,
        'metadata_reader': file_binding(Path(__file__).resolve().with_name('read_native_base_inventory.py')),
        'readback_declaration': readback_declaration,
        'source_evidence': source_evidence, 'registered_receipt_identity': identity_ref,
        'tokenizer_relocation': tokenizer_ref,
        'source_fingerprint': row['tokenizer_contract']['source_fingerprint'],
        'expanded_fingerprint': row['tokenizer_contract']['expanded_fingerprint'],
        'tokenizer_files': tokenizer_files,
        'weights_copied': False, 'training_admission': False}
    if proof['metadata_reader']['sha256'] != MODEL_CONFIG_READER_SHA256:
        raise ValueError('native model metadata reader source changed after exact readback')
    path = output / f'base-inventory-{model}.json'
    write_json_durable(path, proof)
    return path


def convert(*, root: Path, settings_path: Path, settings_sha256: str,
            tokenizer_manifest: Path, protection_manifest: Path,
            output: Path, batch: int, number: int,
            relocation_manifest: Path | None = None) -> dict:
    root = root.resolve(strict=True)
    settings_ref = file_binding(settings_path)
    if settings_ref['sha256'] != settings_sha256:
        raise ValueError('internal teacher source settings differ from the requested exact binding')
    settings = read_bound(settings_ref)
    if settings.get('schema') != 'natlang.selected-teacher-admission-settings/1':
        raise ValueError('internal source settings schema differs')
    continuation = settings.get('internal_continuation') or {}
    if continuation.get('schema') != 'natlang.teacher-internal-source-continuation/1' or continuation.get('training_admission') is not False:
        raise ValueError('internal conversion requires the explicitly bound internal source continuation')
    original = read_bound(continuation['original_settings'])
    if original.get('teacher_source') != settings.get('teacher_source') or original.get('actor_remote') != settings.get('actor_remote') or original.get('identity_remote') != settings.get('identity_remote'):
        raise ValueError('internal storage continuation changed selected teacher or actor identities')
    publisher = read_bound(continuation['publisher_config'])
    if publisher.get('directory_root') != settings['spool']:
        raise ValueError('internal spool differs from its publisher authority')
    ctx = Tranche(batch, number, teacher_namespace=settings['teacher_namespace'])
    spool = Path(settings['spool'])
    source_dir = ctx.admission(spool)
    source_proof_ref = file_binding(source_dir / 'canonical-source-verification.json')
    source_proof = read_bound(source_proof_ref)
    source = source_proof.get('source') or {}
    if (source_proof.get('schema') != 'natlang.teacher-live-tranche-canonical-source-verification/1'
            or source_proof.get('status') != 'passed' or source_proof.get('training_admission') is not False
            or source.get('path') != str(source_dir / 'native-trajectories.jsonl')
            or any(file_binding(source['path']).get(key) != source.get(key) for key in ('path', 'sha256', 'bytes'))):
        raise ValueError('conversion source lacks exact full-trajectory/oracle/protection verification')
    source_file = Path(source['path'])
    tokenizers = _tokenizers(tokenizer_manifest, root)
    protection_ref = file_binding(protection_manifest)
    current_protection = source_proof.get('current_protection') or {}
    if (current_protection.get('path') != protection_ref['path']
            or current_protection.get('sha256') != protection_ref['sha256']):
        raise ValueError('converter protection manifest differs from exact source verification')
    admitted_inputs, cached_ids, admitted_ids, rejected, same_actor_boundary = _registered_inputs(
        root, spool, ctx, relocation_manifest, tokenizers, source_proof)
    outputs = output
    if outputs.exists() and any(outputs.iterdir()):
        raise FileExistsError(f'conversion output is nonempty: {outputs}')
    outputs.mkdir(parents=True, exist_ok=True)
    model_identity_refs = {model: _registered_model_identity(root, model, admitted_inputs[model], outputs)
                           for model in ('lfm25', 'mellum21')}
    model_dir = tokenizer_manifest.parent.parent / 'model-inventories'
    base_inventory_refs = {model: _base_inventory_reference(
        model, model_dir, outputs, tokenizer_manifest, model_identity_refs[model])
        for model in ('lfm25', 'mellum21')}
    code_root = Path(__file__).resolve().parents[2]
    converter = code_root / 'scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py'
    converter_ref = file_binding(converter)
    records = []
    for model in ('lfm25', 'mellum21'):
        dedup = [*_legacy_inputs(root, model), *admitted_inputs[model]]
        args = [sys.executable, str(converter), '--source', str(source_file),
                '--output', str(outputs / model), '--base-tokenizer', str(tokenizers[model]),
                '--protection-manifest', str(protection_manifest), '--model-key', model,
                '--materializer-root', str(root), '--base-inventory-reference', str(base_inventory_refs[model])]
        # The existing LFM lineage explicitly uses the keep rendering profile.
        # Masking failed assistant targets is independent and does not erase it.
        args.extend(history_policy_args(model))
        for path in dedup:
            args.extend(['--deduplicate-against', str(path)])
        for corpus_id in cached_ids[model]:
            args.extend(['--deduplicate-against-registered-cache', corpus_id])
        if model == 'lfm25':
            args.extend(['--deduplicate-against-registered-cache', LFM_HISTORY_KEEP])
        log = outputs / f'{model}.conversion.log'
        with log.open('xb') as stream:
            result = subprocess.run(args, cwd=root,
                env={**os.environ, 'PYTHONPATH': os.pathsep.join((str(code_root),
                    str(code_root / 'training/neuralese')))},
                stdout=stream, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise RuntimeError(f'{model} native graph conversion failed; see {log}')
        receipt_ref = file_binding(outputs / model / 'receipt.json')
        receipt = read_bound(receipt_ref)
        if (receipt.get('status') != 'passed' or receipt.get('model_key') != model
                or receipt.get('source', {}).get('sha256') != source['sha256']
                or receipt.get('tokenizer', {}).get('expanded_fingerprint') !=
                   {'lfm25': 'be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105',
                    'mellum21': 'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f'}[model]):
            raise ValueError(f'{model} converted receipt differs from exact source/tokenizer authority')
        records.append({'model': model, 'receipt': receipt_ref, 'log': file_binding(log),
                        'deduplication_inputs': len(receipt['deduplication']['inputs'])})
    if (file_binding(converter) != converter_ref
            or any(file_binding(source_file).get(key) != source.get(key) for key in ('path', 'sha256', 'bytes'))):
        raise ValueError('source or converter changed during conversion')
    result = {'schema': 'natlang.teacher-internal-both-port-conversion/1',
        'settings': settings_ref, 'source_proof': source_proof_ref,
        'source': source, 'converter': converter_ref, 'tokenizer_relocation': file_binding(tokenizer_manifest),
        'base_inventory_references': {model: file_binding(path) for model, path in base_inventory_refs.items()},
        'protection_manifest': protection_ref, 'batch': batch, 'tranche': number,
        'admitted_predecessors': admitted_ids, 'nonadmitted_predecessors': rejected,
        'same_actor_source_boundary_inputs': same_actor_boundary,
        'models': records, 'training_registration': False, 'training_exposure': False,
        'prefix_debt': 'Earlier retained source boundaries remain deferred custody; this tranche operation does not mark them exported or admitted.'}
    result_path = outputs / 'both-port-conversion.json'
    write_json_durable(result_path, result)
    return {**result, 'result': file_binding(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=resolve('repo'))
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--settings-sha256', required=True)
    parser.add_argument('--tokenizer-manifest', type=Path, required=True)
    parser.add_argument('--protection-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--relocation-manifest', type=Path)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--number', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(root=args.repository, settings_path=args.settings,
        settings_sha256=args.settings_sha256, tokenizer_manifest=args.tokenizer_manifest,
        protection_manifest=args.protection_manifest, output=args.output,
        batch=args.batch, number=args.number,
        relocation_manifest=args.relocation_manifest), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
