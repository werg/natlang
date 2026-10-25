"""Convert a source-verified internal Qwen36 tranche with complete native dedup inputs."""

from __future__ import annotations

import argparse
import errno
import hashlib
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
    preceding_tranches, read_bound, verify_teacher_model_actions,
)
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import (
    _content_index_policy_sha256, _valid_content_index, _validated_content_fingerprints, graph_content_sha256,
    file_rows, open_jsonl_text, provision, registered_cached_content_index,
    reviewed_guided_decision_authority, tokenizer_fingerprint,
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


def _verified_internal_predecessors(root: Path, ctx: Tranche, source_proof: dict,
                                   contracts: dict[str, dict], rows: list[dict]):
    """Authenticate internal registrations before using their graphs or boundaries.

    Internal package IDs intentionally differ from the historical teacher
    namespace. A namespace relocation does not change source identity, and it
    must not make an already admitted predecessor invisible to deduplication.
    The immutable registration proof and both model receipts bind that
    relocation to the exact source actor/runtime and current rendering law.
    """
    current_protection = (source_proof.get('current_protection') or {}).get('sha256')
    if not isinstance(current_protection, str) or len(current_protection) != 64:
        raise ValueError('current source proof lacks exact protection identity')
    registry_path = root / 'training/neuralese_corpora.json'
    registry_ref = file_binding(registry_path)
    registry = read_bound(registry_ref)
    manifest_root = root / 'training/corpus-manifests'
    checked = []
    def relocated_exact(declared: dict, actual: dict) -> bool:
        try:
            same_location = Path(declared.get('path', '')).resolve(strict=True) == Path(actual['path']).resolve(strict=True)
        except (OSError, RuntimeError):
            same_location = False
        return (same_location and declared.get('sha256') == actual.get('sha256')
                and declared.get('bytes') == actual.get('bytes'))
    present = {row.get('id') for row in rows}
    for entry in registry.get('corpora', []):
        ident = entry.get('id')
        if (not isinstance(ident, str)
                or not ident.startswith('qwen36-verified-positive-native-graphs-20261023-internal-')
                or entry.get('training_admission') is not True or ident in present):
            continue
        try:
            batch = int(ident.rsplit('batch', 1)[1].split('-', 1)[0])
            number = int(ident.rsplit('tranche', 1)[1][:4])
        except (IndexError, ValueError):
            raise ValueError(f'internal predecessor ID has no exact batch/tranche position: {ident}')
        if (batch, number) >= (ctx.batch, ctx.number):
            continue
        manifest_ref = file_binding(manifest_root / f'{ident}.json')
        manifest = read_bound(manifest_ref)
        folder = root / entry['path']
        if manifest.get('id') != ident or manifest.get('path') != entry.get('path'):
            raise ValueError(f'internal predecessor manifest differs from registry: {ident}')
        members = {item['path']: item for item in manifest.get('files', [])}
        outputs = {}
        for model in ('lfm25', 'mellum21'):
            rel = f'{model}/train/native-graphs.jsonl.zst'
            member = members.get(rel)
            if member is None:
                raise ValueError(f'internal predecessor manifest omits {model} graphs: {ident}')
            graph_ref = file_binding(folder / rel)
            if any(graph_ref.get(key) != member.get(key) for key in ('sha256', 'bytes')):
                raise ValueError(f'internal predecessor graph differs from manifest: {ident}/{model}')
            outputs[model] = graph_ref
        rows.append({'id': ident, 'outputs': outputs, 'registry_entry': entry})
        present.add(ident)
    for row in rows:
        ident = row.get('id')
        if not isinstance(ident, str) or not ident.startswith('qwen36-verified-positive-native-graphs-20261023-internal-'):
            continue
        entry = row.get('registry_entry')
        if entry is None:
            entry = next((item for item in registry['corpora'] if item.get('id') == ident), None)
        if not isinstance(entry, dict) or entry.get('training_admission') is not True:
            raise ValueError(f'internal predecessor is not an admitted registry entry: {ident}')
        manifest_ref = file_binding(manifest_root / f'{ident}.json')
        manifest = read_bound(manifest_ref)
        folder = root / entry['path']
        if manifest.get('id') != ident or manifest.get('path') != entry.get('path'):
            raise ValueError(f'internal predecessor manifest differs from registry: {ident}')
        members = {item['path']: item for item in manifest.get('files', [])}
        proof_rel = 'evidence/registration-proof.json'
        proof_member = members.get(proof_rel)
        if proof_member is None:
            raise ValueError(f'internal predecessor has no manifest-bound registration proof: {ident}')
        binding = (entry.get('source_binding') or {}).get('registration_proof')
        if (not isinstance(binding, dict)
                or binding.get('path') != f"{entry['path']}/{proof_rel}"
                or any(binding.get(key) != proof_member.get(key) for key in ('sha256', 'bytes'))):
            raise ValueError(f'internal predecessor registry does not bind its exact registration proof: {ident}')
        proof_path = folder / proof_rel
        proof_ref = file_binding(proof_path)
        if any(proof_ref.get(key) != proof_member.get(key) for key in ('sha256', 'bytes')):
            raise ValueError(f'internal predecessor registration proof differs from manifest: {ident}')
        proof = read_bound(proof_ref)
        try:
            batch, number = int(ident.rsplit('batch', 1)[1].split('-', 1)[0]), int(ident.rsplit('tranche', 1)[1][:4])
        except (IndexError, ValueError):
            raise ValueError(f'internal predecessor ID has no exact batch/tranche position: {ident}')
        if (proof.get('schema') != 'natlang.teacher-internal-tranche-registration-proof/1'
                or proof.get('status') != 'passed' or proof.get('training_admission') is not True
                or proof.get('training_exposure') is not False or proof.get('corpus_id') != ident
                or proof.get('source_namespace') != ctx.teacher_namespace
                or proof.get('batch') != batch or proof.get('tranche') != number
                or not isinstance(proof.get('current_protection_manifest_sha256'), str)
                or len(proof['current_protection_manifest_sha256']) != 64):
            raise ValueError(f'internal predecessor proof identity/protection differs: {ident}')
        if (batch, number) >= (ctx.batch, ctx.number):
            continue
        source_rel = 'evidence/native-trajectories.jsonl'
        source_member = members.get(source_rel)
        if source_member is None:
            raise ValueError(f'internal predecessor does not retain its exact raw source: {ident}')
        source_ref = file_binding(folder / source_rel)
        source_proof_member = members.get('evidence/canonical-source-verification.json')
        result_member = members.get('evidence/source-result.json')
        conversion_member = members.get('evidence/both-port-conversion.json')
        if source_proof_member is None or result_member is None or conversion_member is None:
            raise ValueError(f'internal predecessor source verification/result is not retained: {ident}')
        source_proof_ref = file_binding(folder / 'evidence/canonical-source-verification.json')
        source_result_ref = file_binding(folder / 'evidence/source-result.json')
        conversion_ref = file_binding(folder / 'evidence/both-port-conversion.json')
        if (any(source_ref.get(k) != source_member.get(k) for k in ('sha256', 'bytes'))
                or any(source_proof_ref.get(k) != source_proof_member.get(k) for k in ('sha256', 'bytes'))
                or any(source_result_ref.get(k) != result_member.get(k) for k in ('sha256', 'bytes'))
                or any(conversion_ref.get(k) != conversion_member.get(k) for k in ('sha256', 'bytes'))
                or not relocated_exact(proof.get('source') or {}, source_ref)
                or not relocated_exact(proof.get('source_verification') or {}, source_proof_ref)
                or not relocated_exact(proof.get('source_result') or {}, source_result_ref)
                or not relocated_exact(proof.get('conversion') or {}, conversion_ref)):
            raise ValueError(f'internal predecessor source custody differs from its registration proof: {ident}')
        conversion = read_bound(conversion_ref)
        if (conversion.get('schema') != 'natlang.teacher-internal-both-port-conversion/1'
                or conversion.get('batch') != batch or conversion.get('tranche') != number
                or conversion.get('training_registration') is not False
                or conversion.get('training_exposure') is not False
                or conversion.get('source', {}).get('sha256') != source_ref['sha256']):
            raise ValueError(f'internal predecessor conversion does not bind its registered source boundary: {ident}')
        source_verification = read_bound(source_proof_ref)
        source_result = read_bound(source_result_ref)
        actor = (source_verification.get('actor') or {}).get('sha256')
        runtime = (source_verification.get('execution_identity') or {}).get('sha256')
        source_protection = (source_verification.get('current_protection') or {}).get('sha256')
        if (source_verification.get('status') != 'passed' or source_verification.get('training_admission') is not False
                or source_verification.get('source', {}).get('sha256') != source_ref['sha256']
                or source_verification.get('source', {}).get('bytes') != source_ref['bytes']
                or not isinstance(actor, str) or len(actor) != 64
                or not isinstance(runtime, str) or len(runtime) != 64
                or source_protection != proof.get('current_protection_manifest_sha256')
                or source_result.get('training_admission') is not False
                or source_result.get('training_exposure') is not False
                or any((source_result.get('source_proof') or {}).get(key) != source_proof_ref.get(key)
                       for key in ('sha256', 'bytes'))):
            raise ValueError(f'internal predecessor source/actor/runtime binding differs: {ident}')
        proof_models = proof.get('models')
        if not isinstance(proof_models, dict) or set(proof_models) != {'lfm25', 'mellum21'}:
            raise ValueError(f'internal predecessor does not bind both native model views: {ident}')
        for model in ('lfm25', 'mellum21'):
            receipt_rel = f'{model}/receipt.json'
            graph_rel = f'{model}/train/native-graphs.jsonl.zst'
            receipt_member, graph_member = members.get(receipt_rel), members.get(graph_rel)
            declared = proof_models[model]
            if receipt_member is None or graph_member is None:
                raise ValueError(f'internal predecessor lacks exact {model} graph/receipt: {ident}')
            receipt_ref = file_binding(folder / receipt_rel)
            graph_ref = file_binding(folder / graph_rel)
            conversion_receipts = [item.get('receipt') for item in conversion.get('models', [])
                                   if item.get('model') == model]
            if (any(receipt_ref.get(k) != receipt_member.get(k) for k in ('sha256', 'bytes'))
                    or any(graph_ref.get(k) != graph_member.get(k) for k in ('sha256', 'bytes'))
                    or not relocated_exact(declared.get('receipt') or {}, receipt_ref)
                    or not relocated_exact(declared.get('train') or {}, graph_ref)
                    or len(conversion_receipts) != 1
                    or any((conversion_receipts[0] or {}).get(key) != receipt_ref.get(key)
                           for key in ('sha256', 'bytes'))):
                raise ValueError(f'internal predecessor {model} graph/receipt differs from registration proof: {ident}')
            receipt = read_bound(receipt_ref)
            expected_explicit = 'keep' if model == 'lfm25' else None
            if (receipt.get('status') != 'passed' or receipt.get('model_key') != model
                    or receipt.get('source', {}).get('sha256') != source_ref['sha256']
                    or receipt.get('tokenizer', {}).get('expanded_fingerprint') != contracts[model]['tokenizer_fingerprint']
                    or receipt.get('explicit_history_reasoning_policy') != expected_explicit
                    or receipt.get('failed_assistant_history_policy') != 'mask-runtime-failures'):
                raise ValueError(f'internal predecessor {model} receipt has incompatible source/tokenizer/history policy: {ident}')
        checked.append({'id': ident, 'batch': batch, 'tranche': number, 'registration_proof': proof_ref,
                        'manifest': manifest_ref})
    if file_binding(registry_path) != registry_ref:
        raise ValueError('registry changed while authenticating internal predecessors')
    return checked


def _registered_dedup_input(root, ident, model, fingerprint, policy, declared, *,
                            registry_ref, predecessor, manifest_ref, manifest):
    """Verify a historical cache receipt against an available original when restored.

    Cache-only selection remains restricted to unavailable sources. Historical
    receipts keep their recorded identity; a restored source adds an exact byte
    check before reusing its existing parser/policy-addressed content index.
    """
    relative = f'{model}/train/native-graphs.jsonl.zst'
    source = root / predecessor['path'] / relative
    try:
        source.stat()
    except OSError as error:
        if error.errno not in (errno.ENOENT, errno.EIO):
            raise
        cached, contents = registered_cached_content_index(
            root, ident, model, fingerprint, policy, registry_ref=registry_ref,
            registry_entry=predecessor, manifest_ref=manifest_ref,
            manifest=manifest, verify_bindings=False)
        return cached, contents, []
    members = [item for item in manifest.get('files', []) if item.get('path') == relative]
    actual = file_binding(source)
    if (manifest.get('id') != ident or manifest.get('path') != predecessor['path']
            or len(members) != 1
            or any(members[0].get(key) != actual[key] for key in ('sha256', 'bytes'))):
        raise ValueError('restored dedup source differs from its admitted manifest')
    key = {'source_sha256': actual['sha256'], 'tokenizer_fingerprint': fingerprint,
           'policy_sha256': policy, 'kind': 'teacher-content-fingerprints-v1'}
    cache_ref = declared.get('content_index_cache') or {}
    if file_binding(cache_ref.get('path', '')) != cache_ref:
        raise ValueError('historical dedup content index differs from its recorded bytes')
    from natlang_neuralese.common import row_index
    index = row_index.load(Path(cache_ref['path']))
    if not _valid_content_index(index, key):
        raise ValueError('restored dedup source lacks its exact current-law content index')
    cached = {'path': f'registered-cache:{ident}/{relative}',
              'sha256': actual['sha256'], 'bytes': actual['bytes'], 'rows': index['rows'],
              'content_index_cache': cache_ref, 'content_index_key': key}
    return cached, index['fingerprints'], [actual]


def _verified_zero_conversion(root: Path, path: Path, ctx: Tranche,
                             current_source_proof: dict, contracts: dict[str, dict]) -> dict:
    """Prove a prior boundary emitted no new graphs for either student."""
    conversion_ref = file_binding(path)
    conversion = read_bound(conversion_ref)
    source_ref = conversion.get('source') or {}
    source_proof_ref = conversion.get('source_proof') or {}
    settings_ref = conversion.get('settings') or {}
    settings = read_bound(settings_ref)
    if (conversion.get('schema') != 'natlang.teacher-internal-both-port-conversion/1'
            or conversion.get('training_registration') is not False
            or conversion.get('training_exposure') is not False
            or type(conversion.get('batch')) is not int or type(conversion.get('tranche')) is not int
            or conversion.get('batch') != ctx.batch or conversion['tranche'] > ctx.number
            or settings.get('teacher_namespace') != ctx.teacher_namespace
            or (settings.get('internal_continuation') or {}).get('schema') != 'natlang.teacher-internal-source-continuation/1'
            or (settings.get('internal_continuation') or {}).get('training_admission') is not False):
        raise ValueError('zero boundary conversion has invalid position or admission state')
    source_proof = read_bound(source_proof_ref)
    source_file = file_binding(source_ref.get('path', ''))
    bound_inputs = [conversion_ref, source_proof_ref, source_file, settings_ref]
    actor = (source_proof.get('actor') or {}).get('sha256')
    runtime = (source_proof.get('execution_identity') or {}).get('sha256')
    protection = (source_proof.get('current_protection') or {}).get('sha256')
    if (source_proof.get('status') != 'passed' or source_proof.get('training_admission') is not False
            or source_proof.get('source') != source_ref
            or source_file.get('sha256') != source_ref.get('sha256')
            or source_file.get('bytes') != source_ref.get('bytes')
            or not isinstance(actor, str) or len(actor) != 64
            or not isinstance(runtime, str) or len(runtime) != 64
            or not isinstance(protection, str) or len(protection) != 64
            or conversion.get('protection_manifest', {}).get('sha256') != protection):
        raise ValueError('zero boundary source/actor/runtime/protection custody differs')
    rows = conversion.get('models')
    if (not isinstance(rows, list) or len(rows) != 2
            or {row.get('model') for row in rows} != {'lfm25', 'mellum21'}):
        raise ValueError('zero boundary lacks both exact student receipts')
    admitted = set(conversion.get('admitted_predecessors') or [])
    admitted.update(row.get('corpus_id') for row in conversion.get('same_actor_source_boundary_inputs', [])
                    if isinstance(row, dict))
    registry_ref = file_binding(root / 'training/neuralese_corpora.json')
    registry = read_bound(registry_ref)
    by_id = {row.get('id'): row for row in registry.get('corpora', [])}
    manifest_refs = []
    raw_rows, raw_row_shas = {}, {}
    for raw, source_row in file_rows(Path(source_ref['path'])):
        ident = source_row.get('id')
        if not isinstance(ident, str) or ident in raw_rows:
            raise ValueError('zero-boundary source has missing or duplicate trajectory IDs')
        raw_rows[ident] = source_row
        raw_row_shas[ident] = hashlib.sha256(raw.rstrip(b'\r\n')).hexdigest()
    source_actions = sum(len(row.get('trajectory', [])) for row in raw_rows.values())
    if (len(raw_rows) != source_ref.get('rows') or len(raw_rows) != source_proof.get('rows')
            or source_actions != source_proof.get('teacher_action_turns')):
        raise ValueError('zero-boundary raw source rows/actions differ from canonical source proof')
    builder = Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py'
    builder_ref = file_binding(builder)
    if any(conversion.get('converter', {}).get(key) != builder_ref.get(key)
           for key in ('sha256', 'bytes')):
        raise ValueError('zero-boundary converter implementation differs from exact current action verifier')

    def exact_decisions(receipt: dict) -> dict:
        declared = receipt.get('decision_admission') or {}
        decision_receipt_ref = file_binding(declared.get('path', ''))
        if decision_receipt_ref.get('sha256') != declared.get('sha256'):
            raise ValueError('zero-boundary decision receipt differs from its converter binding')
        materialized = read_bound(decision_receipt_ref)
        if (materialized.get('schema') != 'natlang.verified-teacher-decision-materialization/1'
                or materialized.get('identity', {}).get('source_sha256') != source_ref.get('sha256')
                or materialized.get('identity', {}).get('direct_answers') is not True
                or materialized.get('summary') != declared.get('summary')
                or materialized.get('output_sha256') != declared.get('output_sha256')):
            raise ValueError('zero-boundary materialized decision proof differs from exact source receipt')
        runtime = materialized.get('identity', {}).get('runtime')
        if (not isinstance(runtime, dict) or not runtime
                or any(not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                       for name, digest in runtime.items())):
            raise ValueError('zero-boundary decision proof lacks its exact reader runtime inventory')
        output_path = Path(decision_receipt_ref['path']).parent / materialized['summary']['output']
        output_ref = file_binding(output_path)
        if output_ref.get('sha256') != materialized.get('output_sha256'):
            raise ValueError('zero-boundary materialized decisions differ from their receipt')
        decisions = {}
        for raw_decision in output_path.open():
            record = json.loads(raw_decision)
            reference, decision = record.get('source_ref') or {}, record.get('decision') or {}
            key = (reference.get('trajectory_id'), decision.get('index'))
            if not isinstance(key[0], str) or type(key[1]) is not int or key in decisions:
                raise ValueError('zero-boundary decisions lack unique exact source action identities')
            decisions[key] = {
                'approved': (record.get('training_admission') or {}).get('approved') is True
                    and decision.get('training_approved') is True and decision.get('failed_action') is False,
                'reason': (record.get('training_admission') or {}).get('reason'),
                'invocation_id': reference.get('invocation_id'),
                'raw_response_sha256': decision.get('source_raw_response_sha256'),
                **({'parent_quality_hold': reference['parent_quality_hold'], 'parent_task_qualified': False,
                    **({'assistant_history_input_sha256': reference['assistant_history_input_sha256']}
                       if reference.get('assistant_history_input_sha256') else {})}
                   if reference.get('parent_quality_hold') else {}),
                **reviewed_guided_decision_authority(
                    record, review_bound='decision_review' in materialized.get('identity', {})),
                'materialized_decision_id': record['id'],
                'materialized_decision_sha256': canonical_json_sha256_hex(record),
            }
        if len(decisions) != materialized.get('summary', {}).get('training_decisions'):
            raise ValueError('zero-boundary materialized decision count differs from exact receipt')
        return decisions

    for row in rows:
        model = row['model']
        bound_inputs.append(row['receipt'])
        receipt = read_bound(row['receipt'])
        dedup = receipt.get('deduplication') or {}
        omissions = dedup.get('omissions')
        inputs = {item.get('path'): item for item in dedup.get('inputs', [])
                  if isinstance(item, dict) and isinstance(item.get('path'), str)}
        expected_history = 'keep' if model == 'lfm25' else None
        if (receipt.get('status') != 'passed' or receipt.get('model_key') != model
                or receipt.get('source', {}).get('sha256') != source_ref.get('sha256')
                or receipt.get('tokenizer', {}).get('expanded_fingerprint') != contracts[model]['tokenizer_fingerprint']
                or receipt.get('explicit_history_reasoning_policy') != expected_history
                or receipt.get('failed_assistant_history_policy') != 'mask-runtime-failures'
                or receipt.get('outputs', {}).get('train', {}).get('rows') != 0
                or receipt.get('outputs', {}).get('test', {}).get('rows') != 0
                or dedup.get('emitted_rows') != 0 or type(dedup.get('omitted_rows')) is not int
                or dedup.get('omitted_rows') < 1 or not isinstance(omissions, list)
                or len(omissions) != dedup.get('omitted_rows')):
            raise ValueError(f'{model} zero-boundary receipt does not prove complete exact deduplication')
        bindings = receipt.get('source', {}).get('row_bindings')
        expected_bindings = [{'id': ident, 'sha256': raw_row_shas[ident]}
                             for ident in raw_rows]
        if (not isinstance(bindings, list) or len(bindings) != len(expected_bindings)
                or any(not isinstance(actual, dict) or actual.get('id') != expected['id']
                       or actual.get('sha256') != expected['sha256']
                       for actual, expected in zip(bindings, expected_bindings))):
            raise ValueError(f'{model} zero-boundary receipt does not bind every exact source row')
        verify_teacher_model_actions([], receipt, raw_rows=raw_rows, raw_row_shas=raw_row_shas,
            source_sha256=source_ref['sha256'], decision_keys=exact_decisions(receipt), hold_ids=set())
        receipt_path = Path(row['receipt']['path'])
        for split in ('train', 'test'):
            output = receipt.get('outputs', {}).get(split) or {}
            output_path = receipt_path.parent / split / 'native-graphs.jsonl.zst'
            output_ref = file_binding(output_path)
            if (output.get('path') != output_ref.get('path') or output.get('rows') != 0
                    or output.get('tokens') != 0 or output.get('max_row_tokens') != 0
                    or output_ref.get('sha256') != output.get('sha256')
                    or output_ref.get('bytes') != output.get('bytes')):
                raise ValueError(f'{model} zero-boundary {split} file differs from its exact receipt')
            with open_jsonl_text(output_path) as stream:
                if any(line.strip() for line in stream):
                    raise ValueError(f'{model} zero-boundary {split} stream is not actually empty')
            bound_inputs.append(output_ref)

        legacy = {str(path): file_binding(path) for path in _legacy_inputs(root, model)}
        target_representations = {
            item.get('content_sha256'): item.get('represented_by')
            for item in omissions if isinstance(item, dict)
        }
        seen_contents = {}
        input_ids = set()
        policy = _content_index_policy_sha256()
        for input_binding in dedup.get('inputs', []):
            input_path = input_binding.get('path')
            if not isinstance(input_path, str):
                raise ValueError(f'{model} zero-boundary dedup input lacks its exact path')
            if input_path.startswith('registered-cache:'):
                target = input_path.removeprefix('registered-cache:')
                parts = target.split('/')
                if len(parts) != 4 or parts[1:] != [model, 'train', 'native-graphs.jsonl.zst']:
                    raise ValueError(f'{model} zero-boundary cache input has a mismatched model/member')
                ident = parts[0]
                if (ident not in admitted
                        and not (model == 'lfm25' and ident == LFM_HISTORY_KEEP)):
                    raise ValueError(f'{model} zero-boundary cache input is outside admitted predecessor closure')
                predecessor = by_id.get(ident)
                if predecessor is None or predecessor.get('training_admission') is not True:
                    raise ValueError(f'{model} zero-boundary cache input is not an admitted registry entry: {ident}')
                manifest_ref = file_binding(root / 'training/corpus-manifests' / f'{ident}.json')
                manifest_refs.append(manifest_ref)
                manifest = read_bound(manifest_ref)
                cached, contents, restored_inputs = _registered_dedup_input(
                    root, ident, model, receipt['tokenizer']['expanded_fingerprint'], policy,
                    input_binding, registry_ref=registry_ref, predecessor=predecessor,
                    manifest_ref=manifest_ref, manifest=manifest)
                bound_inputs.extend(restored_inputs)
                if any(input_binding.get(key) != cached.get(key)
                       for key in ('path', 'sha256', 'bytes', 'rows', 'content_index_cache', 'content_index_key')):
                    raise ValueError(f'{model} zero-boundary cache input differs from its admitted content index')
                input_ids.add(ident)
                bound_inputs.append(file_binding(cached['content_index_cache']['path']))
            else:
                actual_input = file_binding(input_path)
                if (actual_input.get('sha256') != input_binding.get('sha256')
                        or ('bytes' in input_binding and actual_input.get('bytes') != input_binding.get('bytes'))):
                    raise ValueError(f'{model} zero-boundary dedup input changed from its exact receipt')
                if input_path in legacy:
                    if legacy[input_path].get('sha256') != actual_input.get('sha256'):
                        raise ValueError(f'{model} zero-boundary legacy dedup source differs from its exact v21 receipt')
                    ident = next((part for part in Path(input_path).parts if part.startswith('qwen-verified-positive-native-graphs-20261015-v21')), None)
                    if ident:
                        input_ids.add(ident)
                else:
                    marker = '/runs/'
                    if marker not in input_path:
                        raise ValueError(f'{model} zero-boundary input has no admitted corpus identity')
                    ident = input_path.split(marker, 1)[1].split('/', 1)[0]
                    if ident not in admitted:
                        raise ValueError(f'{model} zero-boundary input is outside admitted predecessor closure')
                    entry = by_id.get(ident)
                    if entry is None or entry.get('training_admission') is not True:
                        raise ValueError(f'{model} zero-boundary input source is not admitted: {ident}')
                    relative = Path(input_path).resolve(strict=True).relative_to(
                        (root / entry['path']).resolve(strict=True)).as_posix()
                    if relative != f'{model}/train/native-graphs.jsonl.zst':
                        raise ValueError(f'{model} zero-boundary input is not its admitted train graph: {ident}')
                    manifest_ref = file_binding(root / 'training/corpus-manifests' / f'{ident}.json')
                    manifest_refs.append(manifest_ref)
                    manifest = read_bound(manifest_ref)
                    members = [member for member in manifest.get('files', [])
                               if member.get('path') == relative]
                    if (manifest.get('id') != ident or manifest.get('path') != entry.get('path')
                            or len(members) != 1 or any(members[0].get(key) != actual_input.get(key)
                                                       for key in ('sha256', 'bytes'))):
                        raise ValueError(f'{model} zero-boundary input differs from admitted manifest: {ident}')
                    input_ids.add(ident)
                # Reuse the exact parser/tokenizer/policy-addressed content
                # index used by native conversion when present. The key binds
                # the admitted graph bytes; otherwise parse the graph stream
                # itself before trusting an omission.
                from natlang_neuralese.common import row_index
                cache_key = {'source_sha256': actual_input['sha256'],
                    'tokenizer_fingerprint': receipt['tokenizer']['expanded_fingerprint'],
                    'policy_sha256': policy, 'kind': 'teacher-content-fingerprints-v1'}
                cache_path = row_index.entry_path('teacher-content', canonical_json_sha256_hex(cache_key))
                cached = row_index.load(cache_path)
                if _valid_content_index(cached, cache_key):
                    contents, count = cached['fingerprints'], cached['rows']
                    bound_inputs.append(file_binding(cache_path))
                else:
                    contents, count = _validated_content_fingerprints(
                        input_path, receipt['tokenizer']['expanded_fingerprint'])
                if count != input_binding.get('rows'):
                    raise ValueError(f'{model} zero-boundary input row count differs from current graph parser')
                bound_inputs.append(actual_input)
            for content, identifier in contents.items():
                if content in target_representations:
                    expected = target_representations[content] or {}
                    actual = {'path': input_path, 'id': identifier}
                    if (expected.get('path') == input_path and expected.get('id') == identifier):
                        seen_contents[content] = actual
            # Content indexes may contain hundreds of thousands of rows.
            # Retain only the omission witnesses while checking each input.
            del contents
            if 'cached' in locals():
                del cached
        if not set(conversion.get('admitted_predecessors', [])) <= input_ids:
            raise ValueError(f'{model} zero-boundary receipt dropped an admitted predecessor input')
        expected_same_actor = {item.get('corpus_id') for item in conversion.get('same_actor_source_boundary_inputs', [])
                               if item.get('model') == model}
        if not expected_same_actor <= input_ids:
            raise ValueError(f'{model} zero-boundary receipt dropped a same-actor source input')
        for omission in omissions:
            represented = omission.get('represented_by') or {}
            input_row = inputs.get(represented.get('path'))
            content = omission.get('content_sha256')
            actual_representation = seen_contents.get(content)
            if (input_row is None or not isinstance(content, str) or len(content) != 64
                    or any(char not in '0123456789abcdef' for char in content)
                    or actual_representation != represented):
                raise ValueError(f'{model} zero-boundary omission is not bound to an exact input graph')
            represented_path = represented['path']
            if represented_path.startswith('registered-cache:'):
                ident = represented_path.removeprefix('registered-cache:').split('/', 1)[0]
                if not (ident in admitted or (model == 'lfm25' and ident == LFM_HISTORY_KEEP)):
                    raise ValueError(f'{model} zero-boundary omission is outside admitted predecessor closure')
            else:
                if '/runs/' not in represented_path:
                    raise ValueError(f'{model} zero-boundary omission path has no admitted corpus identity')
                ident = represented_path.split('/runs/', 1)[1].split('/', 1)[0]
                if ident not in admitted:
                    raise ValueError(f'{model} zero-boundary omission is outside admitted predecessor closure')
            entry = by_id.get(ident)
            if entry is None or entry.get('training_admission') is not True:
                raise ValueError(f'{model} zero-boundary omission is outside admitted predecessor closure')
            manifest_ref = file_binding(root / 'training/corpus-manifests' / f'{ident}.json')
            manifest_refs.append(manifest_ref)
            manifest = read_bound(manifest_ref)
            rel = f'{model}/train/native-graphs.jsonl.zst'
            member = [item for item in manifest.get('files', []) if item.get('path') == rel]
            if manifest.get('id') != ident or manifest.get('path') != entry.get('path') or len(member) != 1:
                raise ValueError(f'{model} zero-boundary omission graph is not an admitted manifest member')
            if represented_path.startswith('registered-cache:'):
                if (member[0].get('sha256') != input_row.get('sha256')
                        or member[0].get('bytes') != input_row.get('bytes')):
                    raise ValueError(f'{model} zero-boundary cached omission differs from its admitted manifest')
            else:
                graph = file_binding(root / entry['path'] / rel)
                bound_inputs.append(graph)
                if (graph.get('sha256') != member[0].get('sha256')
                        or graph.get('bytes') != member[0].get('bytes')
                        or graph.get('sha256') != input_row.get('sha256')):
                    raise ValueError(f'{model} zero-boundary omission graph is not an admitted manifest member')
    if (file_binding(root / 'training/neuralese_corpora.json') != registry_ref
            or any(file_binding(ref['path']) != ref for ref in [*bound_inputs, *manifest_refs])):
        raise ValueError('zero-boundary conversion custody changed during verification')
    return {'batch': conversion['batch'], 'tranche': conversion['tranche'],
            'conversion': conversion_ref, 'source_proof': source_proof_ref,
            'settings': settings_ref,
            'models': {row['model']: row['receipt'] for row in rows},
            'training_admission': False, 'training_exposure': False}


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
    namespaces = tuple(dict.fromkeys((LEGACY_TEACHER_NAMESPACE, FLINT_NAMESPACE, ctx.teacher_namespace)))
    additional = admitted_predecessors(root, ctx.corpus_id, namespaces=namespaces,
                                       relocation_manifest=relocation_manifest,
                                       cache_contracts=contracts)
    internal = _verified_internal_predecessors(root, ctx, source_proof, contracts, additional)
    known_ids = {row['id'] for row in additional}
    accounted_positions = {(item['batch'], item['tranche']) for item in internal}
    zero_boundaries = []
    for number in range(ctx.number):
        if (ctx.batch, number) in accounted_positions:
            continue
        original_id = Tranche(ctx.batch, number, teacher_namespace=ctx.teacher_namespace).corpus_id
        if original_id in known_ids:
            accounted_positions.add((ctx.batch, number))
            continue
        candidates = sorted((spool / 'conversions' / f'batch-{ctx.batch:04d}').glob(
            f'tranche-{number:04d}-attempt-*/both-port-conversion.json'), reverse=True)
        if not candidates:
            continue
        zero = _verified_zero_conversion(root, candidates[0], ctx, source_proof, contracts)
        zero_boundaries.append(zero)
        accounted_positions.add((ctx.batch, number))
    admitted, rejected = preceding_tranches(root, spool, ctx,
        relocation_manifest=relocation_manifest, cache_contracts=contracts,
        accounted_internal_boundaries=accounted_positions)
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
    return inputs, cached_ids, list(by_id), rejected, same_actor_boundary, zero_boundaries


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
    declaration = read_bound(readback_declaration)
    readers = [item['reader'] for item in declaration.get('models', []) if item.get('model') == model]
    if len(readers) != 1:
        raise ValueError('native model inventory lost its exact historical readback reader')
    historical_reader = readers[0]
    if (file_binding(historical_reader['path']) != historical_reader
            or historical_reader['sha256'] != MODEL_CONFIG_READER_SHA256):
        raise ValueError('native model inventory historical reader changed after readback')
    proof = {'schema': 'natlang.teacher-base-inventory-reference/1', 'model_key': model,
        'model_id': model_id, 'architecture': config['architecture'],
        'base_inventory': inventory, 'base_inventory_sha256': config['base_inventory_sha256'],
        'model_config': config_ref, 'metadata_readback': metadata_ref,
        'metadata_reader': historical_reader,
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
    admitted_inputs, cached_ids, admitted_ids, rejected, same_actor_boundary, zero_boundaries = _registered_inputs(
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
        'accounted_zero_boundaries': zero_boundaries,
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
