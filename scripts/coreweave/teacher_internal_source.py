"""Recover and verify one exact Qwen36 tranche into an internal spool.

The old external spool remains recorded in the original settings. Only the
storage destination changes; the selected teacher, actor, runtime and all
source-verification operators remain byte-bound inputs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.paths import resolve
from scripts.coreweave.teacher_tranche_admission import Tranche, file_binding, read_bound

_FROZEN_OPERATOR_ROOT = resolve('data_nvme','natlang-teacher-drain-export-20261022')
_FROZEN_OPERATOR_BINDINGS = {
    'teacher_tranche_recovery.py': ('7fc0d11b52fbc06cd94c777349f488f693b39138ca0cd5bbe916e377b8239dac', 3822),
    'teacher_source_operator.py': ('7d1deed55222410c26e22da6cf16fe839549f7acaf7a019248c04f31eca9fc2f', 13600),
    'teacher_tranche_admission.py': ('9284b699a34082cb4fd9f03e97ee73cf044b0a39f3a76c7d9c93f96b1c5bec7b', 49402),
}
_REMOTE_CANDIDATE_SCAN_CODE_SHA256 = 'fb21932f0e670c0e8574ddaeb51364e3e84302d1f20e1419fc8e621f81d7b6d2'

def verified_zero_positive_source(proof: dict) -> dict | None:
    """Account for a fully verified boundary whose outcomes were all rejected."""
    if proof.get('rows') != 0:
        return None
    source = proof.get('source') or {}
    rejected = proof.get('rejected_outputs')
    selected = proof.get('selected_outputs')
    if (proof.get('schema') != 'natlang.teacher-live-tranche-canonical-source-verification/1'
            or proof.get('status') != 'passed' or proof.get('training_admission') is not False
            or type(proof.get('rows')) is not int or source.get('rows') != 0
            or source.get('bytes') != 0 or proof.get('teacher_action_turns') != 0
            or type(selected) is not int or selected < 1
            or not isinstance(rejected, list) or len(rejected) != selected):
        raise ValueError('zero-positive source lacks complete rejected-outcome accounting')
    identifiers = set()
    for row in rejected:
        identifier = row.get('source_output')
        sha = row.get('sha256')
        if (not isinstance(identifier, str) or not identifier or identifier in identifiers
                or not isinstance(row.get('reason'), str) or not row['reason']
                or not isinstance(sha, str) or len(sha) != 64
                or any(char not in '0123456789abcdef' for char in sha)):
            raise ValueError('zero-positive source has invalid or repeated rejected outcomes')
        identifiers.add(identifier)
    actual = file_binding(source.get('path', ''))
    if any(actual.get(key) != source.get(key) for key in ('path', 'sha256', 'bytes')):
        raise ValueError('zero-positive source bytes differ from their full-source proof')
    return {'status': 'verified-zero-positive-source', 'source': actual,
            'rejected_outputs': selected, 'training_admission': False, 'training_exposure': False}


def _exact_ref(path: Path, expected_sha256: str) -> dict:
    ref = file_binding(path)
    if ref['sha256'] != expected_sha256:
        raise ValueError(f'input binding changed: {path}')
    return ref


def _run(command: list[str], *, cwd: Path, env: dict, output: Path) -> None:
    with output.open('ab', buffering=0) as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f'internal teacher source phase failed ({result.returncode}); see {output}')


def _relocate_ready_packet(original_ready: dict, relocation_path: Path,
                           spool: Path, teacher_sandbox_id: str,
                           source_cases_sha256: str) -> tuple[dict, dict]:
    relocation_ref = file_binding(relocation_path)
    relocation = read_bound(relocation_ref)
    if (relocation.get('schema') != 'natlang.teacher-ready-packet-internal-relocation/1'
            or relocation.get('training_admission') is not False
            or relocation.get('original_ready_packet') != original_ready
            or relocation.get('source_sandbox_id') != teacher_sandbox_id
            or relocation.get('source_cases_sha256') != source_cases_sha256):
        raise ValueError('ready-packet relocation changes its original selected source binding')
    scan = read_bound(relocation['candidate_scan'])
    if (scan.get('schema') != 'natlang.teacher-ready-packet-candidate-scan-evidence/1'
            or scan.get('training_admission') is not False
            or scan.get('source_generation_sandbox_id') != teacher_sandbox_id
            or scan.get('source_root') != '/work/generation/strong-teacher-20261015/source-sequences'
            or scan.get('original_ready_packet') != original_ready
            or scan.get('source_cases_sha256') != source_cases_sha256
            or scan.get('candidate_count') != len(ready_entries := read_bound(original_ready).get('entries', []))):
        raise ValueError('ready-packet candidate scan differs from exact source sequence')
    if scan.get('remote_scan_code_sha256') != _REMOTE_CANDIDATE_SCAN_CODE_SHA256:
        raise ValueError('ready-packet remote candidate scan code binding changed')
    scanned = scan.get('scan') or {}
    candidates = scanned.get('candidates') or []
    if (scanned.get('schema') != 'natlang.teacher-ready-packet-remote-scan/1'
            or scanned.get('candidate_count') != len(ready_entries)
            or scanned.get('target_cases_sha256') != source_cases_sha256
            or len(candidates) != len(ready_entries)):
        raise ValueError('ready-packet candidate scan is incomplete')
    source_scan_ref = scan.get('source_candidate_scan')
    if source_scan_ref is not None:
        source_scan = read_bound(source_scan_ref)
        source_candidates = (source_scan.get('scan') or {}).get('candidates') or []
        if (source_scan.get('schema') != 'natlang.teacher-ready-packet-candidate-scan-evidence/1'
                or source_scan.get('remote_scan_code_sha256') != _REMOTE_CANDIDATE_SCAN_CODE_SHA256
                or source_scan.get('original_ready_packet') != original_ready
                or len(source_candidates) != len(ready_entries)):
            raise ValueError('derived target scan lacks the exact original 202-candidate scan')
        for derived, base in zip(candidates, source_candidates):
            base_candidate = base.get('candidate_asset') or {}
            derived_asset = derived.get('candidate_asset') or {}
            if (derived.get('ordinal') != base.get('ordinal')
                    or derived.get('candidate_ref') != base.get('candidate_ref')
                    or derived.get('candidate_remote_path') != base.get('candidate_remote_path')
                    or derived_asset != base_candidate
                    or derived.get('cases_binding') != base.get('cases_binding')
                    or derived.get('matches_source_cases') !=
                       (derived.get('cases_binding', {}).get('sha256') == source_cases_sha256)):
                raise ValueError('derived target scan changed an authenticated candidate/source-case binding')
    matches = []
    for ordinal, entry in enumerate(ready_entries):
        candidate = candidates[ordinal]
        reference = entry.get('candidate_packet') or {}
        asset = candidate.get('candidate_asset') or {}
        if (candidate.get('ordinal') != ordinal or candidate.get('candidate_ref') != reference
                or asset.get('sha256') != reference.get('sha256')
                or asset.get('bytes') != reference.get('bytes')
                or not candidate.get('candidate_remote_path', '').startswith(scanned.get('source_root') + '/')
                or Path(candidate.get('candidate_remote_path', '')).name != reference.get('sha256', '') + '.json'
                or candidate.get('matches_source_cases') !=
                   (candidate.get('cases_binding', {}).get('sha256') == source_cases_sha256)):
            raise ValueError('ready-packet remote scan member differs from the original entry binding')
        if candidate['matches_source_cases']:
            matches.append((ordinal, entry, candidate))
    if (len(matches) != 1 or scan.get('matching_ordinals') != [matches[0][0]]
            or relocation.get('selected_ordinal') != matches[0][0]):
        raise ValueError('ready packet does not have exactly one exact source-case match')
    # The selected entry can be from either the original broad-ready sequence
    # or the separately verified recovery-qualified sequence. Bind the asset
    # root to this exact candidate path instead of assuming one historical
    # directory layout.
    selected_asset_root = matches[0][2]['candidate_remote_path'].rsplit('/', 1)[0]
    if relocation.get('source_root') != selected_asset_root:
        raise ValueError('ready-packet relocation source root differs from the exact selected candidate asset')
    ready = read_bound(original_ready)
    selected_ordinal, selected_entry, selected_candidate = matches[0]
    files = {item.get('role'): item for item in relocation.get('files', [])}
    required = {'candidate_packet', 'qualified_packet', 'qualification', 'global_protection_join', 'selection', 'selected_cases'}
    if set(files) != required:
        raise ValueError('ready-packet relocation must bind the complete exact selected packet closure')
    for role, item in files.items():
        original_ref, local_ref = item.get('original_ref') or {}, item.get('local_ref') or {}
        expected_extension = 'jsonl' if role == 'selected_cases' else 'json'
        if (item.get('source_sandbox_id') != teacher_sandbox_id
                or item.get('source_path') != relocation['source_root'] + '/' + original_ref.get('sha256', '') + '.' + expected_extension
                or file_binding(local_ref.get('path', '')) != local_ref
                or any(local_ref.get(key) != original_ref.get(key) for key in ('sha256', 'bytes'))):
            raise ValueError(f'{role} relocation source/member hash differs from exact packet metadata')
    ready = read_bound(original_ready)
    rows = [selected_entry]
    original_packet_ref = rows[0]['qualified_packet']
    if original_packet_ref != files['qualified_packet']['original_ref']:
        raise ValueError('qualified packet relocation differs from ready-sequence reference')
    if selected_entry['candidate_packet'] != files['candidate_packet']['original_ref']:
        raise ValueError('selected candidate proof relocation differs from ready-sequence reference')
    candidate_ref = file_binding(files['candidate_packet']['local_ref']['path'])
    if candidate_ref != files['candidate_packet']['local_ref']:
        raise ValueError('selected candidate packet does not match its exact source scan')
    candidate_proof = read_bound(candidate_ref)
    if candidate_proof.get('cases_binding') != selected_candidate.get('cases_binding'):
        raise ValueError('selected candidate source cases differ from remote scan evidence')
    original_packet = read_bound(files['qualified_packet']['local_ref'])
    if original_packet['qualification'] != files['qualification']['original_ref']:
        raise ValueError('qualified packet relocation changes its reference-qualified proof')
    refs = original_packet.get('references') or {}
    role_to_reference = {'global_protection_join': refs.get('global_protection_join'),
                         'selection': refs.get('selection'), 'selected_cases': refs.get('selected_cases')}
    for role, expected in role_to_reference.items():
        if expected != files[role]['original_ref']:
            raise ValueError(f'{role} relocation differs from original qualified packet')
    if original_packet.get('cases') != files['selected_cases']['original_ref']:
        raise ValueError('selected cases relocation differs from packet case identity')
    packet_copy = dict(original_packet)
    packet_copy['qualification'] = {**packet_copy['qualification'],
                                    'path': files['qualification']['local_ref']['path']}
    packet_copy['cases'] = {**packet_copy['cases'], 'path': files['selected_cases']['local_ref']['path']}
    packet_copy['references'] = dict(refs)
    for role in ('global_protection_join', 'selection', 'selected_cases'):
        packet_copy['references'][role] = {**refs[role], 'path': files[role]['local_ref']['path']}
    packet_copy_path = spool / 'assets/ready-packet/relocated' / f'packet-{selected_ordinal:05d}' / 'prepared.json'
    packet_copy_path.parent.mkdir(parents=True, exist_ok=True)
    packet_bytes = (json.dumps(packet_copy, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    if packet_copy_path.exists():
        if packet_copy_path.read_bytes() != packet_bytes:
            raise ValueError('relocated qualified packet already exists with different data')
    else:
        packet_copy_path.write_bytes(packet_bytes)
    packet_copy_ref = file_binding(packet_copy_path)
    relocated_ready = dict(ready)
    relocated_ready['entries'] = [{**selected_entry, 'qualified_packet': packet_copy_ref}]
    ready_path = spool / 'assets/ready-packet/relocated' / f'packet-{selected_ordinal:05d}' / 'ready-packet-sequence.json'
    ready_bytes = (json.dumps(relocated_ready, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    if ready_path.exists():
        if ready_path.read_bytes() != ready_bytes:
            raise ValueError('relocated ready packet already exists with different data')
    else:
        ready_path.write_bytes(ready_bytes)
    ready_ref = file_binding(ready_path)
    result = {'schema': 'natlang.teacher-ready-packet-relocation-result/1',
        'original_ready_packet': original_ready, 'relocation_declaration': relocation_ref,
        'original_qualified_packet': original_packet_ref, 'relocated_qualified_packet': packet_copy_ref,
        'selected_candidate_packet': candidate_ref, 'candidate_scan': relocation['candidate_scan'],
        'source_cases_sha256': source_cases_sha256,
        'original_entry_count': len(ready_entries), 'filtered_entry_count': 1,
        'unique_source_case_match': True,
        'relocated_ready_packet': ready_ref,
        'source_files': [files[role] for role in sorted(files)],
        'path_only_relocation': True, 'training_admission': False}
    result_path = spool / 'assets/ready-packet/relocated' / f'packet-{selected_ordinal:05d}' / 'relocation-result.json'
    write_json_durable(result_path, result)
    if file_binding(relocation_path) != relocation_ref:
        raise ValueError('ready-packet relocation declaration changed during resolution')
    return ready_ref, file_binding(result_path)


def run(*, original_settings: Path, original_settings_sha256: str,
        publisher_config: Path, publisher_sha256: str,
        deployment: Path, deployment_sha256: str, ready_relocation: Path, spool: Path,
        operator_root: Path, batch: int, number: int) -> dict:
    original_ref = _exact_ref(original_settings, original_settings_sha256)
    original = read_bound(original_ref)
    if original.get('schema') != 'natlang.selected-teacher-admission-settings/1':
        raise ValueError('original selected-teacher settings schema changed')
    publisher_ref = _exact_ref(publisher_config, publisher_sha256)
    publisher = read_bound(publisher_ref)
    deployment_ref = _exact_ref(deployment, deployment_sha256)
    deployed = read_bound(deployment_ref)
    continuation = publisher.get('spool_continuation') or {}
    identity = publisher.get('identity') or {}
    if (publisher.get('schema') != 'natlang.teacher-incremental-export-publisher/1'
            or continuation.get('schema') != 'natlang.teacher-spool-continuation/1'
            or continuation.get('previous_publisher', {}).get('sha256') is None
            or publisher.get('directory_root') != str(spool)
            or deployed.get('spool') != str(spool)
            or deployed.get('first_export_batch') != publisher.get('first_export_batch')
            or publisher.get('first_export_batch') != continuation.get('first_export_batch')
            or batch < publisher['first_export_batch']
            or deployed.get('config_sha256') != publisher_ref['sha256']
            or identity.get('teacher_config_sha256') != original['teacher_source']['sha256']
            or identity.get('actor_sha256') != original['actor_remote']['sha256']
            or identity.get('execution_identity_sha256') != original['identity_remote']['sha256']
            or deployed.get('training_admission') is not False):
        raise ValueError('internal spool continuation changes source, teacher authority, or export boundary')
    previous = continuation['previous_publisher']
    previous_ref = file_binding(previous['path'])
    if any(previous_ref.get(key) != previous.get(key) for key in ('sha256', 'bytes')):
        raise ValueError('previous publisher authority binding changed')
    if original.get('spool') == str(spool):
        raise ValueError('internal continuation must retain the prior spool as historical metadata')

    spool = spool.resolve(strict=True)
    ctx = Tranche(batch, number, teacher_namespace=original['teacher_namespace'])
    operator_root = operator_root.resolve(strict=True)
    if operator_root != _FROZEN_OPERATOR_ROOT:
        raise ValueError('source operator root differs from the selected frozen verifier package')
    operators = [operator_root / 'scripts/coreweave/teacher_tranche_recovery.py',
                 operator_root / 'scripts/coreweave/teacher_source_operator.py',
                 operator_root / 'scripts/coreweave/teacher_tranche_admission.py']
    operator_refs = [file_binding(path) for path in operators]
    for path, reference in zip(operators, operator_refs):
        expected_sha, expected_bytes = _FROZEN_OPERATOR_BINDINGS[path.name]
        if reference['sha256'] != expected_sha or reference['bytes'] != expected_bytes:
            raise ValueError(f'frozen teacher source operator changed: {path.name}')
    workdir = spool / 'source-executions' / ctx.corpus_id
    workdir.mkdir(parents=True, exist_ok=True)
    lock_handle = (workdir / 'pipeline.lock').open('a')
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, 'PYTHONPATH': str(operator_root) + ':' + str(operator_root / 'training/neuralese')
           + ':' + str(Path(__file__).resolve().parents[2] / 'training/neuralese')}
    recovery_python = Path.home() / '.local/share/natlang/cw-venv/bin/python'
    if not recovery_python.is_file():
        recovery_python = Path(sys.executable)
    recovery = [str(recovery_python), '-m', 'scripts.coreweave.teacher_tranche_recovery',
        '--spool', str(spool), '--teacher-namespace', ctx.teacher_namespace,
        '--batch', str(batch), '--number', str(number)]
    _run(recovery, cwd=operator_root, env=env, output=workdir / 'recovery.log')
    teacher_source_path = ctx.admission(spool, recovery=True) / 'restored/context/teacher-source.json'
    teacher_source = json.loads(teacher_source_path.read_bytes())
    if (teacher_source.get('actual_teacher_execution_identity') != original['identity_remote']
            or teacher_source.get('actor') != original['actor_remote']
            or teacher_source.get('teacher_configuration_source') != original['teacher_source_remote']
            or not isinstance(teacher_source.get('source_cases_sha256'), str)
            or len(teacher_source['source_cases_sha256']) != 64):
        raise ValueError('recovered teacher source differs from the selected exact actor/runtime/source authority')
    ready_ref, ready_relocation_ref = _relocate_ready_packet(
        original['ready_packet'], ready_relocation, spool, publisher['sandbox_id'],
        teacher_source['source_cases_sha256'])
    source_settings = {**original, 'spool': str(spool),
        'ready_packet': ready_ref,
        'internal_continuation': {
            'schema': 'natlang.teacher-internal-source-continuation/1',
            'original_settings': original_ref, 'publisher_config': publisher_ref,
            'deployment': deployment_ref, 'previous_publisher': previous_ref,
            'ready_packet_relocation': ready_relocation_ref,
            'external_spool_unavailable': continuation['unavailable_directory'],
            'first_export_batch': publisher['first_export_batch'],
            'prefix_semantics': 'Earlier guardian boundaries are retained as deferred custody evidence; this command makes no export or admission claim for them.',
            'training_admission': False,
        }}
    settings_path = spool / 'operator-settings' / (ctx.corpus_id + '.json')
    if settings_path.exists():
        existing = read_bound(file_binding(settings_path))
        if existing != source_settings:
            if (ctx.admission(spool) / 'canonical-source-verification.json').exists():
                raise ValueError('cannot change internal source settings after canonical source verification')
            existing_continuation = existing.get('internal_continuation') or {}
            expected_continuation = source_settings['internal_continuation']
            if any(existing_continuation.get(key) != value for key, value in expected_continuation.items()
                   if key != 'ready_packet_relocation'):
                raise ValueError('internal source settings already exist with a different binding')
            if 'ready_packet_relocation' in existing_continuation:
                old_relocation_ref = existing_continuation['ready_packet_relocation']
                expected_old_path = spool / 'assets/ready-packet/relocated/relocation-result.json'
                if old_relocation_ref.get('path') == str(expected_old_path):
                    old_ready_ref = existing.get('ready_packet') or {}
                    if not str(old_ready_ref.get('path', '')).startswith(
                            str(spool / 'assets/ready-packet/relocated') + '/'):
                        raise ValueError('previous internal ready-packet copy is outside the relocation bundle')
                else:
                    old_relocation = read_bound(old_relocation_ref)
                    if (old_relocation.get('schema') != 'natlang.teacher-ready-packet-relocation-result/1'
                            or old_relocation.get('original_ready_packet') != original['ready_packet']
                            or old_relocation.get('training_admission') is not False):
                        raise ValueError('previous internal ready-packet relocation has a different source binding')
            elif existing.get('ready_packet') != original['ready_packet']:
                raise ValueError('previous internal settings contain an unbound ready-packet relocation')
            write_json_durable(settings_path, source_settings)
    else:
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_durable(settings_path, source_settings)
    settings_ref = file_binding(settings_path)

    execution = {'schema': 'natlang.teacher-internal-source-execution/1',
        'created_at': time.time(), 'tranche': ctx.corpus_id, 'settings': settings_ref,
        'original_settings': original_ref, 'publisher_config': publisher_ref,
        'deployment': deployment_ref, 'previous_publisher': previous_ref,
        'ready_packet_relocation': ready_relocation_ref,
        'operators': operator_refs, 'training_admission': False}
    execution_path = workdir / 'execution.json'
    # The frozen operator inserts the canonical checkout into sys.path itself.
    # Inject this successor's reviewed admission helper before executing its
    # unchanged body, so path-only asset relocation is understood without
    # changing any source, actor, protection, or trajectory checks.
    admission_module = Path(__file__).resolve().parents[2] / 'scripts/coreweave/teacher_tranche_admission.py'
    wrapper_path = workdir / 'run-frozen-source-operator.py'
    wrapper = (
        'import runpy, sys\n'
        f'sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n'
        'from scripts.coreweave import teacher_tranche_admission as admission\n'
        "sys.modules['scripts.coreweave.teacher_tranche_admission'] = admission\n"
        f'runpy.run_path({str(operators[1])!r}, run_name="__main__")\n'
    ).encode()
    wrapper_path.write_bytes(wrapper)
    execution['runtime_overrides'] = {
        'admission_module': file_binding(admission_module),
        'frozen_operator_wrapper': file_binding(wrapper_path),
        'reason': 'execute unchanged frozen source operator with exact reviewed relocation-aware admission verifier',
    }
    write_json_durable(execution_path, execution)
    source = [sys.executable, str(wrapper_path), '--settings', str(settings_path),
        '--settings-sha256', settings_ref['sha256'], '--batch', str(batch), '--number', str(number)]
    _run(source, cwd=operator_root, env=env, output=workdir / 'source.log')
    for path, reference in zip(operators, operator_refs):
        if file_binding(path) != reference:
            raise ValueError('source operator closure changed during execution')
    proof = read_bound(file_binding(ctx.admission(spool) / 'canonical-source-verification.json'))
    zero_positive = verified_zero_positive_source(proof)
    if (proof.get('schema') != 'natlang.teacher-live-tranche-canonical-source-verification/1'
            or proof.get('status') != 'passed' or proof.get('training_admission') is not False
            or (proof.get('rows', 0) < 1 and zero_positive is None)):
        raise ValueError('internal source phase did not establish exact positive full-trajectory verification')
    result = {'schema': 'natlang.teacher-internal-source-result/1',
        'execution': file_binding(execution_path), 'source_proof': file_binding(ctx.admission(spool) / 'canonical-source-verification.json'),
        'ready_packet_relocation': ready_relocation_ref,
        'source': proof['source'], 'batch': batch, 'tranche': number,
        'source_disposition': 'verified-zero-positive-source' if zero_positive else 'positive-source',
        'training_admission': False, 'training_exposure': False,
        'prefix_debt': 'retained-as-deferred-custody; not exported, admitted, or skipped'}
    result_path = workdir / 'source-result.json'
    write_json_durable(result_path, result)
    lock_handle.close()
    return {**result, 'result': file_binding(result_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-settings', type=Path, required=True)
    parser.add_argument('--original-settings-sha256', required=True)
    parser.add_argument('--publisher-config', type=Path, required=True)
    parser.add_argument('--publisher-sha256', required=True)
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--deployment-sha256', required=True)
    parser.add_argument('--ready-packet-relocation', type=Path, required=True)
    parser.add_argument('--spool', type=Path, required=True)
    parser.add_argument('--operator-root', type=Path, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--number', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(run(original_settings=args.original_settings,
        original_settings_sha256=args.original_settings_sha256,
        publisher_config=args.publisher_config, publisher_sha256=args.publisher_sha256,
        deployment=args.deployment, deployment_sha256=args.deployment_sha256,
        ready_relocation=args.ready_packet_relocation,
        spool=args.spool, operator_root=args.operator_root,
        batch=args.batch, number=args.number), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
