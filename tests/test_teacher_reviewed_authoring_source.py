import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.coreweave.teacher_tranche_admission import file_binding
from scripts.neuralese_data import teacher_file_quality as quality

CRITERION = 'Under policy, decide whether item.notes announce these failures as expected.'


def fixture(tmp_path):
    source = ('const policy="Source-owned policy"; export async function gate(services:'
              '{id:string,notes:string,requests:{status:number}[]}[]):Promise<Record<string,"hold"|"rollback">>{'
              'const out:Record<string,"hold"|"rollback">={};for(const item of services){'
              'const ok=await nl<boolean>`'+CRITERION+'`(item);out[item.id]=ok?"hold":"rollback";}return out;}')
    records = [{'id': 'a', 'notes': '503 failures are expected.', 'requests': [{'status': 503}]},
               {'id': 'b', 'notes': 'Translations updated only.', 'requests': [{'status': 503}]}]
    spec = {'module': 'gate.ts', 'export': 'gate', 'requires': {'nl': True},
            'runs': [{'args': [records], 'expected': {'a': 'hold', 'b': 'rollback'}}],
            'oracle': [{'match': [record['notes']], 'value': value}
                       for record, value in zip(records, ['hold', 'rollback'])]}
    row = {'id': 'teacher:reviewed', 'task': {'program_ir': {'id': 'train:source', 'split': 'train',
        'source_groups': ['source:independent'], 'semantics': {'expected': None, 'authoring': spec}}},
        'trajectory': [{'exact_raw_action': 'unchanged'}], 'provenance': {'actor': 'independent-teacher'},
        'outcome': {'status': 'done', 'accepted': False, 'files': {'gate.ts': source},
            'oracle': {'accepted': False, 'level': 'exact', 'verdict': 'executable-authoring/1'}, 'rejection_reasons': ['files'],
            'quality_pending': [], 'checks': {key: key != 'files' for key in
             ('seeded_failure_requirement', 'expected_status', 'honest_stop', 'effects', 'files',
              'world', 'file_return_consistency', 'answer')}}}
    baseline, _ = quality.authoring_verdict(row)
    assert baseline['ok'] is False and 'oracle has no answer' in baseline['problems'][0]
    row['outcome']['authoring'] = baseline
    raw = tmp_path / 'original.jsonl'; raw.write_text(json.dumps(row)+'\n'); original = file_binding(raw)
    descriptor = {'tag': 'nl', 'typeArguments': ['boolean'], 'template': CRITERION, 'interpolated': False}
    key = quality.canonical_json_sha256_hex(descriptor)
    inventory = {'schema': 'natlang.authored-source-criterion-custody/1', 'training_admission': False,
        'templates': {key: {'criterion': descriptor, 'rows': ['job']}}, 'rows': [{
            'key': 'job', 'original': original, 'row_id': row['id'],
            'program_sha256': quality.canonical_json_sha256_hex(row['task']['program_ir']),
            'authoring_spec': spec, 'captured_file': {'content': source,
            'sha256': hashlib.sha256(source.encode()).hexdigest()}, 'ast': {'tags': [descriptor]}}]}
    ip = tmp_path / 'inventory.json'; ip.write_text(json.dumps(inventory))
    review = {'schema': 'natlang.authored-child-source-semantic-review/1',
        'reviewer': 'independent-root-source-meaning-review', 'training_admission': False,
        'proposition': 'whether-source-notes-announce-these-failures-as-expected',
        'inventory': file_binding(ip), 'approved_templates': {key: {
            'instruction': CRITERION, 'result_kind': 'boolean'}}}
    rp = tmp_path / 'review.json'; rp.write_text(json.dumps(review))
    return row, original, file_binding(rp)


def test_real_original_verdict_all_runs_and_canonical_reconstruction(tmp_path):
    original, binding, review = fixture(tmp_path); before = copy.deepcopy(original)
    derived = quality.continue_authoring_alias_quality(original, binding, source_review=review)
    assert original == before and derived['task'] == original['task']
    assert derived['trajectory'] == original['trajectory'] and derived['provenance'] == original['provenance']
    assert derived['outcome']['files'] == original['outcome']['files']
    assert derived['outcome']['authoring']['results'] == [{'a': 'hold', 'b': 'rollback'}]
    quality.verify_quality_continuation(derived, original, binding)
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import verify_original_row
    assert verify_original_row(derived, binding) == Path(binding['path'])
    derived['outcome']['authoring']['results'][0]['a'] = 'rollback'
    with pytest.raises(ValueError, match='differs'):
        quality.verify_quality_continuation(derived, original, binding)


@pytest.mark.parametrize('change', ['raw', 'review', 'unknown', 'held', 'other_check', 'captured_verdict', 'unknown_oracle'])
def test_raw_authority_review_and_other_checks_cannot_be_waived(tmp_path, change):
    row, binding, review = fixture(tmp_path)
    if change == 'raw': Path(binding['path']).write_text('changed')
    elif change == 'review': Path(review['path']).write_text('{}')
    elif change == 'unknown':
        p = Path(review['path']); v = json.loads(p.read_text()); v['approved_templates'] = {'0'*64: {'instruction': CRITERION, 'result_kind': 'boolean'}}; p.write_text(json.dumps(v)); review = file_binding(p)
    elif change == 'held': row['task']['program_ir']['split'] = 'test'
    elif change == 'other_check': row['outcome']['checks']['answer'] = False
    elif change == 'unknown_oracle': row['outcome']['oracle']['unknown'] = True
    elif change == 'captured_verdict': row['outcome']['authoring']['results'] = [False]
    with pytest.raises(ValueError):
        quality.continue_authoring_alias_quality(row, binding, source_review=review)
