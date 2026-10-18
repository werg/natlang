import hashlib
import json
import subprocess
import shutil
import sys
from pathlib import Path

from scripts.neuralese_data.build_execution_recovery_curriculum import (
    VARIANTS, graph_rows, short_worlds)


def service_rows(service):
    return json.loads(service.split('const ROWS:Row[] = ', 1)[1].split(';\n', 1)[0])


def test_ordinary_curriculum_is_short_broad_exact_train_and_recoverable():
    rows = list(short_worlds())
    assert len(rows) == 24
    assert rows == list(short_worlds())
    assert len({row['id'] for row in rows}) == 24
    assert {row['curriculum']['variant'] for row in rows} == set(VARIANTS)
    assert len({group for row in rows for group in row['source_groups']}) == 8
    assert all(row['split'] == 'train' for row in rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row['source_groups'][0], []).append(row)
        semantics = row['semantics']
        namespace, service = next(iter(semantics['services'].items()))
        records = service_rows(service)
        assert len(records) == 5
        assert len({record['id'] for record in records}) == 5
        assert 'page_count():number {return 2;}' in service
        assert 'ROWS.slice((n-1)*3,n*3)' in service
        assert len(row['curriculum']['reference']['children']) == 5
        policy = semantics['inputs']['policy']
        selected = [record for record in records if record['category'] == policy['category']
                    and not record['blocked'] and (record['waiver'] or
                    (record['active'] and record['value'] >= policy['minimum']))]
        expected = {'ids': sorted(record['id'] for record in selected),
                    'total': sum(record['value'] for record in selected), 'count': len(selected)}
        assert semantics['expected'] == json.loads(semantics['expected_files']['summary.json']) == expected
        assert row['curriculum']['reference']['root'][1][1]['value'] == expected
        for record, child in zip(records, row['curriculum']['reference']['children'], strict=True):
            assert child['match'] == ['select_record(', record['id']]
            assert child['calls'][-1] == ['return_result',
                {'status': 'success', 'value': record in selected}]
        assert semantics['failure_seed']['code']
        assert semantics['folder_files']['summary.json'] == '{}\n'
        assert semantics['expected_files']['summary.json'].endswith('\n')
        assert 'writeText' in row['curriculum']['reference']['root'][0][1]['code']
        assert 'readText' in row['curriculum']['reference']['root'][0][1]['code']
        if row['curriculum']['variant'] == 'page-range-recovery':
            assert semantics['failure_seed']['code'] == f'return {namespace}.page(0);'
        if row['curriculum']['variant'] == 'reserved-service-binding':
            assert semantics['failure_seed']['code'].startswith(f'const {namespace} = ')
        if row['curriculum']['variant'] == 'typed-result-recovery':
            assert 'total:"not-a-number"' in semantics['failure_seed']['code']
    assert all({row['curriculum']['variant'] for row in variants} == set(VARIANTS)
               for variants in grouped.values())


def test_private_state_packet_requires_complete_sampled_graph_topology():
    rows = list(graph_rows())
    assert len(rows) == 3
    assert len({row['id'] for row in rows}) == 3
    for row in rows:
        contract = row['curriculum']['trajectory_contract']
        assert contract['authored_root_eval'] is True
        assert contract['min_child_invocations'] == len(row['curriculum']['reference']['children'])
        assert contract['min_state_edges_from_children'] > 0
        assert 'Neuralese<' in row['curriculum']['reference']['root'][0][1]['code']
        assert row['semantics']['files']['main.nl'].startswith('---\nargs: {}\nreturns: ')
        assert row['split'] == 'train'
        assert 'failure_seed' not in row['semantics']
        assert row['curriculum']['reference']['root'][1][1]['value'] == row['semantics']['expected']
    assert [row['curriculum']['trajectory_contract']['min_state_edges_from_children']
            for row in rows] == [1, 1, 2]


def test_source_builder_emits_two_separate_nonadmitted_sampling_modes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    protection = Path('/home/werg/data/dgxops/acquired-connected-source-protection-20261017-d13-d14-d32-v3/protection-manifest.json')
    if not protection.is_file():
        return
    out = tmp_path / 'curriculum'
    subprocess.run([sys.executable, str(root / 'scripts/neuralese_data/build_execution_recovery_curriculum.py'),
                    '--out', str(out), '--protection', str(protection)], check=True,
                   cwd=root, capture_output=True, text=True)
    prepared = json.loads((out / 'prepared.json').read_text())
    assert prepared['teacher_responses_generated'] == prepared['model_calls'] == 0
    assert prepared['training_admission'] is False and prepared['held_content_used'] is False
    ordinary = prepared['packets']['ordinary']; graph = prepared['packets']['private-state']
    assert ordinary['rows'] == 24 and ordinary['collection_sampling'] == {'group': 2, 'root_mode': 'student'}
    assert graph['rows'] == 3 and graph['collection_sampling'] == {'group': 16, 'root_mode': 'authored'}
    for packet in (ordinary, graph):
        assert hashlib.sha256(Path(packet['cases']['path']).read_bytes()).hexdigest() == packet['cases']['sha256']
        proof = json.loads(Path(packet['candidate_packet']['path']).read_text())
        assert proof['collection_sampling'] == packet['collection_sampling']
        assert proof['training_admission'] is False and proof['teacher_calls'] == 0
