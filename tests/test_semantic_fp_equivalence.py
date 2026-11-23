"""scripts/semantic_fp/equivalence.py: version rows, seeds that pair, per-case reports, the 07 §4.2 acceptance rule."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('equivalence', REPO / 'scripts/semantic_fp/equivalence.py')
eq = importlib.util.module_from_spec(spec)
sys.modules['equivalence'] = eq
spec.loader.exec_module(eq)


def design_row(case='001-a', root='main.nl', oracle='exact', split='dev'):
    return {'id': f'design-briefs-v3:001:{case}', 'family': 'design_brief_001', 'split': split,
            'curriculum': {'rl': {'level': 0}},
            'semantics': {'root': root, 'files': {'main.nl': 'x', 'types.ts': 'export type A = 1;'},
                          'inputs': {}, 'expected': 'pass', 'oracle': oracle}}


def test_seed_depends_on_case_and_sample_only():
    assert eq.case_seed(4200000, 'a', 0) == eq.case_seed(4200000, 'a', 0)
    assert eq.case_seed(4200000, 'a', 0) != eq.case_seed(4200000, 'a', 1)
    assert eq.case_seed(4200000, 'a', 0) != eq.case_seed(4200000, 'b', 0)
    assert 4200000 <= eq.case_seed(4200000, 'z', 7) < 4200000 + eq.SEED_SPAN


def test_program_files_take_sources_only(tmp_path):
    (tmp_path / 'main').mkdir()
    (tmp_path / 'main.nl').write_text('---\nreturns: string\n---\nSay hi.')
    (tmp_path / 'main' / 'helper.ts').write_text('export const f = 1;')
    (tmp_path / 'main' / 'helper.d.ts').write_text('export declare const f: number;')
    (tmp_path / 'natlang.json').write_text('{}')
    (tmp_path / 'DECOMPOSITION.md').write_text('notes')
    (tmp_path / 'node_modules').mkdir()
    (tmp_path / 'node_modules' / 'x.ts').write_text('')
    assert sorted(eq.program_files(tmp_path)) == ['main.nl', 'main/helper.ts', 'natlang.json']
    (tmp_path / 'empty').mkdir()
    (tmp_path / 'empty' / 'README.md').write_text('no sources')
    with pytest.raises(ValueError):
        eq.program_files(tmp_path / 'empty')


def test_with_version_keeps_id_oracle_and_marks_split():
    row = design_row(oracle={'level': 'judged', 'rubric': 'r'})
    files = {'run.nl': 'y', 'run/helper.ts': 'export const h = 1;'}
    out = eq.with_version(row, 'design-brief:001', 'migrated', files, {'main.nl': 'run.nl'})
    assert out['id'] == row['id'] and out['semantics']['oracle'] == row['semantics']['oracle']
    assert out['semantics']['root'] == 'run.nl' and out['semantics']['files'] == files
    assert out['split'] == 'train'
    marker = out['equivalence']
    assert marker['split'] == 'dev' and marker['kind'] == 'workflow' and marker['version'] == 'migrated'
    assert marker['source_sha256'] == eq.files_sha256(files)
    with pytest.raises(ValueError):
        eq.with_version(row, 'design-brief:001', 'migrated', files)   # main.nl is gone and no root map names run.nl


def test_kinds():
    assert eq.case_kind(design_row()) == 'gate'
    assert eq.case_kind(design_row(oracle={'level': 'judged'})) == 'workflow'
    world = {'semantics': {'trusted_exact_verifier': {'path': 'v.mjs'}, 'oracle': None}}
    assert eq.case_kind(world) == 'world' and eq.oracle_level(world) == 'verifier'
    assert eq.program_of_design_row(design_row()) == 'design-brief:001'


def test_judge_pending_only_for_returned_unjudged_rubric_cases():
    done = {'status': 'done', 'oracle': {'accepted': False, 'needs_review': True}}
    assert eq.judge_pending('judged', done, judged=False)
    assert not eq.judge_pending('judged', done, judged=True)
    assert not eq.judge_pending('exact', done, judged=False)
    assert not eq.judge_pending('judged', {'status': 'error', 'oracle': None}, judged=False)
    assert not eq.judge_pending('judged', None, judged=False)


def case(case_id, passes, samples=4, program='design-brief:001', version='original', seeds=None, calls=10.0):
    return {'case_id': case_id, 'program': program, 'version': version, 'kind': 'gate', 'samples': samples,
            'ran': samples, 'scored': samples, 'passes': passes, 'judge_pending': 0,
            'seeds': seeds or [1, 2, 3, 4][:samples], 'calls_mean': calls, 'prompt_tokens_mean': 100.0,
            'completion_tokens_mean': 10.0, 'wall_seconds_mean': 5.0}


def test_compare_rule():
    original = [case('a', 4), case('b', 2), case('c', 0)]
    # One case of slack, and every 4/4 case stays 4/4.
    ok = eq.compare(original, [case('a', 4, version='m', calls=3.0), case('b', 0, version='m'), case('c', 1, version='m')])
    assert ok[0]['accepted'], ok[0]['problems']
    assert ok[0]['expected_passing_first'] == 1.5 and ok[0]['expected_passing_second'] == 1.25
    assert ok[0]['cost']['calls_mean'] == {'first': 10.0, 'second': 7.67}
    lost_full = eq.compare(original, [case('a', 3), case('b', 4), case('c', 4)])
    assert not lost_full[0]['accepted'] and 'a:' in lost_full[0]['problems'][0]
    worse = eq.compare(original, [case('a', 4), case('b', 0), case('c', 0), ])
    assert worse[0]['accepted']   # 1.0 >= 1.5 - 1
    much_worse = eq.compare([case('a', 4), case('b', 4), case('c', 4)], [case('a', 4), case('b', 1), case('c', 1)])
    assert not much_worse[0]['accepted']
    unpaired = eq.compare(original, [case('a', 4, seeds=[9, 9, 9, 9]), case('b', 2), case('c', 0)])
    assert any('seeds differ' in p for p in unpaired[0]['problems'])


def test_summary_counts_programs_without_a_pass():
    rows = [case('a', 0), case('b', 1), case('c', 0, program='sol:x'), {**case('d', 0, program='astra:y'), 'scored': 0}]
    out = eq.summary(rows)
    assert out['programs'] == 3 and out['programs_with_a_passing_case'] == 1
    assert out['programs_scored_without_a_pass'] == ['sol:x']
    assert out['programs_without_a_scored_sample'] == ['astra:y']
    gate = next(t for t in out['by_corpus_kind'] if t['corpus'] == 'design-brief')
    assert gate['passes'] == 1 and gate['scored'] == 8 and gate['pass_rate'] == 0.125


def test_report_reads_a_run(tmp_path):
    """A two-sample run of one gate case: the report pairs queue entries, journals, rewards and results."""
    run = tmp_path / 'run'
    run.mkdir()
    row = eq.with_version(design_row(), 'design-brief:001', 'original')
    (run / 'packet.jsonl').write_text(json.dumps(row) + '\n')
    (run / 'run.json').write_text(json.dumps({'policy_id': 'p' * 64, 'code': {'commit': 'c'}, 'judge_model': None}))
    entries, rewards = [], []
    for sample in range(2):
        out = run / f'row-000/sample-{sample:02d}'
        out.mkdir(parents=True)
        key = f's7-task-000-{sample:02d}'
        entries.append({'key': key, 'index': 0, 'group': 0, 'sample': sample, 'seed': eq.case_seed(1, row['id'], sample),
                        'output': str(out / 'results.jsonl')})
        steps = [{'model_response': {'prompt_tokens': 100, 'completion_tokens': 7}}] * (sample + 1)
        (out / 'results.jsonl').write_text(json.dumps({'trajectory': steps, 'outcome': {
            'status': 'done', 'oracle': {'accepted': sample == 0}}}) + '\n')
        (run / 'workers').mkdir(exist_ok=True)
        (run / 'workers' / f'{key}.journal.jsonl').write_text(json.dumps({'event': 'finish', 'elapsed_seconds': 10.0 * (sample + 1)}) + '\n')
        rewards.append({'group': 0, 'sample': sample, 'outcome': 'task_success' if sample == 0 else 'semantic_mismatch',
                        'reward': 1.0 if sample == 0 else 0.0})
    (run / 'sealed-queue.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    (run / 'rewards.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rewards))
    [out] = eq.report(run)
    assert out['passes'] == 1 and out['scored'] == 2 and out['ran'] == 2 and out['judge_pending'] == 0
    assert out['calls_mean'] == 1.5 and out['prompt_tokens_mean'] == 150.0 and out['wall_seconds_mean'] == 15.0
    assert out['program'] == 'design-brief:001' and out['split'] == 'dev'
