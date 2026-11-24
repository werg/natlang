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
    (tmp_path / 'services.d.ts').write_text('export declare const api: { ping(): void };')
    (tmp_path / 'main.d.nl.ts').write_text('declare const main: () => Promise<string>; export default main;')
    (tmp_path / 'natlang.json').write_text('{}')
    (tmp_path / 'DECOMPOSITION.md').write_text('notes')
    (tmp_path / 'node_modules').mkdir()
    (tmp_path / 'node_modules' / 'x.ts').write_text('')
    assert sorted(eq.program_files(tmp_path)) == ['main.d.nl.ts', 'main.nl', 'main/helper.ts', 'natlang.json', 'services.d.ts']
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
    (run / 'run.json').write_text(json.dumps({'policy_id': 'p' * 64, 'code': {'git_commit': 'abc123'}, 'engine': 'sglang', 'judge_model': None}))
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
    assert out['code'] == 'abc123' and out['engine'] == 'sglang'
    # A requeued attempt (journal gone) no longer counts its old reward.
    (run / 'workers' / 's7-task-000-01.journal.jsonl').unlink()
    [out] = eq.report(run)
    assert out['scored'] == 1 and out['ran'] == 1 and out['passes'] == 1


def test_requeue_forgets_attempts_without_a_result(tmp_path):
    run = tmp_path / 'run'
    (run / 'workers').mkdir(parents=True)
    entries = []
    for key, text in (('a', '{"x": 1}\n'), ('b', ''), ('c', None)):
        out = run / f'row-{key}' / 'sample-00' / 'results.jsonl'
        out.parent.mkdir(parents=True)
        if text is not None:
            out.write_text(text)
        (run / 'workers' / f'{key}.journal.jsonl').write_text(json.dumps({'event': 'finish', 'key': key}) + '\n')
        entries.append({'key': key, 'output': str(out)})
    (run / 'sealed-queue.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    assert eq.requeue(run) == ['b', 'c']
    assert (run / 'workers' / 'a.journal.jsonl').exists() and not (run / 'workers' / 'b.journal.jsonl').exists()
    assert not (run / 'row-b').joinpath('sample-00').exists()


def test_requeue_timeouts_only_when_asked(tmp_path):
    run = tmp_path / 'run'
    (run / 'workers').mkdir(parents=True)
    out = run / 'row-a' / 'sample-00' / 'results.jsonl'
    out.parent.mkdir(parents=True)
    out.write_text('{"x": 1}\n')
    (run / 'workers' / 'a.journal.jsonl').write_text(json.dumps({'event': 'finish', 'key': 'a', 'status': 'timeout'}) + '\n')
    (run / 'sealed-queue.jsonl').write_text(json.dumps({'key': 'a', 'output': str(out)}) + '\n')
    assert eq.requeue(run) == []
    assert eq.requeue(run, timeouts=True) == ['a']


def test_compare_engine_partial_zero_scored_and_float_epsilon():
    c_sglang = {**case('a', 4), 'engine': 'sglang'}
    c_native = {**case('a', 4), 'engine': 'native'}
    cross = eq.compare([c_sglang], [c_native])
    assert not cross[0]['accepted'] and any('another engine' in p for p in cross[0]['problems'])
    allowed = eq.compare([c_sglang], [c_native], allow_other_engine=True)
    assert allowed[0]['accepted']

    # Partial run rejected
    partial = {**case('a', 2), 'scored': 2, 'samples': 4}
    res_partial = eq.compare([case('a', 4)], [partial])
    assert not res_partial[0]['accepted'] and any('partial run' in p for p in res_partial[0]['problems'])

    # Zero scored cases rejected (e.g. all judge_pending)
    pending = {**case('a', 0), 'scored': 0, 'judge_pending': 4, 'samples': 4}
    res_zero = eq.compare([pending], [pending])
    assert not res_zero[0]['accepted'] and any('no scored cases' in p for p in res_zero[0]['problems'])

    # Float epsilon: 3 cases with 3 samples each: rate_a = 1/3 + 1/3 + 1/3 = 1.0, rate_b = 0.0, slack = 1.0
    # 0.0 + 1e-9 >= 1.0 - 1.0 even with float imprecision
    orig = [case('a', 1, samples=3), case('b', 1, samples=3), case('c', 1, samples=3)]
    cand = [case('a', 0, samples=3), case('b', 0, samples=3), case('c', 0, samples=3)]
    eps_res = eq.compare(orig, cand, slack_cases=1.0)
    assert eps_res[0]['accepted']


def test_measure_restores_record_result_on_exception(tmp_path, monkeypatch):
    import types
    from natlang_neuralese.rl import held_eval
    if 'natlang_neuralese.common.sampling' not in sys.modules:
        try:
            __import__('natlang_neuralese.common.sampling')
        except ModuleNotFoundError:
            fake_sampling = types.ModuleType('natlang_neuralese.common.sampling')
            fake_sampling.RL_TOP_P = 0.95
            fake_sampling.sampling_law_of = lambda top_p: {'top_p': top_p}
            monkeypatch.setitem(sys.modules, 'natlang_neuralese.common.sampling', fake_sampling)
    cases_dir = tmp_path / 'cases'
    cases_dir.mkdir()
    (cases_dir / 'cases.jsonl').write_text(json.dumps(eq.with_version(design_row(), 'design-brief:001', 'original')) + '\n')
    (cases_dir / 'manifest.json').write_text(json.dumps({'tasks': [{'id': 't1'}], 'cases': {'path': 'cases.jsonl'}}))
    monkeypatch.setattr(held_eval, '_read_info', lambda client: {'rl_policy_binding': {'policy_id': 'p1', 'engine': 'sglang'}, 'context': held_eval.CONTEXT_TOKENS})
    monkeypatch.setattr(held_eval, 'policy_id_of', lambda binding: 'p1')
    monkeypatch.setattr(held_eval, 'engine_of', lambda binding: 'sglang')
    def fake_entries(spec):
        (Path(spec['dir']) / 'packet.jsonl').write_text('')
        return []
    monkeypatch.setattr(held_eval, '_entries', fake_entries)
    orig = held_eval.record_result
    monkeypatch.setattr(held_eval, 'run', lambda out: (_ for _ in ()).throw(RuntimeError('boom')))
    with pytest.raises(RuntimeError, match='boom'):
        eq.measure(cases_dir, tmp_path / 'run', [('http://127.0.0.1:9', 1)], samples=1)
    assert held_eval.record_result is orig
