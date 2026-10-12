"""In-run task-success evaluation (owner decision 2026-10-12): spec parsing, serving policy and the qualification gate."""
import pytest

from natlang_neuralese.eval.task_cadence import DEFAULT_CASES, parse_task_eval, task_gate, training_prompt_rendering


def test_spec_defaults_to_a_fixed_sixteen_task_subset_and_validates():
    spec = parse_task_eval('{"packet": {"path": "p.jsonl", "sha256": "' + 'a' * 64 + '"}}')
    assert spec['cases'] == list(DEFAULT_CASES) and len(spec['cases']) == 16
    assert spec['min_successes'] == 0 and spec['noise_successes'] == 0 and spec['workers'] == 4
    assert spec['reference_successes'] is None
    assert parse_task_eval(None) is None
    with pytest.raises(ValueError):
        parse_task_eval({'packet': {'path': 'p'}})
    with pytest.raises(ValueError):
        parse_task_eval({'packet': {'path': 'p', 'sha256': 'x'}, 'cases': [1, 1]})


def test_serving_policy_follows_the_rows_the_stage_trains_on():
    assert training_prompt_rendering([{'prompt_rendering': 'program-header'}] * 3 + [{}]) == 'program-header'
    assert training_prompt_rendering([{}, {}]) == 'full'


def test_task_gate_is_relative_to_the_base_and_the_previous_evaluation():
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'reference_successes': 3, 'noise_successes': 2})
    final = lambda n, **k: {'eligible': True, 'task_successes': n, 'cases': 16, 'step': 9, **k}
    assert task_gate(None, None) is None
    assert task_gate(None, spec)['passed'] is False
    assert task_gate(final(3), spec)['passed'] is True
    assert task_gate(final(2), spec)['reasons'] == ['below the base model']
    assert task_gate(final(5, previous_successes=8, regressed=True), spec)['passed'] is False
    assert task_gate(final(6, previous_successes=8, regressed=False), spec)['passed'] is True
    assert task_gate({**final(9), 'eligible': False}, spec)['passed'] is False
    unmeasured = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}})
    assert task_gate(final(9), unmeasured)['reasons'] == ['base reference_successes unmeasured for this line']


def test_previous_evaluation_and_regression_are_read_from_the_stage_log(tmp_path):
    import json
    from natlang_neuralese.eval.task_cadence import TaskEval
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [0], 'noise_successes': 1})
    evaluator = TaskEval.__new__(TaskEval)
    evaluator.spec, evaluator.out = spec, tmp_path
    rows = [{'step': 100, 'eligible': True, 'task_successes': 5, 'packet_sha256': 's', 'case_indices': [0]},
            {'step': 200, 'eligible': False, 'task_successes': 9, 'packet_sha256': 's', 'case_indices': [0]},
            {'step': 150, 'eligible': True, 'task_successes': 7, 'packet_sha256': 'other', 'case_indices': [0]}]
    (tmp_path / 'task-eval.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    assert evaluator.previous(300)['step'] == 100
    assert evaluator.previous(100) is None


def test_the_first_evaluation_repeats_to_measure_noise_and_later_ones_reuse_it(tmp_path):
    from natlang_neuralese.eval.task_cadence import TaskEval
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'cases': [0], 'noise_repeats': 3})
    evaluator = TaskEval.__new__(TaskEval)
    evaluator.spec, evaluator.out, evaluator.root, evaluator.rows = spec, tmp_path, tmp_path / 'task-eval', [{}]
    evaluator.prompt_rendering = 'full'
    counts = iter([4, 6, 5, 3])
    calls = []

    def execute(run):
        run.mkdir(parents=True, exist_ok=True)
        calls.append(run.name)
        n = next(counts)
        return {'task_successes': n, 'task_success_rate': n / 16, 'outcome_counts': {}, 'complete': True,
                'eligible_for_best': True, 'cases': []}, [0]
    evaluator._execute = execute
    first = evaluator.run(10)
    assert calls == ['step-0000010', 'step-0000010-repeat1', 'step-0000010-repeat2']
    assert first['repeat_noise'] == 2 and first['noise_successes'] == 2 and first['task_successes'] == 4
    second = evaluator.run(20)  # 3 successes: within noise of 4, no repeat
    assert calls[-1] == 'step-0000020' and second['repeat_noise'] is None
    assert second['noise_successes'] == 2 and second['previous_successes'] == 4 and second['regressed'] is False
