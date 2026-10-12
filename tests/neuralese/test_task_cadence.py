"""In-run task-success evaluation (owner decision 2026-10-12): spec parsing, serving policy and the qualification gate."""
import pytest

from natlang_neuralese.eval.task_cadence import DEFAULT_CASES, parse_task_eval, task_gate, training_prompt_rendering


def test_spec_defaults_to_a_fixed_sixteen_task_subset_and_validates():
    spec = parse_task_eval('{"packet": {"path": "p.jsonl", "sha256": "' + 'a' * 64 + '"}}')
    assert spec['cases'] == list(DEFAULT_CASES) and len(spec['cases']) == 16
    assert spec['min_successes'] == 1 and spec['workers'] == 4
    assert parse_task_eval(None) is None
    with pytest.raises(ValueError):
        parse_task_eval({'packet': {'path': 'p'}})
    with pytest.raises(ValueError):
        parse_task_eval({'packet': {'path': 'p', 'sha256': 'x'}, 'cases': [1, 1]})


def test_serving_policy_follows_the_rows_the_stage_trains_on():
    assert training_prompt_rendering([{'prompt_rendering': 'program-header'}] * 3 + [{}]) == 'program-header'
    assert training_prompt_rendering([{}, {}]) == 'full'


def test_task_gate_requires_an_eligible_final_evaluation_at_the_floor():
    spec = parse_task_eval({'packet': {'path': 'p', 'sha256': 's'}, 'min_successes': 2})
    assert task_gate(None, None) is None
    assert task_gate(None, spec)['passed'] is False
    assert task_gate({'eligible': True, 'task_successes': 2, 'cases': 16, 'step': 9}, spec)['passed'] is True
    assert task_gate({'eligible': True, 'task_successes': 1, 'cases': 16}, spec)['passed'] is False
    assert task_gate({'eligible': False, 'task_successes': 5, 'cases': 16}, spec)['passed'] is False
