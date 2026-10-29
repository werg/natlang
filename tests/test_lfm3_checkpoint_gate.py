import json

import pytest

from scripts.neuralese_data import lfm3_checkpoint_gate as gate


def run(outcomes, families=None, harness=('p', 't', 19, 0)):
    return {'cases': {str(i): {'family': (families or {}).get(i, f'shape{i % 2}'), 'outcome': outcome}
                      for i, outcome in enumerate(outcomes)}, 'harness': list(harness)}


def test_mcnemar_exact_values():
    assert gate.mcnemar_p(0, 0) == 1.0
    assert gate.mcnemar_p(5, 5) == 1.0
    assert gate.mcnemar_p(6, 0) == pytest.approx(2 / 64)
    assert gate.mcnemar_p(0, 6) == gate.mcnemar_p(6, 0)
    assert gate.mcnemar_p(10, 2) == pytest.approx(2 * (1 + 12 + 66) / 4096)


def test_compare_counts_paired_flips_and_verdicts():
    ok, bad = 'task_success', 'semantic_mismatch'
    base = run([bad] * 8 + [ok] * 4)
    trained = run([ok] * 8 + [ok] * 4)
    result = gate.compare(trained, base)
    assert (result['successes'], result['base_successes']) == (12, 4)
    assert result['gained'] == list(range(8)) and result['lost'] == []
    assert result['verdict'] == 'improved'
    reverse = gate.compare(base, trained)
    assert reverse['verdict'] == 'regressed'
    assert gate.compare(base, base)['verdict'] == 'no-significant-change'


def test_compare_flags_shape_regressions():
    ok, bad = 'task_success', 'budget_exhaustion'
    families = {i: 'a' if i < 3 else 'b' for i in range(20)}
    base = run([ok] * 3 + [bad] * 17, families)
    trained = run([bad] * 3 + [ok] * 17, families)
    result = gate.compare(trained, base)
    assert result['shape_regressions'] == ['a']
    assert result['verdict'] == 'improved-with-shape-regressions'


def test_compare_refuses_mismatched_runs():
    with pytest.raises(RuntimeError, match='different harnesses'):
        gate.compare(run(['task_success']), run(['task_success'], harness=('q', 't', 19, 0)))
    with pytest.raises(RuntimeError, match='different cases'):
        gate.compare(run(['task_success']), run(['task_success'] * 2))


def test_server_args_are_single_sequence_without_prompt_cache_and_apply_logit_bias():
    args = gate.server_args('/bin/llama-server', 'm.gguf', {'chat_template_kwargs': {'preserve_thinking': True},
                                                         'logit_bias': {'125033': -100}}, 8170)
    assert args[args.index('-np') + 1] == '1' and '--no-cache-prompt' in args
    assert args[args.index('--logit-bias') + 1] == '125033-100'
    assert json.loads(args[args.index('--chat-template-kwargs') + 1]) == {'preserve_thinking': True}
    batched = gate.server_args('/bin/llama-server', 'm.gguf', {}, 8170, 'batched')
    slots = int(batched[batched.index('-np') + 1])
    assert slots == gate.SERVING['batched']['workers'] and '--logit-bias' not in batched
    assert int(batched[batched.index('-c') + 1]) == slots * 32768


def test_read_run_refuses_wall_budget_and_mixed_harness(tmp_path):
    probe = tmp_path / 'probe'
    case = probe / 'task-eval/probe/worker-00/case-0000'
    case.mkdir(parents=True)
    tasks = [{'case_index': 0, 'family': 'a', 'outcome': 'task_success', 'model_requests': 3,
              'completion_tokens': 10, 'resource_limit_reason': None}]
    report = {'outcome_counts': {'task_success': 1}, 'task_successes': 1, 'tasks': tasks}
    (probe / 'budget-probe.json').write_text(json.dumps(report))
    row = {'provenance': {'system_prompt_sha256': 's', 'tool_surface_sha256': 't', 'runtime_contract_version': 19,
                          'temperature': 0}}
    (case / 'results.jsonl').write_text(json.dumps(row) + '\n')
    assert gate.read_run(tmp_path)['harness'] == ['s', 't', 19, 0]
    other = probe / 'task-eval/probe/worker-01/case-0001'
    other.mkdir(parents=True)
    (other / 'results.jsonl').write_text(json.dumps({'provenance': {**row['provenance'], 'system_prompt_sha256': 'x'}}))
    with pytest.raises(RuntimeError, match='mixed harness'):
        gate.read_run(tmp_path)
    tasks[0]['resource_limit_reason'] = 'case_wall_budget'
    (probe / 'budget-probe.json').write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match='wall-budget'):
        gate.read_run(tmp_path)


def test_decision_cases_cover_every_packet_row():
    assert gate.decision_cases(gate.DATA / gate.PACKET) == list(range(128))
