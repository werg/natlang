import json
import pytest
from natlang_neuralese.rl.rollout_math import grounded_failure_report_credit, reporting_spread
from natlang_neuralese.rl.native_replay import group_episode_advantages


def turns(reason='ReferenceError: missingName is not defined'):
    return [
        {'invocation_id': 'root', 'target': {'tool_calls': [{'id': 'a', 'function': {
            'name': 'eval', 'arguments': '{"code":"missingName()"}'}}]}},
        {'invocation_id': 'root', 'messages': [{'role': 'tool', 'tool_call_id': 'a',
            'content': 'ReferenceError: missingName is not defined'}],
         'target': {'tool_calls': [{'id': 'b', 'function': {'name': 'return_result',
             'arguments': json.dumps({'status': 'failed', 'reason': reason})}}]}}]


def test_credit_requires_exact_observed_linked_error_and_terminal_report():
    assert grounded_failure_report_credit(turns()) == .02
    assert grounded_failure_report_credit(turns('I am overwhelmed')) == 0
    foreign = turns(); foreign[0]['invocation_id'] = 'other'
    assert grounded_failure_report_credit(foreign) == 0
    unlinked = turns(); unlinked[-1]['messages'][0]['tool_call_id'] = 'invented'
    assert grounded_failure_report_credit(unlinked) == 0
    good_value = turns(); good_value[-1]['messages'][0]['content'] = 'The correct answer is 42'
    assert grounded_failure_report_credit(good_value) == 0
    malformed = turns(); malformed[-1]['target']['tool_calls'][0]['function']['arguments'] = 'bad'
    assert grounded_failure_report_credit(malformed) == 0


@pytest.mark.parametrize('jolt', [False, True])
def test_grounded_reporting_is_small_and_never_overrides_completion(jolt):
    def row(quality, credit):
        return {'reward': {'quality': quality, 'execution_efficiency': {'cost': .05,
                                                                        'failure_report_credit': credit}}}
    failed = [row(0, .02), row(0, 0)]
    assert reporting_spread([r['reward'] for r in failed])
    assert group_episode_advantages(failed, jolt=jolt, mode='grpo') == [.01, -.01]
    mixed = group_episode_advantages([row(1, 0), row(0, .02)], jolt=jolt, mode='grpo')
    assert mixed[0] - mixed[1] >= .93


def test_replay_rejects_invented_reporting_credit_before_graph_admission():
    from natlang_neuralese.rl.native_replay import _require_scored_training_episode
    from natlang_neuralese.rl.task_rollouts import execution_efficiency
    capture = turns('I am overwhelmed')
    telemetry = execution_efficiency({'turns': capture})
    telemetry['failure_report_credit'] = .02
    with pytest.raises(ValueError, match='failure-report credit'):
        _require_scored_training_episode({'split': 'train', 'policy': {'checkpoint_sha256': 'a'},
            'turns': capture, 'reward': {'quality': 0, 'execution_efficiency': telemetry}},
            {'checkpoint_sha256': 'a'})
