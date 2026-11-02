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


def test_efficiency_cost_keeps_resolution_at_actual_long_rollout_lengths():
    from natlang_neuralese.rl.rollout_math import execution_efficiency_cost
    costs=[execution_efficiency_cost(model_calls=n,generated_tokens=t,repeated_actions=0)
           for n,t in [(8,6539),(15,13175),(29,19209),(33,20259),(128,1000000)]]
    assert all(a<b for a,b in zip(costs,costs[1:]))
    assert all(0<c<.03 for c in costs)
    assert execution_efficiency_cost(model_calls=1000000,generated_tokens=1000000,repeated_actions=1000000)<.05
    assert execution_efficiency_cost(model_calls=8,generated_tokens=None,repeated_actions=0)==.0075
    assert execution_efficiency_cost(model_calls=33,generated_tokens=20259,repeated_actions=0,
                                    policy='bounded-clipped-v1')==.03
    with pytest.raises(ValueError,match='unknown execution-efficiency'):
        execution_efficiency_cost(model_calls=1,generated_tokens=1,repeated_actions=0,policy='invented')


def test_new_efficiency_policy_is_bound_and_replay_cannot_reinterpret_legacy():
    from natlang_neuralese.rl.task_rollouts import execution_efficiency
    from natlang_neuralese.rl.native_replay import _require_scored_training_episode
    capture=turns();telemetry=execution_efficiency({'turns':capture})
    assert telemetry['policy']=='bounded-rational-v2'
    row={'split':'train','policy':{'checkpoint_sha256':'a'},'turns':capture,
         'reward':{'quality':0,'execution_efficiency':telemetry}}
    # Removing a current policy must not let its new cost pass the historical reader.
    telemetry.pop('policy')
    with pytest.raises(ValueError,match='cost differs'):
        _require_scored_training_episode(row,{'checkpoint_sha256':'a'})
