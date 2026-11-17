"""A real canonical hosted decision turn reaches shared SFT without tool-loop conversion."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))

from portable_paths import node_executable
import sft_sources
from natlang_neuralese.rl.task_rollouts import rollout_turns


def test_canonical_generated_decision_keeps_its_actual_prompt_and_plain_json_target(tmp_path):
    # The Node integration test owns the real ProgramRecord fixture, local hosted
    # transport, collector execution and cleanup. Reuse that execution rather
    # than inventing another Python/JS collector adapter or a synthetic row.
    evidence_path = tmp_path / 'generated-decision.json'
    env = {**os.environ, 'NATLANG_TEST_FIXTURE_ROW_OUTPUT': str(evidence_path)}
    command = [node_executable(), '--test', '--test-name-pattern',
               'native collector preserves generated decision requests and plain JSON replies',
               str(ROOT / 'ts-host/test/teacher-evaluation-fixture.test.mjs')]
    process = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    assert process.returncode == 0, process.stdout + process.stderr
    assert evidence_path.exists(), 'the native collector decision integration test must execute, not silently skip'
    evidence = json.loads(evidence_path.read_text())
    row = evidence['row']
    assert row['version'] == 'natlang.teacher_trajectory.native/1'
    program = row['task']['program_ir']
    assert program['version'] == 'natlang.program/2'
    assert 'readout: decision' in program['semantics']['files'][program['semantics']['root']]
    assert row['outcome']['accepted'] is True
    assert len(row['trajectory']) == len(evidence['requests']) == 1
    turn = rollout_turns(row)[0]
    transported = evidence['requests'][0]
    # ChatTransport removes internal origin provenance and supplies the two
    # identical reasoning aliases expected by compatible servers
    # (model/chat-completion.ts). No presented content is rewritten.
    wire_messages = [{key: value for key, value in message.items() if key != 'natlang_host_generated'}
                     for message in turn['messages']]
    for message in wire_messages:
        if message['role'] == 'assistant' and isinstance(message.get('reasoning_content'), str):
            message['reasoning'] = message['reasoning_content']
    assert wire_messages == transported['messages']
    assert (turn['tools'] or []) == transported.get('tools', []) == []
    assert turn['messages'][0] == {'role': 'system', 'content': evidence['decision_system_prompt']}
    assert json.loads(turn['target']['content']) == 4
    assert not turn['target'].get('tool_calls')
    assert 'sampling' not in turn
    assert not {'native_decision_capture', 'rl_sampling', 'log_probs', 'logprobs', 'probabilities'} & set(
        row['trajectory'][0]['model_response'])
    semantic = sft_sources.semantic_turn(turn, keep_exact=False)
    assert semantic['messages'] == turn['messages']
    assert semantic['target'] == {'role': 'assistant', 'content': '4'}
    assert not semantic.get('exact')
