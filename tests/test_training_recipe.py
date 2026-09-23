from pathlib import Path
import hashlib
import json

import pytest

from scripts.create_training_pipeline import recipe


def stages_by_id(config):
    return {stage['id']: stage for stage in config['stages']}


def test_default_recipe_is_one_sequential_lora_curriculum(tmp_path):
    config = recipe(tmp_path, python='.venv/bin/python', image='natlang-train-kernels')
    stages = stages_by_id(config)

    assert config['version'] == 'natlang.training_pipeline/1'
    assert [stage['id'] for stage in config['stages']] == [
        'acquire', 'assemble', 'freeze-runtime', 'freeze-failure-corpus', 'observe-source', 'synthetic', 'teacher-seeds', 'prepare',
        'training-readiness', 'render-general', 'audit-general', 'train-general', 'render-coding', 'audit-coding', 'train-coding',
        'teacher', 'materialize-teacher', 'prepare-teacher', 'render-teacher', 'audit-teacher', 'train-teacher',
    ]
    assert '--execute' in stages['observe-source']['command']
    assert '${run}/runtime-host/frozen-runtime.json' in stages['freeze-runtime']['outputs']
    assert '${run}/runtime-host/dist' in stages['freeze-runtime']['outputs']
    assert '${run}/runtime-host/src' in stages['freeze-runtime']['outputs']
    assert '${run}/runtime-host/frozen-runtime.json' in stages['observe-source']['inputs']
    assert '${run}/runtime-host/frozen-runtime.json' in stages['synthetic']['inputs']
    assert any(value.endswith('/runtime-host/scripts/code-corpus/source-cases.mjs')
               for value in stages['observe-source']['command'])
    assert any(value.endswith('/runtime-host/scripts/code-corpus/curriculum.mjs')
               for value in stages['synthetic']['command'])
    assert '${run}/source-observations.jsonl' in stages['synthetic']['command']
    assert '${run}/synthetic/code-proposals.jsonl' in stages['synthetic']['outputs']
    assert '${run}/teacher-programs.jsonl' in stages['teacher-seeds']['outputs']
    assert '${run}/failure-repair-cases.jsonl' in stages['freeze-failure-corpus']['outputs']
    assert '${run}/failure-repair-cases.jsonl' in stages['teacher-seeds']['inputs']
    assert '${run}/failure-repair-cases.jsonl' in stages['teacher-seeds']['command']
    assert '${run}/source-observations.jsonl' in stages['observe-source']['outputs']
    assert '${run}/synthetic/code-proposals.jsonl' in stages['prepare']['inputs']
    for name in ('observe-source', 'synthetic', 'teacher-seeds', 'teacher', 'materialize-teacher'):
        assert '${run}/runtime-host/frozen-runtime.json' in stages[name]['inputs']
        assert not any('/ts-host/dist/' in value or '/ts-host/scripts/' in value or value.endswith('/ts-host/prelude.js')
                       for value in stages[name]['command'])
    assert '--model' in stages['train-general']['command']
    assert stages['train-general']['command'][stages['train-general']['command'].index('--model') + 1] == 'LiquidAI/LFM2.5-350M'
    assert stages['teacher']['command'][stages['teacher']['command'].index('--model-id') + 1] == 'Ternary-Bonsai-2-27B'
    assert stages['teacher']['command'][stages['teacher']['command'].index('--server') + 1] == 'http://127.0.0.1:8081'
    train_general = stages['train-general']
    assert train_general['command'][0:3] == ['docker', 'run', '--rm']
    assert 'natlang-train-kernels' in train_general['command']
    assert train_general['min_free_vram_mib'] == 2048
    for name in ('train-general', 'train-coding', 'train-teacher'):
        command = stages[name]['command']
        assert '--epochs' in command and command[command.index('--epochs') + 1] == '1'
        assert '--no-merge' in command and '--skip-heldout-loss' in command
        assert '--require-audit' in command
        assert '--data-order' in command and command[command.index('--data-order') + 1] == 'source'
        assert not any('baseline' in value or 'benchmark' in value or 'compare' in value for value in command)


def test_recipe_applies_model_teacher_docker_and_training_overrides(tmp_path):
    config = recipe(tmp_path, model='org/student-v2', revision='immutable-commit',
                    image='natlang-train-kernels', teacher_model='org/teacher-27b',
                    teacher_server='http://127.0.0.1:8181', train_args=['--load-in-4bit', '--rank', '64', '--max-len', '4096'],
                    min_free_vram_mib=6144)
    stages = stages_by_id(config)

    render = stages['render-general']['command']
    assert render[render.index('--model') + 1] == 'org/student-v2'
    assert render[render.index('--revision') + 1] == 'immutable-commit'
    train = stages['train-general']['command']
    assert train[train.index('--model') + 1] == 'org/student-v2'
    assert train[train.index('--model-revision') + 1] == 'immutable-commit'
    assert train[-5:] == ['--load-in-4bit', '--rank', '64', '--max-len', '4096']
    assert stages['train-general']['min_free_vram_mib'] == 6144
    assert stages['teacher']['command'][stages['teacher']['command'].index('--model-id') + 1] == 'org/teacher-27b'
    assert stages['teacher']['command'][stages['teacher']['command'].index('--server') + 1] == 'http://127.0.0.1:8181'


@pytest.mark.parametrize('model', ['LiquidAI/LFM2.5-350M', 'org/8B-A1B', '/models/local-student'])
def test_student_selection_propagates_to_every_render_and_training_stage(tmp_path, model):
    revision = 'a' * 40
    stages = stages_by_id(recipe(tmp_path, model=model, revision=revision))
    for phase in ('general', 'coding', 'teacher'):
        for kind, revision_flag in [('render', '--revision'), ('audit', '--revision'), ('train', '--model-revision')]:
            command = stages[f'{kind}-{phase}']['command']
            assert command[command.index('--model') + 1] == model
            assert command[command.index(revision_flag) + 1] == revision
    # Student selection never silently changes the separate teacher.
    teacher = stages['teacher']['command']
    assert teacher[teacher.index('--model-id') + 1] == 'Ternary-Bonsai-2-27B'


def test_lora_stages_chain_checkpoint_adapters_in_order(tmp_path):
    stages = stages_by_id(recipe(tmp_path, image='natlang-train-kernels'))
    coding = stages['train-coding']
    teacher = stages['train-teacher']

    assert coding['command'][coding['command'].index('--init-adapter') + 1] == '${run}/train-general/checkpoint/weights'
    assert teacher['command'][teacher['command'].index('--init-adapter') + 1] == '${run}/train-coding/checkpoint/weights'
    assert '${run}/train-general/checkpoint/state.json' in coding['inputs']
    assert '${run}/train-coding/checkpoint/state.json' in teacher['inputs']
    assert all('--full' not in stage['command'] for stage in stages.values())


def test_recipe_rejects_full_weight_stage_chaining(tmp_path):
    with pytest.raises(ValueError, match='chains LoRA adapters'):
        recipe(tmp_path, train_args=['--full'])


@pytest.mark.parametrize('args', [[], ['--max-len', '2048'], ['--max-len=2048']])
def test_audit_precedes_training_and_uses_effective_context_budget(tmp_path, args):
    config = recipe(tmp_path, train_args=args)
    stages = stages_by_id(config)
    ids = list(stages)
    for phase in ('general', 'coding', 'teacher'):
        audit, train = stages[f'audit-{phase}'], stages[f'train-{phase}']
        assert ids.index(f'render-{phase}') < ids.index(f'audit-{phase}') < ids.index(f'train-{phase}')
        assert audit['command'][audit['command'].index('--max-len') + 1] == ('2048' if args else '8192')
        assert f'${{run}}/{phase}.ready.jsonl' in train['command']
        assert '${run}/training-readiness.json' in train['inputs']
        assert '--gpus' not in stages['training-readiness']['command']


def test_student_cannot_be_changed_only_for_training(tmp_path):
    with pytest.raises(ValueError, match='render, audit and trainer agree'):
        recipe(tmp_path, train_args=['--model=other'])


def test_existing_test_captures_are_inputs_to_observation(tmp_path):
    captured = tmp_path / 'data/direct-code-2026-09-23/fixture/captures.jsonl'
    captured.parent.mkdir(parents=True)
    captured.write_text('')
    observer = stages_by_id(recipe(tmp_path))['observe-source']
    assert str(captured) in observer['inputs']
    assert observer['command'][observer['command'].index('--captures') + 1] == str(captured)
    explicit = tmp_path / 'explicit-calls.jsonl'
    observer = stages_by_id(recipe(tmp_path, captures_override=[explicit]))['observe-source']
    assert str(explicit) in observer['inputs']
    assert str(captured) not in observer['inputs']


def test_legacy_unit_test_turns_require_explicit_override(tmp_path):
    turns = tmp_path / 'data/direct-code-2026-09-23/d3-transpose-pilot-final/native-replay.jsonl.turns.jsonl'
    turns.parent.mkdir(parents=True)
    turns.write_text('')
    prepared = stages_by_id(recipe(tmp_path))['prepare']
    assert str(turns) not in prepared['inputs']
    command = prepared['command']
    assert str(turns) not in command
    explicit = tmp_path / 'extra/native-turns.jsonl'
    prepared = stages_by_id(recipe(tmp_path, verified_turns_override=[explicit]))['prepare']
    assert str(explicit) in prepared['inputs']
    assert str(turns) not in prepared['inputs']


def test_workspace_unit_test_capture_runs_before_coding_and_uses_frozen_runtime(tmp_path):
    workspace = tmp_path / 'app'
    (workspace / 'src').mkdir(parents=True)
    (workspace / 'test').mkdir()
    (workspace / 'package.json').write_text('{"name":"app"}')
    (workspace / 'src/main.mjs').write_text('export function main() { return 1 }')
    (workspace / 'test/main.test.mjs').write_text('')
    case = tmp_path / 'case.json'
    case.write_text(json.dumps({'workspace': str(workspace), 'source': 'src/main.mjs',
                                'test': 'test/main.test.mjs', 'functions': ['main', 'other'],
                                'instruction': 'Return one.'}))
    stages = stages_by_id(recipe(tmp_path, workspace_cases=[case]))
    capture = stages['capture-unit-test-0000']
    assert capture['command'][:2] == ['node', '${run}/runtime-host/scripts/code-corpus/workspace-pilot.mjs']
    assert '--execute' in capture['command']
    assert '--instruction' in capture['command']
    assert capture['command'].count('--function') == 2
    assert '${run}/runtime-host/frozen-runtime.json' in capture['inputs']
    assert '${run}/captured-unit-tests/0000/native-replay.jsonl.turns.jsonl' in stages['prepare']['inputs']
    assert list(stages).index('capture-unit-test-0000') < list(stages).index('prepare')


def test_workspace_unit_test_case_cannot_escape_project(tmp_path):
    workspace = tmp_path / 'app'
    workspace.mkdir()
    (workspace / 'package.json').write_text('{"name":"app"}')
    outside = tmp_path / 'outside.mjs'
    outside.write_text('')
    case = tmp_path / 'case.json'
    case.write_text(json.dumps({'workspace': str(workspace), 'source': '../outside.mjs',
                                'test': '../outside.mjs', 'function': 'main'}))
    with pytest.raises(ValueError, match='workspace-relative'):
        recipe(tmp_path, workspace_cases=[case])


def test_only_verified_workspace_captures_enter_default_coding_inputs(tmp_path):
    corpus = tmp_path / 'data/direct-code-2026-09-23/unit-test-corpus'
    accepted = corpus / 'accepted'
    accepted.mkdir(parents=True)
    turns = accepted / 'native-replay.jsonl.turns.jsonl'
    turns.write_text('{"id":"accepted"}\n')
    (accepted / 'manifest.json').write_text(json.dumps({'native_replay': {
        'accepted': 1, 'turns_sha256': hashlib.sha256(turns.read_bytes()).hexdigest(),
    }}))
    (accepted / 'tasks.jsonl').write_text(json.dumps({'function': {'helpers': [], 'imports': []}}) + '\n')
    (accepted / 'native-replay.jsonl').write_text(json.dumps({'outcome': {'accepted': True}}) + '\n')
    rejected = corpus / 'rejected'
    rejected.mkdir()
    (rejected / 'native-replay.jsonl.turns.jsonl').write_text('')
    (rejected / 'manifest.json').write_text(json.dumps({'native_replay': {'accepted': 0}}))
    prepared = stages_by_id(recipe(tmp_path))['prepare']
    assert str(turns) in prepared['inputs']
    assert str(rejected / 'native-replay.jsonl.turns.jsonl') not in prepared['inputs']
    inlined = corpus / 'inlined'
    inlined.mkdir()
    inlined_turns = inlined / 'native-replay.jsonl.turns.jsonl'
    inlined_turns.write_text('{"id":"inlined"}\n')
    (inlined / 'manifest.json').write_text(json.dumps({'native_replay': {
        'accepted': 1, 'turns_sha256': hashlib.sha256(inlined_turns.read_bytes()).hexdigest(),
    }}))
    (inlined / 'tasks.jsonl').write_text(json.dumps({'function': {'helpers': ['function helper() {}']}}) + '\n')
    inlined_replay = {
        'outcome': {'accepted': True},
        'task': {'program_ir': {'semantics': {'root': {'$lambda': {}}}}},
    }
    (inlined / 'native-replay.jsonl').write_text(json.dumps(inlined_replay) + '\n')
    config = recipe(tmp_path)
    assert str(inlined_turns) not in stages_by_id(config)['prepare']['inputs']
    assert config['unit_test_corpus']['excluded'][0]['reason'] == 'local subfunctions were inlined or imported'
    turns.write_text('tampered\n')
    with pytest.raises(ValueError, match='do not match capture manifest'):
        recipe(tmp_path)
