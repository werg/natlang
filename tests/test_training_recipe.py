from pathlib import Path

import pytest

from scripts.create_training_pipeline import recipe


def stages_by_id(config):
    return {stage['id']: stage for stage in config['stages']}


def test_default_recipe_is_one_sequential_lora_curriculum(tmp_path):
    config = recipe(tmp_path, python='.venv/bin/python', image='natlang-train-kernels')
    stages = stages_by_id(config)

    assert config['version'] == 'natlang.training_pipeline/1'
    assert [stage['id'] for stage in config['stages']] == [
        'acquire', 'assemble', 'freeze-runtime', 'observe-source', 'synthetic', 'teacher-seeds', 'prepare',
        'training-readiness', 'render-general', 'audit-general', 'train-general', 'render-coding', 'audit-coding', 'train-coding',
        'teacher', 'materialize-teacher', 'prepare-teacher', 'render-teacher', 'audit-teacher', 'train-teacher',
    ]
    assert '--execute' in stages['observe-source']['command']
    assert '${run}/runtime-host/frozen-runtime.json' in stages['freeze-runtime']['outputs']
    assert '${run}/runtime-host/dist' in stages['freeze-runtime']['outputs']
    assert '${run}/runtime-host/frozen-runtime.json' in stages['observe-source']['inputs']
    assert '${run}/runtime-host/frozen-runtime.json' in stages['synthetic']['inputs']
    assert any(value.endswith('/runtime-host/scripts/code-corpus/source-cases.mjs')
               for value in stages['observe-source']['command'])
    assert any(value.endswith('/runtime-host/scripts/code-corpus/curriculum.mjs')
               for value in stages['synthetic']['command'])
    assert '${run}/source-observations.jsonl' in stages['synthetic']['command']
    assert '${run}/synthetic/code-proposals.jsonl' in stages['synthetic']['outputs']
    assert '${run}/teacher-programs.jsonl' in stages['teacher-seeds']['outputs']
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
