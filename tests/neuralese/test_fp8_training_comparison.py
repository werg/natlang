import json
import subprocess
from pathlib import Path

from natlang_neuralese.eval import fp8_training_comparison as comparison
from natlang_neuralese.eval.fp8_training_comparison import _profile_identity, summarize_run


def test_short_and_long_profiles_may_change_only_context_lengths():
    short = {'id': 'short', 'stage': {'id': 'fit', 'kind': 'neuralese_direct'},
             'inputs': {'text_data': {'path': 'data', 'sha256': 'a' * 64}},
             'parameters': {'tokens': 1024, 'target_tokens': 128, 'prefix_tokens': 32,
                            'batch': 4, 'backbone_lr': 1e-5},
             'outputs': {'directory': 'short'}, 'runtime': {'image': 'sha256:x'},
             'lineage': {'scope': 'same scope', 'evidence': []}}
    long = json.loads(json.dumps(short))
    long['id'] = 'long'
    long['parameters'].update(tokens=8192, target_tokens=512, prefix_tokens=64)
    long['outputs']['directory'] = 'long'
    assert _profile_identity(short) == _profile_identity(long)
    long['inputs']['text_data']['sha256'] = 'b' * 64
    assert _profile_identity(short) != _profile_identity(long)


def test_summary_uses_updates_after_three_cold_start_steps(tmp_path):
    output = tmp_path / 'run'
    stage = output / 'direct_sft'
    stage.mkdir(parents=True)
    rows = [
        {'step': i + 1, 'seconds': seconds, 'supervised_tokens_per_second': 1000 / seconds,
         'cuda_peak_allocated_gb': 4.0 + i / 10, 'cuda_peak_reserved_gb': 5.0,
         'observed_incremental_peak_bytes': 100 + i}
        for i, seconds in enumerate((10.0, 8.0, 6.0, 4.0, 2.0))
    ]
    (stage / 'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (stage / 'eval.jsonl').write_text(json.dumps({'step': 5, 'loss': 1.2}) + '\n')
    result = summarize_run(output)
    assert result['cold_first_update_seconds'] == 10.0
    assert result['warm_updates_used'] == 2
    assert result['warm_median_update_seconds'] == 3.0
    assert result['peak_cuda_allocated_gb'] == 4.4
    assert result['last_evaluation'] == [{'step': 5, 'loss': 1.2}]


def test_variant_separates_wrapper_and_container_logs_and_parses_container_plan(tmp_path, monkeypatch):
    artifact_root = tmp_path / 'artifacts'
    output_root = artifact_root / 'comparison'
    output = output_root / 'outputs' / 'short' / 'bf16'
    output.mkdir(parents=True)
    recipe_path = output_root / 'recipes' / 'variant.json'
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text('{}')
    recipe = {'id': 'paired-short-bf16', 'outputs': {'directory': str(output)}}
    variant = {'recipe': recipe, 'recipe_path': str(recipe_path), 'recipe_sha256': 'a' * 64}

    def no_gpu_sampling(_stop, _destination, _interval):
        return None

    def fake_run(argv, **kwargs):
        metadata_path = Path(argv[argv.index('--launch-metadata') + 1])
        container_path = Path(argv[argv.index('--log') + 1])
        wrapper_path = output_root / 'launches' / 'paired-short-bf16.wrapper.log'
        assert container_path != wrapper_path
        assert container_path != metadata_path.with_suffix('.log') or container_path.name.endswith('.container.log')
        container_path.parent.mkdir(parents=True, exist_ok=True)
        container_path.write_text(json.dumps({'event': 'execution_plan', 'step': 1,
                                              'context_tokens': 1024, 'target_tokens': 128}) + '\n')
        stage = output / 'direct_sft'
        stage.mkdir()
        (stage / 'train.jsonl').write_text(''.join(json.dumps({'step': i, 'seconds': 1.0,
                                                                'supervised_tokens_per_second': 100.0}) + '\n'
                                           for i in range(1, 6)))
        (stage / 'eval.jsonl').write_text('{}\n')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(comparison, '_gpu_sample', no_gpu_sampling)
    monkeypatch.setattr(comparison.subprocess, 'run', fake_run)
    result = comparison.run_variant(variant, output_root, 'cpu', artifact_root, None, 1.0)
    assert result['execution_plans'] == [{'step': 1, 'source_documents': None, 'mode': None,
                                         'checkpoint_layers': None, 'context_tokens': 1024,
                                         'target_tokens': 128, 'activation_geometry_bytes': None}]
    assert result['launch_log'].endswith('.wrapper.log')
    assert result['container_log'].endswith('.container.log')
