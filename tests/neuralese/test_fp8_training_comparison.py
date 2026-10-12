import json

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
