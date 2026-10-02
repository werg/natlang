import json
from scripts import create_training_pipeline as pipeline


def test_default_recipe_generates_large_optimizer_curriculum_and_feeds_preparation(tmp_path, monkeypatch):
    # Isolate this contract from the separately tested inline-family registry.
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *args, **kwargs: '[{"id":"interpreter","generated_families":["minimal"],"source_families":[]}]')
    (tmp_path/'training').mkdir()
    (tmp_path/'training/data_sources.json').write_text(json.dumps({'decisions':[], 'replacements':{}, 'required_default_inputs':[]}))
    config = pipeline.recipe(tmp_path, teacher_results_override=[])
    stages = {stage['id']: stage for stage in config['stages']}
    build = stages['build-optimizer-curriculum']
    generate = stages['generate-optimizer-curriculum']
    assert '12' in build['command']  #16families x12variants
    assert '--optimizer-model' in generate['command'] and 'gpt-6-luna' in generate['command']
    assert '--executor-endpoint' in generate['command']
    output = '${run}/optimizer-curriculum/generation/verified-turns.jsonl'
    assert output in generate['outputs']
    assert output in stages['prepare']['inputs'] and output in stages['prepare']['command']
    assert config['stages'].index(generate) < config['stages'].index(stages['prepare'])
    assert '${run}/runtime-host/frozen-runtime.json' in generate['inputs']
