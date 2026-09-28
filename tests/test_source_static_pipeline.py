import hashlib
import json

import pytest

from scripts.create_training_pipeline import recipe
from scripts.render_training_corpus import render_turn
from scripts.audit_training_corpus import assess


def test_recipe_connects_a_checked_static_bundle_without_teacher_collection(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.create_training_pipeline.subprocess.check_output', lambda *a, **k:
                        json.dumps([{'id': 'fixture', 'generated_families': ['fixture'], 'source_families': []}]))
    folder = tmp_path / 'data/teacher/source-backed'
    folder.mkdir(parents=True)
    manifest = {'version': 'natlang.source_static_bundle/1', 'cases': 1}
    for field, name in [('ir', 'train.ir.jsonl'), ('results', 'static.results.jsonl')]:
        raw = b'{}\n'
        (folder / name).write_bytes(raw)
        manifest[field] = {'path': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': 1}
    (folder / 'static.manifest.json').write_text(json.dumps(manifest))
    config = recipe(tmp_path, verified_turns_override=[])
    stages = {stage['id']: stage for stage in config['stages']}
    validate = stages['validate-static-sources']
    assert '/runtime-host/scripts/inline-curriculum/static-bundle-input.mjs' in ' '.join(validate['command'])
    assert '${run}/static-source.turns.jsonl' in stages['prepare']['command']
    assert '${run}/runtime-host/frozen-runtime.json' in validate['inputs']
    assert config['static_source_bundle']['model_calls'] == 0
    assert '--model-id' not in validate['command']
    assert 'validate-static-sources' not in {stage['id'] for stage in recipe(tmp_path, static_bundle=False)['stages']}
    (folder / 'static.results.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='checksum/path mismatch'):
        recipe(tmp_path)


class NoteTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        result = ''
        for message in messages:
            result += f'<{message["role"]}>'
            if message.get('thinking'):
                result += message['thinking'] + '</think>'
            result += message.get('content', '') + '<end>'
        return result + ('<assistant>' if add_generation_prompt else '')


def test_staged_renderer_preserves_source_conversion_and_masks_scripted_notes():
    row = {'id': 'source', 'messages': [{'role': 'user', 'content': 'Edit'}], 'tools': [],
           'target': {'role': 'assistant', 'content': 'action'}, 'training_admission': {'approved': True},
           'teacher_reasoning': 'I use write_file.', 'teacher_reasoning_trained': False,
           'license': 'MIT', 'gold_sources': ['source-gold'], 'split': 'train',
           'provenance': {'source_conversion': {'conversion_scope': 'independent_file_creation'}}}
    result = render_turn(row, NoteTokenizer(), '<end>')
    assert result['completion'][result['completion_masked']:] == 'action<end>'
    assert result['source_conversion'] == row['provenance']['source_conversion']
    assert result['gold_sources'] == row['gold_sources']
    assert result['license'] == 'MIT'


def test_staged_renderer_does_not_train_unbounded_synthetic_reasoning():
    class NoBoundary(NoteTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return super().apply_chat_template(messages, **kwargs).replace('</think>', '')
    row = {'id': 'source', 'messages': [{'role': 'user', 'content': 'Edit'}], 'tools': [],
           'target': {'role': 'assistant', 'content': 'action'}, 'training_admission': {'approved': True},
           'teacher_reasoning': 'note', 'teacher_reasoning_trained': False}
    with pytest.raises(ValueError, match='reasoning boundary'):
        render_turn(row, NoBoundary(), '<end>')


def test_token_audit_counts_masked_notes_as_context_not_supervision():
    tokenizer = lambda value, **kwargs: {'input_ids': list(value.encode())}
    row = {'id': 'masked', 'prompt': 'P', 'completion': 'note</think>action!', 'completion_masked': 12,
           'split': 'train', 'training_admission': {'approved': True}}
    result = assess(row, tokenizer, '!', 100)
    assert result['summary']['reason'] is None
    assert result['summary']['prompt_tokens'] == 13
    assert result['summary']['supervised_tokens'] == 7


def test_recipe_includes_both_default_static_bundles(tmp_path, monkeypatch):
    monkeypatch.setattr('scripts.create_training_pipeline.subprocess.check_output', lambda *a, **k:
                        json.dumps([{'id': 'fixture', 'generated_families': ['fixture'], 'source_families': []}]))
    for name in ('source-backed', 'recovered'):
        folder = tmp_path / 'data/teacher' / name
        folder.mkdir(parents=True)
        manifest = {'version': 'natlang.source_static_bundle/1', 'cases': 1}
        for field in ('ir', 'results'):
            raw = b'{}\n'
            (folder / f'{field}.jsonl').write_bytes(raw)
            manifest[field] = {'path': f'{field}.jsonl', 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': 1}
        (folder / 'static.manifest.json').write_text(json.dumps(manifest))
    config = recipe(tmp_path, verified_turns_override=[])
    stages = {s['id']: s for s in config['stages']}
    assert 'validate-static-sources' in stages and 'validate-static-sources-1' in stages
    assert len(config['static_source_bundles']) == 2
    assert '${run}/static-source-1.turns.jsonl' in stages['prepare']['command']
    path = tmp_path / 'data/teacher/recovered/static.manifest.json'
    assert len(recipe(tmp_path, static_bundle=[path], verified_turns_override=[])['static_source_bundles']) == 1
    with pytest.raises(ValueError, match='duplicate static bundle'):
        recipe(tmp_path, static_bundle=[path, path])
