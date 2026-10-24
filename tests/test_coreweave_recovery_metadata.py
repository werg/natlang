import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'coreweave'))
import recovery_metadata as recovery


def test_source_reconstruction_requires_exact_prior_bytes_and_keeps_priority(tmp_path):
    document = {'schema': 'natlang.native-graph-sources/1', 'sources': [
        {'kind': 'native-graphs', 'path': '/old/one.json', 'sha256': 'a' * 64},
        {'kind': 'native-graphs', 'path': '/old/two.json', 'sha256': 'b' * 64, 'training_priority': 4}]}
    raw = json.dumps(document, indent=2, sort_keys=True).encode()
    receipt = {'path': '/old/root.json', 'sha256': hashlib.sha256(raw).hexdigest(), 'sources': [
        {'path': '/old/one.json', 'sha256': 'a' * 64}, {'path': '/old/two.json', 'sha256': 'b' * 64}]}
    assert recovery.source_manifest_bytes(receipt, 'lfm25') == raw
    assert recovery.source_manifest_bytes({**receipt, 'sha256': '0' * 64}, 'lfm25') is None
    result = recovery.recover_sources({'training': {'source_receipts': [receipt]}}, 'lfm25', tmp_path)
    assert Path(result[0]['local']).read_bytes() == raw
    assert result[0]['original']['path'] == '/old/root.json'


def test_summarized_capture_cannot_be_promoted_to_original_metadata(tmp_path):
    original = {'complete': ['one', 'two']}
    raw = json.dumps(original, sort_keys=True, indent=2).encode()
    binding = {'path': '/old/control.json', 'sha256': hashlib.sha256(raw).hexdigest()}
    assert recovery.recover_capture({'files': {'control': {'binding': binding, 'value': {'summary': 2}}}}, tmp_path) == []
    result = recovery.recover_capture({'stdout': json.dumps({'files': {
        'control': {'binding': binding, 'value': original}}})}, tmp_path)
    assert Path(result[0]['local']).read_bytes() == raw


def test_append_recipe_replay_cannot_accept_an_unmatched_recipe_digest():
    prior = {'input_bindings': {}, 'stages': [{'id': 'native_full_sft', 'inputs': {}},
                                             {'id': 'dpo', 'inputs': {}}]}
    metadata = {'training': {'source_receipts': [{'path': '/new/train', 'sha256': 'a' * 64,
                  'sources': [{'path': '/old/train', 'sha256': 'b' * 64}]}],
                'options': {'recipe_binding': json.dumps({'path': '/new/recipe', 'sha256': '0' * 64}),
                    'source_classification_history': [{'proof_sha256': 'c' * 64}],
                    'source_classification_migration': '/new/proof',
                    'ce_supervision_policy_binding': {'path': '/new/ce', 'sha256': 'd' * 64}}}}
    assert recovery.appended_recipe_bytes(prior, metadata, 'next_') is None
    assert prior['stages'][0]['inputs'] == {}  # Original recipe was not mutated.
