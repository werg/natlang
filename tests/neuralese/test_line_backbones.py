"""The shared line-backbone declaration: historical LFM2.5 identities still validate, LFM3 identities are accepted."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from natlang_neuralese.common import line_backbones as lb

REPO = Path(__file__).resolve().parents[2]
LFM25_SNAPSHOT = 'models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f'
# A linked worktree has no models/ directory of its own: look in the checkout and each ancestor (the main checkout).
LFM25_TOKENIZER = [root / LFM25_SNAPSHOT for root in (REPO, *REPO.parents) if (root / LFM25_SNAPSHOT).is_dir()][:1]
LFM3_SNAPSHOT = (Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
                 '40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd')


def _load(relative):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(path.stem + '_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_declaration_is_consistent():
    assert lb.LINES == ('lfm', 'mellum')
    assert set(lb.keys('lfm')) == {'lfm25', 'lfm3'} and lb.keys('mellum') == ('mellum21',)
    assert lb.current_key('lfm') == 'lfm3' and lb.active_lines() == ('lfm',)
    assert lb.backbone('mellum21')['status'] == lb.HISTORICAL  # the retired Mellum line
    with pytest.raises(ValueError, match='retired'):
        lb.current_key('mellum')
    with pytest.raises(ValueError, match='retired'):
        lb.key_for_role('mellum')
    assert lb.key_for_role('mellum', architecture='mellum') == 'mellum21'  # historical rows still resolve
    assert lb.current_model_keys() == ('lfm3',)
    for key, value in lb.BACKBONES.items():
        assert value['model_key'] == key and lb.line_of(key) in lb.LINES
        for field in ('source_fingerprint', 'expanded_fingerprint'):
            assert re.fullmatch('[0-9a-f]{64}', value[field])
        assert value['source_fingerprint'] != value['expanded_fingerprint']
        assert re.fullmatch('[0-9a-f]{64}', value['base_inventory_sha256'])
        assert lb.model_id(key) == value['model_id_architecture'] + ':' + value['base_inventory_sha256']
    ids = [lb.backbone(key)['role_token_ids'] for key in ('lfm25', 'lfm3')]
    assert all(len(set(row)) == 4 for row in ids)
    assert len({value['expanded_fingerprint'] for value in lb.BACKBONES.values()}) == 3
    assert set(lb.model_key_pairs()) == {frozenset({'lfm25', 'mellum21'}), frozenset({'lfm3', 'mellum21'})}


def test_historical_lfm25_identities_are_unchanged():
    # The exact values earlier receipts, packages and checkpoints recorded.
    lfm25 = lb.backbone('lfm25')
    assert lfm25['source_fingerprint'] == 'e52de346c5a9d6fe666b5a412be077525daee0ee40f81a91c36728cfb4cbf008'
    assert lfm25['expanded_fingerprint'] == 'be949ec48f44e17b2b17e1131aeb1dab4262175ba20f9783ecd8e59c27a58105'
    assert lb.model_id('lfm25') == 'lfm2:f40d7a5bf651a03618ae8c06bda24a180e22fa79de964e55862c2c4984249da6'
    assert lb.model_id('mellum21') == 'mellum:bc700045baee1c7ea1869675c38efedd59d9db09975ea0b38cff8db152d6148f'
    assert lfm25['evaluator_sandbox_id'] == '393f9b75-2666-456c-a850-1b46a8b7d147'
    assert lfm25['production_dir'] == 'lfm25' and lfm25['explicit_history_reasoning_policy'] == 'keep'
    assert lb.backbone('mellum21')['expanded_fingerprint'] == \
        'fbf22d8773d09be802cb9977895067db36cdfba747fcb24b02ddbf1312d3778f'


def test_model_id_and_architecture_resolution_accepts_both_lfm_generations():
    assert lb.key_for_model_id('lfm2:' + '0' * 64) == 'lfm25'
    assert lb.key_for_model_id(lb.model_id('lfm3')) == 'lfm3'
    assert lb.key_for_model_id('mellum:' + '1' * 64) == 'mellum21'
    for bad in ('lfm:abc', 'lfm3', None, 'qwen:1'):
        with pytest.raises(ValueError):
            lb.key_for_model_id(bad)
    assert {'lfm2', 'lfm2_moe', 'lfm3'} <= lb.architectures('lfm')
    assert lb.architectures('lfm', status=lb.HISTORICAL) == {'lfm2', 'lfm2_moe'}
    assert lb.architectures('mellum') == {'mellum', 'maple'}
    assert lb.key_for_architecture('lfm3') == 'lfm3' and lb.key_for_architecture('lfm2_moe') == 'lfm25'
    assert set(lb.all_architectures()) == {'lfm2', 'lfm2_moe', 'lfm3', 'mellum', 'maple'}
    assert lb.key_for_role('lfm', model_id='lfm2:' + '0' * 64) == 'lfm25'
    assert lb.key_for_role('lfm', model_id=lb.model_id('lfm3')) == 'lfm3'
    assert lb.key_for_role('lfm', model_id='lfm:' + '2' * 64) == 'lfm3'  # role-prefixed ids use the current backbone
    assert lb.key_for_role('lfm') == 'lfm3' and lb.key_for_role('lfm', production_dir='lfm25') == 'lfm25'
    with pytest.raises(ValueError):
        lb.key_for_role('lfm', model_id='mellum:' + '1' * 64)
    assert lb.key_for_expanded_fingerprint(lb.backbone('lfm3')['expanded_fingerprint']) == 'lfm3'


@pytest.mark.skipif(not LFM25_TOKENIZER or not LFM3_SNAPSHOT.is_dir(), reason='pinned tokenizers are not local')
@pytest.mark.parametrize('key,directory', [('lfm25', LFM25_TOKENIZER[0] if LFM25_TOKENIZER else None),
                                           ('lfm3', LFM3_SNAPSHOT)])
def test_fingerprints_are_what_the_shared_functions_compute(key, directory):
    from transformers import AutoTokenizer
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from natlang_neuralese.model.shared_state import provision_role_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False, local_files_only=True)
    declared = lb.backbone(key)
    assert tokenizer_fingerprint(tokenizer) == declared['source_fingerprint']
    assert len(tokenizer.get_vocab()) == declared['vocab_size']
    assert provision_role_tokenizer(tokenizer) == declared['role_token_ids']
    assert tokenizer_fingerprint(tokenizer) == declared['expanded_fingerprint']


@pytest.mark.skipif(not LFM3_SNAPSHOT.is_dir(), reason='LFM3 snapshot is not local')
def test_lfm3_pinned_files_match_the_declaration():
    import hashlib
    declared = lb.backbone('lfm3')
    assert hashlib.sha256((LFM3_SNAPSHOT / 'tokenizer.json').read_bytes()).hexdigest() == declared['tokenizer_json_sha256']
    assert LFM3_SNAPSHOT.name == declared['revision']
    config = json.loads((LFM3_SNAPSHOT / 'config.json').read_text())
    assert config['model_type'] in declared['architectures']
