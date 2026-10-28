"""The shared line-backbone declaration: historical LFM2.5 identities still validate, LFM3 identities are accepted."""
import copy
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from natlang_neuralese.common import line_backbones as lb

REPO = Path(__file__).resolve().parents[2]
LFM25_TOKENIZER = [path for path in (
    REPO / 'models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f',
    Path('/home/werg/natlang/models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f'),
) if path.is_dir()]
LFM3_SNAPSHOT = Path('/home/werg/.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/'
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
    assert lb.current_key('lfm') == 'lfm3' and lb.current_key('mellum') == 'mellum21'
    assert lb.current_model_keys() == ('lfm3', 'mellum21')
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


def test_ferry_accepts_lfm25_and_lfm3_native_identities(tmp_path):
    from natlang_neuralese.common import memetic_teacher_ferry as ferry
    assert ferry.EVALUATOR_IDS['lfm'] == lb.backbone('lfm25')['evaluator_sandbox_id']
    assert ferry.EVALUATOR_IDS['mellum'] == lb.backbone('mellum21')['evaluator_sandbox_id']
    manifest = json.loads(ferry.DEPLOYMENTS_PATH.read_text())
    assert ferry.load_deployments()['evaluators']['lfm']['native_identity']['architecture'] == 'lfm2'
    for architecture, ok in (('lfm2', True), ('lfm3', True), ('mellum', False), ('qwen', False)):
        changed = copy.deepcopy(manifest)
        changed['evaluators']['lfm']['native_identity']['architecture'] = architecture
        path = tmp_path / f'{architecture}.json'
        path.write_text(json.dumps(changed))
        if ok:
            ferry.load_deployments(path)
        else:
            with pytest.raises(ValueError, match='native identity'):
                ferry.load_deployments(path)
    changed = copy.deepcopy(manifest)
    changed['evaluators']['mellum']['native_identity'] = {
        'checkpoint_sha256': 'a' * 64, 'tokenizer_fingerprint': 'b' * 64,
        'embedding_weight_sha256': 'c' * 64, 'architecture': 'lfm3'}
    path = tmp_path / 'mellum-as-lfm3.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='native identity'):
        ferry.load_deployments(path)


def test_architecture_dispatch_sets_cover_lfm3():
    import inspect
    from natlang_neuralese.eval import native_async, native_async_service
    assert 'set(all_architectures())' in inspect.getsource(native_async)
    parser_source = inspect.getsource(native_async_service)
    assert "choices=all_architectures()" in parser_source
    dispatch = _load('scripts/task_eval_dispatch.py')
    request = {'stage_out': 'x', 'step': 1, 'native_state': True,
               'evaluation_provenance': {'model_config': {'architecture': 'lfm3'}}}
    assert dispatch.unit_name(Path('.'), request).startswith('taskeval-native-lfm3-')
    request['evaluation_provenance']['model_config']['architecture'] = 'lfm2'
    assert dispatch.unit_name(Path('.'), request).startswith('taskeval-native-lfm2-')
    request['evaluation_provenance']['model_config']['architecture'] = 'mystery'
    assert dispatch.unit_name(Path('.'), request).startswith('taskeval-native-unknown-')


def test_capacity_composition_pins_stay_with_the_backbone_they_were_reviewed_for():
    from natlang_neuralese.train import native_capacity_continuation as continuation
    assert continuation.REVIEWED_BACKBONES == {'lfm': 'lfm25', 'mellum': 'mellum21'}
    assert continuation._composition_applies('lfm', 'lfm2') and continuation._composition_applies('lfm', 'lfm2_moe')
    assert continuation._composition_applies('mellum', 'maple') and continuation._composition_applies('mellum', 'mellum')
    assert not continuation._composition_applies('lfm', 'lfm3')  # needs its own independent review
    assert continuation._composition_applies('lfm', None)  # nothing declared: the pinned code hashes still decide


def test_native_resident_d31_model_views_include_lfm3():
    import inspect
    from natlang_neuralese.train import native_resident
    source = inspect.getsource(native_resident._d31_reference_qa_geometry_part)
    assert 'line_backbones.BACKBONES' in source


def test_private_append_constants_mirror_the_declaration():
    append = _load('scripts/neuralese_data/prepare_private_teacher_append.py')
    assert append.SUPPORTED_MODEL_ROLES == set(lb.BACKBONES)
    assert append.LFM_MODEL_ROLES == set(lb.keys('lfm'))
    assert append.EXPLICIT_KEEP_HISTORY_ROLES == {
        key for key, value in lb.BACKBONES.items() if value['explicit_history_reasoning_policy'] == 'keep'}
    assert append.BACKBONE_MODEL_TYPE == {key: value['model_id_architecture'] for key, value in lb.BACKBONES.items()}


@pytest.mark.parametrize('lineage,ok', [('lfm25', True), ('lfm3', True), ('mellum21', False)])
def test_cursor_export_accepts_each_lfm_backbone_namespace(tmp_path, lineage, ok):
    export = _load('scripts/coreweave/cursor_checkpoint_export.py')
    production = tmp_path / 'production'
    output = production / lineage / 'native_full_sft'
    output.mkdir(parents=True)
    argv = tmp_path / 'argv.json'
    argv.write_text(json.dumps(['python', '--out', str(output)]))
    import hashlib
    reference = {'path': str(argv), 'sha256': hashlib.sha256(argv.read_bytes()).hexdigest(),
                 'allowed_output_root': str(production / lineage)}
    if ok:
        assert export.bound_output_checkpoint(reference, 'lfm', production_root=production,
                                              control_root=production / 'control') == output / 'checkpoint.pt'
    else:
        with pytest.raises(ValueError):
            export.bound_output_checkpoint(reference, 'lfm', production_root=production,
                                           control_root=production / 'control')
    # the declared production directories are what the self-contained function spells
    assert {lb.backbone(key)['production_dir'] for key in lb.keys('lfm')} == {'lfm3', 'lfm25'}


def test_teacher_conversion_and_admission_select_views_by_declaration():
    sys.path.insert(0, str(REPO))
    conversion = importlib.import_module('scripts.coreweave.teacher_internal_conversion')
    assert conversion.history_policy_args('lfm25') == [
        '--failed-assistant-history-policy', 'mask-runtime-failures', '--history-reasoning-policy', 'keep']
    assert conversion.history_policy_args('lfm3') == ['--failed-assistant-history-policy', 'mask-runtime-failures']
    assert conversion.history_policy_args('mellum21') == ['--failed-assistant-history-policy', 'mask-runtime-failures']
    with pytest.raises(ValueError):
        conversion.history_policy_args('lfm2')
    admission = importlib.import_module('scripts.coreweave.teacher_tranche_admission')
    by_path = {'lfm25/train/native-graphs.jsonl.zst': {}, 'mellum21/train/native-graphs.jsonl.zst': {}}
    assert admission._predecessor_models(None, by_path) == ('lfm25', 'mellum21')
    assert admission._predecessor_models({'lfm3': {}, 'mellum21': {}}, by_path) == ('lfm3', 'mellum21')
    with pytest.raises(ValueError):
        admission._predecessor_models({'lfm9': {}}, by_path)
