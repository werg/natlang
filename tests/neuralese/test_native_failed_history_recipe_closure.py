"""Recipe/source-host closure includes the same exact inputs as provider closure."""
import json
import pytest
from test_native_failed_history import fixture
from natlang_neuralese.train.native_failed_history import binding
from natlang_neuralese.train.native_supervision import ce_policy_dependencies
from natlang_neuralese.train.recipe import _bind_native_source_dependencies,_binding_files


def test_recipe_binds_manifest_and_every_decoder_asset_before_plan(tmp_path):
    _,_,manifest,policy,_=fixture(tmp_path)
    reference=binding(policy)
    resolved=_bind_native_source_dependencies({'ce_supervision_policy':reference})
    assert set(item['path']for item in _binding_files(resolved['ce_supervision_policy']))=={
        item['path']for item in ce_policy_dependencies(binding(policy))}
    assert manifest['path']in{item['path']for item in reference['dependencies']}
    asset=next(item for item in reference['dependencies']if item['path'].endswith('tokenizer.json'))
    from pathlib import Path
    Path(asset['path']).write_text('{}')
    with pytest.raises(ValueError):_bind_native_source_dependencies({'ce_supervision_policy':binding(policy)})


def test_legacy_policy_binding_has_no_artificial_dependency_change(tmp_path):
    _,_,_,policy,_=fixture(tmp_path)
    value=json.loads(policy.read_text());value.pop('failed_history_masks');policy.write_text(json.dumps(value))
    original=binding(policy)
    assert _bind_native_source_dependencies({'ce_supervision_policy':dict(original)})=={'ce_supervision_policy':original}


def test_dependency_enumeration_prepares_each_mask_once_and_preserves_exact_bindings(tmp_path,monkeypatch):
    from natlang_neuralese.train import native_failed_history as masks
    _,_,manifest,policy,_=fixture(tmp_path)
    expected=[binding(policy),*masks.prepare_masks(json.loads(policy.read_text()))]
    original=masks.load_manifest;calls=[]
    def counted(reference):
        calls.append(reference)
        return original(reference)
    monkeypatch.setattr(masks,'load_manifest',counted)
    assert ce_policy_dependencies(binding(policy))==expected
    assert calls==[manifest]


@pytest.mark.parametrize('damage',['policy','manifest','decoder','extra-decoder-file'])
def test_dependency_enumeration_rechecks_artifacts_after_single_preparation(tmp_path,monkeypatch,damage):
    from pathlib import Path
    from natlang_neuralese.train import native_failed_history as masks
    _,_,manifest,policy,_=fixture(tmp_path)
    reference=binding(policy);original=masks.prepare_masks
    def prepare(value):
        result=original(value)
        if damage=='policy':target=policy
        elif damage=='manifest':target=Path(manifest['path'])
        else:
            asset=next(x for x in result if x['path'].endswith('tokenizer.json'))
            target=Path(asset['path'])
            if damage=='extra-decoder-file':
                (target.parent/'unbound.json').write_text('{}')
                return result
        target.write_bytes(target.read_bytes()+b' ')
        return result
    monkeypatch.setattr(masks,'prepare_masks',prepare)
    with pytest.raises(ValueError):ce_policy_dependencies(reference)
