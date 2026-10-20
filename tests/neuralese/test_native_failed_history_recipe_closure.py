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
