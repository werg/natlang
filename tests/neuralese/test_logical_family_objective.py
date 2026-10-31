import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))
from build_logical_family_objective import family_of  # noqa: E402


def test_cohort_leaves_map_to_logical_families():
    assert family_of(Path('c/t1-communication-03.jsonl.zst')) == 'native-communication'
    assert family_of(Path('c/t5-broad-00.jsonl.zst')) == 'coding'
    assert family_of(Path('c/t6-broad-01.jsonl.zst')) == 'external-agent'
    assert family_of(Path('c/t2-scope-11.jsonl.zst')) == 'natlang'
    with pytest.raises(ValueError):
        family_of(Path('elsewhere/native-graphs.jsonl.zst'))


def test_port_coding_targets_use_bounded_coding_mass_without_renaming_interface():
    from build_logical_family_objective import feature_family
    assert feature_family({'features': {'family': 'port_record_calls', 'domain': 'trajectory_continuation_coding'}}) == 'coding'
    assert feature_family({'features': {'family': 'port_record_calls', 'domain': 'trajectory_continuation_research'}}) is None
    assert feature_family({'interface_family': 'port_record_calls', 'logical_family': 'natlang'}) == 'natlang'
    with pytest.raises(ValueError):
        feature_family({'interface_family': 'port_record_calls', 'logical_family': 'unbound'})


def test_capped_scales_keep_mean_one_and_true_cap():
    from build_logical_family_objective import capped_scales
    mass = {'native-communication': 1.0, 'natlang': 99.0}
    scales, capped = capped_scales({'native-communication': 0.4, 'natlang': 0.6}, mass, 8)
    assert capped == ['native-communication'] and scales['native-communication'] == 8
    assert abs(sum(scales[f] * mass[f] for f in mass) - sum(mass.values())) < 1e-9
    scales, capped = capped_scales({'a': 0.5, 'b': 0.5}, {'a': 40.0, 'b': 60.0}, 8)
    assert not capped and abs(scales['a'] * 40 - 50) < 1e-9


def test_recovered_feature_inputs_use_the_bound_source_map(tmp_path):
    import hashlib
    import json
    from build_logical_family_objective import feature_overrides
    original = tmp_path/'original-features.jsonl'
    staged = tmp_path/'recovered'; staged.mkdir()
    payload = staged/'features.jsonl'
    payload.write_text(json.dumps({'id':'actual-row', 'interface_family':'port_record_calls',
                                   'logical_family':'coding'})+'\n')
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    manifest = staged/'train-sources.json'
    manifest.write_text(json.dumps({'logical_family_features':[{'path':str(original),'sha256':digest}]}))
    (staged/'.natlang-source-map.json').write_text(json.dumps({'schema':'natlang.staged-source-map/1',
        'paths':{str(original):str(payload)}}))
    assert not original.exists()
    overrides, bindings = feature_overrides(manifest)
    assert overrides == {'actual-row':'coding'}
    assert bindings[0]['path'] == str(payload) and bindings[0]['sha256'] == digest
