"""Incremental fits (plan §8): pinned nodes stay frozen at earlier artifacts; warm-started nodes continue honestly."""
import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for, tensors
from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.trainer import JointTrainer, Recipe

ADAM = {'name': 'adam', 'learning_rate': 1e-2}


def earlier_fit(engine, parameterization):
    trainer = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3) + rows_for(engine, 'by', 3, offset=2),
                           Recipe(parameterization=parameterization, batch_rows=6, steps=2, optimizer=ADAM))
    trainer.run()
    return trainer.state


@pytest.mark.parametrize('parameterization', ['native-subspace', 'free-lora'])
def test_leaf_candidate_trains_only_its_leaf_against_frozen_parents(tmp_path, parameterization):
    engine = engine_fixture(tmp_path, 'lfm')
    earlier = earlier_fit(engine, parameterization)
    pinned = {node: earlier['node_artifacts'][node] for node in ('tree:', 'folder:a', 'folder:b', 'lambda:y')}
    warm = {'lambda:x': {'artifact': earlier['node_artifacts']['lambda:x'], 'optimizer': earlier['optimizer_states']['lambda:x']}}
    frozen_before = {node: {key: value.clone() for key, value in tensors(engine, artifact).items()} for node, artifact in pinned.items()}
    trainer = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3),
                           Recipe(parameterization=parameterization, batch_rows=3, steps=2, optimizer=ADAM), pinned=pinned, warm=warm)
    # The warm node starts exactly where the earlier fit left it, with its moments continued (same artifact, same algorithm).
    assert trainer.state['node_artifacts']['lambda:x'] == earlier['node_artifacts']['lambda:x']
    assert trainer.state['optimizer_states'] == {'lambda:x': earlier['optimizer_states']['lambda:x']}
    for record in trainer.run():
        assert record['active_nodes'] == ['lambda:x']
        assert record['updated_nodes'] == ['lambda:x']
    for node, artifact in pinned.items():
        assert trainer.state['node_artifacts'][node] == artifact
        assert all(torch.equal(value, frozen_before[node][key]) for key, value in tensors(engine, artifact).items())
    assert trainer.state['node_artifacts']['lambda:x'] != earlier['node_artifacts']['lambda:x']
    receipt = trainer.receipt()
    assert receipt['incremental'] == {'pinned': pinned, 'warm': {'lambda:x': {'artifact': warm['lambda:x']['artifact'],
                                                                              'optimizer': 'continued'}}, 'cold': {}}
    # The release still lists every node, the pinned ones at their exact earlier artifacts.
    listed = {item['node']: item['artifact'] for item in receipt['artifacts']}
    assert {node: listed[node] for node in pinned} == pinned
    unpinned = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3),
                            Recipe(parameterization=parameterization, batch_rows=3, steps=2, optimizer=ADAM))
    assert unpinned.input_sha256 != trainer.input_sha256, 'pins are part of what a fit is'


def test_warm_start_restarts_moments_or_starts_cold_when_the_contract_does_not_hold(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    earlier = earlier_fit(engine, 'native-subspace')
    warm = {node: {'artifact': earlier['node_artifacts'][node], 'optimizer': earlier['optimizer_states'][node]}
            for node in ('tree:', 'folder:a', 'lambda:x')}
    sgd = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3),
                       Recipe(batch_rows=3, steps=1, optimizer={'name': 'sgd', 'learning_rate': 1e-2}), warm=warm)
    assert {node: item['optimizer'] for node, item in sgd.state['incremental']['warm'].items()} == dict.fromkeys(warm, 'new')
    assert all(sgd.state['optimizer_states'][node] is None for node in warm)
    # A different lambda rank cannot seed the node: it starts cold and says why; same-rank nodes still warm-start.
    ranked = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3),
                          Recipe(batch_rows=3, steps=1, optimizer=ADAM, ranks={'tree': 8, 'folder': 4, 'lambda': 4}), warm=warm)
    assert 'rank' in ranked.state['incremental']['cold']['lambda:x']
    assert ranked.state['node_artifacts']['lambda:x'] != earlier['node_artifacts']['lambda:x']
    assert ranked.state['node_artifacts']['tree:'] == earlier['node_artifacts']['tree:']
    # A pinned node must compose exactly; a changed rank is an error rather than a silently different stack.
    with pytest.raises(TuneMathError, match='cannot be composed'):
        JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3),
                     Recipe(batch_rows=3, steps=1, ranks={'tree': 8, 'folder': 4, 'lambda': 4}),
                     pinned={'lambda:x': earlier['node_artifacts']['lambda:x']})
    with pytest.raises(TuneMathError, match='either pinned or warm'):
        JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3), Recipe(batch_rows=3, steps=1),
                     pinned={'tree:': earlier['node_artifacts']['tree:']}, warm={'tree:': {'artifact': earlier['node_artifacts']['tree:']}})


def test_shared_refit_trains_the_dirty_ancestor_and_its_dependents_only(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    earlier = earlier_fit(engine, 'native-subspace')
    # folder:a changed: it and its dependent lambda:x refit; tree:, folder:b and lambda:y stay frozen.
    pinned = {node: earlier['node_artifacts'][node] for node in ('tree:', 'folder:b', 'lambda:y')}
    warm = {node: {'artifact': earlier['node_artifacts'][node]} for node in ('folder:a', 'lambda:x')}
    trainer = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 3) + rows_for(engine, 'by', 2, offset=1),
                           Recipe(batch_rows=5, steps=1, optimizer=ADAM), pinned=pinned, warm=warm)
    record = trainer.step()
    assert record['updated_nodes'] == ['folder:a', 'lambda:x']
    assert all(trainer.state['node_artifacts'][node] == artifact for node, artifact in pinned.items())
