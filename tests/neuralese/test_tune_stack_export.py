"""Stack members read from release files equal the trainer's in-engine stack (plan §7)."""
import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for
from natlang_neuralese.tune.build import persist_artifacts
from natlang_neuralese.tune.stack import stack_factors
from natlang_neuralese.tune.stack_export import members_from_artifacts, load_artifacts
from natlang_neuralese.tune.trainer import JointTrainer, Recipe


@pytest.mark.parametrize('parameterization', ['native-subspace', 'free-lora'])
def test_stack_members_from_files_match_the_trained_stack(tmp_path, parameterization):
    engine = engine_fixture(tmp_path, 'lfm')
    trainer = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 4) + rows_for(engine, 'by', 4),
                           Recipe(parameterization=parameterization, steps=2, batch_rows=8))
    while trainer.state['step'] < trainer.recipe.steps:
        trainer.step()
    release = tmp_path / 'release'
    persist_artifacts(engine, list(trainer.state['node_artifacts'].values()), release / 'artifacts')
    entries = [{'id': trainer.state['node_artifacts'][node], 'scale': scale, 'node': node}
               for node, scale in (('tree:', 1.), ('folder:a', .5), ('lambda:x', 1.))]
    # Reference: the trainer's own members for the same nodes and scales, through the same exact algebra.
    reference, bases = trainer._members([('node', node) for node in ('tree:', 'folder:a', 'lambda:x')])
    reference = [type(member)(member.node, member.level, entry['scale'], member.deltas) for member, entry in zip(reference, entries)]
    members, file_bases, shapes = members_from_artifacts(load_artifacts(release / 'artifacts'), entries)
    expected, actual = stack_factors(reference, bases, shapes), stack_factors(members, file_bases, shapes)
    for name, shape in shapes.items():
        assert torch.allclose(actual[name].dense(shape).double(), expected[name].dense(shape).double(), atol=1e-6), name
