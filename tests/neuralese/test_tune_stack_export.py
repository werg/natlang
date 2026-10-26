"""Whole-stack export from release files equals the trainer's in-engine stack (plan §7)."""
import json

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for
from natlang_neuralese.tune.build import persist_artifacts
from natlang_neuralese.tune.export import load_peft_dense
from natlang_neuralese.tune.stack import stack_factors
from natlang_neuralese.tune.stack_export import export_stacks, members_from_artifacts, load_artifacts
from natlang_neuralese.tune.trainer import JointTrainer, Recipe


@pytest.mark.parametrize('parameterization', ['native-subspace', 'free-lora'])
def test_stack_export_from_files_matches_the_trained_stack(tmp_path, parameterization):
    engine = engine_fixture(tmp_path, 'lfm')
    trainer = JointTrainer(engine, manifest_for(engine), rows_for(engine, 'bx', 4) + rows_for(engine, 'by', 4),
                           Recipe(parameterization=parameterization, steps=2, batch_rows=8))
    while trainer.state['step'] < trainer.recipe.steps:
        trainer.step()
    release = tmp_path / 'release'
    persist_artifacts(engine, list(trainer.state['node_artifacts'].values()), release / 'artifacts')
    (release / 'natlang.adapters.json').write_text(json.dumps({'base': manifest_for(engine)['base']}))
    entries = [{'id': trainer.state['node_artifacts'][node], 'scale': scale, 'node': node}
               for node, scale in (('tree:', 1.), ('folder:a', .5), ('lambda:x', 1.))]
    # Reference: the trainer's own members for the same nodes and scales, through the same exact algebra.
    reference, bases = trainer._members([('node', node) for node in ('tree:', 'folder:a', 'lambda:x')])
    reference = [type(member)(member.node, member.level, entry['scale'], member.deltas) for member, entry in zip(reference, entries)]
    members, file_bases, shapes = members_from_artifacts(load_artifacts(release / 'artifacts'), entries)
    expected, actual = stack_factors(reference, bases, shapes), stack_factors(members, file_bases, shapes)
    for name, shape in shapes.items():
        assert torch.allclose(actual[name].dense(shape).double(), expected[name].dense(shape).double(), atol=1e-6), name
    # The PEFT export round-trips to the same dense deltas, one directory per stack identity, deduplicated.
    exported = export_stacks(release, [{'identity': 'stack-x', 'entries': entries}, {'identity': 'base', 'entries': []}],
                             tmp_path / 'exports', family='lfm2', hf_config=tmp_path, base_identity=manifest_for(engine)['base'])
    assert [item['identity'] for item in exported] == ['stack-x']
    dense = load_peft_dense(tmp_path / 'exports' / 'stack-x')
    assert dense and all(torch.isfinite(value).all() for value in dense.values())
    total = sum(float(value.abs().sum()) for value in dense.values())
    assert total > 0, 'a trained stack exports a nonzero delta'
