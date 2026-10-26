"""Cross-program families: gauge-invariant discovery, family fits against the frozen base, fixed amplitude, member
refits with the family frozen (plan §5.4, §5.5)."""
import json

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for, tensors
from natlang_neuralese.tune import TuneMathError
from natlang_neuralese.tune.build import persist_artifacts
from natlang_neuralese.tune.families import (FAMILY_NODE, balance_rows, discover, family_manifest, normalized_family)
from natlang_neuralese.tune.gates import family_norm
from natlang_neuralese.tune.similarity import delta_cosine
from natlang_neuralese.tune.trainer import JointTrainer, Recipe


def test_similarity_is_invariant_under_factor_gauge():
    torch.manual_seed(0)
    q = {'w': 1 / 48, 'v': 1 / 30}
    first = {'w': (torch.randn(6, 3, dtype=torch.float64), torch.randn(3, 8, dtype=torch.float64)),
             'v': (torch.randn(5, 2, dtype=torch.float64), torch.randn(2, 6, dtype=torch.float64))}
    second = {'w': (torch.randn(6, 4, dtype=torch.float64), torch.randn(4, 8, dtype=torch.float64))}
    Q = torch.randn(3, 3, dtype=torch.float64) + 3 * torch.eye(3, dtype=torch.float64)
    regauged = {**first, 'w': (first['w'][0] @ Q, torch.linalg.solve(Q, first['w'][1]))}
    assert torch.allclose(delta_cosine(first, second, q), delta_cosine(regauged, second, q), atol=1e-10)
    dense = lambda item: {name: B @ A for name, (B, A) in item.items()}
    a, b = dense(first), dense(second)
    reference = sum(q[n] * (a[n] * b[n]).sum() for n in b) / torch.sqrt(
        sum(q[n] * (a[n] ** 2).sum() for n in a) * sum(q[n] * (b[n] ** 2).sum() for n in b))
    assert torch.allclose(delta_cosine(first, second, q), reference, atol=1e-10)
    # Raw-factor cosine is not invariant: the reason it is not used.
    flat = lambda item: torch.cat([item['w'][0].reshape(-1), item['w'][1].reshape(-1)])
    assert not torch.allclose(torch.nn.functional.cosine_similarity(flat(first), flat(second)[:flat(first).numel()], 0),
                              torch.nn.functional.cosine_similarity(flat(regauged), flat(second)[:flat(first).numel()], 0))


def _release(engine, tmp_path, name, rows, steps=2, base_override=None):
    """A trained program release directory (manifest + artifacts) like `natlang tune` stores."""
    manifest = manifest_for(engine)
    trainer = JointTrainer(engine, manifest, rows, Recipe(steps=steps, batch_rows=4, seed=hash(name) % 97))
    trainer.run()
    release = tmp_path / name
    persist_artifacts(engine, list(trainer.state['node_artifacts'].values()), release / 'artifacts')
    fragment = trainer.manifest_fragment()
    (release / 'natlang.adapters.json').write_text(json.dumps({**manifest, **fragment,
        **({'base': base_override} if base_override else {})}))
    return release


def test_discovery_clusters_across_programs_on_one_base(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    rows = rows_for(engine, 'bx', 4) + rows_for(engine, 'by', 4, offset=2)
    first, second = _release(engine, tmp_path, 'p1', rows), _release(engine, tmp_path, 'p2', rows)
    result = discover([{'id': 'p1', 'release': str(first)}, {'id': 'p2', 'release': str(second)}], threshold=-1.)
    assert result['members'] == [['p1', 'bx'], ['p1', 'by'], ['p2', 'bx'], ['p2', 'by']]
    assert all(item['a'][0] != item['b'][0] for item in result['similarities']), 'pairs are across programs'
    assert all(-1 - 1e-9 <= item['cosine'] <= 1 + 1e-9 for item in result['similarities'])
    assert result['clusters'] == [[['p1', 'bx'], ['p1', 'by'], ['p2', 'bx'], ['p2', 'by']]]
    assert discover([{'id': 'p1', 'release': str(first)}, {'id': 'p2', 'release': str(second)}], threshold=1.01)['clusters'] == []
    tagged = discover([{'id': 'p1', 'release': str(first)}], tags={'shout': [['p1', 'bx']]})
    assert tagged['tags'] == {'shout': [['p1', 'bx']]}
    with pytest.raises(TuneMathError, match='without a tuned release'):
        discover([{'id': 'p1', 'release': str(first)}], tags={'shout': [['p1', 'nope']]})
    other = dict(json.loads((first / 'natlang.adapters.json').read_text())['base'], checkpoint_sha256='0' * 64)
    third = _release(engine, tmp_path, 'p3', rows, steps=1, base_override=other)
    with pytest.raises(TuneMathError, match='another base'):
        discover([{'id': 'p1', 'release': str(first)}, {'id': 'p3', 'release': str(third)}])


def test_family_fit_trains_only_the_shared_node_and_publishes_a_fixed_amplitude(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    members = ['p1/bx', 'p2/by']
    manifest = family_manifest({key: engine.native_identity[key] for key in engine.native_identity}, members)
    rows = rows_for(engine, 'p1/bx', 6) + rows_for(engine, 'p2/by', 2, offset=1)
    balanced = balance_rows(rows)
    totals = {}
    for row in balanced:
        totals[row['binding'].split('/')[0]] = totals.get(row['binding'].split('/')[0], 0) + row['evidence_weight']
    assert totals['p1'] == pytest.approx(totals['p2']), 'each program carries an equal share'
    recipe = Recipe(steps=3, batch_rows=4, ranks={'tree': 4, 'folder': 0, 'lambda': 0})
    trainer = JointTrainer(engine, manifest, balanced, recipe)
    initial = dict(trainer.state['node_artifacts'])
    assert list(initial) == [FAMILY_NODE], 'no private program/folder/lambda residual exists in a family fit'
    trainer.run()
    assert trainer.state['node_artifacts'][FAMILY_NODE] != initial[FAMILY_NODE]
    artifact, rho = normalized_family(trainer)
    record = engine.store.get_native_tensor_artifact(artifact).artifact
    norm = float(family_norm(trainer._node_factors(artifact, record.tensors), trainer.q))
    assert norm == pytest.approx(rho, rel=1e-5)
    # A later version is normalized to the recorded amplitude, not to its own norm.
    again, kept = normalized_family(trainer, rho=rho * 2)
    assert kept == rho * 2
    assert float(family_norm(trainer._node_factors(again, tensors(engine, again)), trainer.q)) == pytest.approx(rho * 2, rel=1e-5)
    # A zero fit is absent, never published.
    fresh = JointTrainer(engine, manifest, balanced, Recipe(steps=0, batch_rows=4, ranks={'tree': 4, 'folder': 0, 'lambda': 0}))
    assert normalized_family(fresh)[0] is None


def test_member_refit_trains_gates_with_the_published_family_frozen(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    family_trainer = JointTrainer(engine, family_manifest(dict(engine.native_identity), ['p1/bx']),
                                  rows_for(engine, 'p1/bx', 4), Recipe(steps=2, batch_rows=4, ranks={'tree': 4, 'folder': 0, 'lambda': 0}))
    family_trainer.run()
    artifact, rho = normalized_family(family_trainer)
    frozen = {key: value.clone() for key, value in tensors(engine, artifact).items()}
    families = [{'family': 'shout', 'version': 'v1', 'artifact': artifact, 'amplitudeHex': rho.hex()}]
    memberships = [{'binding': 'bx', 'family': 'shout', 'version': 'v1', 'gate': 1.}]
    refit = JointTrainer(engine, manifest_for(engine, families, memberships), rows_for(engine, 'bx', 4),
                         Recipe(steps=2, batch_rows=4, tau=1e-3, a_max=2.))  # headroom above the initial gate 1
    key = json.dumps(['bx', 'shout', 'v1'])
    refit.run()
    assert float.fromhex(refit.state['gates'][key]) != 1., 'the gate trains'
    assert all(torch.equal(tensors(engine, artifact)[name], value) for name, value in frozen.items()), 'family stays frozen'
    assert artifact not in refit.state['node_artifacts'].values()
    members, _ = refit.stack_members('bx')
    assert members[0].level == 'family' and members[0].scale == float.fromhex(refit.state['gates'][key])
