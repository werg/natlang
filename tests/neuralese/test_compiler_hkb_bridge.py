"""The compiler's scoped experience KB over hindsight-kb (`compiler/hkb_bridge.py`): held admission through
``held_eval.training_exclusion_ids`` (siblings by split and source group), the audit on open, scopes as hindsight-kb
namespaces, exclusion before top-k, and the Stage A / Stage C outputs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.compiler import hkb_dependency

try:
    hkb_dependency.require()
except ModuleNotFoundError as error:  # pragma: no cover - the checkout is a declared dependency
    pytest.skip(str(error), allow_module_level=True)

from natlang_neuralese.compiler.hkb_bridge import (  # noqa: E402
    ExperienceEntry, HeldAdmission, ScopedExperienceKB, record_key)
from natlang_neuralese.rl import held_eval  # noqa: E402

ROWS = [
    {'id': 'held-x', 'curriculum': {'split_group': 'g-held'}},
    {'id': 'held-sib', 'curriculum': {'split_group': 'g-held'}},          # sibling by split group
    {'id': 'src-sib', 'source_groups': ['s-held']},                       # sibling by source group
    {'id': 't1'}, {'id': 't2'}, {'id': 't3'}, {'id': 'doc-task'},
]


@pytest.fixture
def held(tmp_path: Path, monkeypatch):
    """A held manifest holding ``held-x`` (its split group excluded), whose own cases carry source group ``s-held``;
    the registry's held roles are replaced by it alone."""
    pool = tmp_path / 'pool.jsonl'
    pool.write_text(''.join(json.dumps(row) + '\n' for row in ROWS))
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'held-x', 'source_groups': ['s-held']}) + '\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'tasks': [{'id': 'held-x'}], 'excluded_split_groups': ['g-held'],
                                    'pool': {'path': str(pool), 'sha256': held_eval._sha256(pool)},
                                    'cases': {'path': 'cases.jsonl'}}))
    monkeypatch.setattr(held_eval, 'held_manifests', lambda **_: [])
    return manifest


def entry(record_id, scope, key, value, tasks, **kw):
    return ExperienceEntry(record_id, scope, torch.tensor(key, dtype=torch.float32),
                           torch.full((8,), float(value)), kw.pop('text', record_id), tuple(tasks), **kw)


def test_admission_is_the_training_exclusion_closure(held, tmp_path):
    admission = HeldAdmission.from_rows(ROWS, held)
    assert admission.excluded == held_eval.training_exclusion_ids(ROWS, held)
    assert admission.excluded == {'held-x', 'held-sib', 'src-sib'}
    kb = ScopedExperienceKB(tmp_path / 'kb.sqlite', admission=admission, key_dim=4, value_dim=8)
    for task in ('held-x', 'held-sib', 'src-sib'):
        with pytest.raises(ValueError, match='held'):
            kb.add(entry(f'e-{task}', 'natlang:global', [1, 0, 0, 0], 1, [task]))
    with pytest.raises(ValueError, match='not among the admission rows'):
        kb.add(entry('e-unknown', 'natlang:global', [1, 0, 0, 0], 1, ['elsewhere']))
    # one refused record refuses the whole batch
    with pytest.raises(ValueError):
        kb.add_many([entry('e-ok', 'group:sql', [1, 0, 0, 0], 1, ['t1']),
                     entry('e-bad', 'group:sql', [1, 0, 0, 0], 1, ['held-sib'])])
    assert len(kb) == 0
    assert kb.add(entry('e-zero', 'group:sql', [1, 0, 0, 0], 1, ['t1'], strength=0.0)) is False
    with pytest.raises(ValueError, match='scope'):
        kb.add(entry('e-scope', 'sql', [1, 0, 0, 0], 1, ['t1']))
    kb.close()


def test_open_audits_stored_tasks_against_the_current_held_closure(held, tmp_path):
    path = tmp_path / 'kb.sqlite'
    loose = HeldAdmission(frozenset(row['id'] for row in ROWS), frozenset())
    with ScopedExperienceKB(path, admission=loose, key_dim=4, value_dim=8) as kb:
        kb.add(entry('e-later-held', 'group:sql', [1, 0, 0, 0], 1, ['held-sib']))
    with pytest.raises(ValueError, match='held'):
        ScopedExperienceKB(path, admission=HeldAdmission.from_rows(ROWS, held), key_dim=4, value_dim=8)


def test_scopes_exclusion_before_top_k_and_stage_outputs(held, tmp_path):
    kb = ScopedExperienceKB(tmp_path / 'kb.sqlite', admission=HeldAdmission.from_rows(ROWS, held),
                            key_dim=4, value_dim=8)
    kb.add_many([
        entry('e-self', 'group:sql', [1, 0, 0, 0], 1, ['t1'], text='self'),
        entry('e-peer', 'group:sql', [0.9, 0.4, 0, 0], 2, ['t2'], text='Use COALESCE on nullable sums.',
              kind='repair'),
        entry('e-global', 'natlang:global', [0.8, 0.6, 0, 0], 3, ['t3']),
        entry('e-repo', 'repo:abc', [1, 0, 0, 0], 4, ['doc-task'], kind='doc'),
        entry('e-far', 'natlang:global', [0, 0, 1, 0], 5, ['t3']),
    ])
    assert len(kb) == 5
    # the records live in hindsight-kb under their scopes, task provenance indexed
    assert set(kb.kb.scopes()) == {'group:sql', 'natlang:global', 'repo:abc'}
    stored = kb.kb.fetch_records([record_key('group:sql', 'e-peer')], scope='group:sql')[0]
    assert stored.metadata['hkb_task_ids'] == ['t2'] and stored.metadata['hkb_kind'] == 'experience'

    query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    # top_k=1: excluding t1 before selection leaves the peer, not an empty group scope
    hits, gate = kb.retrieve(query, scopes=['natlang:global', 'group:sql'], exclude_tasks={'t1'}, top_k=1)
    assert [h.record_id for h in hits] == ['e-peer']
    assert hits[0].kind == 'repair' and hits[0].text.startswith('Use COALESCE') and hits[0].scope == 'group:sql'
    assert 0.0 < gate < 1.0 and hits[0].weight == pytest.approx(gate)

    hits, gate = kb.retrieve(query, scopes=['natlang:global', 'group:sql'], exclude_tasks={'t1'}, top_k=4)
    assert [h.record_id for h in hits] == ['e-peer', 'e-global']           # e-far is below the threshold
    assert sum(h.weight for h in hits) == pytest.approx(gate)
    assert hits[0].similarity > hits[1].similarity
    assert all(h.scope != 'repo:abc' for h in hits)

    z, k, w, valid = kb.retrieve_unified_slots(query, scopes=['natlang:global', 'group:sql'],
                                              exclude_tasks={'t1'}, max_slots=4)
    assert z.shape == (1, 4, 8) and k.shape == (1, 4, 4) and w.shape == (1, 4) and valid.shape == (1, 4)
    assert valid[0].tolist() == [True, True, False, False]
    assert torch.allclose(z[0, 0], torch.full((8,), 2.0)) and torch.allclose(z[0, 1], torch.full((8,), 3.0))
    assert float(w.sum()) == pytest.approx(gate) and float(w[0, 2:].abs().sum()) == 0.0

    z, _, _, valid = kb.retrieve_unified_slots(query, scopes=['repo:abc'], max_slots=2)
    assert valid[0].tolist() == [True, False] and torch.allclose(z[0, 0], torch.full((8,), 4.0))
    kb.close()


def test_readd_keeps_identity_and_records_provenance(held, tmp_path):
    kb = ScopedExperienceKB(tmp_path / 'kb.sqlite', admission=HeldAdmission.from_rows(ROWS, held),
                            key_dim=4, value_dim=8)
    kb.add(entry('e', 'fn:site-1', [1, 0, 0, 0], 1, ['t1']))
    kb.add(entry('e', 'fn:site-1', [1, 0, 0, 0], 2, ['t2']))
    assert len(kb) == 1
    # task lineage is a union across rewrites (hindsight-kb's fail-safe provenance)
    hits, _ = kb.retrieve(torch.tensor([1.0, 0, 0, 0]), scopes=['fn:site-1'], exclude_tasks={'t1'})
    assert hits == []
    prov = kb.provenance()
    assert prov['hindsight_kb']['schema'] == 'natlang.hindsight-kb-provenance/1' and prov['generation'] >= 2
    kb.close()
