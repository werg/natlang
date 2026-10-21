"""The initial selected packet has the same real tail capacity as later packets."""
import copy
import json
from pathlib import Path

import pytest
from scripts.coreweave import teacher_generation_guardian as g
from scripts.coreweave import teacher_source_sequence as seq
from test_teacher_source_sequence import setup, put


def anchor_pool(tmp_path, monkeypatch):
    folder, state, config = setup(tmp_path, packet_count=3)
    feeder = tmp_path/'feed-evidence-anchor.py'; feeder.write_text('# pinned feeder\n')
    worker = tmp_path/'queue-worker.py'; worker.write_text('# pinned worker\n')
    config.update(primary_parallel_batches=2, primary_worker_capacity=32, overlap_workers=16,
        primary_tail_max_running=6, feeder=str(feeder), overlap_feeder=g.binding(feeder),
        collection_repo=str(tmp_path), collection_worker=g.binding(worker), queue_worker=str(worker),
        node_bin='/node', drain_file=str(tmp_path/'DRAIN'))
    home = Path(config['root'])/'evidence-collections-v2/batch-0021'
    initial = {'cases': str(home/'source-cases.jsonl'),
               'cases_sha256': config['source_sequence']['anchor']['cases_sha256'], 'rows': 1}
    target = Path(config['source_sequence']['source_configuration'])
    config['source_sequence']['initial_configuration'] = put(target, initial)
    owner = {'pid': 121, 'start_ticks': '21', 'boot_id': 'boot'}
    state.update(active_batch=21, owner=owner, feeder=g.binding(feeder), task_runtime=config['task_runtime'])
    state.pop('last_boundary')
    state['config_snapshot'] = put(folder/'config.json', config)
    state['primary_phases'] = {'21': dict(batch=21, owner=owner, feeder=g.binding(feeder),
        guardian_config=state['config_snapshot'], task_runtime=config['task_runtime'])}
    owners = {121: {**owner, 'argv': [b'python', str(feeder).encode()]}}
    monkeypatch.setattr(g, 'alive', lambda identity: bool(identity) and identity['pid'] in owners)
    monkeypatch.setattr(seq, 'alive', g.alive)
    monkeypatch.setattr(g, 'process', lambda pid: owners.get(int(pid)))
    (home/'queue.jsonl').write_text(''.join(json.dumps({'key': f'anchor-{i}'})+'\n' for i in range(8)))
    events = []
    for i in range(7):
        events.append({'event': 'start', 'key': f'anchor-{i}'})
        if i < 6: events.append({'event': 'finish', 'key': f'anchor-{i}'})
    (home/'journal-worker-0.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    put(home/'teacher-source.json', dict(batch=21, source_cases_sha256=initial['cases_sha256'],
        source_provenance=initial, actual_task_runtime=config['task_runtime'], concurrency=12))
    return folder, state, config, owners, home


def test_anchor_tail_prefetches_and_adopts_only_after_real_closure(tmp_path, monkeypatch):
    folder, state, config, owners, home = anchor_pool(tmp_path, monkeypatch)
    before = {p: p.read_bytes() for p in home.iterdir()}
    ambient = Path(config['source_sequence']['source_configuration']); original = ambient.read_bytes()
    seq.observe_source_sequence(folder, state, config)
    sequence = state['source_sequence']
    assert sequence['next_ordinal'] == 0 and 'pending' not in sequence
    assert seq._json(sequence['started']['21'], config['source_sequence'])['ordinal'] == -1
    spawned = []
    monkeypatch.setattr(g, '_spawn_primary_prefetch', lambda *args: spawned.append(args[3]['batch']))
    g.maintain_primary_prefetch(folder, state, config)
    assert spawned == [22] and state['primary_prefetch']['batch'] == 22
    admission = json.loads(Path(state['primary_prefetch']['capacity_admission']['path']).read_bytes())
    assert admission['potential_workers'] == 2 and admission['new_workers'] == 16
    reserved = sequence['prefetch']
    assert reserved['ordinal'] == 0 and sequence['next_ordinal'] == 0
    assert ambient.read_bytes() == original and {p: p.read_bytes() for p in home.iterdir()} == before
    with pytest.raises(ValueError, match='genuinely closed'):
        seq.adopt_prefetched_source(folder, state, config, 22)
    value = seq._json(seq._json(reserved['intent'], config['source_sequence'])['new_configuration'], config['source_sequence'])
    future = Path(config['root'])/'evidence-collections-v2/batch-0022'; future.mkdir()
    (future/'source-cases.jsonl').write_bytes(Path(value['cases']).read_bytes())
    put(future/'queue.jsonl', {'key': 'next'})
    put(future/'teacher-source.json', dict(batch=22, source_cases_sha256=value['cases_sha256'], actual_task_runtime=config['task_runtime']))
    seq.observe_prefetched_source(folder, state, config)
    boundary = put(folder/'boundaries/batch-0021.json', dict(batch=21, batch_complete=True,
        bindings={name: g.binding(home/name) for name in ('queue.jsonl', 'teacher-source.json')}))
    state['last_boundary'] = boundary
    seq.observe_source_sequence(folder, state, config)
    assert 'anchor_closed' not in sequence  # Even an observed boundary cannot close a live owner.
    owners.pop(121); state.update(owner=None, active_batch=None)
    seq.observe_source_sequence(folder, state, config)
    assert sequence['anchor_closed'] and not sequence.get('closed')
    assert sequence['next_ordinal'] == 0 and ambient.read_bytes() == original
    seq.adopt_prefetched_source(folder, state, config, 22)
    state['active_batch'] = 22
    seq.observe_source_sequence(folder, state, config)
    assert sequence['pending'] == reserved and json.loads(ambient.read_bytes()) == value
    assert seq._closed_packet_ordinals(folder, sequence, config['source_sequence'],
        sequence['observed_entries']) == {}
    assert {p: p.read_bytes() for p in home.iterdir()} == before
    # Restarting observation preserves the exact original receipt and next reservation.
    reloaded = json.loads((folder/'state.json').read_bytes())
    seq.observe_source_sequence(folder, reloaded, config)
    assert reloaded['source_sequence'] == sequence


@pytest.mark.parametrize('changed', ['source_provenance', 'source_cases_sha256', 'actual_task_runtime'])
def test_anchor_rejects_changed_original_source(tmp_path, monkeypatch, changed):
    folder, state, config, owners, home = anchor_pool(tmp_path, monkeypatch)
    source = json.loads((home/'teacher-source.json').read_bytes()); source[changed] = 'changed'
    put(home/'teacher-source.json', source)
    with pytest.raises(ValueError, match='original sealed configuration'):
        seq.observe_source_sequence(folder, state, config)
    assert not state.get('source_sequence')


def test_anchor_cannot_change_its_pinned_owner_or_use_foreign_boundary(tmp_path, monkeypatch):
    folder, state, config, owners, home = anchor_pool(tmp_path, monkeypatch)
    seq.observe_source_sequence(folder, state, config)
    original = copy.deepcopy(state['owner']); state['owner'] = {**original, 'pid': 999}
    with pytest.raises(ValueError, match='pinned primary owner'):
        seq.observe_source_sequence(folder, state, config)
    state.update(owner=None, active_batch=None); owners.pop(121)
    state['last_boundary'] = put(folder/'foreign-boundary.json', dict(batch=21, batch_complete=True))
    with pytest.raises(ValueError, match='canonical guardian boundary'):
        seq.observe_source_sequence(folder, state, config)


def test_fresh_epoch_observes_inherited_priority_catalog_without_inventing_service(tmp_path, monkeypatch):
    folder, state, config, owners, home = anchor_pool(tmp_path, monkeypatch)
    declaration = config['source_sequence']
    manifest = json.loads(Path(declaration['ready_manifest']).read_bytes())
    original = put(tmp_path/'original-ready.json', {**manifest, 'entries': manifest['entries'][:2]})
    declaration['priority_packets'] = put(tmp_path/'priority.json', dict(
        schema='natlang.teacher-priority-packet-continuation/1', original_ready=original,
        entries=manifest['entries'][2:], minimum_original_ordinal=0, max_consecutive=1,
        training_admission=False))
    state['config_snapshot'] = put(folder/'current-priority-config.json', config)
    seq.observe_source_sequence(folder, state, config)
    sequence = state['source_sequence']
    assert sequence['observed_entries'] == manifest['entries'][:2]
    assert sequence['initial_priority_catalog'] and not sequence.get('closed')
    reserved = seq.prepare_prefetched_source(folder, state, config, 22)
    assert reserved['ordinal'] == 2  # The existing priority rule, not a new selector.
    assert sequence['next_ordinal'] == 0 and 'pending' not in sequence
    assert not sequence.get('anchor_closed')
    assert seq._json(reserved['intent'], declaration)['priority']['resume_ordinal'] == 0


def test_anchor_declaration_requires_current_guardian_configuration(tmp_path, monkeypatch):
    folder, state, config, owners, home = anchor_pool(tmp_path, monkeypatch)
    config['source_sequence']['on_unready'] = 'unreviewed change'
    with pytest.raises(ValueError, match='guardian configuration'):
        seq.observe_source_sequence(folder, state, config)


def test_publisher_retains_anchor_history_after_later_packet_closures(tmp_path):
    from scripts.coreweave.teacher_incremental_export import _guardian_closed_sources
    anchor = put(tmp_path/'anchor-closed.json', dict(batch=21, started={'sha256': 'a'*64}))
    later = put(tmp_path/'later-closed.json', dict(batch=22, started={'sha256': 'b'*64}))
    state = {'source_sequence': {'anchor_closed': anchor, 'closed': {'22': later}}}
    before = copy.deepcopy(state)
    assert _guardian_closed_sources(state, 21) == {'21': anchor, '22': later}
    assert state == before
    state['source_sequence']['closed']['21'] = later
    with pytest.raises(ValueError, match='conflicts'):
        _guardian_closed_sources(state, 21)
