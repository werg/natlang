"""Actual full-state derivatives retain source and consumed-input authority."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from natlang_neuralese.model.native_bundle import write_actor_view
from natlang_neuralese.model.native_checkpoint_control import (
    _file_binding, source_stat, read_checkpoint_control, write_checkpoint_control,
    resolve_checkpoint_controls)
from natlang_neuralese.rl.native_raw_policy import prepare_raw_policy
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.train.loop import capture_training_rng_state

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location('actual_checkpoint_control_dependencies', ROOT/'scripts/coreweave/provider_dependencies.py')
DEPENDENCIES = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEPENDENCIES
SPEC.loader.exec_module(DEPENDENCIES)


@pytest.fixture(params=['lfm', 'mellum'])
def parent(tmp_path, request):
    from test_native_serving_adapters import engine_fixture
    engine = engine_fixture(tmp_path, request.param)
    state = CheckpointPolicy.load(engine.native_checkpoint)
    old = tmp_path/'carried-train.jsonl'; old.write_text('exact old complete training row\n')
    held = tmp_path/'carried-held.jsonl'; held.write_text('exact protected held row\n')
    pending = tmp_path/'whole-pending.jsonl'; pending.write_text('whole long admitted graph\n')
    pending_manifest = tmp_path/'whole-pending-sources.json'
    pending_manifest.write_text(json.dumps({'schema': 'natlang.native-graph-sources/1',
        'sources': [{'kind': 'native-graphs', **_file_binding(pending)}]}))
    declaration = tmp_path/'pending-declaration.json'
    declaration.write_text(json.dumps({'schema': 'natlang.native-pending-sources/1',
        'tokenizer_fingerprint': state['model_config']['tokenizer_fingerprint'],
        'sources': [{'full_cohort': _file_binding(pending_manifest),
            'pending_native_graphs': _file_binding(pending_manifest),
            'rows': [{'id': 'long', 'binding': '2'*64, 'required_visits': 1}]}]}))
    carried = {'required_visits': {'old': 1, 'long': 1}, 'exposure_counts': {},
        'source_receipts': [_file_binding(old), _file_binding(held)],
        'pending_source_declarations': {'whole-long': {'binding': _file_binding(declaration)}}}
    state.update(step=71, training={'schema': TRAINING_SCHEMA, 'global_step': 71,
        'optimizer': {'state': {'actual_moment': torch.arange(19.)}},
        'rng': capture_training_rng_state('cpu'),
        'source_receipts': {'actual_source': 'retained'}, 'sft_progress': {'carried': carried}})
    path = tmp_path/'full.pt'
    torch.save(state, path)
    state = CheckpointPolicy.load(path)
    stat = path.stat()
    view = write_actor_view(path, tmp_path/'views', validated_state=state,
        decoded_source_stat=(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    options = {'namespace': str(tmp_path), 'readonly_input_roots': [str(ROOT)]}
    yield path, state, view, options
    engine.store.close()


def publish(parent, root):
    path, state, view, options = parent
    return write_checkpoint_control(path, root, validated_state=state,
        decoded_source_stat=source_stat(path), actor_view=view,
        dependency_collector=DEPENDENCIES.dependency_closure, **options)


def test_actual_both_ports_prepare_without_full_decode_and_preserve_exact_closure(parent, tmp_path, monkeypatch):
    path, state, view, options = parent
    expected = DEPENDENCIES.dependency_closure({}, _checkpoint_payload=state, **options)
    assert {Path(item['path']).name for item in expected['files']} >= {
        'carried-train.jsonl', 'carried-held.jsonl', 'whole-pending.jsonl',
        'whole-pending-sources.json', 'pending-declaration.json'}
    ref = publish(parent, tmp_path/'controls-v1')
    value = read_checkpoint_control(path, ref)
    assert value['dependencies'] == expected
    assert publish(parent, tmp_path/'controls-v1') == ref
    before_rng = torch.get_rng_state().clone()
    original = CheckpointPolicy.load
    loads = []
    def compact_only(p, *args, **kwargs):
        assert Path(p).resolve() != path.resolve(), 'repeated full source decode'
        loads.append(Path(p))
        return original(p, *args, **kwargs)
    monkeypatch.setattr(CheckpointPolicy, 'load', compact_only)
    actual = prepare_raw_policy(path, base=state['model_config']['base'],
        tokenizer_out=tmp_path/'reloaded-tokenizer', checkpoint_control=ref)
    assert actual['native_identity'] == value['raw_policy']['native_identity']
    assert actual['history_reasoning'] == value['raw_policy']['history_reasoning']
    assert actual['actor_view'] == view and len(loads) == 1
    assert torch.equal(before_rng, torch.get_rng_state())
    assert state['training']['optimizer']['state']['actual_moment'].equal(torch.arange(19.))


@pytest.mark.parametrize('field', ['source', 'history', 'view', 'producer', 'step'])
def test_control_tampering_rejects_actual_source(parent, tmp_path, field):
    path, _, _, _ = parent
    ref = publish(parent, tmp_path/'controls')
    value = json.loads(Path(ref['path']).read_text())
    if field == 'source': value['source']['sha256'] = '0'*64
    elif field == 'history': value['raw_policy']['history_reasoning'] = {'policy': 'invented'}
    elif field == 'view': value['actor_view']['sha256'] = '0'*64
    elif field == 'producer': value['producer']['sha256'] = '0'*64
    else: value['step'] += 1
    target = tmp_path/'tampered.json'; target.write_text(json.dumps(value))
    with pytest.raises(ValueError): read_checkpoint_control(path, _file_binding(target))


def test_no_work_selection_is_frozen_and_versioned_publication_supports_refreshed_controls(parent, tmp_path):
    path, _, _, _ = parent
    root = tmp_path/'controls-v1'
    config = {'checkpoint': str(path), 'native_checkpoint_control_directory': str(root)}
    first = resolve_checkpoint_controls(config, _file_binding(path))
    assert 'native_checkpoint_controls' not in first
    assert 'native_checkpoint_control_directory' not in first
    ref = publish(parent, root)
    # A request already frozen onto full loading cannot pick up later metadata.
    assert resolve_checkpoint_controls(first, _file_binding(path)) == first
    chosen = resolve_checkpoint_controls(config, _file_binding(path))
    assert chosen['native_checkpoint_controls'][_file_binding(path)['sha256']] == ref
    second = publish(parent, tmp_path/'controls-v2')
    assert second != ref
    assert read_checkpoint_control(path, second)['source'] == read_checkpoint_control(path, ref)['source']


def test_source_mutation_with_restored_mtime_rejects_even_after_cached_read(parent, tmp_path):
    import os
    path, _, _, _ = parent
    ref = publish(parent, tmp_path/'controls')
    read_checkpoint_control(path, ref)
    old = path.stat()
    with path.open('r+b') as stream:
        first = stream.read(1); stream.seek(0); stream.write(bytes([first[0]^1]))
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError): read_checkpoint_control(path, ref)


def test_older_qualified_hasher_falls_back_to_existing_full_hasher(tmp_path, monkeypatch):
    from natlang_neuralese.model import native_checkpoint_control as control
    path = tmp_path/'payload'; path.write_bytes(b'actual')
    def old_cached(path): pytest.fail('legacy archive cache must not attest controls')
    monkeypatch.setattr(control, 'sha256_file_hex_cached', old_cached)
    assert control._file_binding(path)['sha256'] == __import__('hashlib').sha256(b'actual').hexdigest()


def test_provider_full_and_compact_consumed_closures_match_and_context_change_falls_back(parent, tmp_path, monkeypatch):
    path, state, _, options = parent
    ref = publish(parent, tmp_path/'controls')
    request = {'parent_checkpoint': _file_binding(path)}
    # A real round config selects controls; a standalone checkpoint request
    # does not grant an unrelated derivative authority.
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'native_checkpoint_controls': {_file_binding(path)['sha256']: ref}}))
    request.update(kind='round', config=_file_binding(config))
    full_config = tmp_path/'full-config.json'; full_config.write_text('{}')
    full = DEPENDENCIES.dependency_closure({**request, 'config': _file_binding(full_config)}, **options)
    original = CheckpointPolicy.load
    calls = []
    def counted(p, *a, **k): calls.append(Path(p)); return original(p, *a, **k)
    monkeypatch.setattr(CheckpointPolicy, 'load', counted)
    compact = DEPENDENCIES.dependency_closure(request, **options)
    assert not calls
    full_inputs = {x['path']: x for x in full['files'] if x['path'] != str(full_config)}
    compact_inputs = {x['path']: x for x in compact['files']}
    assert all(compact_inputs[name] == item for name, item in full_inputs.items())
    assert [item for item in compact['assets'] if item['kind'] != 'input-file'] == full['assets']
    DEPENDENCIES.dependency_closure(request, namespace=str(tmp_path),
        readonly_input_roots=[str(ROOT), str(tmp_path/'extra-readonly')])
    assert calls == [path]


def test_actual_exporter_controls_reuse_single_decode_and_optional_failure_preserves_full_pin(parent, tmp_path, monkeypatch, capsys):
    import shutil
    path, _, _, _ = parent
    spec = importlib.util.spec_from_file_location('actual_controls_cursor', ROOT/'scripts/coreweave/cursor_checkpoint_export.py')
    cursor = importlib.util.module_from_spec(spec); spec.loader.exec_module(cursor)
    root = tmp_path/'production'
    slot = root/'rounds/lfm/production/round-000/learner/checkpoint.pt'
    slot.parent.mkdir(parents=True); shutil.copyfile(path, slot)
    controls_spec = tmp_path/'control-spec.json'
    controls_spec.write_text(json.dumps({'producer': _file_binding(ROOT/'training/neuralese/natlang_neuralese/model/native_checkpoint_control.py'),
        'dependency_producer': _file_binding(ROOT/'scripts/coreweave/provider_dependencies.py'),
        'directory': str(tmp_path/'published-controls'), 'namespace': str(tmp_path),
        'readonly_input_roots': [str(ROOT)]}))
    original = torch.load; loads = []
    def counted(p, *a, **k): loads.append(Path(p)); return original(p, *a, **k)
    monkeypatch.setattr(torch, 'load', counted)
    code = cursor.remote_code(role='lfm', root=root, known={}, last=-1, extra=None,
        pin_directory=tmp_path/'pins', minimum_available_bytes=0,
        actor_view_directory=tmp_path/'views-exported', actor_view_runtime=ROOT/'training/neuralese',
        checkpoint_control_spec=_file_binding(controls_spec))
    scope = {}
    try: exec(compile(code, '<actual-controls-cursor>', 'exec'), scope)
    finally: scope['remote_lock'].close()
    chosen = json.loads(capsys.readouterr().out.splitlines()[-1])['chosen']
    assert 'checkpoint_control_error' not in chosen, chosen
    assert sum('cursor-observation' in p.name for p in loads) == 1
    assert len(loads) == 2  # full state once + actual compact view readback
    value = read_checkpoint_control(chosen['pin'], chosen['checkpoint_control'])
    assert value['step'] == 71 and value['source']['bytes'] == chosen['bytes']
    # An explicitly changed optional spec never blocks the durable full source.
    controls_spec.write_text('corrupt optional spec')
    scope = {}
    try: exec(compile(code, '<actual-controls-cursor-fallback>', 'exec'), scope)
    finally: scope['remote_lock'].close()
    failed = json.loads(capsys.readouterr().out.splitlines()[-1])['chosen']
    assert failed['checkpoint_control_error']['fallback'] == 'complete-checkpoint'
    assert Path(failed['pin']).is_file() and failed['actor_view'] == chosen['actor_view']


def test_actual_bridge_seals_full_fallback_before_later_publication(parent, tmp_path):
    from natlang_neuralese.rl.native_bridge import write_request
    path, _, _, _ = parent
    config_path = tmp_path/'round-config.json'
    directory = tmp_path/'later-controls'
    config_path.write_text(json.dumps({'checkpoint': str(path),
        'native_checkpoint_control_directory': str(directory)}))
    first = write_request(tmp_path/'queue', kind='round', out=tmp_path/'operation', config=config_path)
    request = json.loads(first.read_text())
    frozen_path = Path(request['config']['path'])
    frozen = json.loads(frozen_path.read_text())
    assert 'native_checkpoint_control_directory' not in frozen
    assert 'native_checkpoint_controls' not in frozen
    original = frozen_path.read_bytes()
    reference = publish(parent, directory)
    assert frozen_path.read_bytes() == original
    second = write_request(tmp_path/'queue', kind='round', out=tmp_path/'next-operation', config=config_path)
    selected = json.loads(Path(json.loads(second.read_text())['config']['path']).read_text())
    assert selected['native_checkpoint_controls'][_file_binding(path)['sha256']] == reference
    assert selected['native_checkpoint_control_resolution']['source'] == _file_binding(path)


def test_changed_dependency_verifier_uses_actual_full_checkpoint_path(parent, tmp_path, monkeypatch):
    path, _, _, options = parent
    ref = publish(parent, tmp_path/'controls')
    copy_path = tmp_path/'later_dependency_verifier.py'
    copy_path.write_text((ROOT/'scripts/coreweave/provider_dependencies.py').read_text()+'\n# later immutable verifier revision\n')
    spec = importlib.util.spec_from_file_location('later_checkpoint_dependencies', copy_path)
    later = importlib.util.module_from_spec(spec); sys.modules[spec.name] = later; spec.loader.exec_module(later)
    config = tmp_path/'selected-control.json'
    config.write_text(json.dumps({'native_checkpoint_controls': {_file_binding(path)['sha256']: ref}}))
    original = CheckpointPolicy.load; calls = []
    def counted(p, *a, **k): calls.append(Path(p)); return original(p, *a, **k)
    monkeypatch.setattr(CheckpointPolicy, 'load', counted)
    later.dependency_closure({'kind': 'round', 'parent_checkpoint': _file_binding(path),
        'config': _file_binding(config)}, **options)
    assert calls == [path]


def test_transport_uses_actual_full_parent_alias_without_ferrying_original_pin(parent, tmp_path, monkeypatch):
    import shutil
    from natlang_neuralese.model.native_checkpoint_control import checkpoint_control_dependencies
    path, _, _, _ = parent
    reference = publish(parent, tmp_path/'controls')
    alias = tmp_path/'actual-learner-parent.pt'; shutil.copyfile(path, alias)
    path.unlink()  # Immutable transport source may legitimately be retired.
    control = read_checkpoint_control(alias, reference)
    monkeypatch.setattr(CheckpointPolicy, 'load', lambda *a, **k: pytest.fail('transport decoded weights'))
    files = checkpoint_control_dependencies(reference, checkpoint=alias)
    assert str(path) not in {row['path'] for row in files}
    assert str(alias) not in {row['path'] for row in files}
    assert control['source']['sha256'] == _file_binding(alias)['sha256']


def test_explicit_old_runtime_hash_sidecar_reuses_strong_cache_without_replacing_source_module(tmp_path, monkeypatch):
    import builtins
    from natlang_neuralese.model import native_checkpoint_control as control
    from natlang_neuralese.common import hashing
    original_cached = hashing.sha256_file_hex_cached
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path/'strong-hashes.jsonl'))
    monkeypatch.setattr(control, 'sha256_file_hex_cached', lambda path: pytest.fail('legacy receipt cache'))
    monkeypatch.setattr(control, 'sha256_file_hex', hashing.sha256_file_hex)
    monkeypatch.setattr(control, '_HASHING_SIDECAR', None)
    helper = _file_binding(ROOT/'training/neuralese/natlang_neuralese/common/hashing.py')
    assert control.install_control_hashing(helper) == helper
    payload = tmp_path/'full-source'; payload.write_bytes(b'actual full source bytes')
    real_open = builtins.open; reads = []
    def tracked(p, *a, **k):
        if Path(p) == payload and a and a[0] == 'rb': reads.append(p)
        return real_open(p, *a, **k)
    monkeypatch.setattr(builtins, 'open', tracked)
    assert control._file_binding(payload) == control._file_binding(payload)
    assert len(reads) == 1
    assert hashing.sha256_file_hex_cached is original_cached
    with pytest.raises(ValueError):
        control.install_control_hashing({**helper, 'sha256': '0'*64})


def test_actual_exporter_rejects_mutation_across_the_original_full_decode(parent, tmp_path, monkeypatch, capsys):
    import shutil
    import os
    path, _, _, _ = parent
    spec = importlib.util.spec_from_file_location('actual_controls_mutated_cursor', ROOT/'scripts/coreweave/cursor_checkpoint_export.py')
    cursor = importlib.util.module_from_spec(spec); spec.loader.exec_module(cursor)
    root = tmp_path/'production'; slot = root/'rounds/lfm/production/round-000/learner/checkpoint.pt'
    slot.parent.mkdir(parents=True); shutil.copyfile(path, slot)
    producer_spec = tmp_path/'spec.json'
    producer_spec.write_text(json.dumps({'producer': _file_binding(ROOT/'training/neuralese/natlang_neuralese/model/native_checkpoint_control.py'),
        'dependency_producer': _file_binding(ROOT/'scripts/coreweave/provider_dependencies.py'),
        'hashing_sidecar': _file_binding(ROOT/'training/neuralese/natlang_neuralese/common/hashing.py'),
        'directory': str(tmp_path/'controls'), 'namespace': str(tmp_path), 'readonly_input_roots': [str(ROOT)]}))
    original = torch.load
    def mutate_after_actual_decode(p, *a, **k):
        result = original(p, *a, **k)
        if 'cursor-observation' in Path(p).name:
            old = Path(p).stat()
            with Path(p).open('r+b') as stream:
                first = stream.read(1); stream.seek(0); stream.write(bytes([first[0]^1]))
            os.utime(p, ns=(old.st_atime_ns, old.st_mtime_ns))
        return result
    monkeypatch.setattr(torch, 'load', mutate_after_actual_decode)
    code = cursor.remote_code(role='lfm', root=root, known={}, last=-1, extra=None,
        pin_directory=tmp_path/'pins', minimum_available_bytes=0,
        actor_view_directory=tmp_path/'views', actor_view_runtime=ROOT/'training/neuralese',
        checkpoint_control_spec=_file_binding(producer_spec))
    scope = {}
    try:
        with pytest.raises(ValueError, match='changed across decode'):
            exec(compile(code, '<mutated-decoded-source>', 'exec'), scope)
    finally: scope['remote_lock'].close()
    assert not list((tmp_path/'pins').glob('*.pt'))
    assert not (tmp_path/'controls').exists()


def test_exporter_retains_pre_view_pin_witness_instead_of_resynthesizing_it(parent, tmp_path, monkeypatch):
    import shutil
    import os
    from natlang_neuralese.model import native_bundle
    path, _, _, _ = parent
    spec = importlib.util.spec_from_file_location('actual_controls_pre_hook_cursor', ROOT/'scripts/coreweave/cursor_checkpoint_export.py')
    cursor = importlib.util.module_from_spec(spec); spec.loader.exec_module(cursor)
    root = tmp_path/'production'; slot = root/'rounds/lfm/production/round-000/learner/checkpoint.pt'
    slot.parent.mkdir(parents=True); shutil.copyfile(path, slot)
    producer_spec = tmp_path/'spec.json'; producer_spec.write_text('{}')
    original = native_bundle.write_actor_view
    def mutate_between_view_and_controls(source, *a, **k):
        result = original(source, *a, **k)
        source = Path(source); old = source.stat()
        with source.open('r+b') as stream:
            first = stream.read(1); stream.seek(0); stream.write(bytes([first[0]^1]))
        os.utime(source, ns=(old.st_atime_ns, old.st_mtime_ns))
        return result
    monkeypatch.setattr(native_bundle, 'write_actor_view', mutate_between_view_and_controls)
    code = cursor.remote_code(role='lfm', root=root, known={}, last=-1, extra=None,
        pin_directory=tmp_path/'pins', minimum_available_bytes=0,
        actor_view_directory=tmp_path/'views', actor_view_runtime=ROOT/'training/neuralese',
        checkpoint_control_spec=_file_binding(producer_spec))
    scope = {}
    try:
        with pytest.raises(ValueError, match='pin changed during optional actor view'):
            exec(compile(code, '<mutated-before-hook>', 'exec'), scope)
    finally: scope['remote_lock'].close()
    assert not (tmp_path/'controls').exists()
