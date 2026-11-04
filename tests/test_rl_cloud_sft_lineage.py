"""Cloud joint-SFT lineages: window boundaries, staging, prune and exact fetch (scripts/coreweave/rl_cloud_sandbox.py),
and the sandbox form of DGX controls (scripts/coreweave/rl_cloud.py)."""
import hashlib
import io
import json
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.coreweave import artifact_zip_delta
from scripts.coreweave import rl_cloud_sandbox as sandbox
from scripts.coreweave.rl_cloud import sandbox_command, sandbox_controls


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_zip(path, entries):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def sft_lineage(tmp_path):
    L = tmp_path / 'root' / 'joint'
    window = L / 'w000000'
    window.mkdir(parents=True)
    parent = L / 'parent' / 'checkpoint.pt'
    parent.parent.mkdir()
    write_zip(parent, {'model/data/0': b'\x01' * 4096, 'model/data/1': b'\x02' * 4096})
    config = {'out': str(window), 'checkpoint': str(parent), 'mandatory_sft_only': True, 'id': 'lfm3-cloud-joint-w000000'}
    (window / 'config.json').write_text(json.dumps(config))
    (L / 'lineage.json').write_text(json.dumps({'schema': sandbox.LINEAGE_SCHEMA, 'windows': [
        {'index': 0, 'out': str(window), 'config': str(window / 'config.json')}]}))
    return L, window, parent


def test_sft_boundaries_select_the_actual_resume_checkpoints_and_skip_unchanged(tmp_path):
    L, window, parent = sft_lineage(tmp_path)
    exports = tmp_path / 'root' / '.exports' / 'joint'
    assert sandbox.progress(L) == {'out': str(window), 'config': str(window / 'config.json'), 'kind': 'sft',
                                   'phase': None, 'trained_rounds': 0, 'window_complete': False,
                                   'window_updates': None, 'rolling_slot': False, 'bound_rolling_resume': False}
    first = sandbox.stage(L, exports)
    assert first['checkpoints'] == [str(parent)] and first['window'] == 0
    assert sandbox.stage(L, exports)['seq'] == first['seq']  # unchanged boundary: no new export

    # an interrupted window: its rolling slot and the exact parent its rolling resume binds
    learner = window / 'learner'
    learner.mkdir()
    slot = learner / 'checkpoint.pt'
    write_zip(slot, {'model/data/0': b'\x01' * 4096, 'model/data/1': b'\x03' * 4096})
    (learner / '.checkpoint.pt.actor-view-source-x').write_bytes(b'copy')
    (learner / 'checkpoint.pt.reservation.lock').write_bytes(b'')
    (learner / 'progress.json').write_text('{}')
    (window / 'state.json').write_text(json.dumps({'checkpoint': str(parent), 'phase': 'updating_mandatory_sft',
                                                   'round': 0, 'trained_rounds': 0}))
    assert sandbox.progress(L)['rolling_slot'] is True
    rolling = sandbox.stage(L, exports)
    assert rolling['checkpoints'] == [str(slot), str(parent)] and rolling['seq'] == first['seq'] + 1
    checkpoint_blobs = [b for b in rolling['blobs'] if b['kind'] == 'checkpoint']
    assert [b['sha256'] for b in checkpoint_blobs] == [sha(slot), sha(parent)]
    window_blob = next(b for b in rolling['blobs'] if b['kind'] == 'window')
    listing = subprocess.run(f'zstd -dc {exports}/cas/{window_blob["sha256"]} | tar -t', shell=True,
                             capture_output=True, text=True, check=True).stdout.split()
    assert 'w000000/learner/progress.json' in listing and 'w000000/state.json' in listing
    assert not any(name.endswith('.pt') or '.pt.' in name for name in listing)

    # the window reaches its semantic-refresh boundary with optimizer updates
    (learner / 'report.json').write_text(json.dumps({'continuation_boundary': {
        'kind': 'semantic-refresh', 'reached': True, 'new_optimizer_updates': 3}}))
    (window / 'state.json').write_text(json.dumps({'checkpoint': str(parent), 'phase': 'awaiting_semantic_refresh',
                                                   'pending_native_checkpoint': str(slot), 'round': 0,
                                                   'trained_rounds': 0}))
    done = sandbox.progress(L)
    assert done['window_complete'] is True and done['window_updates'] == 3 and done['rolling_slot'] is None
    end = sandbox.stage(L, exports)
    assert end['checkpoints'] == [str(slot)] and end['phase'] == 'awaiting_semantic_refresh'


def test_window_activation_and_prune_keep_live_and_exported_checkpoints(tmp_path):
    L, window, parent = sft_lineage(tmp_path)
    exports = tmp_path / 'root' / '.exports' / 'joint'
    slot = window / 'learner' / 'checkpoint.pt'
    slot.parent.mkdir()
    write_zip(slot, {'model/data/0': b'\x09' * 4096})
    (window / 'state.json').write_text(json.dumps({'checkpoint': str(parent), 'phase': 'awaiting_semantic_refresh',
                                                   'pending_native_checkpoint': str(slot)}))
    end = sandbox.stage(L, exports)
    nxt = L / 'w000001'
    nxt.mkdir()
    (nxt / 'config.json').write_text(json.dumps({'out': str(nxt), 'checkpoint': str(slot), 'mandatory_sft_only': True}))
    sandbox.activate(L, nxt, nxt / 'config.json')
    assert sandbox.controller(L) == (nxt, nxt / 'config.json')
    with pytest.raises(ValueError, match='earlier window'):
        sandbox.activate(L, window, window / 'config.json')
    # the new window's first rolling slot, then its rolling control replaces the active config in place
    (nxt / 'learner').mkdir()
    write_zip(nxt / 'learner' / 'checkpoint.pt', {'model/data/0': b'\x0a' * 4096})
    (nxt / 'state.json').write_text(json.dumps({'checkpoint': str(slot), 'phase': 'updating_mandatory_sft'}))
    rolling = nxt / 'config-rolling.json'
    rolling.write_text(json.dumps({'out': str(nxt), 'checkpoint': str(slot), 'mandatory_sft_only': True,
                                   'rolling_sft_resume': {'schema': 'natlang.rolling-sft-resume/1'}}))
    sandbox.activate(L, nxt, rolling)
    index = json.loads((L / 'lineage.json').read_text())
    assert [w['out'] for w in index['windows']] == [str(window), str(nxt)]
    assert index['windows'][1]['config'] == str(rolling) and index['windows'][1]['controls'] == [str(nxt / 'config.json')]
    assert sandbox.progress(L)['bound_rolling_resume'] is True

    # nothing uploaded yet: nothing pruned
    sandbox.prune(L, exports)
    assert parent.exists() and slot.exists()
    staged = sandbox.stage(L, exports)
    (exports / f'{staged["seq"]:06d}.uploaded').write_text('')
    sandbox.prune(L, exports)
    # window 0's slot is window 1's parent (live) and its rolling resume's bound parent: kept, as is the new slot;
    # blobs only the superseded export named are dropped from the sandbox CAS
    assert slot.exists() and (nxt / 'learner' / 'checkpoint.pt').exists()
    assert staged['checkpoints'] == [str(nxt / 'learner' / 'checkpoint.pt'), str(slot)]
    remaining = {p.name for p in (exports / 'cas').iterdir()}
    assert remaining == {b['sha256'] for b in staged['blobs']} and end['seq'] < staged['seq']


def tar_zst(path, members, base):
    listing = '\n'.join(members)
    subprocess.run(f'tar -C {base} --no-recursion -cf - -T - | zstd -q -c > {path}', shell=True, check=True,
                   input=listing, text=True)
    return path


def blob(path, **extra):
    data = Path(path).read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
            'parts': [{'index': 0, 'offset': 0, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                       'url': Path(path).resolve().as_uri()}], **extra}


def test_fetch_reconstructs_delta_chain_and_restores_exact_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_CW_DELTA_MIN_FREE_BYTES', '0')
    store = tmp_path / 'store'
    store.mkdir()
    full = write_zip(store / 'full.pt', {'a': os.urandom(8192), 'b': b'\x00' * 8192})
    middle = store / 'middle.pt'
    with zipfile.ZipFile(full) as source:
        write_zip(middle, {'a': source.read('a'), 'b': b'\x00' * 4096 + b'\x05' * 4096})
    target = store / 'target.pt'
    with zipfile.ZipFile(middle) as source:
        write_zip(target, {'a': source.read('a'), 'b': b'\x07' * 8192, 'c': b'new'})
    specs = {}
    for name, base, out in (('mid', full, middle), ('tgt', middle, target)):
        built = artifact_zip_delta.build(base, out, store / name, min_free_bytes=0)
        specs[name] = {'encoding': 'zip-delta', 'sha256': sha(out), 'bytes': out.stat().st_size, 'base': sha(base),
                       'receipt': blob(store / name / 'receipt.json'),
                       'payload': blob(store / name / built['payload']['path'])}

    root = tmp_path / 'root'
    L = root / 'joint'
    seed_dir = tmp_path / 'seed' / 'joint'
    window = seed_dir / 'w000000'
    window.mkdir(parents=True)
    (window / 'config.json').write_text(json.dumps({'out': str(L / 'w000000'), 'checkpoint': str(L / 'parent/checkpoint.pt'),
                                                    'mandatory_sft_only': True}))
    (seed_dir / 'lineage.json').write_text(json.dumps({'schema': sandbox.LINEAGE_SCHEMA, 'windows': [
        {'index': 0, 'out': str(L / 'w000000'), 'config': str(L / 'w000000/config.json')}]}))
    top = tar_zst(store / 'top.tar.zst', ['lineage.json', 'w000000', 'w000000/config.json'], seed_dir)

    data = tmp_path / 'dgx' / 'data'
    data.mkdir(parents=True)
    (data / 'graphs.jsonl').write_text('{"row": 1}\n')
    (data / 'control.json').write_text('{"k": 2}')
    restored = tmp_path / 'restored'
    file_input = restored / 'graphs.jsonl'
    archived = restored / 'sub' / 'control.json'
    archived.parent.mkdir(parents=True)
    archived.write_text('{"k": 2}')
    archive = tar_zst(store / 'closure.tar.zst', [str(archived).lstrip('/')], '/')
    archived.unlink()
    runtime_src = tmp_path / 'rt'
    (runtime_src / 'runtime').mkdir(parents=True)
    (runtime_src / 'runtime' / 'runtime-input-closure.json').write_text('{"commit": "abc"}')
    (runtime_src / 'runtime-manifest.json').write_text('{}')
    runtime_archive = tar_zst(store / 'runtime.tar.zst', ['runtime', 'runtime/runtime-input-closure.json',
                                                           'runtime-manifest.json'], runtime_src)
    runtime = restored / 'runtime-v38'

    capability = {'schema': 'natlang.rl-cloud-parent/1', 'lineage_dir': str(L), 'seq': 'seed', 'trained_rounds': 0,
                  'state_sha256': None,
                  'blobs': [{'kind': 'top', 'encoding': 'tar.zst', **blob(top)},
                            {'kind': 'checkpoint', 'path': str(L / 'parent/checkpoint.pt'), **specs['tgt']}],
                  'bases': {sha(full): {'encoding': 'raw', **blob(full)}, sha(middle): specs['mid']},
                  'inputs': [{'kind': 'runtime', 'path': str(runtime), 'closure_sha256': sha(runtime_src / 'runtime/runtime-input-closure.json'), **blob(runtime_archive)},
                             {'kind': 'file', 'path': str(file_input), **blob(data / 'graphs.jsonl')},
                             {'kind': 'archive', 'path': str(root / '.inputs/joint-closure'),
                              'files': [{'path': str(archived), 'sha256': sha(data / 'control.json')}], **blob(archive)}],
                  'links': []}
    parent = tmp_path / 'rl-parent'
    parent.write_text(json.dumps(capability))
    assert sandbox.fetch(parent, L) == 0
    assert sha(L / 'parent/checkpoint.pt') == sha(target)
    assert file_input.read_text() == '{"row": 1}\n' and archived.read_text() == '{"k": 2}'
    assert json.loads((runtime / 'runtime-input-closure.json').read_text()) == {'commit': 'abc'}
    assert sandbox.controller(L) == (L / 'w000000', L / 'w000000/config.json')
    assert not (root / '.fetch-joint').exists()  # reconstructed bases were intermediates

    # a corrupted delta never becomes a checkpoint
    bad = json.loads(parent.read_text())
    bad['blobs'][1]['sha256'] = '0' * 64
    (L / 'parent/checkpoint.pt').unlink()
    parent.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        sandbox.fetch(parent, L)
    assert not (L / 'parent/checkpoint.pt').exists()


def test_sandbox_controls_drop_host_only_wrappers_and_relocate_resident_plan(tmp_path):
    wrapper = tmp_path / 'admit-learner-100g.py'
    wrapper.write_text("subprocess.run(['python','scripts/memory_ledger.py','adopt'])\n")
    other = tmp_path / 'tool.py'
    other.write_text('print(1)\n')
    lock = '/w/scripts/update_lock.sh'
    learner = ['py', str(wrapper), '--', lock, 'py', '-m', 'natlang_neuralese.rl.native_stage', '--out', '{learner_out}']
    assert sandbox_command(learner) == ['py', '-m', 'natlang_neuralese.rl.native_stage', '--out', '{learner_out}']
    assert sandbox_command(['py', str(other), '--', 'x']) == ['py', str(other), '--', 'x']
    config = {'learner_command': learner, 'seed_command': ['py', str(wrapper), '--', 'py', '-m', 'sft'],
              'llama_cpp_converter': '/dgx/llama', 'resident_data_directory': '/dgx/ops/resident-data',
              'trainer_parameters': {'resident_data_directory': '/dgx/ops/resident-data', 'seed': 1},
              'seed_trainer_parameters': {'resident_data_directory': '/dgx/ops/resident-data'}}
    result = sandbox_controls(config, tmp_path / 'w000000')
    assert result['seed_command'] == ['py', '-m', 'sft'] and result['llama_cpp_converter'] is None
    resident = str(tmp_path / 'w000000' / 'resident-data')
    assert result['resident_data_directory'] == resident
    assert result['trainer_parameters'] == {'resident_data_directory': resident, 'seed': 1}
    assert result['seed_trainer_parameters'] == {'resident_data_directory': resident}
    assert config['trainer_parameters']['resident_data_directory'] == '/dgx/ops/resident-data'  # input untouched
    with pytest.raises(ValueError, match='update lock'):
        sandbox_controls({'actor_command': ['py', lock]}, tmp_path)


def test_select_boundary_follows_the_live_dgx_window(tmp_path):
    from scripts.coreweave.rl_cloud import select_boundary
    ops, runs = tmp_path / 'ops', tmp_path / 'runs'
    recipe = tmp_path / 'recipe.json'
    recipe.write_text(json.dumps({'stages': [{'id': 'sft', 'parameters': {'refresh_after_updates': 64}}]}))

    def window(name, checkpoint, prior=None, extra=None):
        out = runs / name
        (out / 'learner').mkdir(parents=True)
        config = {'out': str(out), 'checkpoint': str(checkpoint), 'mandatory_sft_only': True, 'runtime': '/rt/ts-host',
                  'production_recipe': str(recipe), 'production_stage': 'sft', **(extra or {})}
        if prior is not None:
            config['branch_origin'] = {'prior_control': {'path': str(prior)}}
        (ops / name).mkdir(parents=True)
        (ops / name / 'config.json').write_text(json.dumps(config))
        return out, ops / name / 'config.json'

    base = write_zip(tmp_path / 'base.pt', {'w': b'\x01' * 64})
    first, first_control = window('w84', base)
    earned = write_zip(first / 'learner' / 'checkpoint.pt', {'w': b'\x02' * 64})
    (first / 'state.json').write_text(json.dumps({'phase': 'awaiting_semantic_refresh',
                                                  'pending_native_checkpoint': str(earned)}))
    done = select_boundary({'controller': str(first_control)})
    assert (done['mode'], done['previous_controller'], done['parent_sha256']) == ('next-window', str(first_control), sha(earned))
    assert done['dgx_window_in_flight'] is None

    second, second_control = window('w179', earned, prior=first_control)
    (second / 'state.json').write_text(json.dumps({'phase': 'updating_mandatory_sft', 'checkpoint': str(earned)}))
    with pytest.raises(SystemExit, match='successor'):
        select_boundary({'controller': str(first_control)})
    rebuilt = select_boundary({'controller': str(second_control)})
    assert (rebuilt['mode'], rebuilt['previous_controller'], rebuilt['window']) == ('same-window', str(first_control), 64)
    assert rebuilt['parent_sha256'] == sha(earned) and rebuilt['dgx_window_in_flight']['phase'] == 'updating_mandatory_sft'

    # the DGX retries the window in its rolling slot: the retry control names the window's prior control too
    write_zip(second / 'learner' / 'checkpoint.pt', {'w': b'\x03' * 64})
    retry_dir = ops / 'w179-rolling186'
    retry_dir.mkdir()
    retry = json.loads(second_control.read_text())
    retry['rolling_sft_resume'] = {'schema': 'natlang.rolling-sft-resume/1'}
    (retry_dir / 'config.json').write_text(json.dumps(retry))
    (retry_dir / 'preparation.json').write_text(json.dumps({'prior_control': {'path': str(second_control)},
                                                            'slot_global_step': 186}))
    with pytest.raises(SystemExit, match='successor'):
        select_boundary({'controller': str(second_control)})
    rolling = select_boundary({'controller': str(retry_dir / 'config.json')})
    assert rolling['mode'] == 'same-window' and rolling['parent_sha256'] == sha(earned)
    assert rolling['dgx_window_in_flight']['rolling_slot_global_step'] == 186
    assert rolling['dgx_window_in_flight']['rolling_slot'] == str(second / 'learner' / 'checkpoint.pt')


def test_tune_shard_exports_its_stage_directory_and_final_status(tmp_path):
    L = tmp_path / 'root' / 'tune-s0'
    L.mkdir(parents=True)
    stage_dir = tmp_path / 'stage'
    arms = stage_dir / 'programs' / 'refund-decision' / '.natlang' / 'tune' / 'evaluations' / 'development'
    arms.mkdir(parents=True)
    (stage_dir / 'natlang-config').mkdir()
    (stage_dir / 'natlang-config' / 'config.json').write_text('{}')
    (stage_dir / 'hash-cache.jsonl').write_text('{}\n')
    (stage_dir / 'operation.json').write_text('{"schema": "x"}')
    (L / 'tune.json').write_text(json.dumps({'stage_dir': str(stage_dir), 'argv': [], 'module': 'm'}))
    exports = tmp_path / 'root' / '.exports' / 'tune-s0'
    first = sandbox.stage(L, exports)
    assert first['kind'] == 'tune' and first['phase'] == 'running' and first['final'] is False
    assert sandbox.stage(L, exports)['seq'] == first['seq']
    (arms / 'untuned.json').write_text('{"roots": 3}')
    (stage_dir / 'tunes').mkdir()
    (stage_dir / 'tunes' / 'refund-decision.timing.json').write_text('{}')
    (L / 'tune-status.json').write_text(json.dumps({'status': 'done', 'rc': 0}))
    final = sandbox.stage(L, exports)
    assert final['final'] is True and final['phase'] == 'done'
    assert final['tune_progress']['programs'] == {'refund-decision': {'development': ['untuned']}}
    assert final['tune_progress']['tunes'] == ['refund-decision.timing.json']
    stage_blob = next(b for b in final['blobs'] if b['kind'] == 'stage')
    listing = subprocess.run(f'zstd -dc {exports}/cas/{stage_blob["sha256"]} | tar -t', shell=True,
                             capture_output=True, text=True, check=True).stdout.split()
    assert 'operation.json' in listing and not any(n.startswith(('natlang-config', 'hash-cache')) for n in listing)
    sandbox.prune(L, exports)  # no controller config: CAS only

    # a replacement sandbox resumes the stage directory from the export
    restored = tmp_path / 'elsewhere' / 'stage'
    capability = {'schema': 'natlang.rl-cloud-parent/1', 'lineage_dir': str(tmp_path / 'r2' / 'tune-s0'), 'seq': final['seq'],
                  'trained_rounds': 0, 'state_sha256': None, 'inputs': [], 'links': [],
                  'blobs': [{**blob(exports / 'cas' / b['sha256']), 'kind': b['kind'], 'encoding': b['encoding'],
                             **({'path': str(restored)} if b['kind'] == 'stage' else {})} for b in final['blobs']]}
    parent = tmp_path / 'rl-parent'
    parent.write_text(json.dumps(capability))
    assert sandbox.fetch(parent, tmp_path / 'r2' / 'tune-s0') == 0
    assert (restored / 'programs/refund-decision/.natlang/tune/evaluations/development/untuned.json').read_text() == '{"roots": 3}'
    assert json.loads((tmp_path / 'r2' / 'tune-s0' / 'tune-status.json').read_text())['status'] == 'done'


def test_tune_shard_launches_always_restore_the_seed_parent(tmp_path, monkeypatch):
    from scripts.coreweave import rl_cloud
    monkeypatch.setattr(rl_cloud, 'hub', lambda run: tmp_path)
    parent = {'kind': 'checkpoint', 'encoding': 'raw', 'sha256': 'a' * 64, 'bytes': 1, 'path': '/p/checkpoint.pt'}
    (tmp_path / 'exports' / 'tune-s0').mkdir(parents=True)
    (tmp_path / 'exports' / 'tune-s0' / 'seed.json').write_text(json.dumps(
        {'seq': 'seed', 'lineage': 'tune-s0', 'kind': 'tune', 'blobs': [{'kind': 'top', 'encoding': 'tar.zst', 'sha256': 'b' * 64}, parent]}))
    export = {'seq': 3, 'lineage': 'tune-s0', 'kind': 'tune', 'blobs': [{'kind': 'top', 'encoding': 'tar.zst', 'sha256': 'c' * 64},
                                                                      {'kind': 'stage', 'encoding': 'tar.zst', 'sha256': 'd' * 64}]}
    assert rl_cloud.launch_blobs({}, export)[-1] == parent
    assert rl_cloud.record_digests({}, export) == {'a' * 64, 'c' * 64, 'd' * 64}
    sft = {'seq': 3, 'lineage': 'joint', 'blobs': export['blobs']}
    assert rl_cloud.launch_blobs({}, sft) == export['blobs']


def test_interleaved_controller_exports_its_active_sft_window_rolling_slot(tmp_path):
    out = tmp_path / 'w000004'
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'parent')
    window = out / 'sft-window-000000'
    (window / 'learner').mkdir(parents=True)
    slot = window / 'learner' / 'checkpoint.pt'
    slot.write_bytes(b'rolling')
    config = {'interleaved_rl_stage': 'interleaved_jolt_rl_v1'}
    state = {'checkpoint': str(parent), 'interleave': {'active_window': {'window': 0, 'directory': str(window)}}}
    assert sandbox.resume_checkpoints(out, config, state) == [slot, parent]
    assert sandbox.resume_checkpoints(out, config, {'checkpoint': str(parent), 'interleave': {}}) == [parent]
    assert sandbox.resume_checkpoints(out, {}, state) == [parent]


def test_top_export_carries_lineage_level_session_controls(tmp_path):
    L, window, parent = sft_lineage(tmp_path)
    session = L / 'diloco-s2'
    (session / 'workers' / 'worker-0').mkdir(parents=True)
    (session / 'recipe-runtime.json').write_text('{}')
    (session / 'workers' / 'worker-0' / 'config.json').write_text('{}')
    (session / 'x.lock').write_text('')
    exports = tmp_path / 'root' / '.exports' / 'joint'
    staged = sandbox.stage(L, exports)
    top = next(b for b in staged['blobs'] if b['kind'] == 'top')
    listing = subprocess.run(f'zstd -dc {exports}/cas/{top["sha256"]} | tar -t', shell=True,
                             capture_output=True, text=True, check=True).stdout.split()
    assert 'diloco-s2/recipe-runtime.json' in listing and 'diloco-s2/workers/worker-0/config.json' in listing
    assert not any(m.startswith(('w000000', 'parent')) for m in listing) and 'diloco-s2/x.lock' not in listing
