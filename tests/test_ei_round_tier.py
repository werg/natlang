import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'ei_round_tier.py'


def load():
    spec = importlib.util.spec_from_file_location('ei_round_tier', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # the copy's global os.sync() waits for every filesystem: on a host whose HDD is saturated (2026-11-24 22:40, a
    # live collection writing there) a test hung on it for minutes. Its files are fdatasync'ed one by one already
    module.os = SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith('__')})
    module.os.sync = lambda: None
    return module


def sha(c):
    return c * 64


def write(path, data=b'x'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def driver(tmp_path):
    root = tmp_path / 'ops' / 'ei' / 'aei9'
    lineage = [{'round_id': None, 'sha256': sha('0'), 'version': 0}] + \
        [{'round_id': f'r000{i}', 'sha256': sha(str(i)), 'version': i} for i in (1, 2, 3, 4)]
    state = {'lineage': lineage, 'active_round': {'round_id': 'r0005', 'base_sha256': sha('4')},
             'measuring': {'candidate': {'sha256': sha('2')}, 'reference': {'sha256': sha('0')}}}
    write(root / 'phase-state.json', json.dumps(state).encode())
    for i in range(5):
        for suffix in ('', '-remote0'):
            c = root / 'collect' / f'v000{i}-{sha(str(i))[:12]}{suffix}'
            write(c / 'runs.jsonl')
            write(c / 'groups' / 'g000001' / 'rollouts.jsonl', b'r' * 100)
            write(c / 'actor' / 'sglang-hf' / 'model.safetensors', b'w' * 50)
            write(c / 'actor' / 'blocks' / 'b.safetensors', b'b' * 50)
    for i in range(1, 7):
        r = root / f'r000{i}'
        write(r / 'policy' / 'checkpoint.pt', b'c' * 30)
        write(r / 'policy' / 'sglang-hf' / 'model.safetensors', b'h' * 30)
        write(r / 'policy' / 'gguf' / 'm.gguf', b'g' * 30)
        write(r / 'sft' / 'learner-state' / 'learner_manifest.json')
        write(r / 'sft' / 'factors' / 'adapter_factors_fp32.safetensors', b'f' * 20)
        write(r / 'sft' / 'train.jsonl')
        write(r / 'mix' / 'chains.pkl', b'p' * 40)
        write(r / 'mix' / 'rows.jsonl.zst', b'z')
        write(r / 'mix' / 'trained-units.json', b'{}')
        write(r / 'filled' / 'chains.pkl', b'q' * 10)
    return root, state


def test_plan_keeps_current_previous_and_measured(tmp_path):
    m = load()
    root, state = driver(tmp_path)
    acts = {(a['op'], str(Path(a['path']).relative_to(root))) for a in m.plan(root, state, None)}
    collections = {p for op, p in acts if p.startswith('collect/')}
    # policies 3 and 4 (lineage tail) and the active base 4 stay; 0, 1, 2 go (2 is held's candidate: checkpoint only)
    assert ('move', 'collect/v0001-111111111111') in acts and ('move', 'collect/v0002-222222222222-remote0') in acts
    assert ('delete', 'collect/v0000-000000000000/actor/sglang-hf') in acts
    assert not any(p.startswith(('collect/v0003', 'collect/v0004')) for p in collections)
    # rounds older than the previous one: r0001, r0002 (r0003/r0004 lineage tail, r0005 active, r0006 newer)
    assert {p.split('/')[0] for op, p in acts if p.startswith('r')} == {'r0001', 'r0002'}
    assert ('delete', 'r0001/policy/sglang-hf') in acts and ('delete', 'r0002/policy/gguf') in acts
    assert ('move', 'r0001/policy/checkpoint.pt') in acts and ('move', 'r0001/sft/factors') in acts
    assert ('move', 'r0002/policy/checkpoint.pt') not in acts       # held's candidate: rollback needs it
    assert ('move', 'r0002/sft/factors') not in acts
    assert ('move', 'r0002/mix/chains.pkl') in acts and ('move', 'r0002/filled') in acts
    assert not any(p.endswith(('rows.jsonl.zst', 'trained-units.json', 'train.jsonl')) for op, p in acts)


def test_plan_never_tiers_the_collections_of_the_rollback_targets(tmp_path):
    """2026-11-24 19:12: aei1 rolled back to r1, whose collections ei_round_tier had moved to the HDD that morning
    (only the last two lineage entries were kept); its live collectors then wrote there. The reference and start
    policies' collections (and NVMe continuations ``-nN``) stay."""
    m = load()
    root, state = driver(tmp_path)
    state['reference_policy'] = {'sha256': sha('1'), 'checkpoint': '/r1.pt'}
    state['start_policy'] = {'sha256': sha('0'), 'checkpoint': '/s.pt'}
    write(root / 'collect' / f'v0001-{sha("1")[:12]}-n1' / 'runs.jsonl')
    acts = {str(Path(a['path']).relative_to(root)) for a in m.plan(root, state, None)}
    assert not any(p.startswith(('collect/v0000', 'collect/v0001')) for p in acts)
    assert 'collect/v0002-222222222222' in acts


def test_apply_moves_with_symlinks_and_keeps_referenced(tmp_path, monkeypatch):
    m = load()
    root, state = driver(tmp_path)
    monkeypatch.setattr(m, 'OPS', tmp_path / 'ops')
    monkeypatch.setattr(m, 'DEST', tmp_path / 'hdd' / 'dgx-moved-ei')
    monkeypatch.setattr(m, 'LOG', tmp_path / 'log.jsonl')
    busy = root / 'collect' / 'v0001-111111111111'
    monkeypatch.setattr(m, 'process_references', lambda: [('1', 'cwd', str(busy / 'groups'))])
    monkeypatch.setattr(m, 'unit_commands', lambda: [])
    (root / 'r0001' / 'filled' / '.keep-on-nvme').write_text('keep\n')
    m.run(SimpleNamespace(apply=True, max_minutes=5, hdd_floor_gb=0))
    moved = root / 'collect' / 'v0000-000000000000'
    assert moved.is_symlink() and (moved / 'runs.jsonl').exists()
    assert (moved / 'groups' / 'g000001' / 'rollouts.jsonl').read_bytes() == b'r' * 100
    assert not (moved / 'actor' / 'sglang-hf').exists() and (moved / 'actor' / 'blocks' / 'b.safetensors').exists()
    assert busy.is_dir() and not busy.is_symlink()                       # a process works in it
    ckpt = root / 'r0001' / 'policy' / 'checkpoint.pt'
    assert ckpt.is_symlink() and ckpt.read_bytes() == b'c' * 30
    assert not (root / 'r0001' / 'policy' / 'sglang-hf').exists()
    assert json.loads((root / 'r0001' / 'policy' / 'sglang-hf.removed.json').read_text())['bytes'] == 30
    assert not (root / 'r0001' / 'filled').is_symlink()                  # marker keeps it
    assert (root / 'r0001' / 'mix' / 'chains.pkl').is_symlink()
    assert not (root / 'r0001' / 'mix' / 'rows.jsonl.zst').is_symlink()
    assert not (root / 'r0003' / 'mix' / 'chains.pkl').is_symlink()      # previous round untouched
    assert not os.path.exists(str(moved) + '.ei-moving')


def test_hdd_floor_refuses_moves(tmp_path, monkeypatch):
    m = load()
    root, state = driver(tmp_path)
    monkeypatch.setattr(m, 'OPS', tmp_path / 'ops')
    monkeypatch.setattr(m, 'DEST', tmp_path / 'hdd' / 'dgx-moved-ei')
    monkeypatch.setattr(m, 'LOG', tmp_path / 'log.jsonl')
    monkeypatch.setattr(m, 'process_references', lambda: [])
    monkeypatch.setattr(m, 'unit_commands', lambda: [])
    (tmp_path / 'hdd').mkdir()
    summary = m.run(SimpleNamespace(apply=True, max_minutes=5, hdd_floor_gb=10**9))
    assert summary['moved_bytes'] == 0 and summary['errors'] > 0
    assert not (root / 'collect' / 'v0000-000000000000').is_symlink()


def test_tree_copy_resumes_a_killed_partial_and_bounds_writeback(tmp_path, monkeypatch):
    m = load()
    monkeypatch.setattr(m, 'FLUSH_BYTES', 1000)
    src = tmp_path / 'nvme' / 'v0002'
    (src / 'blocks').mkdir(parents=True)
    (src / 'blocks' / 'a.bin').write_bytes(b'a' * 5000)
    (src / 'blocks' / 'b.bin').write_bytes(b'b' * 3000)
    (src / 'link').symlink_to('blocks/a.bin')
    dest = tmp_path / 'hdd' / 'v0002'
    partial = dest.with_name('v0002.partial')
    (partial / 'blocks').mkdir(parents=True)
    done = partial / 'blocks' / 'a.bin'                 # finished by the killed pass: reused
    done.write_bytes(b'a' * 5000)
    os.utime(done, ns=(src.joinpath('blocks/a.bin').stat().st_atime_ns, src.joinpath('blocks/a.bin').stat().st_mtime_ns))
    (partial / 'blocks' / 'b.bin').write_bytes(b'b' * 100)  # cut off mid-write: copied again
    copied = []
    real = m._copy_bounded
    monkeypatch.setattr(m, '_copy_bounded', lambda s, o: copied.append(s.name) or real(s, o))
    assert m.copy_tree_verified(src, dest) == 8000
    assert copied == ['b.bin'] and not partial.exists()
    assert (dest / 'blocks' / 'b.bin').read_bytes() == b'b' * 3000 and os.readlink(dest / 'link') == 'blocks/a.bin'


def test_tree_copy_drops_a_file_that_does_not_verify(tmp_path, monkeypatch):
    m = load()
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'x.bin').write_bytes(b'x' * 100)
    real = m._sha
    monkeypatch.setattr(m, '_sha', lambda p: 'bad' if '.partial' in str(p) else real(p))
    dest = tmp_path / 'dest'
    try:
        m.copy_tree_verified(src, dest)
        raise AssertionError('should not verify')
    except RuntimeError:
        pass
    assert not dest.exists() and not (tmp_path / 'dest.partial' / 'x.bin').exists()


def _collection(root):
    c = root / 'v0003-6f57450a2aad'
    s = c / 'groups' / 'g000001' / 'row-000' / 'sample-00'
    (s / 'jobs').mkdir(parents=True)
    (s / 'results.jsonl').write_bytes(b'{"turns": 1}\n')
    (s / 'jobs' / '000000-aa.result.json').write_bytes(b'{"turns": 1}\n')      # byte copy: goes
    (s / 'jobs' / '000000-aa.trace.jsonl').write_bytes(b't\n')
    (s / 'jobs' / '000000-bb.partial.json.tmp-123-0f0e0d0c-1111-2222-3333-444455556666').write_bytes(b'x')
    (s / 'collector.log').write_text('log')
    other = c / 'groups' / 'g000001' / 'row-000' / 'sample-01'
    (other / 'jobs').mkdir(parents=True)
    (other / 'results.jsonl').write_bytes(b'{"turns": 2}\n')
    (other / 'jobs' / '000000-cc.result.json').write_bytes(b'{"turns": 3}\n')  # differs: stays
    (other / 'jobs' / '000000-cc.partial.json').write_bytes(b'{}')               # partial episode: stays
    (other / 'jobs' / '000000-cc.partial.json.evidence-x.jsonl').write_bytes(b'{}\n')
    (other / 'jobs' / '000000.error.json').write_bytes(b'{}')
    (c / 'groups' / 'g000001' / 'workers').mkdir()
    (c / 'groups' / 'g000001' / 'workers' / 's7-task-000-00.log').write_text('w')
    (c / 'groups' / 'g000001' / 'workers' / 's7-task-000-00.journal.jsonl').write_text('{}\n')
    (c / 'groups' / 'g000001' / 'rollouts.jsonl').write_text('{}\n')
    (c / 'actor' / 'blocks' / 'ab').mkdir(parents=True)
    (c / 'actor' / 'blocks' / 'ab' / 'nzs1_abcdef.safetensors').write_bytes(b'b' * 10)
    (c / 'actor' / 'store.sqlite').write_bytes(b's')
    (c / 'actor.log').write_text('a')
    log = c / 'collect.log'
    log.write_text('done')
    os.utime(log, (time.time() - 7200, time.time() - 7200))
    return c


def test_trim_removes_only_reproducible_and_diagnostic_files(tmp_path):
    m = load()
    c = _collection(tmp_path)
    before = {p for p in c.rglob('*') if p.is_file()}
    assert sum(m.trim(c, apply=False).values()) > 0 and {p for p in c.rglob('*') if p.is_file()} == before
    kinds = m.trim(c, apply=True)
    assert set(kinds) == {'duplicate-result', 'trace', 'log', 'tmp', 'native-state-cache'}
    left = {str(p.relative_to(c)) for p in c.rglob('*') if p.is_file()}
    assert left == {'groups/g000001/row-000/sample-00/results.jsonl',
                    'groups/g000001/row-000/sample-01/results.jsonl',
                    'groups/g000001/row-000/sample-01/jobs/000000-cc.result.json',
                    'groups/g000001/row-000/sample-01/jobs/000000-cc.partial.json',
                    'groups/g000001/row-000/sample-01/jobs/000000-cc.partial.json.evidence-x.jsonl',
                    'groups/g000001/row-000/sample-01/jobs/000000.error.json',
                    'groups/g000001/workers/s7-task-000-00.journal.jsonl', 'groups/g000001/rollouts.jsonl',
                    'actor/store.sqlite', 'collect.log', 'trimmed.json'}
    record = json.loads((c / 'trimmed.json').read_text())
    assert len(record['passes'][0]['files']) == 7


def test_trim_refuses_an_open_collection_and_plan_trims_moved_ones(tmp_path):
    m = load()
    c = _collection(tmp_path / 'hdd')
    (c / 'collect.log').write_text('still writing')
    try:
        m.trim(c, apply=True)
        raise AssertionError('an open collection must be refused')
    except RuntimeError:
        pass
    assert (c / 'actor' / 'blocks' / 'ab' / 'nzs1_abcdef.safetensors').exists()
    driver = tmp_path / 'aei1'
    (driver / 'collect').mkdir(parents=True)
    (driver / 'collect' / c.name).symlink_to(c)
    state = {'lineage': [{'sha256': 'f' * 64, 'round_id': 'r0007'}, {'sha256': 'e' * 64, 'round_id': 'r0008'}]}
    actions = m.plan(driver, state, None)
    assert [a['op'] for a in actions] == ['trim'] and actions[0]['path'] == str(c.resolve())
