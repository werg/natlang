import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'ei_round_tier.py'


def load():
    spec = importlib.util.spec_from_file_location('ei_round_tier', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
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
