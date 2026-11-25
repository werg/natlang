import importlib.util
import json
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prune_stale_checkouts.py'


def load():
    spec = importlib.util.spec_from_file_location('prune_stale_checkouts', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pin(root, name, age_hours):
    path = root / name
    path.mkdir(parents=True)
    (path / '.natlang-pin.json').write_text(json.dumps({'commit': '0' * 40, 'created': time.time() - age_hours * 3600}))
    return path


def test_dry_run_removes_only_the_unreferenced_stale_pin(tmp_path, monkeypatch, capsys):
    m = load()
    ops = tmp_path / 'ops'
    busy = pin(ops / 'ei', 'pin-busy00000000', 30)
    watched = pin(ops / 'ei', 'pin-watched00000', 30)
    kept = pin(ops / 'ei', 'pin-kept00000000', 30)
    stale = pin(ops / 'ei', 'pin-stale0000000', 30)
    young = pin(ops / 'ei', 'pin-young0000000', 1)
    (kept / '.natlang-keep').write_text('driver awaiting its verdict\n')
    watch = ops / 'ei' / 'watch'
    watch.mkdir(parents=True)
    (watch / 'guard.sh').write_text(f'exec python {watched}/scripts/x.py\n')

    monkeypatch.setattr(m, 'OPS', ops)
    monkeypatch.setattr(m, 'PIN_ROOTS', (ops,))
    monkeypatch.setattr(m, 'EXCLUDE_FILE', tmp_path / 'absent.txt')
    monkeypatch.setattr(m, 'worktrees', lambda: {})
    monkeypatch.setattr(m, 'on_root_fs', lambda path: True)
    monkeypatch.setattr(m, 'unit_commands', lambda: [])
    monkeypatch.setattr(m, 'process_references', lambda: [('4242', 'cwd', str(busy / 'ts-host'))])
    removed = []
    monkeypatch.setattr(m, 'remove', lambda path, tree: removed.append(path))

    assert m.main([]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    verdicts = {Path(r['path']).name: r for r in rows if 'path' in r}
    assert verdicts['pin-stale0000000'].get('would_remove') is True
    assert 'pid 4242 cwd' in verdicts['pin-busy00000000']['kept']
    assert 'state' in verdicts['pin-watched00000']['kept']
    assert verdicts['pin-kept00000000']['kept'] == 'marker .natlang-keep'
    assert verdicts['pin-young0000000']['kept'] == 'young'
    assert [name for name, r in verdicts.items() if r.get('would_remove')] == ['pin-stale0000000']
    assert removed == [] and stale.exists()


def test_check_reports_command_line_references(monkeypatch, capsys, tmp_path):
    m = load()
    target = tmp_path / 'ops' / 'grpo' / 'actor'
    monkeypatch.setattr(m, 'process_references', lambda: [('7', 'cmdline', f'python x.py --export {target}/hf')])
    monkeypatch.setattr(m, 'unit_commands', lambda: [])
    monkeypatch.setattr(m, 'state_files', lambda days, now: [])
    assert m.main(['--check', str(target)]) == 1
    assert 'pid 7 cmdline' in capsys.readouterr().out
    monkeypatch.setattr(m, 'process_references', lambda: [])
    assert m.main(['--check', str(target)]) == 0


def test_pin_walk_never_follows_symlinks(tmp_path, monkeypatch):
    """~/data symlinks into the external HDD: the walk must not descend through them (2026-11-23 sweep wedge)."""
    m = load()
    data = tmp_path / 'data'
    real = pin(data / 'run', 'pin-real00000000', 30)
    elsewhere = tmp_path / 'hdd'
    pin(elsewhere / 'deep', 'pin-hdd000000000', 30)
    (data / 'archived').symlink_to(elsewhere / 'deep')
    (data / 'run' / 'linked').symlink_to(elsewhere)
    monkeypatch.setattr(m, 'PIN_ROOTS', (data,))
    assert m.pin_dirs() == [str(real)]


def test_configured_consumers_and_failed_units_reference(tmp_path, monkeypatch, capsys):
    """A periodic consumer between runs (a keeper relaunching from its config, a failed attempt) still references."""
    m = load()
    unit_commands = m.unit_commands
    ops = tmp_path / 'ops'
    corpora = tmp_path / 'corpora'
    target = tmp_path / 'pins' / 'reader-929'
    globbed = tmp_path / 'campaign' / 'a-config.json'
    globbed.parent.mkdir(parents=True)
    globbed.write_text(json.dumps({'source_pin': str(target)}))
    (ops / 'consumers.d').mkdir(parents=True)
    (ops / 'consumers.d' / 'fill.json').write_text(json.dumps({'configs': [str(tmp_path / 'campaign' / '*-config.json'),
                                                                           str(tmp_path / 'absent.json')]}))
    corpora.mkdir()
    (corpora / 'ds.json').write_text('{}')
    script = tmp_path / 'run.sh'
    script.write_text('exec true\n')
    units = [('u.service', f'{{ path=/bin/sh ; argv[]=/bin/sh {script} --cfg={globbed} ; }}')]
    found = m.configured_consumers(units, ops=ops, corpora=corpora)
    assert found == [ops / 'consumers.d' / 'fill.json', globbed, corpora / 'ds.json', script]
    monkeypatch.setattr(m, 'OPS', ops)
    monkeypatch.setattr(m, 'CORPORA', corpora)
    monkeypatch.setattr(m, 'process_references', lambda: [])
    monkeypatch.setattr(m, 'unit_commands', lambda: [])
    monkeypatch.setattr(m, 'state_files', lambda days, now: [])
    assert m.main(['--check', str(target)]) == 1
    assert 'a-config.json' in capsys.readouterr().out
    assert m.main(['--no-consumer-configs', '--check', str(target)]) == 0

    # unit_commands: a failed service counts (its keeper relaunches the same command); an inactive one does not
    def run(argv, **kw):
        if 'list-units' in argv:
            out = ('fill-1.service loaded failed failed x\nold.service loaded inactive dead x\n'
                   'live.service loaded active running x\n')
        else:
            out = ''.join(f'Id={u}\nExecStart=cmd {u}\n\n' for u in argv if u.endswith('.service'))
        return type('R', (), {'stdout': out})()
    monkeypatch.setattr(m.subprocess, 'run', run)
    assert [u for u, _ in unit_commands()] == ['fill-1.service', 'live.service']
