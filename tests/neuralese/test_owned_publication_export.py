"""Actual saved publications export recovered steps without another tensor decode."""
import json
import contextlib
import hashlib
import io
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from scripts.coreweave import owned_publication_export as module
from scripts.coreweave import cursor_checkpoint_export as cursor
from natlang_neuralese.model.native_checkpoint_control import (
    _file_binding, source_stat, write_checkpoint_control,
)
from test_native_checkpoint_control import parent, publication_parent, DEPENDENCIES


@pytest.fixture
def publication(parent, tmp_path):
    path, state, previous, options = publication_parent(parent)
    role = 'lfm' if state['model_config']['architecture'] == 'lfm2' else 'mellum'
    out = tmp_path/'control'/(role+'-recovered-current')/'learner'
    out.mkdir(parents=True)
    target = out/'checkpoint.pt'
    path.replace(target)
    ref = write_checkpoint_control(target, out/'checkpoint-controls', validated_state=state,
        decoded_source_stat=source_stat(target), publication_only=True,
        dependency_collector=DEPENDENCIES.dependency_closure, **options)
    value = json.loads(Path(ref['path']).read_text())
    (out/'checkpoint-publication-status.json').write_text(json.dumps(dict(
        passed=True, control=ref, checkpoint=_file_binding(target))))
    argv = tmp_path/'effective-argv.json'
    argv.write_text(json.dumps(['--out', str(out)]))
    spec = dict(schema='natlang.owned-publication-export/1', role=role,
        production_root=str(tmp_path), control_root=str(tmp_path/'control'),
        effective_argv={**_file_binding(argv), 'allowed_output_root': str(out.parent)},
        parent=previous, pin_directory=str(tmp_path/'pins'), reserve_bytes=0,
        publication=dict(schema='natlang.native-checkpoint-publication-selection/1',
            producer=value['producer'], dependency_producer=value['dependency_producer'], **options))
    return spec, target, state


def test_actual_recovered_publication_uses_sha_and_never_decodes(publication, monkeypatch):
    spec, source, state = publication
    def forbidden(*args, **kwargs):
        raise AssertionError('no extra optimizer/model deserialization')
    monkeypatch.setattr(torch, 'load', forbidden)
    value = module.observe_publication(spec, known_sha256='f'*64)
    assert value['step'] == 72 and value['tensor_decodes'] == 0
    pin = Path(value['pin'])
    assert pin.read_bytes() == source.read_bytes()
    assert pin.stat().st_ino != source.stat().st_ino
    assert module.observe_publication(spec, known_sha256=value['sha256']) is None
    assert module.observe_publication(spec) == value
    # Atomic replacement of the learner slot leaves its independently pinned full state intact.
    replacement = source.with_suffix('.replacement')
    replacement.write_bytes(b'later checkpoint publication still pending')
    replacement.replace(source)
    assert _file_binding(pin)['sha256'] == value['sha256']
    assert module.observe_publication(spec, known_sha256=value['sha256']) is None


@pytest.mark.parametrize('change', ['parent', 'producer', 'argv', 'foreign-status', 'conflicting-pin'])
def test_wrong_authority_or_pin_is_never_exported(publication, change):
    spec, source, state = publication
    if change == 'parent':
        spec['parent']['sha256'] = '0'*64
    elif change == 'producer':
        spec['publication']['producer']['sha256'] = '0'*64
    elif change == 'argv':
        Path(spec['effective_argv']['path']).write_text('[]')
    elif change == 'foreign-status':
        status = source.parent/'checkpoint-publication-status.json'
        value = json.loads(status.read_text())
        value['checkpoint']['path'] = str(source.parent/'foreign.pt')
        status.write_text(json.dumps(value))
    else:
        pins = Path(spec['pin_directory']); pins.mkdir()
        (pins/(spec['role']+'-'+_file_binding(source)['sha256']+'.pt')).write_bytes(b'conflicting immutable pin')
    with pytest.raises(ValueError):
        module.observe_publication(spec)


def test_atomic_source_advance_after_snapshot_keeps_exact_old_publication(publication, monkeypatch):
    spec, source, state = publication
    original = module.subprocess.run
    def racing(*args, **kwargs):
        result = original(*args, **kwargs)
        replacement = source.with_suffix('.replacement')
        replacement.write_bytes(b'atomic later source')
        replacement.replace(source)
        return result
    monkeypatch.setattr(module.subprocess, 'run', racing)
    value = module.observe_publication(spec)
    pin = Path(value['pin'])
    assert _file_binding(pin)['sha256'] == value['sha256']
    assert pin.read_bytes() != source.read_bytes()
    assert pin.stat().st_ino != source.stat().st_ino


def test_newer_or_mismatched_snapshot_cannot_use_captured_old_control(publication, monkeypatch):
    spec, source, state = publication
    original = module.subprocess.run
    def advance_before_copy(argv, **kwargs):
        replacement = source.with_suffix('.replacement')
        replacement.write_bytes(b'new save before cp opens the slot')
        replacement.replace(source)
        return original(argv, **kwargs)
    monkeypatch.setattr(module.subprocess, 'run', advance_before_copy)
    with pytest.raises(ValueError):
        module.observe_publication(spec)
    assert not list(Path(spec['pin_directory']).glob('*.pt'))


def test_forged_status_control_reference_is_rejected_before_copy(publication, monkeypatch):
    spec, source, _ = publication
    status = source.parent/'checkpoint-publication-status.json'
    value = json.loads(status.read_text())
    value['control'] = {**value['control'], 'sha256': '0'*64}
    status.write_text(json.dumps(value))
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k:pytest.fail('must reject forged control before copying'))
    with pytest.raises(ValueError):
        module.observe_publication(spec)


def test_same_continuation_cannot_regress_its_own_step(publication):
    spec, source, state = publication
    with pytest.raises(ValueError, match='regressed'):
        module.observe_publication(spec, known_step=72, known_sha256='f'*64)


def test_copy_preserves_declared_storage_reserve(publication, monkeypatch):
    spec, source, state = publication
    monkeypatch.setattr(module.shutil,'disk_usage',lambda path:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match='insufficient storage'):
        module.observe_publication(spec)
    assert not list(Path(spec['pin_directory']).iterdir())


def test_actual_cli_pins_reads_back_and_rejects_foreign_journal(publication, tmp_path, monkeypatch):
    import scripts.coreweave as package
    from natlang_neuralese.train.native_resident import code_inventory, digest
    spec, source, state = publication
    root = Path(__file__).parents[2]
    spec.update(runtime=str(root), runtime_code_sha256=digest(code_inventory(root/'training/neuralese/natlang_neuralese')))
    selection = tmp_path/'selection.json'; selection.write_text(json.dumps(spec))
    hash_cache=tmp_path/'cache'/'raw-provider-hashes.jsonl'
    # If setup order regresses, an imported hash helper would write here.
    # Keep that fallback inside this test's temporary directory.
    prior_cache=tmp_path/'prior-cache-value'
    monkeypatch.setenv('NATLANG_HASH_CACHE',str(prior_cache))
    captured={}
    def execute(argv, **kwargs):
        assert argv[:2] == ['python','-c']
        captured['code']=argv[2]
        output = io.StringIO()
        # The actual generated remote program runs against the actual tiny
        # publication. All optimizer deserialization is forbidden in this path.
        def forbidden(*a, **kw):
            raise AssertionError('owned publication CLI must not decode tensors')
        # The supervisor test process has already imported the hashing
        # module. Evict the package modules temporarily so this generated
        # remote program exercises its own import order, as a fresh worker
        # would, then restore the host's exact module objects.
        prior_modules={name:value for name,value in sys.modules.items()
            if name=='natlang_neuralese' or name.startswith('natlang_neuralese.')}
        for name in prior_modules: del sys.modules[name]
        try:
            with monkeypatch.context() as scoped:
                scoped.setattr(torch, 'load', forbidden)
                with contextlib.redirect_stdout(output):
                    exec(argv[2], {})
        finally:
            for name in tuple(sys.modules):
                if name=='natlang_neuralese' or name.startswith('natlang_neuralese.'):
                    del sys.modules[name]
            sys.modules.update(prior_modules)
        result = SimpleNamespace(returncode=0,stdout=output.getvalue(),stderr='')
        return SimpleNamespace(result=lambda **kw:result)
    fake = SimpleNamespace(load_env_file=lambda:None, require_key=lambda:None,
                           read_sandbox=lambda *a,**kw:SimpleNamespace(exec=execute))
    monkeypatch.setitem(sys.modules,'scripts.coreweave.cwctl',fake)
    monkeypatch.setattr(package,'cwctl',fake,raising=False)
    original = module.subprocess.run
    transfers = []
    def transfer(argv, **kwargs):
        if argv[0] != 'test-private-uploader':
            return original(argv, **kwargs)
        path = Path(argv[1]); payload = path.read_bytes()
        identity = dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
        env = kwargs['env']
        assert identity == dict(sha256=env['CW_EXPORT_EXPECTED_SHA256'],bytes=int(env['CW_EXPORT_EXPECTED_BYTES']))
        receipt = dict(source=str(path),identity=identity,private=True,role=spec['role'],sandbox_id='owned-test',
                       independent_verification=identity.copy())
        Path(env['CW_EXPORT_RECEIPT_FILE']).write_text(json.dumps(receipt))
        transfers.append(identity)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',transfer)
    journal = tmp_path/'fresh-owned-journal'
    argv = ['--role',spec['role'],'--sandbox-id','owned-test','--root',str(tmp_path),
            '--state-dir',str(journal),'--publication-spec',str(selection),
            '--publication-spec-sha256',_file_binding(selection)['sha256'],
            '--owned-hash-cache',str(hash_cache),
            '--export-command',json.dumps(['test-private-uploader','{pin}']),'--once']
    assert cursor.main(argv) == 0 and len(transfers) == 1
    assert captured['code'].index('os.environ["NATLANG_HASH_CACHE"]') < captured['code'].index('from natlang_neuralese')
    assert os.environ['NATLANG_HASH_CACHE']==str(hash_cache)
    assert hash_cache.exists() and prior_cache.parent == tmp_path
    assert cursor.main(argv) == 0 and len(transfers) == 1
    saved = json.loads(cursor.paths_for(journal,spec['role'])['state'].read_text())
    actual_source = _file_binding(source)
    assert saved['last_verified_step'] == 72
    assert saved['receipt']['identity'] == {key:actual_source[key] for key in ('sha256','bytes')}
    changed = dict(spec,parent={**spec['parent'],'sha256':'0'*64})
    selection.write_text(json.dumps(changed))
    argv[argv.index('--publication-spec-sha256')+1] = _file_binding(selection)['sha256']
    with pytest.raises(ValueError,match='another continuation'):
        cursor.main(argv)
    # Old step29861 remains historical; it cannot silently become a new journal.
    old = tmp_path/'old-journal'; old.mkdir()
    cursor.paths_for(old,spec['role'])['state'].write_text(json.dumps({'last_verified_step':29861}))
    argv[argv.index('--state-dir')+1] = str(old)
    with pytest.raises(ValueError,match='fresh journal'):
        cursor.main(argv)


def test_fixed_owned_publication_remote_error_is_saved_with_traceback(publication, tmp_path, monkeypatch):
    import scripts.coreweave as package
    spec, source, state = publication
    selection=tmp_path/'selection.json';selection.write_text(json.dumps(spec))
    captured={}
    class Rpc:
        def result(self,timeout):
            return SimpleNamespace(returncode=1,stdout='',
                stderr='Traceback (most recent call last):\n  File "observer.py", line 2\nValueError: access_token=private-secret')
    def execute(argv,*args,**kwargs):
        captured['code']=argv[2]
        return Rpc()
    fake=SimpleNamespace(load_env_file=lambda:None,require_key=lambda:None,
        read_sandbox=lambda *args,**kwargs:SimpleNamespace(exec=execute))
    monkeypatch.setitem(sys.modules,'scripts.coreweave.cwctl',fake)
    monkeypatch.setattr(package,'cwctl',fake,raising=False)
    journal=tmp_path/'fixed-publication-journal'
    hash_cache=tmp_path/'cache'/'raw-provider-hashes.jsonl'
    argv=['--role',spec['role'],'--sandbox-id','fixture','--root',str(tmp_path),
        '--state-dir',str(journal),'--publication-spec',str(selection),
        '--publication-spec-sha256',_file_binding(selection)['sha256'],
        '--owned-hash-cache',str(hash_cache),'--export-command','["true"]','--once']
    with pytest.raises(RuntimeError,match='remote checkpoint observation failed'):
        cursor.main(argv)
    assert captured['code'].index('os.environ["NATLANG_HASH_CACHE"]') < captured['code'].index('from natlang_neuralese')
    saved=json.loads(cursor.paths_for(journal,spec['role'])['error'].read_text())
    assert saved['type']=='RuntimeError' and 'observer.py' in saved['traceback']
    assert 'private-secret' not in saved['traceback'] and '[REDACTED]' in saved['traceback']


def test_failed_publication_is_a_visible_retry_not_an_absent_first_save(publication, monkeypatch):
    spec, source, _ = publication
    status = source.parent/'checkpoint-publication-status.json'
    status.unlink()
    assert module.observe_publication(spec) is None
    status.write_text(json.dumps({'passed':False,'error_type':'ValueError','full_checkpoint_preserved':True}))
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('failed controls must not decode state'))
    with pytest.raises(ValueError,match='owned checkpoint publication failed: ValueError'):
        module.observe_publication(spec)
    assert not Path(spec['pin_directory']).exists()
