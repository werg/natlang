"""Real disappearing-proc discovery plus strict transition custody negatives."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.coreweave.teacher_incremental_export import discover_guardian_batch


def transition_fixture(monkeypatch, *, probe_mode='missing', mutation=None):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    child = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(.1)'])
    ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    child.wait()
    assert not Path(f'/proc/{child.pid}').exists()
    control, root = '/work/control', '/work/generation'
    config = json.dumps({'control': control, 'root': root, 'feeder': '/guard/feed.py',
                         'source_sequence': {}}).encode()
    config_ref = {'path': control+'/config.json', 'sha256': hashlib.sha256(config).hexdigest(),
                  'bytes': len(config)}
    old = {'schema': 'natlang.teacher-generation-guardian/1', 'config': config_ref,
           'active_batch': 26, 'next_batch': 26,
           'owner': {'pid': child.pid, 'start_ticks': ticks, 'boot_id': boot}}
    source = root+'/evidence-collections-v2/batch-0026'
    boundary = {'schema': 'natlang.teacher-generation-boundary/1', 'batch': 26,
                'completion': {'can_advance': True}, 'individual_outcomes': [],
                'bindings': {name: {'path': source+'/'+name, 'sha256': 'a'*64, 'bytes': 1}
                             for name in ('queue.jsonl', 'journal.jsonl', 'score.json', 'teacher-source.json')}}
    boundary_raw = json.dumps(boundary).encode()
    reference = {'path': control+'/boundaries/batch-0026.json',
                 'sha256': hashlib.sha256(boundary_raw).hexdigest(), 'bytes': len(boundary_raw)}
    fresh = {**old, 'active_batch': 27, 'next_batch': 27,
             'owner': {'pid': 9, 'start_ticks': '22', 'boot_id': boot}, 'last_boundary': reference}
    remote = {config_ref['path']: config, reference['path']: boundary_raw,
              '/proc/sys/kernel/random/boot_id': (boot+'\n').encode()}
    if mutation == 'config':
        fresh['config'] = {**config_ref, 'sha256': 'f'*64}
    elif mutation == 'boundary_sha':
        fresh['last_boundary'] = {**reference, 'sha256': 'f'*64}
    elif mutation == 'boundary_batch':
        boundary['batch'] = 25
        remote[reference['path']] = json.dumps(boundary).encode()
        fresh['last_boundary'] = {**reference, 'sha256': hashlib.sha256(remote[reference['path']]).hexdigest(),
                                  'bytes': len(remote[reference['path']])}
    elif mutation == 'new_boot':
        fresh['owner'] = {**fresh['owner'], 'boot_id': 'other'}
    elif mutation == 'boot_during_transition':
        remote['/proc/sys/kernel/random/boot_id'] = b'other\n'
    elif mutation == 'source_binding':
        boundary['bindings']['queue.jsonl']['path'] = '/wrong/queue.jsonl'
        remote[reference['path']] = json.dumps(boundary).encode()
        fresh['last_boundary'] = {**reference, 'sha256': hashlib.sha256(remote[reference['path']]).hexdigest(),
                                  'bytes': len(remote[reference['path']])}
    elif mutation == 'unchanged':
        fresh = old
    remote[root+'/feed-evidence-phase.json'] = json.dumps({'batch':26,'phase':'collecting'}).encode()
    if mutation == 'complete_pending':
        fresh = old
        remote[root+'/feed-evidence-phase.json'] = json.dumps({'batch':26,'phase':'batch_complete'}).encode()
    state_reads = 0

    def reader(path):
        nonlocal state_reads
        if path == control+'/state.json':
            state_reads += 1
            return json.dumps(old if state_reads == 1 else fresh).encode()
        return remote[path]

    probes = []

    def execute(argv, **kwargs):
        if argv[0] == 'cat':
            result = SimpleNamespace(returncode=0, stdout=boot+'\n', stderr='')
        elif probe_mode == 'nonzero':
            # This is the old deployed missing-/proc failure, not a fabricated
            # return-code fixture: execute a genuinely missing process read.
            result = subprocess.run([sys.executable, '-c',
                f"from pathlib import Path;Path('/proc/{child.pid}/stat').read_text()"],
                capture_output=True, text=True)
            assert result.returncode != 0 and 'FileNotFoundError' in result.stderr
        elif probe_mode == 'wrong_birth':
            result = SimpleNamespace(returncode=0, stdout=json.dumps({'valid': False,
                'identity': {'pid': child.pid, 'start_ticks': 'wrong', 'boot_id': boot}}), stderr='')
        else:
            result = subprocess.run([sys.executable, *argv[1:]], capture_output=True, text=True)
            assert result.returncode == 0
            assert json.loads(result.stdout)['missing'] is True
        probes.append(result)
        return SimpleNamespace(result=lambda **_: result)

    publisher = {'schema': 'natlang.teacher-incremental-export-publisher/1', 'sandbox_id': 'sid',
        'sandbox_boot_id': boot, 'guardian_config': config_ref, 'control_dir': control,
        'generation_root': root, 'anchor_batch': 21, 'anchor_cases_sha256': 'a'*64,
        'directory_root': '/external/export', 'role': 'bulk', 'chunk_size': 32,
        'exporter_path': '/owner/export.py', 'identity': {}}
    return publisher, SimpleNamespace(exec=execute), reader, reference


@pytest.mark.parametrize('mode', ['missing', 'nonzero'])
def test_genuine_missing_proc_follows_only_authenticated_closed_transition(monkeypatch, mode):
    publisher, sandbox, reader, reference = transition_fixture(monkeypatch, probe_mode=mode)
    result = discover_guardian_batch(publisher, sandbox, reader)
    assert result['draining'] and result['verified_owner_transition']
    assert result['batch'] == 26 and result['guardian_state']['active_batch'] == 27
    assert result['closed_boundaries'] == [reference]
    assert 'queue_path' not in result  # stale active rows are never exported


@pytest.mark.parametrize('mutation', ['config', 'boundary_sha', 'boundary_batch', 'new_boot',
    'boot_during_transition', 'source_binding', 'unchanged'])
def test_missing_proc_does_not_waive_state_or_closed_boundary_authority(monkeypatch, mutation):
    publisher, sandbox, reader, _ = transition_fixture(monkeypatch, mutation=mutation)
    with pytest.raises(ValueError, match='transition|boundary|authority'):
        discover_guardian_batch(publisher, sandbox, reader)


def test_wrong_live_birth_is_fatal_even_if_an_advanced_state_exists(monkeypatch):
    publisher, sandbox, reader, _ = transition_fixture(monkeypatch, probe_mode='wrong_birth')
    with pytest.raises(ValueError, match='PID/start/boot/argv/batch differs'):
        discover_guardian_batch(publisher, sandbox, reader)


def test_missing_completed_owner_drains_without_deadline_or_export_custody(monkeypatch):
    publisher, sandbox, reader, _ = transition_fixture(monkeypatch, mutation='complete_pending')
    for _ in range(3):
        result = discover_guardian_batch(publisher, sandbox, reader)
        assert result['draining'] and not result['verified_owner_transition']
        assert result['closed_boundaries'] == []
        assert 'queue_path' not in result


def test_wrong_birth_remains_fatal_even_with_completed_phase(monkeypatch):
    publisher, sandbox, reader, _ = transition_fixture(monkeypatch, probe_mode='wrong_birth',mutation='complete_pending')
    with pytest.raises(ValueError, match='PID/start/boot/argv/batch differs'):
        discover_guardian_batch(publisher,sandbox,reader)


@pytest.mark.parametrize('phase',[{'phase':'waiting_strong_server_ready'},
    {'phase':'batch_complete','batch':26}])
def test_exact_live_owner_phase_grace_grants_no_source_or_export_custody(monkeypatch,phase):
    publisher,sandbox,reader,_=transition_fixture(monkeypatch,mutation='complete_pending')
    original=sandbox.exec
    def execute(argv,**kwargs):
        if argv[0]=='cat':return original(argv,**kwargs)
        return SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,
            stdout=json.dumps({'valid':True,'identity':{'pid':1}}),stderr=''))
    sandbox.exec=execute
    def read(path):
        if path.endswith('/feed-evidence-phase.json'):return json.dumps(phase).encode()
        if '/evidence-collections-v2/'in path:pytest.fail('preparation grants no source/row read')
        return reader(path)
    result=discover_guardian_batch(publisher,sandbox,read)
    assert result['draining']and not result['verified_owner_transition']
    assert 'queue_path'not in result and 'identity'not in result


@pytest.mark.parametrize('mutation',['config','new_boot'])
def test_live_preparation_does_not_waive_owner_or_config_transition(monkeypatch,mutation):
    publisher,sandbox,reader,_=transition_fixture(monkeypatch,mutation=mutation)
    original=sandbox.exec
    sandbox.exec=lambda argv,**kwargs:(original(argv,**kwargs)if argv[0]=='cat'else
        SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,stdout=json.dumps({'valid':True}),stderr='')))
    def read(path):
        if path.endswith('/feed-evidence-phase.json'):return json.dumps({'phase':'waiting_strong_server_ready'}).encode()
        return reader(path)
    with pytest.raises(ValueError):discover_guardian_batch(publisher,sandbox,read)


def test_bad_live_birth_is_fatal_before_preparation_grace(monkeypatch):
    publisher,sandbox,reader,_=transition_fixture(monkeypatch,probe_mode='wrong_birth',mutation='complete_pending')
    def read(path):
        if path.endswith('/feed-evidence-phase.json'):return json.dumps({'phase':'waiting_strong_server_ready'}).encode()
        return reader(path)
    with pytest.raises(ValueError,match='PID/start/boot'):discover_guardian_batch(publisher,sandbox,read)


@pytest.mark.parametrize('tamper',[False,True])
def test_next_live_preparation_requires_exact_previous_boundary(monkeypatch,tamper):
    publisher,sandbox,reader,reference=transition_fixture(monkeypatch,mutation='complete_pending')
    original=sandbox.exec
    sandbox.exec=lambda argv,**kwargs:(original(argv,**kwargs)if argv[0]=='cat'else
        SimpleNamespace(result=lambda **_:SimpleNamespace(returncode=0,stdout=json.dumps({'valid':True}),stderr='')))
    def read(path):
        raw=reader(path)
        if path.endswith('/state.json'):
            value=json.loads(raw);value.update(active_batch=27,next_batch=27,last_boundary=reference)
            if tamper:value['last_boundary']={**reference,'sha256':'0'*64}
            return json.dumps(value).encode()
        return raw
    if tamper:
        with pytest.raises(ValueError,match='predecessor changed'):discover_guardian_batch(publisher,sandbox,read)
    else:
        result=discover_guardian_batch(publisher,sandbox,read)
        assert result['preparing']and result['batch']==27 and 'queue_path'not in result
