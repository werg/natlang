import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.rl.native_partial_custody import retain_partial_artifacts
from natlang_neuralese.model.native_artifact import (NativeTensorArtifact, encode_native_tensor_artifact,
                                                    native_tensor_artifact_id, native_tensor_dependencies)
from natlang_neuralese.model.state_block import native_state_dependencies
from natlang_neuralese.serve.store import (TensorStore, encode_native_state, decode_native_state,
                                         native_state_id)
from test_native_state_store import _block


@pytest.mark.parametrize('failure', [False, True])
def test_real_collector_session_drains_orphans_without_stopping_other_worker(tmp_path, monkeypatch, failure):
    from natlang_neuralese.rl import production
    monkeypatch.setattr(production, 'root', lambda _:Path(__file__).resolve().parents[2])
    _run_collector=production._run_collector
    from scripts.coreweave.teacher_engine_guard import session_process
    ready = tmp_path / 'owned.json'
    grand_ready = tmp_path / 'grand-ready'
    grand = ('import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
             f'pathlib.Path({str(grand_ready)!r}).touch();time.sleep(60)')
    script = f'''
import os,sys,signal,subprocess,time,json,pathlib
signal.signal(signal.SIGTERM,signal.SIG_IGN)
p=subprocess.Popen([sys.executable,'-c',{grand!r}])
while not pathlib.Path({str(grand_ready)!r}).exists():time.sleep(.01)
pathlib.Path({str(ready)!r}).write_text(json.dumps([os.getpid(),p.pid]))
{'time.sleep(60)' if failure else 'raise SystemExit(0)'}
'''
    sibling = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'], start_new_session=True)
    owned = []
    class Control:
        def wait(self, child):
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:time.sleep(.01)
            assert ready.exists()
            raise RuntimeError('actual collector wait failed')
    try:
        with (tmp_path / 'worker.log').open('w') as log:
            if failure:
                with pytest.raises(RuntimeError, match='actual collector wait failed'):
                    _run_collector([sys.executable, '-c', script], log, Control(), shutdown_grace=.05)
            else:
                assert _run_collector([sys.executable, '-c', script], log, None, shutdown_grace=.05) == 0
        owned = json.loads(ready.read_text())
        assert all((p := session_process(pid)) is None or p['state'] in ('Z', 'X') for pid in owned)
        assert sibling.poll() is None, 'session cleanup stopped another concurrent worker'
    finally:
        if ready.exists():owned = json.loads(ready.read_text())
        for pid in owned:
            try:os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:pass
        sibling.kill();sibling.wait()


@pytest.mark.parametrize('change', ['old-api', 'other-runtime'])
def test_collector_rejects_unbound_custody_helper_before_process_launch(tmp_path, monkeypatch, change):
    from natlang_neuralese.rl import production
    from scripts.coreweave import teacher_engine_guard as guard
    if change == 'old-api':
        monkeypatch.setattr(guard, 'cleanup_session', lambda owner, child, grace: None)
    else:
        copied = tmp_path / 'teacher_engine_guard.py'
        copied.write_bytes(Path(guard.__file__).read_bytes())
        monkeypatch.setattr(guard, '__file__', str(copied))
    calls = []
    monkeypatch.setattr(production.subprocess, 'Popen', lambda *a, **kw: calls.append((a, kw)))
    with (tmp_path / 'worker.log').open('w') as log:
        with pytest.raises(RuntimeError, match='exact runtime session-custody helper'):
            production._run_collector([sys.executable, '-c', 'pass'], log, None)
    assert calls == []


class Actor:
    def __init__(self):
        self.store = TensorStore()
        self.puts = []
        self.missing = set()

    def get_bytes(self, route):
        ident = route.rsplit('/', 1)[-1]
        if ident in self.missing:
            raise FileNotFoundError(ident)
        if ident.startswith('nzs1_'):
            record = self.store.get_native_state(ident)
            if record is not None:
                return encode_native_state(record.block)
        else:
            record = self.store.get_native_tensor_artifact(ident)
            if record is not None:
                return encode_native_tensor_artifact(record.artifact)
        raise FileNotFoundError(ident)

    def put_bytes(self, route, raw, *, owner=None):
        ident = route.rsplit('/', 1)[-1]
        if ident.startswith('nzs1_'):
            block, actual = decode_native_state(raw)
            assert all(self.store.get_native_state(x) or self.store.get_native_tensor_artifact(x)
                       for x in native_state_dependencies(block))
            self.store.put_native_state(block, owner)
        else:
            from natlang_neuralese.model.native_artifact import decode_native_tensor_artifact
            block, actual = decode_native_tensor_artifact(raw)
            assert all(self.store.get_native_state(x) or self.store.get_native_tensor_artifact(x)
                       for x in native_tensor_dependencies(block))
            self.store.put_native_tensor_artifact(block, owner)
        assert ident == actual
        self.puts.append((ident, owner))
        return {'id': ident}


def fixture(tmp_path):
    actor = Actor()
    block = _block()
    state = actor.store.put_native_state(block).id
    identity = {k: getattr(block, k) for k in ('checkpoint_sha256', 'tokenizer_fingerprint',
                                             'embedding_weight_sha256', 'architecture')}
    artifact = NativeTensorArtifact('state-gradient', identity,
        {'embeddings': torch.ones_like(block.embeddings)}, state, {'scope': 'retained-partial-test'})
    artifact_id = actor.store.put_native_tensor_artifact(artifact).id
    jobs = tmp_path / 'jobs'
    jobs.mkdir()
    # The artifact exists only in authenticated raw evidence, not response text.
    evidence = (json.dumps({'raw_response': {'native_state_gradient': artifact_id}}) + '\n').encode()
    (jobs / 'trace.jsonl').write_bytes(evidence + b'{"uncommitted_later_row":true}\n')
    partial = {'version': 'natlang.teacher_partial.native/1', 'program_id': 'producer-consumer',
        'provenance': {'policy': identity['checkpoint_sha256']},
        'turns': [{'request_sha256': 'a' * 64, 'response': {'content': 'exact retained reply'}}],
        'evidence_snapshots': [{'path': 'trace.jsonl', 'bytes': len(evidence),
                                'sha256': hashlib.sha256(evidence).hexdigest()}]}
    path = jobs / 'case.partial.json'
    path.write_text(json.dumps(partial))
    return actor, state, artifact_id, path, [{'jobs': str(jobs)}], identity


def test_actual_native_dependency_bundle_survives_actor_loss_before_replay(tmp_path):
    actor, state, artifact, partial, entries, identity = fixture(tmp_path)
    before = partial.read_bytes()
    policy = identity['checkpoint_sha256']
    binding = {'cases': 'exact', 'policy': policy}
    first = retain_partial_artifacts(entries, tmp_path, policy_id=policy, binding=binding, servers=[actor])
    assert set(first['blocks']) == {state, artifact}
    # Brand-new actor store: the old actor is unavailable. Same verified bytes,
    # dependency-first insertion and GET equality are required before replay.
    fresh = Actor()
    restored = retain_partial_artifacts(entries, tmp_path, policy_id=policy, binding=binding,
                                        servers=[fresh], restore=True)
    assert [x[0] for x in fresh.puts] == [state, artifact]
    assert all(owner == 'retained-policy:' + policy for _, owner in fresh.puts)
    assert fresh.get_bytes('/v1/neuralese/native-states/' + state) == actor.get_bytes('/v1/neuralese/native-states/' + state)
    assert restored['collection_binding'] == binding
    assert partial.read_bytes() == before
    fresh.store.close(); actor.store.close()


def test_missing_recursive_ancestor_is_fatal_before_any_restore(tmp_path):
    actor, state, artifact, partial, entries, identity = fixture(tmp_path)
    actor.missing.add(state)
    with pytest.raises(RuntimeError, match='retained artifact unavailable'):
        retain_partial_artifacts(entries, tmp_path, policy_id=identity['checkpoint_sha256'],
                                 binding={}, servers=[actor], restore=True)
    assert actor.puts == []
    assert not (tmp_path / 'retained-partial-blocks/manifest.json').exists()
    actor.store.close()


@pytest.mark.parametrize('change', ['policy', 'collection', 'evidence', 'traversal'])
def test_retained_authority_and_evidence_tamper_fails_closed(tmp_path, change):
    actor, state, artifact, partial, entries, identity = fixture(tmp_path)
    policy = identity['checkpoint_sha256']; binding = {'cases': 'exact'}
    retain_partial_artifacts(entries, tmp_path, policy_id=policy, binding=binding, servers=[actor])
    if change == 'policy': policy = 'f' * 64
    elif change == 'collection': binding = {'cases': 'different'}
    else:
        v = json.loads(partial.read_text())
        v['evidence_snapshots'][0]['sha256' if change == 'evidence' else 'path'] = ('0' * 64 if change == 'evidence' else '../outside')
        partial.write_text(json.dumps(v))
    with pytest.raises((ValueError, FileNotFoundError)):
        retain_partial_artifacts(entries, tmp_path, policy_id=policy, binding=binding, servers=[actor], restore=True)
    assert actor.puts == []
    actor.store.close()


def test_custody_manifest_binding_is_in_the_single_atomic_publication(tmp_path, monkeypatch):
    actor, state, artifact, partial, entries, identity = fixture(tmp_path)
    policy = identity['checkpoint_sha256']; binding = {'cases': 'exact'}
    retain_partial_artifacts(entries, tmp_path, policy_id=policy, binding=binding, servers=[actor])
    path = tmp_path / 'retained-partial-blocks/manifest.json'; before = path.read_bytes()
    def fail(path, value):
        assert value['collection_binding'] == binding
        assert value['retained_partial_sources']
        raise OSError('publication interrupted')
    monkeypatch.setattr('natlang_neuralese.rl.graph_replay.write_json_durable', fail)
    with pytest.raises(OSError, match='publication interrupted'):
        retain_partial_artifacts(entries, tmp_path, policy_id=policy, binding=binding, servers=[actor])
    assert path.read_bytes() == before
    actor.store.close()


def test_real_collect_seam_cannot_launch_before_missing_partial_closure(tmp_path, monkeypatch):
    from natlang_neuralese.rl import production
    actor, state, artifact, partial, entries, identity = fixture(tmp_path)
    actor.missing.add(state)
    repo = Path(__file__).resolve().parents[2]
    cases = tmp_path / 'cases.jsonl'; cases.write_text('{}\n')
    runtime = tmp_path / 'runtime'; (runtime / 'dist/teacher').mkdir(parents=True)
    (runtime / 'dist/teacher/cli.js').write_text('// fixture')
    info = {'rl_policy_binding': {'schema': 'natlang.rl-policy-binding/2',
        'checkpoint_sha256': identity['checkpoint_sha256'], 'communication': 'native-state',
        'native_identity': identity}}
    original_get = actor.get_bytes
    actor.get_bytes = lambda route: json.dumps(info).encode() if route.endswith('/info') else original_get(route)
    monkeypatch.setattr('natlang_neuralese.rl.grpo.Server', lambda _: actor)
    monkeypatch.setattr(production, 'root', lambda _: repo)
    def prepare(args):
        (args.out / 'queue.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in entries))
    monkeypatch.setattr(production, 'prepare', prepare)
    launches = []
    monkeypatch.setattr(production.subprocess, 'run', lambda *a, **k: launches.append(a))
    with pytest.raises(RuntimeError, match='retained artifact unavailable'):
        production.collect(cases=cases, out=tmp_path, endpoints=['http://same-actor'],
            policy_id=identity['checkpoint_sha256'], rows=1, group=1, workers=1, runtime=runtime)
    assert launches == []
    actor.store.close()


def test_collect_stop_archives_new_partial_then_fresh_actor_restores_before_worker(tmp_path, monkeypatch):
    from natlang_neuralese.rl import production
    old, state, artifact, partial, entries, identity = fixture(tmp_path)
    retained_bytes = partial.read_bytes(); partial.unlink()
    repo = Path(__file__).resolve().parents[2]
    cases = tmp_path / 'cases.jsonl'; cases.write_text('{}\n')
    runtime = tmp_path / 'runtime'; (runtime / 'dist/teacher').mkdir(parents=True)
    (runtime / 'dist/teacher/cli.js').write_text('// fixture')
    info = {'rl_policy_binding': {'schema': 'natlang.rl-policy-binding/2',
        'checkpoint_sha256': identity['checkpoint_sha256'], 'communication': 'native-state',
        'native_identity': identity}}
    fresh = Actor(); current = [old]; launches = []
    for actor in (old, fresh):
        original = actor.get_bytes
        actor.get_bytes = lambda route, original=original: (json.dumps(info).encode()
            if route.endswith('/info') else original(route))
    monkeypatch.setattr('natlang_neuralese.rl.grpo.Server', lambda _: current[0])
    monkeypatch.setattr(production, 'root', lambda _: repo)
    def prepare(args):
        (args.out / 'queue.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in entries))
    monkeypatch.setattr(production, 'prepare', prepare)
    def score(out, path):
        path.write_text('')
        return {}
    monkeypatch.setattr(production, 'score', score)
    stop = SimpleNamespace(requested=False)
    def launch(*args, **kwargs):
        if current[0] is fresh:
            # The real production call reaches Popen only after recursive native
            # restoration. No model sampling is needed to recover the input.
            assert fresh.store.get_native_state(state) is not None
            assert fresh.store.get_native_tensor_artifact(artifact) is not None
        launches.append(current[0])
        partial.write_bytes(retained_bytes)
        return control.wait(SimpleNamespace())
    monkeypatch.setattr(production, '_run_collector', launch)
    def wait(child):
        stop.requested = True
        return 0
    control = SimpleNamespace(stop=stop, wait=wait)
    args = dict(cases=cases, out=tmp_path, endpoints=['http://same-policy'],
                policy_id=identity['checkpoint_sha256'], rows=1, group=1, workers=1,
                runtime=runtime, control=control)
    first = production.collect(**args)
    assert first['collection_complete'] is False
    manifest = json.loads((tmp_path / 'retained-partial-blocks/manifest.json').read_text())
    assert set(manifest['blocks']) == {state, artifact}
    # Retire the source store completely; only the authenticated archive remains.
    old.store.close(); current[0] = fresh; stop.requested = False
    second = production.collect(**args)
    assert second['collection_complete'] is False
    assert launches == [old, fresh]
    assert [ident for ident, _ in fresh.puts] == [state, artifact]
    assert partial.read_bytes() == retained_bytes
    fresh.store.close()


def test_wait_exception_drains_actual_child_before_native_partial_archive(tmp_path, monkeypatch):
    from natlang_neuralese.rl import production
    actor, state, artifact, partial, entries, identity = fixture(tmp_path)
    raw = partial.read_bytes();partial.unlink()
    repo = Path(__file__).resolve().parents[2]
    cases = tmp_path / 'cases.jsonl';cases.write_text('{}\n')
    runtime = tmp_path / 'runtime';(runtime / 'dist/teacher').mkdir(parents=True)
    (runtime / 'dist/teacher/cli.js').write_text('// fixture')
    info = {'rl_policy_binding': {'schema': 'natlang.rl-policy-binding/2',
        'checkpoint_sha256': identity['checkpoint_sha256'], 'communication': 'native-state',
        'native_identity': identity}}
    children = []
    original_get = actor.get_bytes
    def get(route):
        if children:
            assert children[0].poll() is not None, 'archive began while collector could still mutate partials'
        return json.dumps(info).encode() if route.endswith('/info') else original_get(route)
    actor.get_bytes = get
    monkeypatch.setattr('natlang_neuralese.rl.grpo.Server', lambda _: actor)
    monkeypatch.setattr(production, 'root', lambda _: repo)
    def prepare(args):
        (args.out / 'queue.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in entries))
    monkeypatch.setattr(production, 'prepare', prepare)
    actual_run = production._run_collector
    script = ('import signal,time,pathlib;signal.signal(signal.SIGTERM,signal.SIG_IGN);'
              f'pathlib.Path({str(partial)!r}).write_bytes({raw!r});time.sleep(60)')
    def run(_command, log, control, **kwargs):
        return actual_run([sys.executable, '-c', script], log, control, shutdown_grace=.05, **kwargs)
    monkeypatch.setattr(production, '_run_collector', run)
    class Control:
        stop = SimpleNamespace(requested=False)
        def wait(self, child):
            children.append(child)
            deadline = time.monotonic() + 10
            while not partial.exists() and time.monotonic() < deadline:time.sleep(.01)
            assert partial.exists()
            raise RuntimeError('real wait observer failed')
    try:
        with pytest.raises(RuntimeError, match='real wait observer failed'):
            production.collect(cases=cases, out=tmp_path, endpoints=['http://same-actor'],
                policy_id=identity['checkpoint_sha256'], rows=1, group=1, workers=1,
                runtime=runtime, control=Control())
        assert children[0].poll() is not None
        manifest = json.loads((tmp_path / 'retained-partial-blocks/manifest.json').read_text())
        assert set(manifest['blocks']) == {state, artifact}
        assert partial.read_bytes() == raw
    finally:
        for child in children:
            if child.poll() is None:child.kill()
            child.wait()
        actor.store.close()
