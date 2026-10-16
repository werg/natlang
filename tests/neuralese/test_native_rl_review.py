"""Independent regressions for actual shared participant binding and retry durability."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex, canonical_json_sha256_hex
from natlang_neuralese.rl.production import expected_native_actor, validate_actor_receipts


def test_shared_file_digest_does_not_qualify_wrong_participant_or_adapter(tmp_path):
    checkpoint = tmp_path/'shared.pt'; checkpoint.write_bytes(b'shared actual byte binding')
    identity = {'checkpoint_sha256': sha256_file_hex(checkpoint), 'tokenizer_fingerprint': '1'*64,
                'embedding_weight_sha256': '2'*64, 'architecture': 'lfm2'}
    adapter = tmp_path/'adapter.json'
    binding = {'path': str(checkpoint), 'sha256': identity['checkpoint_sha256']}
    adapter.write_text(json.dumps({'schema': 'natlang.native-semantic-adapter/1', 'checkpoint': binding,
        'models': {'student': {'checkpoint': {**binding, 'participant': 'student'}, 'native_identity': identity}}}))
    expected = expected_native_actor(adapter, 'student', binding['sha256'])
    receipt = {'schema': 'natlang.rl-policy-binding/2', 'communication': 'native-state',
        'checkpoint_sha256': binding['sha256'], **expected}
    validate_actor_receipts({'endpoint': receipt}, policy_id=binding['sha256'], neuralese_server=True, expected_native=expected)
    for change in ({'native_participant': 'other'}, {'semantic_adapter_sha256': '3'*64},
                   {'native_identity': {**identity, 'architecture': 'mellum'}}):
        with pytest.raises(ValueError, match='requested shared participant'):
            validate_actor_receipts({'endpoint': {**receipt, **change}}, policy_id=binding['sha256'], neuralese_server=True, expected_native=expected)


@pytest.mark.parametrize('rl,failure_kind', [(False, 'exit'), (True, 'exit'),
    (False, 'launch'), (True, 'launch'), (False, 'missing-completion'), (False, 'invalid-completion')])
def test_failed_child_retries_preserved_work_without_marking_success(tmp_path, monkeypatch, rl, failure_kind):
    from natlang_neuralese.train import native_readiness as controller, loop
    out = tmp_path/'controller'; out.mkdir()
    curriculum = tmp_path/'curriculum.json'; curriculum.write_text('{}')
    config = {'schema': 'natlang.native-shared-stage-controller/1', 'out': str(out),
        'command': ['semantic-fixture'], 'curriculum': str(curriculum), 'participants': [], 'poll_seconds': 1}
    if failure_kind == 'invalid-completion':
        (out/'shared-stage-completion.json').write_text('{}')
    if rl:
        config['rl_bootstrap'] = {'path': 'fixture-plan', 'sha256': '1'*64}
        (out/'shared-stage-completion.json').write_text(json.dumps({
            'participants_signature': canonical_json_sha256_hex([{'model_id': 'a'}, {'model_id': 'b'}]),
            'curriculum_signature': canonical_json_sha256_hex({})}))
        def validate(path, published):
            key, value = next(iter(published.items()))
            return {'fixture': True}, value['execution_signature']
        monkeypatch.setattr(controller, 'validate_completion', validate)
        monkeypatch.setattr(controller, 'rl_bootstrap_command', lambda *args: (['rl-fixture'], 'exact-rl-signature'))
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    monkeypatch.setattr(controller, 'ready_participants', lambda config: ([{'model_id': 'a'}, {'model_id': 'b'}], []))
    stop = SimpleNamespace(requested=False, install=lambda: None, restore=lambda: None)
    monkeypatch.setattr(loop, 'StopSignal', lambda: stop)
    clock, launches = [100.], []
    monkeypatch.setattr(controller.time, 'time', lambda: clock[0])
    def sleep(seconds):
        saved = json.loads((out/'controller.json').read_text())
        assert saved['retry']['attempts'] == 1
        assert saved['retry']['exit_code'] == (None if failure_kind == 'launch' else
                                               7 if failure_kind == 'exit' else 0)
        if failure_kind == 'launch':
            assert 'launch failed' in saved['retry']['reason']
        if failure_kind.endswith('completion'):
            assert 'completion rejected' in saved['retry']['reason']
            assert saved['last_successful_participants'] is None
        assert saved['last_successful_rl_bootstrap'] is None
        clock[0] += seconds
    monkeypatch.setattr(controller.time, 'sleep', sleep)
    class Child:
        pid = 123456
        def __init__(self, code): self.returncode = code
        def poll(self): return self.returncode
        def terminate(self): pass
    def launch(argv, **kwargs):
        launches.append(argv)
        if len(launches) == 2:
            stop.requested = True
        if failure_kind == 'launch' and len(launches) == 1:
            raise FileNotFoundError('repairable missing worker executable')
        return Child(7 if len(launches) == 1 and failure_kind == 'exit' else 0)
    monkeypatch.setattr(controller.subprocess, 'Popen', launch)
    result = controller.run(path)
    assert len(launches) == 2 and launches[0] == launches[1] == (['rl-fixture'] if rl else ['semantic-fixture'])
    assert result['status'] == 'stopped-after-checkpoint-request'
    assert result['last_successful_rl_bootstrap'] is None
    assert result['retry']['attempts'] == 1


def test_child_launch_retries_transient_proc_identity_before_persisting(tmp_path, monkeypatch):
    import sys
    from natlang_neuralese.train import native_readiness as controller
    actual = controller._owned_process_identity
    reads = []
    def transient(pid, launch, **kwargs):
        reads.append(pid)
        return None if len(reads) <= 2 else actual(pid, launch, **kwargs)
    monkeypatch.setattr(controller, '_owned_process_identity', transient)
    argv = [sys.executable, '-c', 'import time; time.sleep(300)']
    child, launch = controller._launch_child(argv, 'native-semantic', {}, tmp_path, 'execution', None)
    try:
        assert len(reads) >= 3
        assert launch['identity']['pid'] == child.pid
        assert json.loads((tmp_path/'child-launch.json').read_text())['identity'] == launch['identity']
        assert controller._recover_child(launch).pid == child.pid
    finally:
        child.terminate(); child.wait(timeout=5)


def test_child_launch_identity_miss_does_not_block_supervision(tmp_path, monkeypatch):
    from natlang_neuralese.train import native_readiness as controller
    class Child:
        pid = 2147483647
        def poll(self): return None
    child = Child()
    monkeypatch.setattr(controller.subprocess, 'Popen', lambda *args, **kwargs: child)
    monkeypatch.setattr(controller, '_owned_process_identity', lambda *args, **kwargs: None)
    monkeypatch.setattr(controller.time, 'sleep', lambda _: pytest.fail('absent process must not be retried'))
    observed, launch = controller._launch_child(['missing-fixture'], 'native-semantic', {}, tmp_path, 'execution', None)
    assert observed is child and launch['identity'] is None
    assert launch['status'] == 'running' and launch['token']


def test_wrapper_argv_identity_miss_is_bounded_and_owned_tree_recoverable(tmp_path):
    import sys
    from natlang_neuralese.train import native_readiness as controller
    script = tmp_path/'worker'
    script.write_text('#!' + sys.executable + '\nimport time\ntime.sleep(300)\n')
    script.chmod(0o700)
    child, launch = controller._launch_child([str(script)], 'native-semantic', {}, tmp_path, 'execution', None)
    try:
        assert child.poll() is None and launch['identity'] is None
        recovered = controller._recover_child(launch)
        assert recovered is not None
        assert any(entry['pid'] == child.pid for entry in controller._owned_tree(launch))
    finally:
        child.terminate(); child.wait(timeout=5)


def test_restart_adopts_exact_live_child_and_drains_without_duplicate_launch(tmp_path, monkeypatch):
    import sys
    from natlang_neuralese.train import native_readiness as controller, loop
    out = tmp_path/'out'; out.mkdir()
    curriculum = tmp_path/'curriculum.json'; curriculum.write_text('{}')
    config = {'schema': 'natlang.native-shared-stage-controller/1', 'out': str(out),
        'command': [sys.executable, '-c', 'import time; time.sleep(300)'],
        'curriculum': str(curriculum), 'participants': [], 'poll_seconds': 1}
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    child, launch = controller._launch_child(config['command'], 'native-semantic', config, out, 'bound-execution', None)
    try:
        assert launch['identity']['pid'] == child.pid
        assert controller._recover_child(launch).pid == child.pid
        # A PID match is insufficient: boot/start identity and argv are bound.
        with pytest.raises(RuntimeError, match='identity changed'):
            controller._recover_child({**launch, 'identity': {**launch['identity'], 'start_ticks': '0'}})
        assert controller._owned_process_identity(child.pid, {**launch, 'argv': [sys.executable, '-c', 'wrong']}) is None
        # Simulate controller death after launch.json, before controller.json.
        assert not (out/'controller.json').exists()
        monkeypatch.setattr(controller, 'ready_participants', lambda config: ([], []))
        monkeypatch.setattr(loop, 'StopSignal', lambda: SimpleNamespace(requested=True, install=lambda: None, restore=lambda: None))
        monkeypatch.setattr(controller.time, 'sleep', lambda seconds: child.wait(timeout=5))
        monkeypatch.setattr(controller.subprocess, 'Popen', lambda *a, **k: pytest.fail('duplicate child launch'))
        result = controller.run(path)
        assert result['status'] == 'stopped-after-checkpoint-request'
        assert result['child_pid'] is None
        assert child.wait(timeout=5) != 0
        assert json.loads((out/'child-launch.json').read_text())['status'] == 'exited'
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=5)


def test_actual_shared_actor_attests_loaded_participant_and_adapter(tmp_path):
    """Real CPU model and persisted VAE loading; no learned-quality claim."""
    import copy
    import torch
    from test_native_serve import _tiny_source
    from natlang_neuralese.model.native_model import load_native_model, model_state, native_identity, SHARED_CHECKPOINT_SCHEMA
    from natlang_neuralese.model.shared_state import Invocation, compile_graph, capture_block
    from natlang_neuralese.model.semantic_vae import SemanticStateVAE, StateLayout
    from natlang_neuralese.serve import load_native_engine
    base = tmp_path/'base'; base.mkdir(); _tiny_source(base)
    model, tokenizer, roles, config = load_native_model(base, dtype=torch.float32, rank=4)
    peer = copy.deepcopy(model)
    with torch.no_grad():
        peer.port.hf.get_input_embeddings().weight.add_(0.01)
        result = model(compile_graph([Invocation('writer', (roles.produce_open, 4, roles.produce_close), producing=((1, 2),))]))
        block = capture_block(result, 'writer', 1, 2, text='t4',
            **native_identity(model, config, checkpoint_sha256='0'*64))
    vae = SemanticStateVAE({name: StateLayout.from_block(block) for name in ('student', 'peer')}, latent_width=4, hidden_width=8)
    checkpoint = tmp_path/'shared.pt'
    torch.save({'schema': SHARED_CHECKPOINT_SCHEMA, 'training': {'step': 0},
        'participants': {name: {'model_config': config, 'model_state': model_state(value)}
                         for name, value in [('student', model), ('peer', peer)]},
        'vae_specification': vae.specification(), 'vae_state': vae.state_dict()}, checkpoint)
    digest = sha256_file_hex(checkpoint)
    tokenizers = tmp_path/'tokenizers'; tokenizer.save_pretrained(tokenizers)
    tokenizer_binding = {'path': 'tokenizers', 'files': {str(path.relative_to(tokenizers)): sha256_file_hex(path)
                          for path in tokenizers.rglob('*') if path.is_file()}}
    adapter = tmp_path/'adapter.json'
    binding = {'path': str(checkpoint), 'sha256': digest}
    adapter.write_text(json.dumps({'schema': 'natlang.native-semantic-adapter/1', 'checkpoint': binding,
        'specification_sha256': canonical_json_sha256_hex(vae.specification()),
        'models': {name: {'native_identity': native_identity(value, config, checkpoint_sha256=digest),
                         'checkpoint': {**binding, 'participant': name}}
                   for name, value in [('student', model), ('peer', peer)]},
        'tokenizers': {name: tokenizer_binding for name in ('student', 'peer')}}))
    engine = load_native_engine(str(checkpoint), participant='student', semantic_adapter=str(adapter), dtype=torch.float32)
    expected = expected_native_actor(adapter, 'student', digest)
    validate_actor_receipts({'actual-engine': engine.rl_policy_binding}, policy_id=digest,
                           neuralese_server=True, expected_native=expected)
    assert engine.dialect == 'semantic-state/' + expected['semantic_adapter_sha256']
    with pytest.raises(ValueError, match='requested shared participant'):
        validate_actor_receipts({'actual-engine': engine.rl_policy_binding}, policy_id=digest,
            neuralese_server=True, expected_native=expected_native_actor(adapter, 'peer', digest))
    from natlang_neuralese.model.native_bundle import export_bundle,verify_bundle
    import shutil
    original_write = engine.native_write_text('t4')
    exported = export_bundle(checkpoint,tmp_path/'bundle',semantic_adapter=adapter)
    moved = tmp_path/'remote'
    (tmp_path/'bundle').rename(moved)
    checkpoint.unlink();adapter.unlink();shutil.rmtree(tokenizers)
    base.rename(tmp_path/'remote-base')
    portable = verify_bundle(moved)
    assert exported['checkpoint']['sha256']==portable['checkpoint']['sha256']==digest
    assert portable['semantic_adapter']['sha256']==expected['semantic_adapter_sha256']
    relocated = load_native_engine(portable['checkpoint']['resolved'],participant='student',
        semantic_adapter=portable['semantic_adapter']['resolved'],base=str(tmp_path/'remote-base'),dtype=torch.float32)
    assert relocated.rl_policy_binding==engine.rl_policy_binding
    assert expected_native_actor(portable['semantic_adapter']['resolved'],'student',digest)==expected
    assert relocated.native_write_text('t4')['id']==original_write['id']
    relocated.store.close()
    engine.store.close()


def test_orphaned_descendant_is_drained_before_retry(tmp_path):
    import sys
    import time
    from natlang_neuralese.train import native_readiness as controller
    out = tmp_path/'out'; out.mkdir()
    # The command exits while its subprocess retains exact launch ownership.
    script = ('import subprocess,sys; '
              'subprocess.Popen([sys.executable,"-c","import time; time.sleep(300)"])')
    child, launch = controller._launch_child([sys.executable, '-c', script], 'native-rl', {}, out, 'execution', 'rl')
    child.wait(timeout=5)
    recovered = controller._recover_child(launch)
    assert recovered is not None and recovered.identity is None
    try:
        recovered.terminate()
        deadline = time.monotonic() + 5
        while recovered.poll() is None:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert recovered.returncode != 0  # no inferred successful training
        assert controller._owned_tree(launch) == []
    finally:
        recovered.terminate()
