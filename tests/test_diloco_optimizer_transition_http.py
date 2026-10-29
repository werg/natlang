"""Authenticated mode transition retains the exact drained worker and outer sequence."""
import argparse
import hashlib
import json
import socket
import threading
import time

import pytest
import torch

from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport
from natlang_neuralese.train.native_diloco import identity
from natlang_neuralese.rl import diloco_contract as dc


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_http_rounding_transition_requires_exact_drained_identity(tmp_path, monkeypatch):
    model = torch.nn.Linear(2, 1, bias=False)
    seed = tmp_path / 'seed.pt'
    torch.save({'model': model.state_dict()}, seed)
    recipe, source, held = (tmp_path / name for name in ('recipe', 'source.jsonl', 'held.jsonl'))
    for path in (recipe, source, held):
        path.write_text(path.name)
    binding = {'path': str(recipe), 'sha256': ref(recipe)['sha256'],
               'file_sha256': ref(recipe)['sha256'], 'lineage_sha256': 'b' * 64}
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    settings = DilocoSettings(run_id='rounding-switch', workers=4, worker_id=2,
                              coordinator_url=url, allow_private_http=True, local_steps=1,
                              wire_dtype='fp32', heartbeat_seconds=.1)
    out = tmp_path / 'worker'
    out.mkdir()
    argv = ['--recipe-binding', json.dumps(binding), '--stage-id', 'sft',
            '--native-graphs', str(source), '--held-native-graphs', str(held),
            '--continue-from', str(seed), '--out', str(out),
            '--muon-rounding', 'stochastic', '--diloco', json.dumps(settings.__dict__)]
    old_identity = identity(seed, argv, settings)
    old_role = {'kind': 'sft', 'identity': old_identity,
                'dependencies': [ref(recipe), ref(source), ref(held)]}
    authority = {'schema': 'natlang.diloco-heterogeneous-identity/1',
                 'baseline_sha256': ref(seed)['sha256'], 'recipe_lineage_sha256': 'b' * 64,
                 'diloco': settings.identity(), 'roles': {str(i): [old_role] for i in range(4)}}
    root = tmp_path / 'outer'
    initialize_coordinator(root, run_id='rounding-switch', masters=TensorSurface.from_module(model).masters,
                           identity=authority, workers=4, min_quorum=1, grace_seconds=0,
                           fragment_bytes=1024, seed_source={'sha256': ref(seed)['sha256']})
    stop = threading.Event()
    thread = threading.Thread(target=serve_coordinator, args=(root, '127.0.0.1', port, 'x' * 32),
                              kwargs={'stop': stop}, daemon=True)
    thread.start()
    transport = Transport(url, 'x' * 32, 10, allow_http=True)
    for _ in range(100):
        try:
            transport.json('GET', '/v1/manifest')
            break
        except OSError:
            time.sleep(.02)
    monkeypatch.setattr(dc, 'transport_from_env', lambda _: transport)
    learner = DilocoLearner(settings, surface=TensorSurface.from_module(model), identity=old_identity,
                            output=tmp_path / 'learner', transport=transport)
    try:
        learner.start()
        with torch.no_grad():
            model.weight.add_(.1)
        learner.after_step(tokens=10)
        assert learner.drain_for_contract()
        saved = learner.state_dict()
        slot = out / 'checkpoint.pt'
        torch.save({'training': {'diloco': saved}}, slot)
        transition = {'schema': 'natlang.diloco-optimizer-contract-transition/1', 'verified': True,
                      'checkpoint_sha256': ref(slot)['sha256'], 'from': 'stochastic', 'to': 'kahan'}
        new_argv = list(argv)
        new_argv[new_argv.index('--muon-rounding') + 1] = 'kahan'
        args = argparse.Namespace(continue_from=seed, out=out)
        bad_saved = {**saved, 'steps_since': [1] * len(saved['steps_since'])}
        with pytest.raises((ValueError, RuntimeError)):
            dc.register_optimizer_transition(args, new_argv, bad_saved, transition)
        with pytest.raises((ValueError, RuntimeError)):
            dc.register_optimizer_transition(args, new_argv, saved, {**transition, 'checkpoint_sha256': '0' * 64})
        with pytest.raises((ValueError, RuntimeError)):
            dc.register_optimizer_transition(args, new_argv, saved, {**transition, 'to': 'stochastic'})
        bad_source = tmp_path / 'other.jsonl'
        bad_source.write_text('other')
        altered = list(new_argv)
        altered[altered.index('--native-graphs') + 1] = str(bad_source)
        with pytest.raises((ValueError, RuntimeError, OSError)):
            dc.register_optimizer_transition(args, altered, saved, transition)
        altered = list(new_argv) + ['--learning-rate', '0.01']
        with pytest.raises((ValueError, RuntimeError, OSError)):
            dc.register_optimizer_transition(args, altered, saved, transition)
        result, new_identity = dc.register_optimizer_transition(args, new_argv, saved, transition)
        assert result['status'] == 'registered'
        previous_sequences = list(saved['sequences'])
        assert learner.switch_role_contract(new_identity) == result['sha256']
        assert learner.state_dict()['sequences'] == previous_sequences
        with torch.no_grad():
            model.weight.add_(.1)
        learner.after_step(tokens=10)
        assert learner.drain_for_contract()
        after = learner.state_dict()
        assert all(a >= b for a, b in zip(after['sequences'], previous_sequences))
        assert any(a > b for a, b in zip(after['sequences'], previous_sequences))
        assert after['contract_history'][-1]['from'] == saved['contract_sha256']
        assert after['contract_history'][-1]['to'] == result['sha256']
    finally:
        learner.close()
        stop.set()
        thread.join(timeout=5)
