import socket
import threading
import time
import json
import hashlib
import multiprocessing

import torch
import pytest

from natlang_neuralese.train.diloco.coordinator import initialize_coordinator, serve_coordinator
from natlang_neuralese.train.diloco.protocol import TensorSurface
from natlang_neuralese.train.diloco.worker import DilocoLearner, DilocoSettings, Transport
from natlang_neuralese.train.native_diloco import validate_new_outer, trainer_argv


def _owned_reanchor_process(worker, url, root, result):
    try:
        torch.set_num_threads(1)
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(worker + 1.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        model(torch.ones(1, 2)).sum().backward(); optimizer.step(); optimizer.zero_grad()
        before = model.weight.detach().clone()
        state = optimizer.state_dict()
        rng = torch.get_rng_state().clone()
        settings = DilocoSettings(run_id='process', workers=4, worker_id=worker,
                                  coordinator_url=url, local_steps=100, wire_dtype='fp32', heartbeat_seconds=.1)
        learner = DilocoLearner(settings, surface=TensorSurface.from_module(model),
                                identity={'seed':'authenticated'}, output=root / f'peer-{worker}',
                                transport=Transport(url, 'x'*32, 10))
        learner.reanchor_fresh()
        checkpoint = root / f'peer-{worker}.pt'
        torch.save({'model':model.state_dict(), 'optimizer':state, 'rng':rng,
                    'exposure':{'row':worker}, 'diloco':learner.state_dict()}, checkpoint)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        restored = torch.load(checkpoint, weights_only=True)
        assert torch.equal(restored['model']['weight'], before)
        assert torch.equal(restored['rng'], rng) and restored['exposure'] == {'row':worker}
        assert restored['optimizer']['state'] and restored['diloco']['local_steps'] == 0
        learner.await_reanchor_cohort(digest, source_hash='source', held_sha256='held')
        model(torch.ones(1, 2)).sum().backward(); optimizer.step(); learner.after_step(tokens=worker+1)
        if worker < 3:
            learner._queue(0)
            assert learner.quiesce_for_checkpoint(timeout_seconds=15)
        result.put((worker, 'ok', learner.local_steps))
        learner.close()
    except BaseException as error:
        result.put((worker, type(error).__name__, str(error)))


def test_four_owned_processes_commit_before_quorum_updates(tmp_path):
    seed = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad(): seed.weight.fill_(1.)
    root = tmp_path / 'coordinator'
    initialize_coordinator(root, run_id='process', masters=TensorSurface.from_module(seed).masters,
                           identity={'seed':'authenticated'}, workers=4, min_quorum=3,
                           fragment_bytes=1024, grace_seconds=0, outer_lr=1, outer_momentum=0)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    stop = threading.Event()
    server = threading.Thread(target=serve_coordinator, args=(root,'127.0.0.1',port,'x'*32),
                              kwargs={'stop':stop}, daemon=True); server.start()
    url = f'http://127.0.0.1:{port}'
    transport = Transport(url, 'x'*32, 10)
    for _ in range(100):
        try: transport.json('GET','/v1/status'); break
        except OSError: time.sleep(.02)
    ctx = multiprocessing.get_context('spawn'); results = ctx.Queue()
    processes = [ctx.Process(target=_owned_reanchor_process,args=(i,url,tmp_path,results)) for i in range(4)]
    try:
        for process in processes: process.start()
        received = [results.get(timeout=40) for _ in processes]
        assert sorted(received) == [(i,'ok',1) for i in range(4)]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
        status = transport.json('GET','/v1/status')
        assert status['reanchor_release'] and status['versions'] == [1]
    finally:
        for process in processes:
            if process.is_alive(): process.terminate(); process.join()
        stop.set(); server.join(timeout=3)


def test_new_outer_authenticates_parent_seed_and_history(tmp_path):
    def pin(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return {'path':str(path), 'bytes':len(data), 'sha256':hashlib.sha256(data).hexdigest()}
    old = pin('old', b'historical pending and outer state')
    parent = pin('parent', b'full peer checkpoint')
    seed = pin('seed', b'common seed')
    slot = tmp_path / 'slot'
    slot.write_bytes(b'full peer checkpoint')
    history = [{'checkpoint':old}]
    proof = {'schema':'natlang.diloco-new-outer/1', 'worker':2, 'parent':parent,
             'historical_outer_checkpoint':old, 'seed_sha256':seed['sha256'],
             'source_hash':'a'*64, 'held_sha256':'b'*64}
    declaration = tmp_path / 'declaration.json'
    declaration.write_text(json.dumps(proof))
    kwargs = dict(slot=slot, seed=seed['path'], worker=2, source_hash='a'*64,
                  held_sha256='b'*64, history=history, saved=None)
    assert not validate_new_outer(declaration, **kwargs)[2]
    with pytest.raises(ValueError, match='source'):
        validate_new_outer(declaration, **{**kwargs, 'source_hash':'wrong'})
    slot.write_bytes(b'different peer')
    with pytest.raises(ValueError, match='resumed full checkpoint'):
        validate_new_outer(declaration, **kwargs)


def test_recipe_and_direct_recovery_share_per_peer_identity():
    from natlang_neuralese.train.recipe import stage_input_args
    from scripts.coreweave.diloco_direct_recovery import _identity_argv
    left = ['--continue-from','/shared/seed', '--diloco-new-outer','/peer/0.json',
            '--standalone-diloco-parent','/peer/0.pt', '--diloco','{}']
    right = ['--continue-from','/shared/seed', '--diloco-new-outer','/peer/1.json',
             '--standalone-diloco-parent','/peer/1.pt', '--diloco','{}']
    assert trainer_argv(left) == trainer_argv(right) == _identity_argv(left) == _identity_argv(right)
    assert stage_input_args({'diloco_new_outer':{'path':'/peer/0.json'}}, 'native_state_sft') == [
        '--diloco-new-outer','/peer/0.json']


def test_fresh_outer_retains_peer_weights_until_actual_update(tmp_path):
    seed = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        seed.weight.fill_(1)
    root = tmp_path / 'coordinator'
    initialize_coordinator(root, run_id='new', masters=TensorSurface.from_module(seed).masters,
                           identity={'seed': 'authenticated'}, workers=4, min_quorum=3,
                           fragment_bytes=1024, grace_seconds=0, outer_lr=1, outer_momentum=0)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    stop = threading.Event()
    server = threading.Thread(target=serve_coordinator, args=(root, '127.0.0.1', port, 'x'*32),
                              kwargs={'stop': stop}, daemon=True)
    server.start()
    url = f'http://127.0.0.1:{port}'
    transport = Transport(url, 'x'*32, 10)
    for _ in range(100):
        try:
            transport.json('GET', '/v1/status')
            break
        except OSError:
            time.sleep(.02)
    learners = []
    try:
        for worker in range(4):
            model = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(worker + 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            checkpoint = tmp_path / f'peer-{worker}.pt'
            rng = torch.get_rng_state().clone()
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                        'rng':rng, 'exposure':{'row':worker + 7}}, checkpoint)
            saved = torch.load(checkpoint, weights_only=True)
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            before = model.weight.detach().clone()
            settings = DilocoSettings(run_id='new', workers=4, worker_id=worker,
                                      coordinator_url=url, local_steps=100,
                                      wire_dtype='fp32', heartbeat_seconds=.1)
            learner = DilocoLearner(settings, surface=TensorSurface.from_module(model),
                                    identity={'seed': 'authenticated'}, output=tmp_path / str(worker),
                                    transport=transport)
            deviations = learner.reanchor_fresh()
            learners.append((model, learner))
            assert torch.equal(before, model.weight)
            assert deviations[0]['max_abs'] == pytest.approx(float((before - 1).abs().max()))
            assert learner.local_steps == 0 and not learner._flushable(0)
            assert torch.equal(learner.baselines[0], seed.weight.flatten())
            assert saved['exposure'] == {'row':worker + 7}
            assert torch.equal(saved['rng'], rng)
            assert optimizer.state_dict()['state']
            transitioned = tmp_path / f'transition-{worker}.pt'
            torch.save({'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                        'rng':rng, 'exposure':saved['exposure'], 'diloco':learner.state_dict()}, transitioned)
            restored = torch.load(transitioned, weights_only=True)
            assert torch.equal(restored['model']['weight'], saved['model']['weight'])
            assert torch.equal(restored['rng'], saved['rng']) and restored['exposure'] == saved['exposure']
            assert restored['diloco']['local_steps'] == 0 and not restored['diloco']['pending']
            for key in saved['optimizer']['state']:
                for name, value in saved['optimizer']['state'][key].items():
                    assert torch.equal(restored['optimizer']['state'][key][name], value)
        # A peer can restart from its durable transition before release.
        model, prior = learners[3]
        prior.close()
        restored = torch.load(tmp_path / 'transition-3.pt', weights_only=True)
        resumed = DilocoLearner(prior.settings, surface=TensorSurface.from_module(model),
                                identity={'seed':'authenticated'}, output=tmp_path / 'resume-before-release',
                                transport=transport)
        resumed.resume(restored['diloco'])
        learners[3] = (model, resumed)
        ready = []
        for worker, (_, learner) in enumerate(learners):
            digest = hashlib.sha256((tmp_path / f'transition-{worker}.pt').read_bytes()).hexdigest()
            thread = threading.Thread(target=learner.await_reanchor_cohort, args=(digest,),
                                      kwargs={'source_hash':'source', 'held_sha256':'held'})
            thread.start()
            ready.append(thread)
        for thread in ready:
            thread.join(timeout=10)
            assert not thread.is_alive()
        # A process restarted after durable release resumes its saved outer
        # state rather than trying to reanchor a now-active coordinator.
        model, prior = learners[3]
        prior.close()
        saved = torch.load(tmp_path / 'transition-3.pt', weights_only=True)
        resumed = DilocoLearner(prior.settings, surface=TensorSurface.from_module(model),
                                identity={'seed':'authenticated'}, output=tmp_path / 'resume-3',
                                transport=transport)
        resumed.resume(saved['diloco'])
        learners[3] = (model, resumed)
        resumed.await_reanchor_cohort(hashlib.sha256((tmp_path / 'transition-3.pt').read_bytes()).hexdigest(),
                                     source_hash='source', held_sha256='held')
        expected = torch.stack([model.weight.detach().flatten() - .1 for model, _ in learners[:3]]).mean(0)
        for model, learner in learners[:3]:
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            optimizer.zero_grad()
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            learner.after_step(tokens=1)
            learner._queue(0)
        for _ in range(200):
            if transport.json('GET', '/v1/status')['versions'] == [1]:
                break
            time.sleep(.02)
        assert transport.json('GET', '/v1/status')['versions'] == [1]
        update = learners[0][1]._fetch(0, -1)
        assert torch.allclose(update[1], expected, atol=1e-6)
        model, prior = learners[3]
        retained = model.weight.detach().clone()
        prior.close()
        resumed = DilocoLearner(prior.settings, surface=TensorSurface.from_module(model),
                                identity={'seed':'authenticated'}, output=tmp_path / 'resume-after-merge',
                                transport=transport)
        resumed.resume(saved['diloco'])
        learners[3] = (model, resumed)
        assert torch.allclose(model.weight.flatten(), expected + retained.flatten() - 1, atol=1e-6)
        resumed.await_reanchor_cohort(hashlib.sha256((tmp_path / 'transition-3.pt').read_bytes()).hexdigest(),
                                     source_hash='source', held_sha256='held')
        with pytest.raises(ValueError, match='unused coordinator'):
            learners[0][1].reanchor_fresh()
    finally:
        for _, learner in learners:
            learner.close()
        stop.set()
        server.join(timeout=3)
