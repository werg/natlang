"""The reference engine admits queued requests into its batch only while their worst-case memory fits, requeues an
out-of-memory prefill instead of failing it, and answers /health while the engine thread is busy."""
import json
import threading
import time
import urllib.request
from concurrent.futures import Future

import pytest
import torch

from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.engine import GenerationRequest, Sequence
from natlang_neuralese.serve.http import serve
from test_native_serve import _tiny_source

GB = 2**30


@pytest.fixture(scope='module')
def engine(tmp_path_factory):
    root = tmp_path_factory.mktemp('admission')
    base = root / 'base'
    base.mkdir()
    _tiny_source(base)
    model, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    checkpoint = root / 'native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, checkpoint)
    return load_native_engine(str(checkpoint), device='cpu', dtype=torch.float32)


def _seq(engine, prompt=10, max_tokens=20):
    request = GenerationRequest(messages=[{'role': 'user', 'content': 't5'}], max_tokens=max_tokens)
    return Sequence(request, Future(), prompt_plan=[('tokens', list(range(prompt)))])


@pytest.fixture
def cuda_memory(engine, monkeypatch):
    """A CUDA device as admission sees it: allocated and free bytes the test sets."""
    state = {'allocated': 4 * GB, 'reserved': 4 * GB, 'free': 100 * GB}
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device=None: state['allocated'])
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device=None: state['reserved'])
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device=None: (state['free'], 128 * GB))
    monkeypatch.setattr(engine, 'device', 'cuda')
    monkeypatch.setattr(engine, 'prefix_cache', None)
    monkeypatch.setattr(engine, 'memory_budget_bytes', 9 * GB)
    monkeypatch.setattr(engine, 'context', 32768)
    monkeypatch.setattr(engine, '_kv_bytes_per_token', None)
    monkeypatch.setattr(engine, '_prefill_workspace_bytes', 0)
    monkeypatch.setattr(engine, 'admission_bounded', True)
    monkeypatch.setattr(engine, '_active', [])
    monkeypatch.setattr(engine, '_waiting', __import__('collections').deque())
    return state


def test_admission_is_bounded_by_the_memory_cap_and_the_oldest_request_always_runs(engine, cuda_memory):
    waiting = [_seq(engine, prompt=1000, max_tokens=1000) for _ in range(6)]
    engine._waiting.extend(waiting)
    # Nothing measured yet: the oldest runs alone (its prefill measures the cache's bytes per token).
    engine._admit()
    assert engine._active == waiting[:1] and len(engine._waiting) == 5
    # 2000 worst-case positions x 1 MiB = ~2 GiB per request; 9 GiB cap - 4 GiB allocated - 0.9 GiB watermark leaves
    # room for the running one plus one more.
    engine._kv_bytes_per_token = float(2**20)
    engine._admit()
    assert engine._active == waiting[:2] and engine._admission_deferrals >= 1
    # The device's own free memory bounds it too (another process holds the rest of the GPU).
    cuda_memory['allocated'] = cuda_memory['reserved'] = 1 * GB
    cuda_memory['free'] = 3 * GB
    engine._admit()
    assert engine._active == waiting[:2]
    cuda_memory['free'] = 100 * GB
    engine._admit()
    assert engine._active == waiting[:3]
    # A request whose client went away leaves the queue without running.
    engine._waiting[0].request.cancelled = True
    engine._active = []
    engine._admit()
    assert waiting[3].future.exception() is not None and engine._active[0] is waiting[4]
    # --unbounded-admission: every queued request joins the batch.
    engine.admission_bounded = False
    engine._admit()
    assert not engine._waiting


def test_an_out_of_memory_prefill_is_requeued_while_others_run_and_fails_alone(engine, cuda_memory, monkeypatch):
    running, victim = _seq(engine), _seq(engine)
    running.phase = 'text'
    engine._active.extend([running, victim])

    def oom(seq):
        raise torch.OutOfMemoryError('CUDA out of memory (test)')
        yield
    monkeypatch.setattr(engine, '_native_sequence_step', lambda seq: oom(seq) if seq is victim else iter(()))
    engine._native_round_group([running, victim])
    assert victim.requeued and not victim.future.done()
    fresh = engine._waiting[0]
    assert fresh is not victim and fresh.future is victim.future and fresh.request is victim.request
    assert fresh.phase == 'prefill' and engine._oom_requeues == 1
    # Alone, the same failure is the request's error (no requeue loop).
    engine._active[:] = [fresh]
    engine._waiting.clear()
    victim = fresh
    engine._native_round_group([fresh])
    assert isinstance(fresh.future.exception(), torch.OutOfMemoryError) and not engine._waiting


def test_health_answers_while_the_engine_thread_is_busy(engine, monkeypatch):
    """/health never waits for the scheduler: a round that blocks the engine thread leaves it answering."""
    release, entered = threading.Event(), threading.Event()

    def stuck_round(sequences):
        entered.set()
        release.wait(30)
        for seq in sequences:
            seq.future.set_exception(RuntimeError('released'))
    monkeypatch.setattr(engine, '_round', stuck_round)
    monkeypatch.setattr(engine, '_active', [])
    monkeypatch.setattr(engine, '_waiting', __import__('collections').deque())
    engine.start()
    server = serve(engine, '127.0.0.1', 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        engine.submit(GenerationRequest(messages=[{'role': 'user', 'content': 't5'}], max_tokens=2))
        assert entered.wait(30)
        started = time.monotonic()
        with urllib.request.urlopen(endpoint + '/health', timeout=5) as response:
            assert json.loads(response.read()) == {'status': 'ok'}
        assert time.monotonic() - started < 2
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        engine.stop()
