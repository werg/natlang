"""A native completion allowance never runs past the model context (H5, 2026-11-03)."""
import json
import threading
import urllib.request

import torch

from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.http import serve
from tests.neuralese.test_native_serve import _tiny_source


def test_completion_budget_is_clamped_to_the_remaining_context(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    base.mkdir()
    _tiny_source(base)
    model, _, _roles, config = load_native_model(base, dtype=torch.float32, rank=4)
    checkpoint = tmp_path / 'native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, checkpoint)
    engine = load_native_engine(str(checkpoint), device='cpu', dtype=torch.float32)
    prompt = [{'role': 'user', 'content': 't5 t6 t7 t8 t9 t10'}]
    prompt_positions = engine.prompt_positions(prompt, None)
    engine.context = prompt_positions + 5
    # A sampler that never stops: only the context allowance can end the reply.
    monkeypatch.setattr(engine, '_native_token', lambda _seq: (7, -.5))
    engine.start()
    server = serve(engine, '127.0.0.1', 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f'http://{server.server_address[0]}:{server.server_address[1]}'
    try:
        request = urllib.request.Request(endpoint + '/v1/chat/completions', headers={'content-type': 'application/json'},
            data=json.dumps({'messages': prompt, 'max_tokens': 100000, 'temperature': 1.0, 'seed': 3}).encode())
        with urllib.request.urlopen(request, timeout=60) as response:
            reply = json.loads(response.read())
        assert reply['choices'][0]['finish_reason'] == 'length'
        assert reply['usage']['completion_tokens'] == 5
        assert reply['usage']['prompt_tokens'] + reply['usage']['completion_tokens'] <= engine.context
    finally:
        server.shutdown()
        engine.stop() if hasattr(engine, 'stop') else None
