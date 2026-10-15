"""Actual native forwards through TypeScript child/result/archive/parent import.

Token selection and typed tool readout are deterministic fixture controls;
this checks real runtime state transport, not learned task performance.
"""
import json
from pathlib import Path
import shutil
import subprocess
import threading

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from test_shared_state import tiny
from natlang_neuralese.model.shared_state import SharedStateModel, RoleTokens, ROLE_MARKERS
from natlang_neuralese.model.native_model import native_identity, model_state
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.serve.engine import Engine
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.serve.http import serve


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_real_native_state_typed_child_archive_parent_import(tmp_path, monkeypatch, family):
    repo = Path(__file__).resolve().parents[2]
    fixture = repo / 'ts-host/test/fixtures/native-state-child-parent.mjs'
    if not (repo / 'ts-host/dist/native/neuralese.js').is_file():
        pytest.fail('build:node must run before the integrated native harness test')
    node = shutil.which('node')
    if node is None:
        pytest.fail('Node is required for the actual TypeScript native harness')
    port = tiny(family)
    model = SharedStateModel(port, (60, 61, 62, 63), rank=4).eval()
    torch.nn.init.normal_(model.adapter.up.weight, std=.02)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    vocabulary = {f't{i}': i for i in range(port.embedding_weight.shape[0])}
    for marker, token in zip(ROLE_MARKERS, model.marker_ids):
        del vocabulary[f't{token}']
        vocabulary[marker] = token
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token='t0')),
        unk_token='t0', bos_token='t1', eos_token='t2', pad_token='t3', additional_special_tokens=list(ROLE_MARKERS))
    last_only = ' and loop.last' if family == 'lfm' else ''
    tokenizer.chat_template = ("{% for message in messages %}"
        "{% if message['role'] == 'assistant' and message.get('reasoning_content')" + last_only + " %}"
        "{{ message['reasoning_content'] }}{% endif %}{{ message['content'] }}{% endfor %}assistant:")
    config = {'architecture': port.config.model_type, 'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer)}
    weights = tmp_path / 'weights.pt'
    torch.save(model_state(model), weights)
    identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex(weights))
    engine = Engine(port, None, tokenizer, TensorStore(), f"native-state/{config['architecture']}",
        shared_model=model, role_tokens=RoleTokens(*model.marker_ids), model_config=config,
        checkpoint_sha256=identity['checkpoint_sha256'], embedding_weight_sha256=identity['embedding_weight_sha256'])
    # One payload token is actually forwarded/captured; parent then samples EOS.
    samples = iter([(60, -.1), (5, -.2), (61, -.3), (2, -.4), (2, -.5)])
    monkeypatch.setattr(engine, '_native_token', lambda _sequence: next(samples))
    imports, original_consume = [], engine._native_consume
    def record_consume(sequence, block_id):
        original_consume(sequence, block_id)
        imports.extend(sequence.native_imports[-1:])
    monkeypatch.setattr(engine, '_native_consume', record_consume)
    engine.start()
    server = serve(engine, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        command = [node, str(fixture), f'http://127.0.0.1:{server.server_address[1]}', str(tmp_path / 'archive')]
        process = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=60)
        assert process.returncode == 0, process.stdout + process.stderr
        receipt = json.loads(process.stdout.splitlines()[-1])
        assert imports == [{'id': receipt['state_id'], 'mode': 'exact-native-state'}]
        assert receipt['parent_result'] == 'consumed' and receipt['archived']
        block = engine.store.get_native_state(receipt['state_id']).block
        assert block.token_ids.tolist() == [5]
        assert any(layer['kind'] == 'attention' for layer in block.layers)
        if family == 'lfm':
            assert any(layer['kind'] == 'finite_convolution' for layer in block.layers)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        engine.stop()
