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
from tokenizers.models import WordLevel, BPE
from tokenizers.pre_tokenizers import WhitespaceSplit
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
@pytest.mark.parametrize('mode', ['handoff', 'recorded', 'forced', 'compact'])
def test_real_native_state_typed_child_archive_parent_import(tmp_path, monkeypatch, family, mode):
    repo = Path(__file__).resolve().parents[2]
    fixture = repo / ('ts-host/test/fixtures/native-state-child-parent.mjs' if mode in {'handoff', 'recorded'} else 'ts-host/test/fixtures/native-state-forced-text.mjs')
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
    if mode=='recorded':
        for spelling,index in [('<|im_start|>',4),('<|im_end|>',10)]:
            del vocabulary[f't{index}'];vocabulary[spelling]=index
    raw_tokenizer = Tokenizer(WordLevel(vocabulary, unk_token='t0'))
    if mode=='recorded':
        raw_tokenizer.pre_tokenizer=WhitespaceSplit()
    if mode not in {'handoff', 'recorded'}:
        # A real BPE merge crosses an ordinary-text body boundary, while actual
        # producer role tokens delimit independent prefix/body/suffix runs.
        for token, index in [('t', 10), ('5', 11), ('6', 12), ('7', 13), ('8', 14), ('t5t6', 15)]:
            del vocabulary[f't{index}']
            vocabulary[token] = index
        raw_tokenizer = Tokenizer(BPE(vocabulary, [('t', '5'), ('t', '6'), ('t', '7'), ('t', '8'), ('t5', 't6')], unk_token='t0'))
        raw_tokenizer.pre_tokenizer = WhitespaceSplit()
        assert raw_tokenizer.encode('t5t6').ids == [15]
        assert raw_tokenizer.encode('t5').ids + raw_tokenizer.encode('t6').ids == [5, 6]
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw_tokenizer,
        unk_token='t0', bos_token='t1', eos_token='t2', pad_token='t3', additional_special_tokens=list(ROLE_MARKERS)+(['<|im_start|>','<|im_end|>'] if mode=='recorded' else []))
    last_only = ' and loop.last' if family == 'lfm' else ''
    tokenizer.chat_template = ("{% for message in messages %}"
        "{% if message['role'] == 'assistant' and message.get('reasoning_content')" + last_only + " %}"
        "{{ message['reasoning_content'] }}{% endif %}{{ message['content'] }}{% endfor %}assistant:")
    if mode not in {'handoff', 'recorded'}:
        tokenizer.chat_template = ("{% for message in messages %}{{ message['role'] }}:"
            "{% if message['role'] == 'assistant' and message.get('reasoning_content')" + last_only + " %}"
            "{{ message['reasoning_content'] }}{% endif %}{{ message['content'] }}"
            "{% for tool in message.get('tool_calls', []) %}"
            "{{ tool['function']['name'] }}({{ tool['function']['arguments'] | tojson }})"
            "{% endfor %}{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")
    if mode=='recorded':
        tokenizer.chat_template=("{% for message in messages %}<|im_start|>{{ message['role'] }} "
            "{% if message['role'] == 'assistant' and message.get('reasoning_content')"+last_only+" %}"
            "{{ message['reasoning_content'] }} {% endif %}{{ message['content'] }}<|im_end|>{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}")
    config = {'architecture': port.config.model_type, 'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer)}
    weights = tmp_path / 'weights.pt'
    torch.save(model_state(model), weights)
    identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex(weights))
    engine = Engine(port, None, tokenizer, TensorStore(), f"native-state/{config['architecture']}",
        shared_model=model, role_tokens=RoleTokens(*model.marker_ids), model_config=config,
        checkpoint_sha256=identity['checkpoint_sha256'], embedding_weight_sha256=identity['embedding_weight_sha256'])
    engine.native_identity = identity
    if mode=='recorded':
        engine.context=8192
    if mode not in {'handoff', 'recorded'}:
        engine.context = 2048
        engine.max_block = 8
    if mode == 'compact':
        from dataclasses import replace
        from natlang_neuralese.model.semantic_vae import SemanticStateVAE, StateLayout
        from natlang_neuralese.model.semantic_channel import SemanticChannel
        raw = engine.native_write_text('t5')
        captured = engine.store.get_native_state(raw['id']).block
        other = {**identity, 'checkpoint_sha256': '9' * 64}
        vae = SemanticStateVAE({'source': StateLayout.from_block(captured), 'peer': StateLayout.from_block(captured)},
                              latent_width=4, hidden_width=16)
        engine.semantic_channel = SemanticChannel(vae, {'source': identity, 'peer': other},
            {'source': tokenizer, 'peer': tokenizer}, adapter_sha256='a' * 64)
        engine.semantic_participant = 'source'
        engine.dialect = 'semantic-state/' + 'a' * 64
    # One payload token is actually forwarded/captured; parent then samples EOS.
    samples = iter([(5, -.2), (61, -.3), (2, -.4)] * 4 if mode not in {'handoff', 'recorded'} else [(60, -.1), (5, -.2), (61, -.3), (2, -.4), (2, -.5)])
    def forced_sample(sequence):
        if sequence.request.messages[-1].get('content') == 'empty-fixture':
            return ((61, -.3) if sequence.native_capture_open else
                    (60, -.1) if sequence.generated_positions == 0 else (2, -.4))
        return ((5, -.2) if not sequence.native_capture_ids else (61, -.3)) if sequence.native_capture_open else (2, -.4)
    def scripted_token(sequence):
        token, logp = next(samples)
        return token, None if mode == 'recorded' else logp
    monkeypatch.setattr(engine, '_native_token', forced_sample if mode not in {'handoff', 'recorded'} else scripted_token)
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
        if mode == 'recorded':
            command.append('recorded')
        process = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=60)
        assert process.returncode == 0, process.stdout + process.stderr
        receipt = json.loads(process.stdout.splitlines()[-1])
        if mode not in {'handoff', 'recorded'}:
            assert receipt == {'forced': True, 'contextual': True, 'view_parts': 2}
            assert imports and all(item['mode'] == ('learned-semantic-state-deltas' if mode == 'compact' else 'exact-native-state') for item in imports)
        else:
            assert imports[0]['id'] == imports[0]['observed_native_state_id'] == receipt['state_id']
            assert imports[0]['mode'] == 'exact-native-state'
            assert receipt['parent_result'] == 'consumed' and receipt['archived']
            if mode == 'recorded':
                assert receipt['recorded'] is True
                from natlang_neuralese.rl.native_replay import compile_replay
                recorded = json.loads((tmp_path / 'archive/native-recorded-turns.json').read_text())
                turns = [{**turn['native']['graph'], 'sampling': turn['native']['sampling']} for turn in recorded if turn['native']['kind']=='generation']
                def fetch(block_id):
                    return engine.store.get_native_state(block_id).block
                with pytest.raises(ValueError, match='unsupported sampling schema'):
                    compile_replay(turns, identity=identity, roles=engine.role_tokens, fetch_block=fetch)
                replay = compile_replay(turns, identity=identity, roles=engine.role_tokens,
                                        fetch_block=fetch, structural=True)
                assert not replay.actions
                assert len(replay.plan.invocations) == 2
                assert any(node.imports for node in replay.plan.invocations)
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
