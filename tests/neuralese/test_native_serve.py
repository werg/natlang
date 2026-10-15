"""The ordinary chat server loads and transports the trained native state end to end."""
import torch
import json
import threading
import urllib.request
from dataclasses import replace
from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA, load_native_model, model_state
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.http import serve
from natlang_neuralese.serve.store import decode_native_state


def _tiny_source(path):
    config = Lfm2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        layer_types=['conv', 'full_attention'], conv_L_cache=4, max_position_embeddings=128)
    torch.manual_seed(83)
    Lfm2ForCausalLM(config).save_pretrained(path)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {f't{i}': i for i in range(64)}, unk_token='t0')), unk_token='t0',
        bos_token='t1', eos_token='t2', pad_token='t3')
    tokenizer.chat_template = ("{% for message in messages %}"
        "{% if message['role'] == 'assistant' and message.get('reasoning_content') and loop.last %}"
        "{{ message['reasoning_content'] }}{% endif %}{{ message['content'] }}{% endfor %}assistant:")
    tokenizer.save_pretrained(path)


def test_native_checkpoint_chat_produces_captures_and_consumes_exact_stored_state(tmp_path, monkeypatch):
    base = tmp_path / 'base'
    base.mkdir()
    _tiny_source(base)
    model, _, roles, config = load_native_model(base, dtype=torch.float32, rank=4)
    checkpoint = tmp_path / 'native.pt'
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, checkpoint)
    engine = load_native_engine(str(checkpoint), device='cpu', dtype=torch.float32)
    engine.start()
    server = serve(engine, '127.0.0.1', 0)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint = f'http://{server.server_address[0]}:{server.server_address[1]}'

    def post(body):
        request = urllib.request.Request(endpoint + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    try:
        produced = iter([(roles.produce_open, -.1), (5, -.2), (roles.produce_close, -.3), (2, -.4)])
        monkeypatch.setattr(engine, '_native_token', lambda _seq: next(produced))
        reply = post({'messages': [{'role': 'user', 'content': 'write'}],
                      'max_tokens': 8, 'temperature': 1.0, 'seed': 13, 'x_natlang_rl_capture': True})
        record = engine.store.get_native_state(reply['native_state']['blocks'][0]['id'])
        capture = reply['x_natlang_rl_capture']
        assert capture['schema'] == 'natlang.native-rl-sampling/1'
        assert capture['policy']['checkpoint_sha256'] == engine.checkpoint_sha256
        assert capture['sampling_law'] == {'temperature': 1.0, 'seed': 13, 'guidance': False,
                                           'probabilities': 'post-processing-normalized'}
        assert capture['finish_reason'] == 'stop' and capture['sample_range_convention'] == 'half-open'
        assert [sample['token_id'] for sample in capture['samples']] == [roles.produce_open, 5, roles.produce_close, 2]
        assert capture['samples'][-1]['terminal'] is True
        assert capture['writes'] == [{'state_id': record.id, 'begin_sample': 0, 'end_sample': 3,
            'payload_begin': 1, 'payload_end': 2,
            'native_identity': {'checkpoint_sha256': record.block.checkpoint_sha256,
                'tokenizer_fingerprint': record.block.tokenizer_fingerprint,
                'embedding_weight_sha256': record.block.embedding_weight_sha256,
                'architecture': record.block.architecture}}]
        assert capture['samples'][1]['input_role'] == 'produce'
        with urllib.request.urlopen(endpoint + f"/v1/neuralese/native-states/{record.id}", timeout=30) as response:
            restored, restored_id = decode_native_state(response.read())
        assert restored_id == record.id and restored.token_ids.tolist() == [5]

        second = iter([(2, -.5)])
        monkeypatch.setattr(engine, '_native_token', lambda _seq: next(second))
        read = post({'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'use '}, {'type': 'native_state', 'id': record.id}]}],
            'max_tokens': 2, 'temperature': 1.0, 'seed': 14, 'x_natlang_rl_capture': True})
        assert read['native_state']['imports'] == [{'id': record.id, 'mode': 'exact-native-state'}]
        prompt = read['x_natlang_rl_capture']['prompt']
        state_at = next(i for i, segment in enumerate(prompt) if segment['kind'] == 'native_state')
        assert prompt[state_at - 1] == {'kind': 'tokens', 'token_ids': [roles.consume_open], 'producing': [False]}
        assert prompt[state_at] == {'kind': 'native_state', 'id': record.id,
                                    'native_identity': capture['writes'][0]['native_identity']}
        assert prompt[state_at + 1] == {'kind': 'tokens', 'token_ids': [roles.consume_close], 'producing': [False]}

        foreign_block = replace(record.block, checkpoint_sha256='4' * 64)
        foreign = engine.store.put_native_state(foreign_block)
        foreign_tokens = iter([(2, -.6)])
        monkeypatch.setattr(engine, '_native_token', lambda _seq: next(foreign_tokens))
        read_foreign = post({'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'use '}, {'type': 'native_state', 'id': foreign.id}]}],
            'max_tokens': 2, 'temperature': 1.0, 'seed': 15})
        assert read_foreign['native_state']['imports'] == [
            {'id': foreign.id, 'mode': 'best-effort-text-retokenization'}]
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        engine.stop()
