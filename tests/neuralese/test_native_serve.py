"""The ordinary chat server loads and transports the trained native state end to end."""
import torch
import pytest
import json
import hashlib
import threading
import urllib.request
from dataclasses import replace
from transformers import Lfm2Config, Lfm2ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

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
    tokenizer.backend_tokenizer.pre_tokenizer = WhitespaceSplit()
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
        return post_path('/v1/chat/completions', body)

    def post_path(path, body):
        request = urllib.request.Request(endpoint + path,
            data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except urllib.request.HTTPError as error:
            raise AssertionError(f'chat HTTP {error.code}: {error.read().decode(errors="replace")}') from error

    def get(path):
        with urllib.request.urlopen(endpoint + path, timeout=30) as response:
            return json.loads(response.read())

    def request(path, method, body):
        request = urllib.request.Request(endpoint + path, data=body, method=method,
            headers={'content-type': 'application/octet-stream'})
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()

    try:
        with urllib.request.urlopen(endpoint + '/v1/neuralese/info', timeout=30) as response:
            info = json.loads(response.read())
        binding = info['rl_policy_binding']
        assert binding['communication'] == 'native-state'
        assert binding['native_identity'] == engine.native_identity
        from natlang_neuralese.rl.production import validate_actor_receipts
        validate_actor_receipts({endpoint: binding}, policy_id=engine.checkpoint_sha256, neuralese_server=True)
        native_token=engine._native_token
        produced = iter([(roles.produce_open, -.1), (5, -.2), (roles.produce_close, -.3), (2, -.4)])
        monkeypatch.setattr(engine, '_native_token', lambda _seq: next(produced))
        reply = post({'messages': [{'role': 'user', 'content': 'write'}],
                      'max_tokens': 8, 'temperature': 1.0, 'seed': 13, 'x_natlang_rl_capture': True})
        with urllib.request.urlopen(endpoint + '/metrics', timeout=5) as response:
            metrics = dict(line.split() for line in response.read().decode().splitlines())
        # Open/payload/close each completed a forward; EOS did not. The private
        # payload was observable even though it was withheld from text deltas.
        assert metrics['natlang_native_decode_tokens_total'] == '3'
        assert metrics['natlang_native_open_produce_spans'] == '0'
        record = engine.store.get_native_state(reply['native_state']['blocks'][0]['id'])
        capture = reply['x_natlang_rl_capture']
        assert capture['schema'] == 'natlang.native-rl-sampling/2'
        assert capture['policy']['checkpoint_sha256'] == engine.checkpoint_sha256
        assert capture['sampling_law'] == {'temperature': 1.0, 'seed': 13, 'guidance': False,
                                           'probabilities': 'post-processing-normalized'}
        assert capture['finish_reason'] == 'stop' and capture['sample_range_convention'] == 'half-open'
        assert [sample['token_id'] for sample in capture['samples']] == [roles.produce_open, 5, roles.produce_close, 2]
        assert capture['samples'][-1]['terminal'] is True
        assert [sample['query_own_position'] for sample in capture['samples']] == sorted(
            sample['query_own_position'] for sample in capture['samples'])
        assert capture['generated_inputs'] == [
            {'token_id': roles.produce_open, 'input_role': 'ordinary', 'sample_index': 0},
            {'token_id': 5, 'input_role': 'produce', 'sample_index': 1},
            {'token_id': roles.produce_close, 'input_role': 'ordinary', 'sample_index': 2}]
        assert capture['writes'] == [{'state_id': record.id, 'begin_sample': 0, 'end_sample': 3,
            'own_begin': capture['samples'][0]['query_own_position'] + 1,
            'own_end': capture['samples'][2]['query_own_position'] + 2,
            'payload_own_begin': capture['samples'][1]['query_own_position'] + 1,
            'payload_own_end': capture['samples'][2]['query_own_position'] + 1,
            'native_identity': {'checkpoint_sha256': record.block.checkpoint_sha256,
                'tokenizer_fingerprint': record.block.tokenizer_fingerprint,
                'embedding_weight_sha256': record.block.embedding_weight_sha256,
                'architecture': record.block.architecture}}]
        assert capture['samples'][1]['input_role'] == 'produce'

        teacher_prompt=[{'role':'user','content':'execute the recorded correction; t8 is already in this prompt'}]
        source={'schema':'natlang.teacher-turn-source/1','source_path':'qwen-turns/row-001.jsonl',
            'source_view':'privileged-hint','source_file_sha256':'f'*64,
            'source_verification_path':'qwen-turns/source-verification.json','source_verification_sha256':'1'*64,
            'transfer_receipt_path':'qwen-turns/transfer-receipt.json','transfer_receipt_sha256':'2'*64,
            'invocation_id':'task/child-1','ordinal':0,'turn_sha256':'a'*64,'task_sha256':'b'*64,
            'prompt_sha256':hashlib.sha256(json.dumps({'messages':teacher_prompt,'tools':[]},
                ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'actor':{'model':'nvidia/Qwen3.6-35B-A3B-NVFP4','revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5',
                'actor_source_path':'qwen-turns/actor.json','actor_source_sha256':'c'*64,
                'execution_identity_path':'qwen-turns/execution.json','execution_identity_sha256':'e'*64}}
        span_marker='NATLANGQWENSPAN0123456789END'
        target={'schema':'natlang.native-teacher-forced-turn/1','source_receipt':source,
            'native_text_spans':[{'path':['content'],'marker':span_marker,'text':'t5 t6 t7'}],
            'assistant':{'role':'assistant','content':'t4 '+span_marker}}
        previous_template=engine.tokenizer.apply_chat_template
        def forced_template(messages,tools=None,tokenize=False,add_generation_prompt=False,**kwargs):
            rendered=''.join(f"<{message['role']}>{''.join(part.get('text','') for part in message.get('content',[])) if isinstance(message.get('content'),list) else message.get('content','')}" +
                ''.join(f"<toolcall>{call.get('function',{}).get('name','')}:{call.get('function',{}).get('arguments','')}</toolcall>" for call in message.get('tool_calls',[])) for message in messages)
            if add_generation_prompt:rendered+='<assistant>'
            if tokenize:return engine.tokenizer(rendered,add_special_tokens=False)['input_ids']
            return rendered
        monkeypatch.setattr(engine.tokenizer,'apply_chat_template',forced_template)
        forced_plan=engine._forced_message_plan(target,None,opening=teacher_prompt)
        assert forced_plan.count(roles.produce_open)==1 and forced_plan.count(roles.produce_close)==1, forced_plan
        assert not any(token in engine.stop_ids for token in forced_plan[
            forced_plan.index(roles.produce_open)+1:forced_plan.index(roles.produce_close)]), (forced_plan,engine.stop_ids)
        arg_marker='NATLANGQWENSPANfedcba9876543210END'
        arg_target={'schema':'natlang.native-teacher-forced-turn/1','source_receipt':source,
            'native_text_spans':[{'path':['tool_calls',0,'arguments','code'],'marker':arg_marker,'text':'t5 t6 t7'}],
            'assistant':{'role':'assistant','content':'','tool_calls':[{'type':'function','function':{
                'name':'eval','arguments':json.dumps({'code':'const body="'+arg_marker+'";'})}}]}}
        arg_plan=engine._forced_message_plan(arg_target,None,opening=teacher_prompt)
        assert arg_plan.count(roles.produce_open)==1 and arg_plan.count(roles.produce_close)==1, arg_plan
        monkeypatch.setattr(engine,'_native_token',native_token)
        repaired=post({'messages':teacher_prompt,'max_tokens':64,'temperature':0.0,
            'x_natlang_native_recording':True,'x_natlang_forced_message':target})
        replay_capture=repaired['x_natlang_rl_capture']
        assert replay_capture['schema']=='natlang.native-teacher-forced-capture/1'
        assert replay_capture['teacher_source_receipt']==source and replay_capture['samples']
        assert replay_capture['replay_prompt_sha256']==hashlib.sha256(json.dumps(
            {'messages':teacher_prompt,'tools':[]},ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert all(sample.get('forced') is True and sample.get('logp') is None for sample in replay_capture['samples'])
        assert any(item['input_role']=='produce' for item in replay_capture['generated_inputs']), (replay_capture['generated_inputs'],replay_capture['samples'])
        forced_block=repaired['native_state']['blocks'][0]
        consumer_prompt=[{'role':'user','content':[{'type':'native_state','id':forced_block['id']}]}]
        consumer_target={'schema':'natlang.native-teacher-forced-turn/1','source_receipt':source,
            'native_text_spans':[],'assistant':{'role':'assistant','content':'t4'}}
        consumed=post({'messages':consumer_prompt,'max_tokens':8,'temperature':0.0,
            'x_natlang_native_recording':True,'x_natlang_forced_message':consumer_target})
        assert consumed['native_state']['imports'][0]['id']==forced_block['id']
        assert consumed['x_natlang_rl_capture']['teacher_source_receipt']==source
        assert len(replay_capture['writes'])==1
        repaired_block=engine.store.get_native_state(replay_capture['writes'][0]['state_id'])
        assert repaired_block.block.text=='t5 t6 t7'
        assert any(part.get('type')=='native_state' and part.get('id')==repaired_block.id
            for part in repaired['choices'][0]['message']['content'])
        monkeypatch.setattr(engine.tokenizer,'apply_chat_template',previous_template)

        with urllib.request.urlopen(endpoint + f"/v1/neuralese/native-states/{record.id}", timeout=30) as response:
            restored, restored_id = decode_native_state(response.read())
        assert restored_id == record.id and restored.token_ids.tolist() == [5]

        second = iter([(2, -.5)])
        monkeypatch.setattr(engine, '_native_token', lambda _seq: next(second))
        read = post({'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': 'use '}, {'type': 'native_state', 'id': record.id}]}],
            'max_tokens': 2, 'temperature': 1.0, 'seed': 14, 'x_natlang_rl_capture': True})
        assert read['native_state']['imports'] == [{'id': record.id, 'mode': 'exact-native-state',
            'observed_native_state_id': record.id}]
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

        info = get('/v1/neuralese/info')
        assert {'template', 'guidance'} <= set(info['capabilities'])
        assert 'guidance' not in info['capability_gaps']
        assert info['native_state']['consumption_modes'] == [
            'exact-native-state', 'same-base-adapter-state', 'best-effort-text-retokenization']
        assert 'adapters' not in info, 'legacy projected-adapter inventory is retired'
        assert 'native.adapters' in info['capabilities'], 'native modelLoRA remains available'
        assert {'native.text-write', 'native.init-body'} <= set(info['capabilities'])
        assert {'decide', 'score'} <= set(info['capabilities'])
        assert 'store.native-tensor-artifact' in info['capabilities']
        assert info['grad'] is True and info['grad_order'] == 1
        assert 'optim' in info['capabilities']
        placeholder = post_path('/v1/neuralese/native-states/write-text',
                                {'text': 'placeholder body', 'type': 'Neuralese<string>'})
        assert placeholder['id'].startswith('nzs1_')
        assert placeholder['native_state']['type'] == 'Neuralese<string>'
        source_segments = placeholder['source_capture']['segments']
        assert [segment['token_ids'] for segment in source_segments] == [
            [roles.produce_open], source_segments[1]['token_ids'], [roles.produce_close]]
        assert source_segments[0]['producing'] == [False] and source_segments[-1]['producing'] == [False]
        assert source_segments[1]['producing'] == [True] * len(source_segments[1]['token_ids'])
        assert placeholder['source_capture']['payload_start'] == 1
        assert placeholder['source_capture']['payload_stop'] == len(source_segments[1]['token_ids']) + 1
        assert placeholder['source_text_sha256']
        literal = post_path('/v1/neuralese/native-states/write-text',
            {'text': 'keep <|neuralese_produce|> literal'})
        literal_record = engine.store.get_native_state(literal['id'])
        assert roles.produce_open not in literal_record.block.token_ids.tolist()
        assert literal_record.block.text == 'keep <|neuralese_produce|> literal'

        from natlang_neuralese.serve.grad import GradSession
        monkeypatch.setattr(GradSession, '_target_items', lambda _self, _messages, _tools, target:
            ([('tok', 4)], [('tok', int(target['content'][1:]))]))
        decision_body = {'messages': [{'role': 'user', 'content': 'choose'}],
                         'options': ['t5', 't6']}
        single = post_path('/v1/neuralese/decide', decision_body)
        batch = post_path('/v1/natlang/score', {'items': [
            {'messages': decision_body['messages'], 'continuations': decision_body['options']},
            {'messages': decision_body['messages'], 'options': []}]})
        assert single['tokens'] == [1, 1]
        assert len(single['log_probs']) == 2 and all(torch.isfinite(torch.tensor(v)) for v in single['log_probs'])
        assert batch['results'][0]['tokens'] == single['tokens']
        assert max(abs(a - b) for a, b in zip(batch['results'][0]['log_probs'], single['log_probs'])) < 1e-6
        assert 'error' in batch['results'][1]
        from natlang_neuralese.model.native_artifact import (NativeTensorArtifact,
            decode_native_tensor_artifact, encode_native_tensor_artifact)
        tensor_artifact = NativeTensorArtifact('parameter-gradient', engine.native_identity,
            {'adapter.up.weight': torch.ones((2, 3))}, source_artifact_id=record.id,
            provenance={'objective': 'service-route-test'})
        encoded_artifact = encode_native_tensor_artifact(tensor_artifact)
        _, artifact_id = decode_native_tensor_artifact(encoded_artifact)
        stored_meta = json.loads(request(f'/v1/neuralese/native-tensor-artifacts/{artifact_id}',
                                        'PUT', encoded_artifact))
        assert stored_meta['id'] == artifact_id and stored_meta['kind'] == 'parameter-gradient'
        fetched_artifact, fetched_id = decode_native_tensor_artifact(
            request(f'/v1/neuralese/native-tensor-artifacts/{artifact_id}', 'GET', b''))
        assert fetched_id == artifact_id and fetched_artifact.native_identity == engine.native_identity
        train_graph = {'schema': 'natlang.neuralese-native-graph/1', 'id': 'http-gradient-train-row',
            'split': 'train', 'source_groups': ['http-gradient-fixture'],
            'tokenizer_fingerprint': config['tokenizer_fingerprint'],
            'nodes': [{'name': 'turn', 'token_ids': [5, 6, 7], 'producing': [[1, 3]],
                       'target_weights': [0, 1, 1]}]}
        gradient_reply = post_path('/v1/neuralese/grad', {'schema': 'natlang.native-gradient-request/1',
            'objective': 'graph-ce', 'graph': train_graph,
            'target': {'kind': 'model-parameters', 'names': ['adapter.up.weight']}})
        assert gradient_reply['objective'] == 'native-graph-token-ce/1'
        gradient_artifact, gradient_id = decode_native_tensor_artifact(request(
            f"/v1/neuralese/native-tensor-artifacts/{gradient_reply['gradient']['id']}", 'GET', b''))
        assert gradient_id == gradient_reply['gradient']['id']
        assert gradient_artifact.kind == 'parameter-gradient'
        assert gradient_artifact.tensors['adapter.up.weight'].norm() > 0
        live_before = engine.shared_model.adapter.up.weight.detach().clone()
        from natlang_neuralese.serve import native_optim
        original_writer = native_optim.write_native_composite_checkpoint
        def observe_immutable_publication(parent, output, updated, config, **kwargs):
            assert updated is not engine.shared_model
            torch.testing.assert_close(engine.shared_model.adapter.up.weight, live_before, rtol=0, atol=0)
            return original_writer(parent, output, updated, config, **kwargs)
        monkeypatch.setattr(native_optim, 'write_native_composite_checkpoint', observe_immutable_publication)
        # A later malformed tensor must not leave an earlier parameter changed.
        down_before = engine.shared_model.adapter.down.weight.detach().clone()
        malformed = NativeTensorArtifact('parameter-gradient', engine.native_identity,
            {'adapter.down.weight': torch.ones_like(down_before), 'adapter.up.weight': torch.ones(1)},
            source_artifact_id=record.id, provenance={'objective': 'rollback-regression'})
        malformed_record = engine.store.put_native_tensor_artifact(malformed)
        from natlang_neuralese.serve.chat import RequestError
        with pytest.raises(RequestError, match='gradient shape'):
            native_optim.native_optimizer_step(engine, {'schema': 'natlang.native-optimizer-request/1',
                'native_identity': engine.native_identity, 'gradient_id': malformed_record.id,
                'optimizer': {'name': 'sgd', 'learning_rate': .1}}, None)
        torch.testing.assert_close(engine.shared_model.adapter.down.weight, down_before, rtol=0, atol=0)
        optimized = post_path('/v1/neuralese/optim', {'schema': 'natlang.native-optimizer-request/1',
            'native_identity': engine.native_identity, 'gradient_id': gradient_id,
            'optimizer': {'name': 'adam', 'learning_rate': 0.01}})
        assert optimized['checkpoint']['schema'] == 'natlang.neuralese-native-composite-checkpoint/1'
        assert optimized['native_identity']['checkpoint_sha256'] == optimized['checkpoint']['sha256']
        assert optimized['optimizer_state_id'].startswith('nnt1_')
        torch.testing.assert_close(engine.shared_model.adapter.up.weight, live_before, rtol=0, atol=0)
        updated_engine = load_native_engine(optimized['checkpoint']['path'], device='cpu', dtype=torch.float32)
        assert updated_engine.native_identity == optimized['native_identity']
        assert not torch.equal(updated_engine.shared_model.adapter.up.weight, live_before)
        assert updated_engine.store.get_native_tensor_artifact(optimized['optimizer_state_id']) is None
        # The next request restores the separately archived optimizer artifact
        # into the new server, then validates it against the current manifest.
        state_artifact, _ = decode_native_tensor_artifact(request(
            f"/v1/neuralese/native-tensor-artifacts/{optimized['optimizer_state_id']}", 'GET', b''))
        updated_engine.store.put_native_tensor_artifact(state_artifact)
        updated_engine.start()
        updated_server = serve(updated_engine, '127.0.0.1', 0)
        updated_thread = threading.Thread(target=updated_server.serve_forever, daemon=True)
        updated_thread.start()
        updated_endpoint = f'http://{updated_server.server_address[0]}:{updated_server.server_address[1]}'
        try:
            new_gradient_req = urllib.request.Request(updated_endpoint + '/v1/neuralese/grad',
                data=json.dumps({'schema': 'natlang.native-gradient-request/1', 'objective': 'graph-ce',
                    'native_identity': updated_engine.native_identity, 'graph': train_graph,
                    'target': {'kind': 'model-parameters', 'names': ['adapter.up.weight']}}).encode(),
                headers={'content-type': 'application/json'})
            with urllib.request.urlopen(new_gradient_req, timeout=30) as response:
                new_gradient = json.loads(response.read())
            second_request = urllib.request.Request(updated_endpoint + '/v1/neuralese/optim',
                data=json.dumps({'schema': 'natlang.native-optimizer-request/1',
                    'native_identity': updated_engine.native_identity, 'gradient_id': new_gradient['gradient']['id'],
                    'optimizer_state_id': optimized['optimizer_state_id'],
                    'optimizer': {'name': 'adam', 'learning_rate': 0.01}}).encode(),
                headers={'content-type': 'application/json'})
            with urllib.request.urlopen(second_request, timeout=30) as response:
                second_update = json.loads(response.read())
            assert second_update['step'] == 2
            assert second_update['native_identity']['checkpoint_sha256'] != optimized['native_identity']['checkpoint_sha256']
        finally:
            updated_server.shutdown()
            updated_thread.join(timeout=5)
        initialized = post_path('/v1/neuralese/native-states/init-body', {
            'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'before '},
                {'type': 'native_state', 'id': placeholder['id']}, {'type': 'text', 'text': ' after'}]}],
            'placeholder': placeholder['id'], 'text': 'soft body', 'type': 'Neuralese<string>'})
        init_record = engine.store.get_native_state(initialized['id'])
        assert init_record.block.text == 'soft body'
        assert initialized['init']['contextual'] and initialized['init']['aligned']
        assert initialized['source_capture']['payload_stop'] > initialized['source_capture']['payload_start']
        roles_in_capture = [role for segment in initialized['source_capture']['segments']
                            if segment['kind'] == 'tokens' for role in segment['producing']]
        assert any(roles_in_capture) and not all(roles_in_capture)
        capture_token_ids = [token for segment in initialized['source_capture']['segments']
                             if segment['kind'] == 'tokens' for token in segment['token_ids']]
        assert roles.produce_open in capture_token_ids and roles.produce_close in capture_token_ids
        assert initialized['gate'] is None and initialized['source_text_sha256']
        original_apply = engine.tokenizer.apply_chat_template
        template_text = ("<|tool_call_start|>[return_result(status='success', value=\""
                         "natlangValue7f3a9c\")]<|tool_call_end|>")
        def apply_template(messages, *args, **kwargs):
            if messages and (messages[-1].get('tool_calls') or []):
                opening = original_apply([{'role': 'user', 'content': 'x'}],
                    tools=kwargs.get('tools'), tokenize=False, add_generation_prompt=True)
                return opening + template_text
            return original_apply(messages, *args, **kwargs)
        monkeypatch.setattr(engine.tokenizer, 'apply_chat_template', apply_template)
        template_prefix, template_suffix = template_text.split('natlangValue7f3a9c')
        monkeypatch.setattr(engine, '_tokens', lambda text: [60] if text == template_prefix else
                            ([61] if text == template_suffix else [5]))
        original_decode = engine.tokenizer.decode
        monkeypatch.setattr(engine.tokenizer, 'decode', lambda ids, **kwargs:
            (template_prefix if list(ids) == [60] else template_suffix if list(ids) == [61] else
             'payload' if list(ids) == [5] else original_decode(ids, **kwargs)))
        generated = iter([(5, None), (roles.produce_close, None), (2, None)])
        monkeypatch.setattr(engine, '_native_token', lambda _seq: next(generated))
        generated = iter([(5, None), (roles.produce_close, None), (2, None)])
        monkeypatch.setattr(engine, '_native_token', lambda _seq: next(generated))
        templated = post({'messages': [{'role': 'user', 'content': 'return the result'}], 'max_tokens': 16,
            'temperature': 1.0, 'x_natlang_rl_capture': True,
            'neuralese_template': {'call': 'return_result', 'arguments': {'status': 'success'}, 'value': 'write'}})
        call = templated['choices'][0]['message'].get('tool_calls', [None])[0]
        assert call is not None, templated['choices'][0]['message']
        template_args = json.loads(call['function']['arguments'])
        assert call['function']['name'] == 'return_result'
        assert template_args['value'] == [{'type': 'native_state',
            'id': templated['native_state']['blocks'][0]['id'], 'value_type': 'string'}]
        template_capture = templated['x_natlang_rl_capture']
        assert template_capture['schema'] == 'natlang.native-rl-sampling/2'
        assert any(item['sample_index'] is None for item in template_capture['generated_inputs'])
        assert template_capture['writes'][0]['own_begin'] < template_capture['writes'][0]['payload_own_end']
        guided_tokens = iter([(5, -.2), (2, -.3)])
        monkeypatch.setattr(engine, '_native_token', lambda _seq: next(guided_tokens))
        guided = post({'messages': [{'role': 'user', 'content': 'guided'}], 'guidance': True,
            'temperature': 1.0, 'max_tokens': 4, 'x_natlang_rl_capture': True})
        assert guided['x_natlang_guidance'] == {'rejections': []}
        assert guided['x_natlang_rl_capture']['sampling_law']['guidance'] is True
        assert guided['x_natlang_rl_capture']['samples'][0]['banned_token_ids'] == []
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        engine.stop()
