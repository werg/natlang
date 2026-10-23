"""Actual native forwards through TypeScript child/result/archive/parent import.

Token selection and typed tool readout are deterministic fixture controls;
this checks real runtime state transport, not learned task performance.
"""
import json
import os
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
    device=os.environ.get('NATLANG_NATIVE_RECORDED_TEST_DEVICE','cpu') if mode=='recorded' else 'cpu'
    if device!='cpu':
        port.hf.to(device=device,dtype=torch.bfloat16 if family=='mellum' else torch.float32)
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
            "{{ message['reasoning_content'] }} {% endif %}{{ message['content'] }}"
            "{% for tool in message.get('tool_calls', []) %}"
            "{{ tool['function']['name'] }}({{ tool['function']['arguments'] | tojson }})"
            "{% endfor %}<|im_end|>{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant {% endif %}")
    config = {'architecture': port.config.model_type, 'tokenizer_fingerprint': tokenizer_fingerprint(tokenizer)}
    weights = tmp_path / 'weights.pt'
    torch.save(model_state(model), weights)
    identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex(weights))
    engine = Engine(port, tokenizer, TensorStore(), f"native-state/{config['architecture']}",
        shared_model=model, role_tokens=RoleTokens(*model.marker_ids), model_config=config,
        checkpoint_sha256=identity['checkpoint_sha256'], embedding_weight_sha256=identity['embedding_weight_sha256'],device=device)
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
    def record_consume(sequence, block_id, **options):
        original_consume(sequence, block_id, **options)
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
        if mode == 'recorded':
            from itertools import cycle
            target = next(name for name in model.named_parameters() if name[0].endswith('q_proj.weight'))[0]
            samples = iter(cycle([(60, None), (5, None), (61, None), (2, None), (2, None)]))
            monkeypatch.setattr(engine, '_native_token', lambda sequence: next(samples))
            binding = tmp_path / 'native-recorded-learning.json'
            binding.write_text(json.dumps({'identity': identity, 'target': target, 'state_id': receipt['state_id']}))
            trained = subprocess.run([node, str(repo / 'ts-host/test/fixtures/native-adapter-recorded-learning.mjs'),
                f'http://127.0.0.1:{server.server_address[1]}', str(tmp_path / 'archive'),
                str(binding)], cwd=repo, text=True, capture_output=True, timeout=120)
            assert trained.returncode == 0, trained.stdout + trained.stderr
            progress = json.loads(trained.stdout.splitlines()[-1])
            assert progress['producer_consumer'] is True
            assert progress['updated_loss'] < progress['initial_loss']
            assert progress['native_state_gradient'] is True
            assert progress['joint_gradient'] is True
            assert progress['multi_target_scope'] is True
            assert progress['higher_order'] is True
            assert progress['scoped_state_writer'] is True
            assert progress['mixed_import_positions'] is True
            assert progress['repeated_scope_imports'] is True
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
                from natlang_neuralese.serve.native_recorded_graph import compile_recorded_graph
                from natlang_neuralese.data.native_graph import parse_graph
                from natlang_neuralese.rl.native_replay import token_action_logps
                events=json.loads((tmp_path / 'archive/native-recorded-graph.json').read_text())
                compiled=compile_recorded_graph(recorded,events,identity=identity,
                    roles=engine.role_tokens,fetch_block=fetch)
                assert len(compiled.replay.plan.invocations)==6
                assert len(compiled.decision_actions)==2
                assert not compiled.replay.external_blocks, 'both decisions consume the actual captured producer'
                only_decisions=[turn for turn in recorded if turn['native']['kind']=='decision']
                with pytest.raises(ValueError,match='whole-episode graph model-turn inventory'):
                    compile_recorded_graph(only_decisions,events,identity=identity,
                        roles=engine.role_tokens,fetch_block=fetch)
                decision_names={turn['native']['graph']['graph_turn'] for turn in only_decisions}
                decision_events=[event for event in events if event['node'] in decision_names
                    or event.get('turn') in decision_names]
                external=compile_recorded_graph(only_decisions,decision_events,identity=identity,
                    roles=engine.role_tokens,fetch_block=fetch)
                assert len(external.replay.external_blocks)==1
                assert len(external.replay.plan.invocations)==4
                assert len(next(iter(external.replay.external_invocations.values())))==4
                assert all(item['invocation'] in {node.name for node in external.replay.plan.invocations}
                    for items in external.replay.external_invocations.values() for item in items)
                assert all(binding['external_read_bindings'] for binding in external.origin_bindings.values())
                external_row=external.row(tokenizer_fingerprint=config['tokenizer_fingerprint'],source_id='external-recorded-graph')
                assert next(iter(external_row['external_states'].values()))['id']==receipt['state_id']
                row=compiled.row(tokenizer_fingerprint=config['tokenizer_fingerprint'],source_id='actual-recorded-graph')
                parsed=parse_graph(row,tokenizer_fingerprint=config['tokenizer_fingerprint'])
                assert parsed.plan==compiled.replay.plan
                with torch.no_grad():
                    result=model(parsed.plan,external_blocks=compiled.replay.external_blocks,retain_states=False)
                    for turn in recorded:
                        if turn['native']['kind']!='decision':continue
                        from natlang_neuralese.serve.native_learning import decide_native
                        cached=decide_native(engine,{'messages':turn['messages'],
                            'options':turn['decision']['options']})
                        actions=compiled.decision_actions[turn['native']['id']]
                        flat=[float(token_action_logps(model,result['h_final'][0],option).sum()) for option in actions]
                        torch.testing.assert_close(torch.tensor(flat),torch.tensor(cached['log_probs']),rtol=2e-5,atol=3e-5)
                import copy
                forged=copy.deepcopy(recorded)
                forged[-1]['native']['graph']['read_bindings']={}
                with pytest.raises(ValueError,match='occurrence maps'):
                    compile_recorded_graph(forged,events,identity=identity,roles=engine.role_tokens,fetch_block=fetch)
                forged=copy.deepcopy(recorded)
                forged[-1]['native']['decision']['prompt']=[segment for segment in
                    forged[-1]['native']['decision']['prompt'] if segment['kind']!='native_state']
                with pytest.raises(ValueError,match='occurrence-matched'):
                    compile_recorded_graph(forged,events,identity=identity,roles=engine.role_tokens,fetch_block=fetch)
                forged=copy.deepcopy(recorded)
                forged[-1]['native']['decision']['sampling_law']={'temperature':1.}
                with pytest.raises(ValueError,match='cannot declare sampled'):
                    compile_recorded_graph(forged,events,identity=identity,roles=engine.role_tokens,fetch_block=fetch)
                with pytest.raises(ValueError,match='repeats an occurrence'):
                    compile_recorded_graph(recorded,events+[events[0]],identity=identity,
                        roles=engine.role_tokens,fetch_block=fetch)
                from natlang_neuralese.serve.native_recorded_objective import lower_recorded_objective
                from natlang_neuralese.serve.native_grad import native_graph_gradient
                generations=[turn for turn in recorded if turn['native']['kind']=='generation']
                decisions=[turn for turn in recorded if turn['native']['kind']=='decision']
                student,teacher=generations[-1],generations[0]
                def side(turn):
                    return {key:turn[key] for key in ('messages','tools','adapters','native_state_modes') if key in turn}
                terms=[{'kind':'crossEntropy',**side(student),'target':student['reply'],
                        'native_turn_id':student['native']['id']},
                    {'kind':'selfDistill',**side(student),'target':student['reply'],
                        'native_turn_id':student['native']['id'],'teacher_native_turn_id':teacher['native']['id'],
                        'teacher_messages':teacher['messages'],
                            **({'teacher_adapters':teacher['adapters']} if 'adapters' in teacher else {}),
                            **({'teacher_native_state_modes':teacher['native_state_modes']} if 'native_state_modes' in teacher else {})},
                    {'kind':'logLikelihood',**side(teacher),'target':teacher['reply'],
                        'native_turn_id':teacher['native']['id'],'likelihood_law':'model-conditional','weight':.2},
                    {'kind':'decision',**side(decisions[0]),'native_turn_id':decisions[0]['native']['id'],
                        'options':decisions[0]['decision']['options'],'rule':'logLoss',
                        'target':{'probabilities':[.2,.8]}},
                    {'kind':'decision',**side(decisions[0]),'native_turn_id':decisions[0]['native']['id'],
                        'options':decisions[0]['decision']['options'],'rule':'brier',
                        'teacher':{**side(decisions[1]),'native_turn_id':decisions[1]['native']['id']}},
                    {'kind':'crossEntropy',**side(student),'native_turn_id':student['native']['id'],
                        'target':{'role':'assistant','content':[{'type':'native_state','id':receipt['state_id']}]}}]
                request=lower_recorded_objective(engine,recorded,events,terms)
                assert request['graph']['target_native_values']==[{'id':receipt['state_id'],
                    'tokenizer_fingerprint':identity['tokenizer_fingerprint'],'mode':'retained-source-token-ids'}]
                target_node=request['graph']['nodes'][-1]
                assert any(target_node['token_ids'][start:stop]==[5] for start,stop in target_node['producing'])
                request['target']={'kind':'model-parameters','names':['adapter.up.weight']}
                forwards=[]
                original_forward=model.forward
                def counted(*args,**kwargs):
                    forwards.append(True)
                    return original_forward(*args,**kwargs)
                monkeypatch.setattr(model,'forward',counted)
                gradient=native_graph_gradient(engine,request,None)
                assert len(forwards)==1, 'all actual producers, expected outputs and teachers use one graph forward'
                assert torch.isfinite(torch.tensor(gradient['loss']))
                artifact=engine.store.get_native_tensor_artifact(gradient['gradient']['id']).artifact
                assert artifact.tensors['adapter.up.weight'].norm()>0
                # The same authenticated external decision capture also runs
                # through the real native higher-order program. The two SGD
                # updates are differentiated through their recorded graph loss.
                from natlang_neuralese.model.native_derivative_contract import PROGRAM
                from natlang_neuralese.model.native_state_derivation import state_values
                from natlang_neuralese.serve.native_derivative import execute_derivative_program
                decision=decisions[0]
                decision_term={'kind':'decision',**side(decision),'native_turn_id':decision['native']['id'],
                    'options':decision['decision']['options'],'rule':'logLoss','target':{'probabilities':[1.,0.]}}
                state_name=next(iter(state_values(engine.store.get_native_state(receipt['state_id']).block)))
                envelope={'schema':'natlang.native-recorded-objective-request/1','turns':only_decisions,
                    'events':decision_events,'terms':[decision_term],
                    'targets':[{'root':'state','kind':'native-state','id':receipt['state_id'],'names':[state_name]}]}
                optimizer={'name':'sgd','learning_rate':.001,'momentum':0.,'eps':1e-8}
                derivative={'schema':PROGRAM,
                    'arithmetic':{'schema':'natlang.native-reference-arithmetic/1','query_chunk_tokens':2},
                    'inputs':{'state':{'kind':'native-state','id':receipt['state_id'],'names':[state_name]}},
                    'nodes':[
                        {'kind':'gradient','name':'g0','request':envelope,'bindings':{},
                         'targets':[{'name':'state-grad-0','value':'state'}]},
                        {'kind':'optimizer','name':'state-1','value':'state','gradient':'state-grad-0',
                         'optimizer':optimizer,'optimizer_state':None},
                        {'kind':'gradient','name':'g1','request':envelope,'bindings':{'state':'state-1'},
                         'targets':[{'name':'state-grad-1','value':'state-1'}]},
                        {'kind':'optimizer','name':'state-2','value':'state-1','gradient':'state-grad-1',
                         'optimizer':optimizer,'optimizer_state':'state-1'}],
                    'outer':{'request':envelope,'bindings':{'state':'state-2'}},'wrt':['state']}
                hyper=execute_derivative_program(engine,derivative)
                hyper_gradient=engine.store.get_native_tensor_artifact(hyper['gradients']['state']['id']).artifact
                assert hyper_gradient.source_artifact_id==receipt['state_id']
                assert torch.isfinite(hyper_gradient.tensors[state_name]).all()
                assert engine.store.get_native_tensor_artifact(hyper['optimizer_states']['state-2']).artifact.provenance['step']==2
                bad=copy.deepcopy(terms)
                bad[2]['likelihood_law']='sampled-policy'
                with pytest.raises(ValueError,match='positive-temperature'):
                    lower_recorded_objective(engine,recorded,events,bad)
                bad=copy.deepcopy(terms)
                bad[0]['messages']=[{'role':'user','content':'forged prompt'}]
                with pytest.raises(ValueError,match='changes its recorded'):
                    lower_recorded_objective(engine,recorded,events,bad)
                bad_turns=copy.deepcopy(recorded)
                first=next(segment for segment in bad_turns[0]['native']['sampling']['prompt'] if segment['kind']=='tokens')
                first['token_ids'][0]=(first['token_ids'][0]+1)%len(vocabulary)
                with pytest.raises(ValueError,match='exact rendered'):
                    lower_recorded_objective(engine,bad_turns,events,terms)
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
