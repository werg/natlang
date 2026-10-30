"""Native admission/coordination keeps actual policies and both learner updates."""
import copy
import json
from pathlib import Path
import pytest
import torch

from natlang_neuralese.rl import production
from natlang_neuralese.common.hashing import sha256_file_hex


@pytest.mark.parametrize('family', ['lfm'])
@pytest.mark.parametrize('mode', ['rft', 'grpo'])
def test_actual_shared_optimizer_updates_native_policy_graph_for_both_ports(tmp_path, family, mode):
    from test_native_rl_replay import actor_rollout
    from natlang_neuralese.rl.native_replay import prepare_objective, NativePolicyObjective
    from natlang_neuralese.data.native_graph import parse_graph
    from natlang_neuralese.train.native_state_sft import make_native_optimizer, packed_graph_loss
    model, roles, identity, _, block, turns = actor_rollout(family)
    if family == 'mellum':
        model.port.hf.natlang_precision = 'bf16'  # tiny CPU dense fixture, not FP8 qualification
    raw = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'],
            'sampling': turn['sampling'], 'replay': {'raw_calls_retained': True}} for turn in turns]
    episodes = [{'task_id': 'fixture-task', 'sample': 0, 'split': 'train', 'group': 'g',
        'reward': {'quality': 1}, 'policy': {'checkpoint_sha256': identity['checkpoint_sha256']}, 'turns': raw}]
    if mode == 'grpo':
        negative = copy.deepcopy(episodes[0]); negative.update(sample=1, reward={'quality': 0})
        episodes.append(negative)
    from test_rl_episode_graph import attach_execution_graph
    attach_execution_graph(episodes)
    graphs, spec = prepare_objective(episodes, identity=identity, roles=roles,
                                    fetch_block=lambda key: block, mode=mode)
    path = tmp_path/'objective.json'; path.write_text(json.dumps(spec))
    examples = [parse_graph(row, tokenizer_fingerprint=identity['tokenizer_fingerprint']) for row in graphs]
    optimizer, named = make_native_optimizer(model, lr=1e-3, in_backward=False)
    before = {name: value.detach().clone() for name, value in named}
    for example in examples:  # every selected episode receives its actual update
        optimizer.zero_grad(set_to_none=True)
        loss, details = packed_graph_loss(model, [example], objective=NativePolicyObjective(path))
        loss.backward(); optimizer.step()
    assert details['sampled_tokens'] == 7 and torch.isfinite(loss)
    assert any(not torch.equal(before[name], value) for name, value in named if name.startswith('backbone.'))
    assert any(not torch.equal(before[name], value) for name, value in named if name.startswith('interface.'))


@pytest.mark.parametrize('shared_parent',[True,False])
def test_native_stage_real_shared_parent_update_and_exact_noop_resume(tmp_path,shared_parent, *, parent_setup=None, extra_args=(),
        preserve_first_operation_report=False):
    """Actual tiny LFM full trainer; fixture reward is NOT an environment result."""
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model, model_state, native_identity, SHARED_CHECKPOINT_SCHEMA
    from natlang_neuralese.model.shared_state import native_step
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.train.loop import capture_training_rng_state
    from natlang_neuralese.data.native_graph import SCHEMA
    from natlang_neuralese.rl.native_stage import main
    base = tmp_path/'base'; tiny_source(base)
    model, tokenizer, roles, config = load_native_model(base, dtype=torch.float32, rank=4)
    optimizer, _ = make_native_optimizer(model, lr=1e-3, weight_decay=0, in_backward=False)
    parent = tmp_path/'shared.pt'
    saved = {'schema': SHARED_CHECKPOINT_SCHEMA,
        'participants': {'lfm': {'model_config': config, 'model_state': model_state(model),
                                  'optimizer': optimizer.state_dict(), 'fp8': None}},
        'training': {'step': 1, 'rng': capture_training_rng_state('cpu'),
                     'config_history': [{'config': {'weight_decay': 0, 'latent_ns': 'quintic'}}]}}
    if not shared_parent:
        from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
        saved = {'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model),
            'training': {'schema':'natlang.neuralese-native-sft-training/1',
                'step': 1, 'global_step': 1, 'optimizer': optimizer.state_dict(),
                'identity': {'schema':'natlang.neuralese-native-sft-training/1',
                    'config':config, 'optimizer':'shared-muon-sr-in-backward-v1',
                    'schedule':'row-visit-deficit/2','seed':1,'rank':4,
                    'fp8':{'kind':'off','recipe':'delayed','amax_history_len':16},
                    'objective':None},
                'source_row_bindings':{'prior-foundation-row':'exact-prior-binding'},
                'source_row_order':['prior-foundation-row'],
                'rng': capture_training_rng_state('cpu'),
                'options': {'weight_decay':0,'latent_ns':'quintic','stage_id':'full','epochs':4},
                'exposure_counts': {'prior-foundation-row':3}}}
    held = tmp_path/'held.jsonl'
    held.write_text(json.dumps({'schema': SCHEMA, 'id': 'held-fixture', 'split': 'test',
        'source_groups': ['held'], 'tokenizer_fingerprint': config['tokenizer_fingerprint'],
        'nodes': [{'name': 'held', 'token_ids': [1, 6, 7], 'target_weights': [0, 0, 1]}]})+'\n')
    if not shared_parent:
        from natlang_neuralese.data.native_sources import NativeGraphSource
        saved['training']['held_row_bindings']=NativeGraphSource(held,config['tokenizer_fingerprint'],
            required_split='test',role_ids=roles.ids).bindings
    if parent_setup is not None:parent_setup(saved,held,roles,config)
    torch.save(saved, parent)
    identity = native_identity(model, config, checkpoint_sha256=sha256_file_hex(parent))
    with torch.no_grad():
        out = native_step(model, torch.tensor([[1, 4]]))
        first = float(model.logits(out['h_final'][:, -1]).float().log_softmax(-1)[0, 5])
        out = native_step(model, torch.tensor([[5]]), out['cache'])
        stop = float(model.logits(out['h_final'][:, -1]).float().log_softmax(-1)[0, 2])
    capture = {'schema': 'natlang.native-rl-sampling/1', 'policy': identity,
        'prompt': [{'kind': 'tokens', 'token_ids': [1, 4], 'producing': [False, False]}],
        'samples': [{'index': 0, 'token_id': 5, 'logp': first, 'input_role': 'ordinary', 'terminal': False},
                    {'index': 1, 'token_id': 2, 'logp': stop, 'input_role': 'ordinary', 'terminal': True}],
        'writes': [], 'sample_range_convention': 'half-open', 'finish_reason': 'stop',
        'sampling_law': {'temperature': 1., 'seed': 3, 'guidance': False, 'probabilities': 'post-processing-normalized'}}
    rollout = {'task_id': 'fixture-only', 'sample': 0, 'group': 0, 'split': 'train',
        'policy': {'checkpoint_sha256': identity['checkpoint_sha256']}, 'reward': {'quality': 1},
        'turns': [{'messages': [], 'target': {}, 'invocation_id':'ordinary-policy',
                   'sampling': capture, 'replay': {'raw_calls_retained': True}}]}
    source = tmp_path/'rollouts.jsonl'; source.write_text(json.dumps(rollout)+'\n')
    saved_tokenizer = tmp_path/'tokenizer'; tokenizer.save_pretrained(saved_tokenizer)
    argv = ['--rollouts', str(source), '--rl-tokenizer', str(saved_tokenizer), '--rl-mode', 'rft',
        '--continue-from', str(parent), *(['--native-participant', 'lfm'] if shared_parent else []), '--held-native-graphs', str(held),
        '--out', str(tmp_path/'learner'), '--device', 'cpu', '--epochs', '1', '--rank', '4',
        '--learning-rate', '1e-3', '--weight-decay', '0', '--eval-every', '1',
        '--communication-foundation', '--foundation-native-graphs', str(tmp_path/'parent-only-source'),
        '--foundation-communication-batches','4',
        '--refresh-foundation-readiness',
        '--resident-data-directory',str(tmp_path/'parent-only-resident'),
        '--resident-geometry-qualification',str(tmp_path/'parent-only-capacity.json'),
        '--memory-qualification',json.dumps({'path':str(tmp_path/'parent-only-sft-memory.json'),
            'sha256':'0'*64,'bytes':1}), *extra_args]
    assert main(argv) == 0
    slot = tmp_path/'learner/checkpoint.pt'
    first_report=(tmp_path/'learner/report.json').read_bytes()
    first_receipt=(tmp_path/'learner/native-training-receipt.json').read_bytes()
    state = torch.load(slot, map_location='cpu', weights_only=False)
    assert state['training']['step'] == 1 and state['training']['global_step'] == 2
    assert state['training']['options']['continuation']['checkpoint_sha256'] == sha256_file_hex(parent)
    assert state['training']['optimizer']
    progress=json.loads((tmp_path/'learner/train.jsonl').read_text().splitlines()[0])
    assert progress['objective_details']['objective']=='rft'
    assert progress['objective_details']['sampled_tokens']==2
    report=json.loads((tmp_path/'learner/report.json').read_text())
    assert report['completed'] is True
    assert report['completion_scope']=='policy-operation'
    if not shared_parent:assert report['mandatory_sft']['complete'] is False
    assert state['training']['options']['resident_data_directory'] is None
    assert state['training']['options']['foundation_communication_batches'] is None
    assert state['training']['options']['refresh_foundation_readiness'] is False
    assert state['training']['options']['resident_geometry_qualification'] is None
    assert state['training']['options']['memory_qualification'] is None
    capacity=json.loads((tmp_path/'learner/rl-policy-capacity-request.json').read_text())
    assert capacity['objective'] is not None
    if not shared_parent:
        from natlang_neuralese.train.native_continuation import progress_obligations, restore_sft_progress
        frame=state['training']['sft_progress']['full']
        assert frame['exposure_counts']['prior-foundation-row']==3
        assert progress_obligations({'full':frame})['remaining_graph_visits']==1
        restored=restore_sft_progress(state['training'],'full')
        assert restored['exposure_counts']['prior-foundation-row']==3
        assert restored['global_step']==2
        assert restored['optimizer'] is state['training']['optimizer']
    assert not state['training']['options']['communication_foundation']
    prior_sha = sha256_file_hex(slot)
    assert main(argv) == 0
    assert sha256_file_hex(slot) == prior_sha  # no repeated update/checkpoint rewrite
    if preserve_first_operation_report:
        # This fixture feeds the real operation into the production controller.
        # Keep the report from the invocation that actually updated; the later
        # idempotence run has zero new updates and is not an RL credit source.
        (tmp_path/'learner/report.json').write_bytes(first_report)
        (tmp_path/'learner/native-training-receipt.json').write_bytes(first_receipt)


def test_policy_completion_uses_bounded_rotating_numeric_held_graphs(tmp_path,monkeypatch):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.native_held import held_selection
    def setup(saved,held,roles,config):
        original=json.loads(held.read_text())
        po,pc,co,cc=config['marker_ids']
        rows=[]
        for index in range(3):
            ordinary=copy.deepcopy(original)
            ordinary['id']='held-fixture' if index==0 else 'ordinary-'+str(index)
            rows.append(ordinary)
            communication={**copy.deepcopy(original),'id':'communication-'+str(index),'nodes':[
                {'name':'writer','token_ids':[1,po,5,6,pc],
                 'target_weights':[0,1,1,1,1],'producing':[[2,4]]},
                {'name':'reader','token_ids':[3,co,cc,11,2],
                 'target_weights':[0,0,0,1,1],
                 'imports':[{'at':2,'source':'writer','start':2,'stop':4}]}]}
            rows.append(communication)
        held.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        source=NativeGraphSource(held,config['tokenizer_fingerprint'],required_split='test',role_ids=roles.ids)
        saved['training']['held_row_bindings']=source.bindings
        _,coverage,_=held_selection(source.entries,source_sha256=source.file_sha256,token_budget=1,graph_budget=2)
        saved['training']['metrics']={'held_coverage':coverage,'held':{'loss':-999,'scope':'full'}}
    test_native_stage_real_shared_parent_update_and_exact_noop_resume(tmp_path,False,
        parent_setup=setup,extra_args=['--held-eval-tokens','1','--held-eval-graphs','2'],
        preserve_first_operation_report=True)
    report=json.loads((tmp_path/'learner/report.json').read_text())
    assert report['continuation_boundary']['new_optimizer_updates']>0
    state=torch.load(tmp_path/'learner/checkpoint.pt',weights_only=False)
    held=state['training']['metrics']['held']
    assert held['scope']=='cadence-subset' and held['examples']==2 and held['source_graphs']==6
    assert held['loss']>=0
    # The rotating selector visits one whole graph from each available class,
    # even though this two-graph minimum exceeds the one-token budget.
    assert held['selection']['classes']['ordinary']['selected_ids']==['ordinary-1']
    assert held['selection']['classes']['communication']['selected_ids']==['communication-1']
    assert held['selection']['whole_graph_class_minimum_over_budget'] is True
    assert report['mandatory_sft']['remaining_graph_visits']==1
    readiness=json.loads((tmp_path/'learner/native-training-receipt.json').read_text())
    assert readiness['completed'] is False and readiness['policy_operation_completed'] is True
    # Consume the actual trainer report through the production controller, rather
    # than a synthetic completed flag that could conceal learner_pending loops.
    import shutil
    from natlang_neuralese.rl import native_raw_policy
    cases=tmp_path/'cases.jsonl';cases.write_text(json.dumps({'id':'fixture-only','split':'train'})+'\n')
    config_path=tmp_path/'production.json'
    config_path.write_text(json.dumps({'native_state':True,'out':str(tmp_path/'production'),
        'checkpoint':str(tmp_path/'shared.pt'),'cases':str(cases),'endpoints':['unused'],
        'rows_per_round':1,'learner_command':['actual-report-fixture'],'after_update':['publish'],'base':'unused'}))
    monkeypatch.setattr(native_raw_policy,'prepare_raw_policy',lambda *a,**kw:{'tokenizer':'bound-parent'})
    def collect(**kwargs):
        shutil.copyfile(tmp_path/'rollouts.jsonl',kwargs['out']/'rollouts.jsonl')
        return {'scored':1,'successes':1,'groups_with_spread':0,
                'groups':[{'task_id':'fixture-only','scored':1,'mean':1.}]}
    def learner(argv,values):
        if argv==['publish']:return
        assert argv==['actual-report-fixture']
        shutil.copytree(tmp_path/'learner',Path(values['learner_out']))
    monkeypatch.setattr(production,'collect',collect)
    monkeypatch.setattr(production,'run_command',learner)
    result=production.run(config_path,rounds=1)
    assert result['round']==1 and result['trained_rounds']==1
    assert result['phase']!='learner_pending'
    assert result['last_learner_progress']['mandatory_sft']['remaining_graph_visits']==1


def test_explicit_task_eval_for_recipe_without_cadence_is_exact_and_bound(tmp_path):
    from natlang_neuralese.rl.production import bind_explicit_task_eval
    import hashlib
    packet=tmp_path/'held.jsonl';packet.write_text('{"split":"test"}\n')
    spec={'packet':{'path':str(packet),'sha256':hashlib.sha256(packet.read_bytes()).hexdigest()},
        'cases':[0],'neuralese_server':True,'reference_successes':None}
    result=bind_explicit_task_eval({'task_eval_every':0},spec,cadence=25)
    assert result['task_eval_every']==25 and result['task_eval']['reference_successes'] is None
    assert result['task_eval']['packet']==spec['packet']
    with pytest.raises(ValueError,match='positive cadence'):bind_explicit_task_eval({},spec,cadence=0)
    packet.write_text('{"split":"train"}\n')
    wrong={**spec,'packet':{'path':str(packet),'sha256':hashlib.sha256(packet.read_bytes()).hexdigest()}}
    with pytest.raises(ValueError,match='protected test'):bind_explicit_task_eval({},wrong,cadence=25)
    packet.write_text('changed')
    with pytest.raises(ValueError,match='exact hash'):bind_explicit_task_eval({},spec,cadence=25)


def test_student_and_self_teacher_entries_keep_the_full_prompt_binding():
    import copy
    entries=[{'key':'student','sample':0},{'key':'student2','sample':1}]
    production.bind_collection_prompt(entries,'full')
    teacher=copy.deepcopy(entries[0]);teacher.update(key='teacher',sample=2,jolt_role='teacher')
    assert all(entry['prompt_rendering']=='full' for entry in [*entries,teacher])
    with pytest.raises(ValueError,match='prompt contract'):
        production.bind_collection_prompt([{'prompt_rendering':'program-header'}],'full')
    with pytest.raises(ValueError,match='full prompt'):
        production.bind_collection_prompt(entries,'program-header')


def test_actual_policy_update_keeps_pending_sft_focus_without_resolving_rl_rows(tmp_path):
    import hashlib
    focus=tmp_path/'focus.json';evidence=tmp_path/'evidence';evidence.write_text('oracle')
    evidence_ref={'path':str(evidence),'sha256':hashlib.sha256(evidence.read_bytes()).hexdigest()}
    focus.write_text(json.dumps({'schema':'natlang.native-pending-focus/1','rows':[{'id':'sft-only',
        'binding':'a'*64,'source_sha256':'b'*64,'split':'train','evidence':[evidence_ref]}]}))
    ref={'path':str(focus),'sha256':hashlib.sha256(focus.read_bytes()).hexdigest()}
    history=[{'schema':'natlang.native-pending-focus-history/1','request':ref,'at_global_step':1,
        'initial_credits':{'sft-only':0},'state':'active','consumed':[],'withdrawn':[]}]
    def setup(saved,held,roles,config):saved['training']['options']['pending_focus_history']=copy.deepcopy(history)
    test_native_stage_real_shared_parent_update_and_exact_noop_resume(tmp_path,False,parent_setup=setup)
    updated=torch.load(tmp_path/'learner/checkpoint.pt',map_location='cpu',weights_only=False)
    assert updated['training']['global_step']==2
    assert updated['training']['options']['pending_focus_history']==history


def test_long_graph_limit_records_and_refuses_on_the_real_stage(tmp_path):
    """--max-graph-forward-tokens: a limit above every graph defers none (recorded); below every graph refuses."""
    test_native_stage_real_shared_parent_update_and_exact_noop_resume(tmp_path/'kept', False,
        extra_args=('--max-graph-forward-tokens', '100000'))
    receipt = json.loads((tmp_path/'kept/learner/rl-data-receipt.json').read_text())
    record = receipt['omissions']['deferred_long_graphs']
    assert record['max_graph_forward_tokens'] == 100000 and record['episodes'] == [] and record['path'] is None
    with pytest.raises(ValueError, match='exceeds --max-graph-forward-tokens'):
        test_native_stage_real_shared_parent_update_and_exact_noop_resume(tmp_path/'refused', False,
            extra_args=('--max-graph-forward-tokens', '1'))


def test_long_graph_deferral_keeps_deferred_episodes_whole(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.rl import native_stage
    import natlang_neuralese.data.native_sources as sources
    import natlang_neuralese.rl.native_replay as replay
    sizes = {'short': 10, 'long': 500}
    class Source:
        def __init__(self, path, *_, **__): self.rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        def iter_from(self, _):
            return [SimpleNamespace(id=row['id'], plan=SimpleNamespace(token_count=sizes[row['id']], slot_sources=()))
                    for row in self.rows]
    monkeypatch.setattr(sources, 'NativeGraphSource', Source)
    monkeypatch.setattr(replay, 'NativePolicyObjective', lambda path: SimpleNamespace(
        identity={'tokenizer_fingerprint': 'f'}, shares_causal_prefixes=lambda examples: False))
    graphs = [{'id': 'short', 'nodes': []}, {'id': 'long', 'nodes': [{'token_ids': [1]}]}]
    objective = {'rows': {'short': {'task_id': 't1', 'sample': 0, 'actions': [1]},
                          'long': {'task_id': 't2', 'sample': 1, 'actions': [1, 2]}}}
    graph_path, objective_path = tmp_path/'g.jsonl', tmp_path/'o.json'
    graph_path.write_text(''.join(json.dumps(row) + '\n' for row in graphs)); objective_path.write_text(json.dumps(objective))
    record = native_stage._defer_long_graphs(SimpleNamespace(max_graph_forward_tokens=100, out=tmp_path),
                                             graphs, objective, graph_path, objective_path)
    assert [row['id'] for row in graphs] == ['short'] and set(objective['rows']) == {'short'}
    assert [json.loads(line)['id'] for line in graph_path.read_text().splitlines()] == ['short']
    assert set(json.loads(objective_path.read_text())['rows']) == {'short'}
    assert record['episodes'] == [{'id': 'long', 'task_id': 't2', 'sample': 1, 'forward_tokens': 500,
                                   'original_tokens': 500, 'sampled_tokens': 2}]
    kept = [json.loads(line) for line in (tmp_path/'deferred-long-graphs.jsonl').read_text().splitlines()]
    assert kept == [{'graph': {'id': 'long', 'nodes': [{'token_ids': [1]}]},
                     'objective_row': {'task_id': 't2', 'sample': 1, 'actions': [1, 2]}}]
    assert record['path']['path'].endswith('deferred-long-graphs.jsonl')
