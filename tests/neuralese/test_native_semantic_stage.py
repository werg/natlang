"""Actual VAE/joint/distill optimizer updates, complete exposure and resume."""
import copy
import json
from pathlib import Path

import torch
import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.train import native_semantic_stage as stage
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from test_native_semantic import setup_pair


def _fixture(tmp_path, monkeypatch):
    participants, row, _ = setup_pair()
    parent = tmp_path/'parent'; parent.write_bytes(b'actual test parent identity')
    bindings = []
    for name, participant in participants.items():
        participant.config['architecture'] = participant.identity['architecture']
        bindings.append({'model_id': name, 'checkpoint': {'path': str(parent), 'sha256': sha256_file_hex(parent)},
                         'native_identity': participant.identity})
    manifest = tmp_path/'participants.json'
    manifest.write_text(json.dumps({'schema': 'natlang.native-shared-participants/1', 'models': bindings,
                                    'signature': canonical_json_sha256_hex(bindings)}))
    def artifact(name, value):
        path = tmp_path/name; path.write_text(json.dumps(value) + '\n')
        return {'path': str(path), 'sha256': sha256_file_hex(path)}
    held = copy.deepcopy(row); held['id'] = 'held:payload'; held['split'] = 'test'; held['source_groups'] = ['held:one']
    for name, view in held['models'].items():
        view['graph'].update(id=name + ':held', split='test', source_groups=['held:one'])
    config = {'schema': stage.CONFIG_SCHEMA,
        'models': {name: {'fp8_training': 'off'} for name in participants},
        'pairs': artifact('train.jsonl', row),
        'held_pairs': artifact('held.jsonl', held), 'backbone_lr': 1e-4, 'vae_lr': 1e-3,
        'latent_width': 6, 'hidden_width': 12, 'checkpoint_every': 1, 'eval_every': 1,
        'phases': [{'id': kind, 'kind': kind, 'epochs': 1} for kind in ('vae', 'joint', 'distill')]}
    config_path = tmp_path/'config.json'; config_path.write_text(json.dumps(config))
    def load(name, binding, config, device):
        fresh = setup_pair()[0][name]
        fresh.config['architecture'] = fresh.identity['architecture']
        if name == 'mellum':
            # The tiny fixture uses dense experts/attention in FP32 for CPU
            # exactness. Select the dense latent policy, never ternary codes.
            fresh.model.port.hf.natlang_precision = 'bf16'
        fresh.optimizer = make_native_optimizer(fresh.model, lr=config['backbone_lr'], in_backward=False)[0]
        return fresh
    monkeypatch.setattr(stage, 'load_participant', load)
    return config_path, manifest, config, row


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    else:
        assert left == right


def test_complete_curriculum_and_interrupted_resume_are_exact(tmp_path, monkeypatch):
    config, manifest, _, row = _fixture(tmp_path, monkeypatch)
    torch.manual_seed(1234)
    uninterrupted = stage.run(config, manifest, tmp_path/'whole', device='cpu')
    assert uninterrupted['completed'] and uninterrupted['step'] == 3
    torch.manual_seed(1234)
    interrupted = stage.run(config, manifest, tmp_path/'continued', max_updates=2, device='cpu')
    assert not interrupted['completed'] and interrupted['missing_exposure'][0]['phase'] == 'distill'
    checkpoint = tmp_path/'continued'/'checkpoint.pt'
    resumed = stage.run(config, manifest, tmp_path/'continued', resume=checkpoint, device='cpu')
    assert resumed['completed'] and resumed['step'] == 3
    whole = CheckpointPolicy.load(tmp_path/'whole'/'checkpoint.pt')
    restored = CheckpointPolicy.load(checkpoint)
    _assert_same(whole['vae_state'], restored['vae_state'])
    _assert_same(whole['vae_optimizer'], restored['vae_optimizer'])
    for name in whole['participants']:
        _assert_same(whole['participants'][name]['model_state'], restored['participants'][name]['model_state'])
        _assert_same(whole['participants'][name]['optimizer'], restored['participants'][name]['optimizer'])
    _assert_same(whole['training']['rng'], restored['training']['rng'])
    assert all(counts[stage.pair_key('lfm', 'mellum')] == 1
               for values in resumed['exposure'].values() for counts in values.values())
    export = json.loads((tmp_path/'continued'/'semantic-adapter.json').read_text())
    assert export['checkpoint']['sha256'] == sha256_file_hex(checkpoint)
    assert {value['native_identity']['checkpoint_sha256'] for value in export['models'].values()} == {
        export['checkpoint']['sha256']}


def test_appended_data_and_larger_epochs_are_not_mistaken_for_finished(tmp_path, monkeypatch):
    path, manifest, config, row = _fixture(tmp_path, monkeypatch)
    first = stage.run(path, manifest, tmp_path/'run', device='cpu')
    assert first['completed']
    added = copy.deepcopy(row); added['id'] = 'additional:payload'
    train = Path(config['pairs']['path'])
    train.write_text(train.read_text() + json.dumps(added) + '\n')
    config['pairs']['sha256'] = sha256_file_hex(train)
    config['phases'][0]['epochs'] = 2
    path.write_text(json.dumps(config))
    report = stage.run(path, manifest, tmp_path/'run', resume=tmp_path/'run'/'checkpoint.pt', device='cpu')
    assert report['completed'] and report['step'] == 8
    for counts in report['exposure']['vae'].values():
        assert counts[stage.pair_key('lfm', 'mellum')] == 2


@pytest.mark.parametrize('changed_count', [1, 2])
def test_refresh_keeps_rl_participant_and_retrains_all_semantic_phases(tmp_path, monkeypatch, changed_count):
    path, manifest, config, _ = _fixture(tmp_path, monkeypatch)
    config['adaptive_refresh']={'updates_per_phase':1}
    path.write_text(json.dumps(config))
    original = stage.run(path, manifest, tmp_path/'original', device='cpu')
    parent = tmp_path/'original/checkpoint.pt'
    parent_sha = sha256_file_hex(parent)
    original_load = stage.load_participant
    bindings = json.loads(manifest.read_text())
    changed_names = {value['model_id'] for value in bindings['models'][:changed_count]}
    for binding in bindings['models'][:changed_count]:
        update = tmp_path/(binding['model_id'] + '-rl-update.pt')
        torch.save({'training': {'options': {'continuation': {'checkpoint_sha256': parent_sha,
            'participant': binding['model_id'], 'objective_transition': True}}}}, update)
        binding['checkpoint'] = {'path': str(update), 'sha256': sha256_file_hex(update)}
        binding['native_identity']['checkpoint_sha256'] = sha256_file_hex(update)
    bindings['signature'] = canonical_json_sha256_hex(bindings['models'])
    manifest.write_text(json.dumps(bindings))
    loaded_values = {}
    def updated_load(name, binding, config, device):
        participant = original_load(name, binding, config, device)
        participant.identity = binding['native_identity']
        if name in changed_names:
            participant.continuation_proof = {'checkpoint_sha256': parent_sha,
                'participant': name, 'objective_transition': True}
            with torch.no_grad():
                participant.model.input_rows.add_(.125)
            loaded_values[name] = participant.model.input_rows.detach().clone()
        return participant
    monkeypatch.setattr(stage, 'load_participant', updated_load)
    # A normal resume must reject a replaced lineage; refresh is explicit.
    import pytest
    with pytest.raises(ValueError, match='changed the declared model lineage'):
        stage.run(path, manifest, tmp_path/'refreshed', resume=parent, device='cpu')
    without_refresh=copy.deepcopy(config);without_refresh.pop('adaptive_refresh')
    path.write_text(json.dumps(without_refresh))
    with pytest.raises(ValueError,match='declared adaptive_refresh'):
        stage.run(path,manifest,tmp_path/'missing-refresh',resume=parent,refresh_participants=True,device='cpu')
    path.write_text(json.dumps(config))
    partial=stage.run(path,manifest,tmp_path/'partial-refresh',resume=parent,refresh_participants=True,device='cpu',max_updates=1)
    assert partial['exposure_complete'] and not partial['completed'] and not partial['refresh_ready']
    assert partial['refresh_progress']['completed_updates']=={'vae':1,'joint':0,'distill':0}
    refreshed = stage.run(path, manifest, tmp_path/'refreshed', resume=parent,
        refresh_participants=True, device='cpu')
    assert refreshed['exposure_complete'] and refreshed['step'] == original['step'] + 3
    state = CheckpointPolicy.load(tmp_path/'refreshed/checkpoint.pt')
    assert state['training']['participant_refreshes'][-1]['changed_models'] == sorted(changed_names)
    for binding in bindings['models'][:changed_count]:
        assert state['participants'][binding['model_id']]['source_identity'] == binding['native_identity']
    # The updated interface survives; old semantic weights were not restored.
    for name in changed_names:
        torch.testing.assert_close(state['participants'][name]['model_state']['interface_state']['input_rows'],
                                   loaded_values[name], atol=.01, rtol=.01)
    assert all(counts[stage.pair_key('lfm', 'mellum')] == 2
               for values in state['training']['exposure'].values() for counts in values.values())


def test_shared_stop_checkpoints_without_publishing_incomplete_curriculum(tmp_path, monkeypatch):
    from types import SimpleNamespace
    path, manifest, config, _ = _fixture(tmp_path, monkeypatch)
    stage.run(path, manifest, tmp_path/'parent-run', device='cpu')
    parent = tmp_path/'parent-run/checkpoint.pt'
    before = CheckpointPolicy.load(parent)
    for phase in config['phases']:
        phase['epochs'] = 2
    path.write_text(json.dumps(config))
    stopped = stage.run(path, manifest, tmp_path/'stopped', resume=parent, device='cpu',
                        stop=SimpleNamespace(requested=True))
    assert not stopped['completed'] and not stopped['production_ready']
    assert stopped['step'] == before['training']['step']
    checkpoint = tmp_path/'stopped/checkpoint.pt'
    saved = CheckpointPolicy.load(checkpoint)
    _assert_same(saved['vae_optimizer'], before['vae_optimizer'])
    for name in before['participants']:
        _assert_same(saved['participants'][name]['optimizer'], before['participants'][name]['optimizer'])
    resumed = stage.run(path, manifest, tmp_path/'stopped', resume=checkpoint, device='cpu',
                        stop=SimpleNamespace(requested=False))
    assert resumed['completed'] and resumed['step'] == stopped['step'] + 3


def test_adaptive_all_phase_quanta_resume_and_carry_whole_cohort(tmp_path,monkeypatch):
    path,manifest,config,row=_fixture(tmp_path,monkeypatch)
    train=Path(config['pairs']['path']);held=Path(config['held_pairs']['path'])
    rows=[copy.deepcopy(row) for _ in range(3)]
    for index,value in enumerate(rows):value['id']='train:'+str(index)
    train.write_text(''.join(json.dumps(value)+'\n' for value in rows))
    config['pairs']['sha256']=sha256_file_hex(train)
    held_row=json.loads(held.read_text());held_rows=[copy.deepcopy(held_row) for _ in range(3)]
    for index,value in enumerate(held_rows):value['id']='held:'+str(index)
    unavailable=copy.deepcopy(held_rows[0]);unavailable['id']='held:other-participants'
    unavailable['models']={('other-'+name):view for name,view in unavailable['models'].items()}
    held_rows.insert(0,unavailable)
    held.write_text(''.join(json.dumps(value)+'\n' for value in held_rows))
    config['held_pairs']['sha256']=sha256_file_hex(held)
    config.update(adaptive_refresh={'updates_per_phase':1},held_eval_rows=1)
    monkeypatch.setattr(stage.PairedSource,'get',lambda *args:pytest.fail('phase changes must restart the stream, never scan the remaining corpus then randomly reread'))
    from natlang_neuralese.train import native_semantic_refresh as scheduler
    summary_calls=[];summarize=scheduler.exposure_summary
    def summary(*args):summary_calls.append(True);return summarize(*args)
    monkeypatch.setattr(scheduler,'exposure_summary',summary)
    path.write_text(json.dumps(config))
    torch.manual_seed(1234)
    whole=stage.run(path,manifest,tmp_path/'whole',device='cpu')
    assert whole['step']==3 and not whole['completed'] and not whole['exposure_complete']
    assert len(summary_calls)==2  # curriculum initialization + final proof, never each committed update
    assert whole['semantic_obligations']['remaining_pair_visits']==6
    assert whole['refresh_progress']['completed_updates']==dict.fromkeys(('vae','joint','distill'),1)
    assert whole['metrics']['held']['scope']=='rotating'
    from natlang_neuralese.train.native_semantic_refresh import validate_boundary
    exported=json.loads((tmp_path/'whole/semantic-adapter.json').read_text())
    # The verifier isolates exposure/phase accounting here; this mechanism
    # fixture has no task/tokenizer export and is NOT a qualified deployment.
    boundary={**whole,'refresh_ready':True}
    assert validate_boundary(boundary,exported,config) is False
    incomplete=copy.deepcopy(boundary);incomplete['refresh_progress']['completed_updates']['joint']=0
    with pytest.raises(ValueError,match='committed work'):
        validate_boundary(incomplete,exported,config)
    dishonest=copy.deepcopy(boundary);dishonest['completed']=True
    with pytest.raises(ValueError,match='full final held'):
        validate_boundary(dishonest,exported,config)
    torch.manual_seed(1234)
    first=stage.run(path,manifest,tmp_path/'resumed',device='cpu',max_updates=2)
    assert not first['refresh_ready']
    resumed=stage.run(path,manifest,tmp_path/'resumed',device='cpu',resume=tmp_path/'resumed/checkpoint.pt')
    assert resumed['step']==3 and resumed['exposure']==whole['exposure']
    a=CheckpointPolicy.load(tmp_path/'whole/checkpoint.pt');b=CheckpointPolicy.load(tmp_path/'resumed/checkpoint.pt')
    _assert_same(a['vae_optimizer'],b['vae_optimizer']);_assert_same(a['vae_state'],b['vae_state'])
    _assert_same(a['training']['rng'],b['training']['rng'])
    assert a['training']['held_coverage']['cursor']==b['training']['held_coverage']['cursor']
    # A new actual shared publication starts a new refresh, preserves every old
    # credit, and consumes row 2 in every phase rather than replaying row 1.
    bindings=json.loads(manifest.read_text())
    exported=json.loads((tmp_path/'resumed/semantic-adapter.json').read_text())
    for binding in bindings['models']:
        entry=exported['models'][binding['model_id']]
        binding.update(checkpoint=entry['checkpoint'],native_identity=entry['native_identity'])
    bindings['signature']=canonical_json_sha256_hex(bindings['models']);manifest.write_text(json.dumps(bindings))
    original=stage.load_participant
    def current(name,binding,config,device):
        value=original(name,binding,config,device);value.identity=binding['native_identity'];return value
    monkeypatch.setattr(stage,'load_participant',current)
    next_report=stage.run(path,manifest,tmp_path/'second',device='cpu',resume=tmp_path/'resumed/checkpoint.pt',refresh_participants=True)
    assert next_report['step']==6 and next_report['semantic_obligations']['remaining_pair_visits']==3
    assert set(next_report['exposure']['joint'])=={'train:0','train:1'}
    assert not next_report['completed']
    exported=json.loads((tmp_path/'second/semantic-adapter.json').read_text())
    for binding in bindings['models']:
        entry=exported['models'][binding['model_id']]
        binding.update(checkpoint=entry['checkpoint'],native_identity=entry['native_identity'])
    bindings['signature']=canonical_json_sha256_hex(bindings['models']);manifest.write_text(json.dumps(bindings))
    third=stage.run(path,manifest,tmp_path/'third',device='cpu',resume=tmp_path/'second/checkpoint.pt',refresh_participants=True)
    assert third['step']==9 and third['completed'] and third['exposure_complete']
    assert third['metrics']['held']['scope']=='full' and third['metrics']['held']['rows']==3
    assert third['semantic_obligations']['remaining_pair_visits']==0
    assert all(len(rows)==3 for rows in third['exposure'].values())
    final_export=json.loads((tmp_path/'third/semantic-adapter.json').read_text())
    assert validate_boundary(third,final_export,config) is True


def test_semantic_only_controller_restart_consumes_real_remaining_phase_rows(tmp_path,monkeypatch):
    """Actual two-port updates/resume; qualification is an explicit control stub.

    The tiny in-memory tokenizer fixture cannot qualify production serving.
    Existing separate real-wire tests exercise that qualification mechanism.
    """
    from natlang_neuralese.train import native_readiness as controller
    from natlang_neuralese.train.native_semantic_refresh import validate_boundary
    path,manifest,config,row=_fixture(tmp_path,monkeypatch)
    train=Path(config['pairs']['path'])
    rows=[copy.deepcopy(row) for _ in range(3)]
    for index,value in enumerate(rows):value['id']='remaining:'+str(index)
    train.write_text(''.join(json.dumps(value)+'\n' for value in rows));config['pairs']['sha256']=sha256_file_hex(train)
    config.update(adaptive_refresh={'updates_per_phase':1},held_eval_rows=1)
    path.write_text(json.dumps(config))
    selected=json.loads(manifest.read_text())['models']
    out=tmp_path/'controller'
    declaration={'schema':'natlang.native-shared-stage-controller/1','out':str(out),
        'participants':[],'curriculum':str(path),'poll_seconds':.001,
        'command':['actual-stage-fixture','{participants}','{out}']}
    controller_path=tmp_path/'controller.json';controller_path.write_text(json.dumps(declaration))
    monkeypatch.setattr(controller,'ready_participants',lambda config:(selected,[]))
    def validate(report_path,published):
        proof=json.loads(Path(report_path).read_text());export=json.loads(Path(proof['adapter']['path']).read_text())
        validate_boundary(proof,export,stage.validate_config(json.loads(Path(proof['curriculum']['path']).read_text())))
        key=(proof['participants_signature'],proof['curriculum_signature'])
        return proof,published[key]['execution_signature']
    monkeypatch.setattr(controller,'validate_completion',validate)
    class Stop:
        requested=False
    stop=Stop();launches=[];outputs=[]
    class Child:
        pid=12345
        def __init__(self,code=0):self.returncode=code
        def poll(self):return self.returncode
        def terminate(self):pass
    def launch(argv,**kwargs):
        child_manifest,child_out=Path(argv[1]),Path(argv[2])
        launches.append(json.loads(child_manifest.read_text()))
        resume=child_out/'checkpoint.pt'
        report=stage.run(path,child_manifest,child_out,device='cpu',resume=resume if resume.is_file() else None,
            max_updates=1 if len(launches)==2 else None)
        # Control-only fixture attestation. This is not a runtime/task pass.
        ready=all(report['refresh_progress']['completed_updates'][key]>=value for key,value in report['refresh_progress']['required_updates'].items())
        report['production_ready']=ready;report['refresh_ready']=ready
        (child_out/'shared-stage-completion.json').write_text(json.dumps(report))
        outputs.append((child_out,report))
        return Child(0 if ready else 75)
    monkeypatch.setattr(controller.subprocess,'Popen',launch)
    def pause(seconds):
        # Stop after the first publication, then simulate a complete supervisor
        # restart using durable state rather than in-memory cursors.
        if len(outputs)==1 and not (out/'stopped-once').exists():
            (out/'stopped-once').write_text('first successful partial publication')
            stop.requested=True
        elif len(outputs)==2 and not (out/'stopped-in-cycle').exists():
            (out/'stopped-in-cycle').write_text('incomplete child checkpoint')
            stop.requested=True
        elif outputs and outputs[-1][1]['completed']:stop.requested=True
    monkeypatch.setattr(controller.time,'sleep',pause)
    first=controller._run(controller_path,stop=stop)
    assert len(outputs)==1 and first['last_semantic_publication']
    initial_dir,initial=outputs[0];initial_sha=sha256_file_hex(initial_dir/'checkpoint.pt')
    assert initial['step']==3 and not initial['completed']
    # Simulate loss of the controller state commit after the successful child
    # report: exact child/report recovery must adopt it rather than replay it.
    state_path=out/'controller.json';lost=json.loads(state_path.read_text())
    lost['last_semantic_publication']=None;lost['last_successful_participants']=None
    state_path.write_text(json.dumps(lost))
    stop.requested=False
    interrupted=controller._run(controller_path,stop=stop)
    assert len(outputs)==2 and outputs[-1][1]['step']==4 and interrupted['retry']
    assert interrupted['last_semantic_publication']==first['last_semantic_publication']
    stop.requested=False
    final=controller._run(controller_path,stop=stop)
    assert len(outputs)==4 and outputs[-1][1]['completed'] and outputs[-1][1]['step']==9
    assert final['last_semantic_publication']['path']==str((outputs[-1][0]/'shared-stage-completion.json').resolve())
    assert len({str(directory) for directory,_ in outputs})==3
    assert outputs[1][0]==outputs[2][0] and outputs[2][1]['step']==6
    assert sha256_file_hex(initial_dir/'checkpoint.pt')==initial_sha
    assert 'semantic_parent' not in launches[0]
    assert all('semantic_parent' in value for value in launches[1:])
    final_state=CheckpointPolicy.load(outputs[-1][0]/'checkpoint.pt')
    assert all(len(values)==3 for values in final_state['training']['exposure'].values())
    assert all(counts[stage.pair_key('lfm','mellum')]==1 for values in final_state['training']['exposure'].values() for counts in values.values())
    # A further supervisor restart observes full completion without relaunch.
    inspection=controller._run(controller_path,once=True,stop=Stop())
    assert len(outputs)==4 and inspection['last_semantic_publication']==final['last_semantic_publication']


def test_adaptive_pending_cursor_avoids_prefix_scan_and_wraps_owed_rows(monkeypatch):
    from natlang_neuralese.train import native_semantic_refresh as refresh
    entries=[{'id':str(index),'models':['a','b']} for index in range(1000)]
    phase={'id':'joint','kind':'joint','epochs':1};key=stage.pair_key('a','b')
    exposure={'joint':{str(index):{key:1} for index in range(900)}}
    cycle={'required_updates':{'joint':16},'completed_updates':{'joint':0}}
    calls=[];original=stage.needed_pairs
    def count(*args):calls.append(args[0]['id']);return original(*args)
    monkeypatch.setattr(stage,'needed_pairs',count)
    chosen=refresh.work([phase],entries,{'a':None,'b':None},exposure,cycle,{'joint':900})
    assert chosen[1]=='900' and calls==['900']
    # A new obligation before the cursor remains owed and is found on wrap.
    exposure['joint']={str(index):{key:1} for index in range(1000)}
    del exposure['joint']['0'];calls.clear()
    chosen=refresh.work([phase],entries,{'a':None,'b':None},exposure,cycle,{'joint':998})
    assert chosen[1]=='0' and calls==['998','999','0']


@pytest.mark.parametrize('parent',[{},False,{'checkpoint':{}}])
def test_invalid_semantic_parent_cannot_silently_restart_training(tmp_path,monkeypatch,parent):
    config,manifest,_,_=_fixture(tmp_path,monkeypatch)
    value=json.loads(manifest.read_text());value['semantic_parent']=parent
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='exact checkpoint, adapter and completion'):
        stage.run(config,manifest,tmp_path/'must-not-start',device='cpu')
    assert not (tmp_path/'must-not-start/checkpoint.pt').exists()


def test_live_parent_change_cannot_reset_current_cycle_credits(tmp_path,monkeypatch):
    path,manifest,config,_=_fixture(tmp_path,monkeypatch)
    config['adaptive_refresh']={'updates_per_phase':1};path.write_text(json.dumps(config))
    actual=stage.joint_loss
    def change(*args,**kwargs):
        value=actual(*args,**kwargs)
        altered=json.loads(manifest.read_text());altered['semantic_parent']={}
        manifest.write_text(json.dumps(altered))
        return value
    monkeypatch.setattr(stage,'joint_loss',change)
    with pytest.raises(ValueError,match='parent changed inside a running cycle'):
        stage.run(path,manifest,tmp_path/'current',device='cpu')
    saved=CheckpointPolicy.load(tmp_path/'current/checkpoint.pt')
    assert saved['training']['step']==1
    assert saved['training']['refresh_cycle']['completed_updates']=={'vae':1,'joint':0,'distill':0}


@pytest.mark.parametrize('change', ['missing-updated','lineage','unchanged-policy'])
def test_semantic_resume_never_relabels_changed_or_unchanged_participant_recipe(tmp_path,monkeypatch,change):
    from natlang_neuralese.train.recipe import native_recipe_binding
    config,manifest,_,_=_fixture(tmp_path,monkeypatch)
    recipe=tmp_path/'recipe.json';recipe.write_text('{}')
    reference=native_recipe_binding(recipe,'a'*64)
    original=stage.load_participant
    def initial(*args):
        result=original(*args);result.recipe_binding=copy.deepcopy(reference);return result
    monkeypatch.setattr(stage,'load_participant',initial)
    stage.run(config,manifest,tmp_path/'lineage-parent',max_updates=1,device='cpu')
    parent=tmp_path/'lineage-parent'/'checkpoint.pt'
    def changed(*args):
        result=initial(*args)
        if args[0]=='lfm':
            if change=='unchanged-policy':result.recipe_binding['sha256']='c'*64
            else:
                result.identity={**result.identity,'checkpoint_sha256':'f'*64}
                result.continuation_proof={'checkpoint_sha256':sha256_file_hex(parent),'participant':'lfm',
                    'objective_transition':True}
                if change=='missing-updated':result.recipe_binding=None
                else:result.recipe_binding['lineage_sha256']='c'*64
        return result
    monkeypatch.setattr(stage,'load_participant',changed)
    before=sha256_file_hex(parent)
    with pytest.raises(ValueError,match='recipe provenance|recipe lineage'):
        stage.run(config,manifest,tmp_path/'changed',resume=parent,refresh_participants=True,device='cpu')
    assert sha256_file_hex(parent)==before


def test_frozen_snapshot_preparation_then_explicit_descendant_optimizer_adoption(tmp_path,monkeypatch):
    """Actual two-port VAE/joint updates; readiness transport is a protocol fixture."""
    path,manifest,config,_=_fixture(tmp_path,monkeypatch)
    config['snapshot_preparation']=True
    bindings=json.loads(manifest.read_text())
    for binding in bindings['models']:
        binding.update(readiness={'path':'protocol-proof','sha256':'a'*64},
            readiness_requirement={'model_id':binding['model_id']})
    bindings['signature']=canonical_json_sha256_hex(bindings['models']);manifest.write_text(json.dumps(bindings))
    # Full readiness cryptography has independent per-row/runtime/task tests.
    # This test measures the actual shared optimizer boundary without fabricating
    # a production CUDA certificate for its tiny CPU ports.
    monkeypatch.setattr(stage,'_qualified_snapshot',lambda name,binding:binding)
    from natlang_neuralese.train import native_continuation
    monkeypatch.setattr(native_continuation,'qualified_refresh_config',lambda *a,**k:(config,[]))
    path.write_text(json.dumps(config))
    frozen=stage.run(path,manifest,tmp_path/'preparation',device='cpu')
    parent=tmp_path/'preparation/checkpoint.pt';saved=CheckpointPolicy.load(parent)
    assert frozen['step']==1 and not frozen['production_ready'] and not frozen['completed']
    assert not any(saved['training'].get('model_updates',{}).values())
    assert set(saved['training']['exposure'])=={'vae'}
    from natlang_neuralese.train import native_readiness
    key=(frozen['participants_signature'],frozen['curriculum_signature'])
    published={key:{'execution_signature':'frozen-preparation-fixture','models':['lfm','mellum']}}
    proof,signature=native_readiness.validate_snapshot_preparation(tmp_path/'preparation/shared-stage-completion.json',published)
    assert signature=='frozen-preparation-fixture' and proof['preparation_complete']
    with pytest.raises(ValueError,match='shared completion'):
        native_readiness.validate_completion(tmp_path/'preparation/shared-stage-completion.json',published)
    controller=tmp_path/'controller.json'
    controller.write_text(json.dumps({'schema':'natlang.native-shared-stage-controller/1',
        'participants':bindings['models'],'out':str(tmp_path/'preparation'),'curriculum':str(path),
        'command':['unused-no-launch-during-once-inspection','--participants','{participants}','--out','{out}']}))
    with monkeypatch.context() as local:
        local.setattr(native_readiness,'ready_participants',lambda config:(bindings['models'],[]))
        inspected=native_readiness.run(controller,once=True)
    assert inspected['child_pid'] is None and inspected['last_snapshot_preparation']
    assert inspected['last_successful_participants'] is None
    assert inspected['last_semantic_publication'] is None
    # A structurally finished preparation is a successful worker operation,
    # while remaining explicitly incomplete as a production curriculum.
    cli=tmp_path/'cli.json';cli.write_text(json.dumps({**config,'task_eval':{'protocol':'not executed here'}}))
    with monkeypatch.context() as local:
        local.setattr(stage,'run',lambda *a,**kw:frozen)
        assert stage.main(['--config',str(cli),'--participants',str(manifest),
            '--out',str(tmp_path/'preparation'),'--device','cpu'])==0
    for participant in saved['participants'].values():
        for key in ('muon','adamw','muonsr'):
            state=participant['optimizer'].get(key)
            if state is not None:assert not state['state']
    config['snapshot_preparation']=False;config['adaptive_refresh']={'updates_per_phase':1}
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='explicit descendant optimizer handoff'):
        stage.run(path,manifest,tmp_path/'without-release',resume=parent,device='cpu',refresh_participants=True)
    # New descendants are produced by a genuine optimizer update on each tiny
    # model. Their full optimizer history and earned exposure must be adopted.
    original=stage.load_participant;loaded={}
    for binding in bindings['models']:
        binding['native_identity']['checkpoint_sha256']='b'*64
    bindings['signature']=canonical_json_sha256_hex(bindings['models']);manifest.write_text(json.dumps(bindings))
    def descendant(name,binding,current,device):
        value=original(name,binding,current,device)
        loss=value.model.input_rows.square().sum();loss.backward();value.optimizer.step();value.optimizer.zero_grad(set_to_none=True)
        value.identity=binding['native_identity']
        value.snapshot_ancestors=[saved['participants'][name]['source_identity']['checkpoint_sha256']]
        value.global_step=17
        value.source_rng=stage.capture_training_rng_state('cpu')
        value.sft_progress={'actual-descendant':{'remaining_ordinary_credit':3}}
        loaded[name]=copy.deepcopy(value.optimizer.state_dict())
        loaded[name+'-rng']=copy.deepcopy(value.source_rng)
        return value
    monkeypatch.setattr(stage,'load_participant',descendant)
    monkeypatch.setattr(stage,'_snapshot_adoption',lambda config,bindings:{'protocol':'real-owner-release independently tested'})
    continued=stage.run(path,manifest,tmp_path/'adopted',resume=parent,refresh_participants=True,
        device='cpu',max_updates=1)
    adopted=CheckpointPolicy.load(tmp_path/'adopted/checkpoint.pt')
    assert continued['step']==2 and not continued['production_ready']
    assert adopted['training']['participant_refreshes'][-1]['changed_models']==['lfm','mellum']
    for name,participant in adopted['participants'].items():
        _assert_same(participant['optimizer'],loaded[name])
        _assert_same(participant['source_rng'],loaded[name+'-rng'])
        assert participant['global_step']==17
        assert participant['sft_progress']=={'actual-descendant':{'remaining_ordinary_credit':3}}
        assert participant['source_identity']['checkpoint_sha256']=='b'*64
    # The first adopted update is a real VAE refresh; joint/distill still remain.
    assert continued['refresh_progress']['completed_updates']['vae']==1


def test_new_frozen_work_refreshes_vae_after_real_descendant_sgd_and_resumes_exact_credit(tmp_path,monkeypatch):
    """Actual tiny two-port optimizer/VAE work; readiness proofs are protocol fixtures."""
    path,manifest,config,_=_fixture(tmp_path,monkeypatch)
    config.update(snapshot_preparation=True,adaptive_refresh={'updates_per_phase':1})
    bindings=json.loads(manifest.read_text())
    for binding in bindings['models']:
        binding.update(readiness={'path':'protocol-proof','sha256':'a'*64},
            readiness_requirement={'model_id':binding['model_id']})
    bindings['signature']=canonical_json_sha256_hex(bindings['models']);manifest.write_text(json.dumps(bindings))
    monkeypatch.setattr(stage,'_qualified_snapshot',lambda name,binding:binding)
    from natlang_neuralese.train import native_continuation
    monkeypatch.setattr(native_continuation,'qualified_refresh_config',lambda *a,**k:(config,[]))
    path.write_text(json.dumps(config))
    first=stage.run(path,manifest,tmp_path/'frozen-first',device='cpu')
    parent=tmp_path/'frozen-first/checkpoint.pt';old=CheckpointPolicy.load(parent)
    assert first['step']==1 and first['preparation_complete']
    assert first['refresh_progress']['required_updates']=={'vae':1}
    original=stage.load_participant;optimizers={}
    row=json.loads(Path(config['pairs']['path']).read_text())
    from natlang_neuralese.data.native_graph import parse_graph,graph_loss
    def descendant(name,binding,current,device):
        participant=original(name,binding,current,device)
        loss,_=graph_loss(participant.model,parse_graph(row['models'][name]['graph'],
            tokenizer_fingerprint=participant.identity['tokenizer_fingerprint']),checkpoint_layers=False)
        loss.backward();participant.optimizer.step();participant.optimizer.zero_grad(set_to_none=True)
        participant.identity=binding['native_identity']
        participant.snapshot_ancestors=[old['participants'][name]['source_identity']['checkpoint_sha256']]
        participant.sft_progress={'actual_descendant':{'earned_updates':1,'remaining_ordinary_credit':3}}
        optimizers[name]=copy.deepcopy(participant.optimizer.state_dict())
        return participant
    monkeypatch.setattr(stage,'load_participant',descendant)
    for binding in bindings['models']:binding['native_identity']['checkpoint_sha256']='b'*64
    bindings['signature']=canonical_json_sha256_hex(bindings['models']);manifest.write_text(json.dumps(bindings))
    config.update(snapshot_work_generation='explicit-new-frozen-work',adaptive_refresh={'updates_per_phase':2})
    path.write_text(json.dumps(config))
    partial=stage.run(path,manifest,tmp_path/'frozen-next',resume=parent,refresh_participants=True,
        device='cpu',max_updates=1)
    checkpoint=tmp_path/'frozen-next/checkpoint.pt';middle=CheckpointPolicy.load(checkpoint)
    assert partial['step']==2 and not partial['preparation_complete']
    assert partial['refresh_progress']['required_updates']=={'vae':2}
    assert partial['refresh_progress']['completed_updates']=={'vae':1}
    from test_native_shared_checkpoint import equal_tree
    for name,participant in middle['participants'].items():
        equal_tree(participant['optimizer'],optimizers[name])
        assert participant['sft_progress']['actual_descendant']['earned_updates']==1
    final=stage.run(path,manifest,tmp_path/'frozen-next',resume=checkpoint,refresh_participants=True,device='cpu')
    saved=CheckpointPolicy.load(checkpoint)
    assert final['step']==3 and final['preparation_complete'] and not final['production_ready']
    assert not any(saved['training'].get('model_updates',{}).values())
    counts=saved['training']['exposure']['vae']
    assert sum(sum(value.values())for value in counts.values())==3
    assert any(not torch.equal(value,saved['vae_state'][key])for key,value in old['vae_state'].items())
    repeated=stage.run(path,manifest,tmp_path/'frozen-next',resume=checkpoint,refresh_participants=True,device='cpu')
    assert repeated['step']==3 and repeated['preparation_complete']
    equal_tree(CheckpointPolicy.load(checkpoint)['training']['exposure'],saved['training']['exposure'])


def test_generated_snapshot_full_multirow_vae_epoch_resumes_after_quantum(tmp_path,monkeypatch):
    from test_native_role_scheduler_deployment import source
    from natlang_neuralese.train.native_deployment import prepare_role_scheduler,binding
    metadata=tmp_path/'metadata';metadata.mkdir()
    plans,controls,curriculum,provider,states,runtime=source(metadata)
    declared=json.loads(curriculum.read_text());declared['adaptive_refresh']={'updates_per_phase':1}
    curriculum.write_text(json.dumps(declared))
    produced=prepare_role_scheduler(plans,controls,curriculum,provider,metadata/'prepared',
        raw_states=states,runtime=runtime,rl_bootstrap=binding(metadata/'rl-bootstrap.json'),handoff_lease_seconds=180)
    controller=json.loads(Path(produced['controller']['path']).read_text())
    snapshot=json.loads(Path(controller['curriculum']).read_text())
    assert 'adaptive_refresh'not in snapshot
    actual=tmp_path/'actual';actual.mkdir()
    path,manifest,config,row=_fixture(actual,monkeypatch)
    train=Path(config['pairs']['path']);rows=[]
    for i in range(3):
        value=copy.deepcopy(row);value['id']='train:'+str(i);rows.append(value)
    train.write_text(''.join(json.dumps(value)+'\n'for value in rows))
    config['pairs']['sha256']=sha256_file_hex(train)
    config['snapshot_preparation']=snapshot['snapshot_preparation']
    if 'adaptive_refresh'in snapshot:config['adaptive_refresh']=snapshot['adaptive_refresh']
    config['inherit_optimizer']=snapshot['inherit_optimizer']
    monkeypatch.setattr(stage,'_qualified_snapshot',lambda name,binding:binding)
    from natlang_neuralese.train import native_continuation
    monkeypatch.setattr(native_continuation,'qualified_refresh_config',lambda *a,**k:(config,[]))
    path.write_text(json.dumps(config));torch.manual_seed(314)
    full=stage.run(path,manifest,actual/'full',device='cpu')
    torch.manual_seed(314)
    paused=stage.run(path,manifest,actual/'paused',device='cpu',max_updates=1)
    assert paused['step']==1 and not paused['preparation_complete']
    resumed=stage.run(path,manifest,actual/'paused',device='cpu',resume=actual/'paused/checkpoint.pt')
    assert full['step']==resumed['step']==3 and full['preparation_complete'] and resumed['preparation_complete']
    assert not resumed['completed'] and not resumed['production_ready']
    assert resumed['semantic_obligations']['remaining_pair_visits']==6
    a=CheckpointPolicy.load(actual/'full/checkpoint.pt');b=CheckpointPolicy.load(actual/'paused/checkpoint.pt')
    _assert_same(a['vae_optimizer'],b['vae_optimizer']);_assert_same(a['vae_state'],b['vae_state'])
    _assert_same(a['training']['rng'],b['training']['rng'])
    _assert_same(a['participants'],b['participants'])
    assert not any(b['training'].get('model_updates',{}).values())
