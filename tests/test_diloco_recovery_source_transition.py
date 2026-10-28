import hashlib,json
from pathlib import Path
import pytest
from test_diloco_recovery import load
from test_diloco_direct_recovery import fixture


def transitioned(tmp_path):
    mod=load('diloco_direct_recovery');path,control,recipe,_=fixture(tmp_path)
    plan_ref=control['bindings']['plan'];plan_path=Path(plan_ref['path']);plan=json.loads(plan_path.read_text())
    roles=plan['stage_inputs']['native_full_sft']
    for name in ('native_graphs','foundation_native_graphs'):
        roles[name]={'path':'/source/new','sha256':'d'*64}
    plan_path.write_text(json.dumps(plan,sort_keys=True));plan_ref['sha256']=hashlib.sha256(plan_path.read_bytes()).hexdigest()
    owner_ref=control['bindings']['owner_journal'];owner_path=Path(owner_ref['path']);owner=json.loads(owner_path.read_text());owner['plan']=plan_ref
    owner_path.write_text(json.dumps(owner,sort_keys=True));owner_ref['sha256']=hashlib.sha256(owner_path.read_bytes()).hexdigest()
    arg_ref=control['bindings']['worker_argv'];arg_path=Path(arg_ref['path']);argv=json.loads(arg_path.read_text())
    for flag in ('--native-graphs','--foundation-native-graphs'):argv[argv.index(flag)+1]='/source/new'
    arg_path.write_text(json.dumps(argv,sort_keys=True));arg_ref['sha256']=hashlib.sha256(arg_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(control));record=mod.record_from_control(path,'run');direct=mod.load_control(path,record)
    return mod,path,direct,recipe,roles


def test_transition_requires_matched_checkpoint_and_committed_coordinator_proof(tmp_path):
    mod,path,direct,recipe,roles=transitioned(tmp_path);service=load('diloco_periodic_recovery')
    assert direct['transition_required']
    original={'schema':'natlang.native-diloco-identity/1','parent_checkpoint_sha256':'e'*64,
              'trainer_argv':direct['original_identity_argv'],'diloco':{'run_id':'run'}}
    current={**original,'trainer_argv':direct['current_identity_argv']}
    base={'schema':'natlang.diloco-source-identity-transition/1','correction_verified':True,
          'checkpoint_sha256':'1'*64,'correction_sha256':'2'*64,'previous_source_hash':'3'*64,
          'source_hash':'4'*64,'original_identity':original,'current_identity':current,
          'stage_id':'native_full_sft','recipe_binding':json.dumps(recipe)}
    coordinator={'fingerprint':'f','runtime_code_sha256':direct['runtime_code_sha256'],'versions':[0],
                 'last_seq':[{}],'queued':[{}],'manifest':{'identity':original},'identity_transitions':{}}
    components={'coordinator':coordinator}
    for i in range(4):
        coordinator['identity_transitions'][str(i)]=[{**base,'worker':i,'fingerprint':'f'}]
        components[f'worker-{i}']={'fingerprint':'f','source_hash':'4'*64,'runtime_code_sha256':direct['runtime_code_sha256'],
          'resident_generation':None,'recipe_binding':json.dumps(recipe),'source_receipt_sha256':[roles['native_graphs']['sha256'],roles['held_native_graphs']['sha256']],
          'ce_supervision_policy_binding':{'sha256':roles['ce_supervision_policy']['sha256']},'versions':[0],
          'sequences':[0],'pending':{},'awaiting_global':[],'identity_transition_history':[base]}
    assert service.compatibility(components,direct)['compatible']
    for changed in (
        {'identity_transition_history':[]},
        {'identity_transition_history':[{**base,'checkpoint_sha256':'9'*64}]},
        {'source_hash':'5'*64},
    ):
        bad={**components,'worker-0':{**components['worker-0'],**changed}}
        assert not service.compatibility(bad,direct)['compatible']
    bad={**components,'coordinator':{**coordinator,'identity_transitions':{}}}
    assert not service.compatibility(bad,direct)['compatible']
    bad={**components,'coordinator':{**coordinator,'last_seq':[{'0':1}]}}
    assert 'uncheckpointed' in service.compatibility(bad,direct)['reason']


def test_transition_rejects_fixed_argument_changes(tmp_path):
    mod,path,direct,recipe,roles=transitioned(tmp_path);control=json.loads(path.read_text());ref=control['bindings']['worker_argv']
    p=Path(ref['path']);args=json.loads(p.read_text());args.extend(['--seed','other']);p.write_text(json.dumps(args));ref['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();path.write_text(json.dumps(control))
    with pytest.raises(ValueError,match='fixed trainer arguments'):mod.record_from_control(path,'run')


def test_operator_argument_filter_matches_actual_runtime():
    from natlang_neuralese.train.native_diloco import _fixed_trainer_argv,trainer_argv
    mod=load('diloco_direct_recovery')
    original=['--seed','1','--diloco','{}','--native-graphs','/old','--dtype','bf16','--ce-supervision-policy=/old']
    assert mod._fixed_identity_argv(mod._identity_argv(original))==_fixed_trainer_argv(trainer_argv(original))


def test_real_checkpoint_and_coordinator_pins_bind_recorded_transition(tmp_path):
    import torch,uuid
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    from natlang_neuralese.train.diloco.coordinator import Coordinator,initialize_coordinator
    mod,path,direct,recipe,roles=transitioned(tmp_path);service=load('diloco_periodic_recovery');pin=load('diloco_recovery_pin')
    package=tmp_path/'package';package.mkdir();(package/'module.py').write_bytes(b'runtime')
    original={'schema':'natlang.native-diloco-identity/1','parent_checkpoint_sha256':'e'*64,
              'trainer_argv':direct['original_identity_argv'],'diloco':{'run_id':'run'}}
    current={**original,'trainer_argv':direct['current_identity_argv']}
    root=tmp_path/'coordinator';manifest=initialize_coordinator(root,run_id='run',masters={'weight':torch.ones(8)},
        identity=original,workers=4,min_quorum=3,fragment_bytes=1024)
    live=Coordinator(root);components={}
    for i in range(4):
        session=str(uuid.uuid4());live.lease(i,session)
        proof={'schema':'natlang.diloco-source-identity-transition/1','correction_verified':True,
          'checkpoint_sha256':'1'*64,'correction_sha256':'2'*64,'previous_source_hash':'3'*64,
          'source_hash':'4'*64,'original_identity':original,'current_identity':current,
          'stage_id':'native_full_sft','recipe_binding':json.dumps(recipe)}
        assert live.identity_transition(i,session,{**proof,'worker':i,'fingerprint':manifest['fingerprint']})['recorded']
        training={'schema':TRAINING_SCHEMA,'global_step':10,'optimizer':{'state':{'p':1}},'rng':{'seed':1},
          'source_receipts':[{'sha256':roles['native_graphs']['sha256']},{'sha256':roles['held_native_graphs']['sha256']}],
          'source_hash':'4'*64,'exposure_counts':{'row':1},
          'options':{'recipe_binding':json.dumps(recipe),'ce_supervision_policy_binding':{'sha256':direct['ce_sha256']},
                     'diloco_identity_transition_history':[proof]},
          'diloco':{'worker':i,'fingerprint':manifest['fingerprint'],'versions':[0],'sequences':[0],'pending':{}}}
        checkpoint=tmp_path/f'worker-{i}.pt';torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':{'source_architecture':'lfm3'},'training':training},checkpoint)
        components[f'worker-{i}']=pin.worker_pin(checkpoint,tmp_path/'pins'/f'{i}.pt',i,manifest['fingerprint'],package)
    components['coordinator']=pin.coordinator_pin(root,tmp_path/'pins/coordinator.tar',package)
    assert service.compatibility(components,direct)['compatible']
    assert len(components['coordinator']['identity_transitions'])==4
    assert all(components[f'worker-{i}']['identity_transition_history'] for i in range(4))
