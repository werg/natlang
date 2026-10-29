import argparse,hashlib,json,os
from pathlib import Path
import pytest
from scripts.coreweave.prepare_fresh_first_outer import prepare


def test_first_outer_builder_exact_receipts_and_common_identity(tmp_path):
    def put(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    owner=put('owner',{'fresh_checkpoint_sha256':'d'*64,'accepted':{'accepted':True,'pid':12,'boot_id':'boot','process_start_ticks':'34'}})
    base='51bacc47b54e937684e8e757dde12e1b01719c8f24e1d52d8f185b236753648f';size=12482163595;train='9d93141df15287e147afb7fa32b499ef4d59d54cdfe5de1755d24b879a821362';held='5053ab2b6754f2998294215e7892b5c18443d973f55e6c04239803ffac7b5e7f';peers=[]
    for rank in range(4):
        sha='a'*64 if rank==0 else base;step=3 if rank==0 else 2;parent={'path':f'/work/peer-{rank}.pt','sha256':sha,'bytes':size,'step':step}
        pin=put(f'pin-{rank}',{**parent,'source_sha256':train,'owner_receipt':owner['path']});private=put(f'private-{rank}',{'private':True,'identity':{'sha256':sha},'independent_verification':{'sha256':sha,'bytes':size,'readback_location':'sandbox'},'verified_at':1});meta=put(f'meta-{rank}',{'checkpoint_sha256':sha,'step':step,'train_sha256':train,'held_sha256':held,'source_hash':'b'*64,'full_state_verified':True,'optimizer_verified':True,'rng_verified':True,'exposure_verified':True});presence=put(f'presence-{rank}',{'sha256':sha,'bytes':size,'owner_access':True,'private_object_present':True,'observed_at':2});peers.append({'worker':rank,'parent':parent,'pin_receipt':pin,'private_receipt':private,'full_state_receipt':meta,'owner_presence_receipt':presence})
    baseline=put('baseline-pin',{'sha256':base,'step':2,'owner_receipt':owner['path']})
    witness=put('witness',{'schema':'natlang.fresh-writer-lineage/1','continuity_verified':True,'owner':owner,'baseline_pin_receipt':baseline,'baseline_sha256':base,'latest_sha256':peers[0]['parent']['sha256'],'initial_checkpoint_sha256':'d'*64,'pid':12,'boot_id':'boot','process_start_ticks':'34'})
    inv=put('inventory',{'schema':'natlang.fresh-first-outer-inventory/1','seed':{'sha256':base,'bytes':size},'peers':peers,'peer1_lineage_witness':witness})
    recipe=Path(os.environ.get('NATLANG_FRESH_RECIPE_FIXTURE','/nonexistent/fresh-recipe.json'))
    if not recipe.is_file():pytest.skip('production recipe fixture not available')
    args=argparse.Namespace(inventory=inv,recipe={'path':str(recipe),'sha256':hashlib.sha256(recipe.read_bytes()).hexdigest()},out=tmp_path/'out',seed_logical='/work/fresh/seed.pt',worker_out='/work/fresh/learner',coordinator_root='/work/fresh/coordinator',coordinator_url='https://example.invalid',run_id='test',local_steps=8)
    result=prepare(args);assert result['quorum']==3 and result['settings']['local_steps']==8 and len(result['peer_declarations'])==4
    args.local_steps=32;args.out=tmp_path/'out32';result32=prepare(args);assert result32['settings']['local_steps']==32
    args.local_steps=0;args.out=tmp_path/'invalid';
    with pytest.raises(ValueError):prepare(args)
    args.local_steps=8
    candidate=os.environ.get('NATLANG_CORRECTED_CE_FIXTURE')
    if candidate:
        raw=Path(candidate).read_bytes();args.ce_policy={'path':candidate,'sha256':hashlib.sha256(raw).hexdigest()};args.out=tmp_path/'corrected';corrected=prepare(args)
        newrecipe=json.loads(Path(corrected['recipe']['path']).read_text());assert newrecipe['input_bindings']['ce_supervision_policy']['sha256']==args.ce_policy['sha256']
        args.ce_policy=None
    declarations=[json.loads(Path(x['path']).read_text()) for x in result['peer_declarations']]
    assert declarations[0]['rng_fork_seed'] is None and len({x['rng_fork_seed'] for x in declarations[1:]})==3
    assert all(x['owner_get']=='pending' for x in result['peers'])
    argv=json.loads(Path(result['worker_args']['path']).read_text());assert argv[argv.index('--resident-data-directory')+1]=='/work/fresh/learner/resident-data';assert argv[argv.index('--muon-rounding')+1]=='kahan';assert argv[argv.index('--weight-ema')+1]=='off'
    from natlang_neuralese.train.native_state_sft import build_parser,validate_static_args
    validate_static_args(build_parser().parse_args(argv),check_output=False)
    badargs=list(argv);badargs[badargs.index('--weight-ema')+1]=json.dumps({'decay':.99,'warmup':0,'eval':'raw'})
    with pytest.raises(ValueError,match='DiLoCo learners do not keep a local weight EMA'):
        validate_static_args(build_parser().parse_args(badargs),check_output=False)
    init=json.loads(Path(result['coordinator_init_args']['path']).read_text());assert init[init.index('--grace-seconds')+1]=='30'
    plan=json.loads(Path(result['worker_plan']['path']).read_text());assert result['worker_plan_target']=='/work/fresh/learner/recipe-plan.json'
    assert plan['stage_inputs']['native_full_sft']['native_graphs']==plan['stage_inputs']['native_full_sft']['foundation_native_graphs'] and plan['code']
    # Launch may use the live, fully decoded newest rank-0 pin while its
    # immutable seed remains independently owner-readback and present.
    fallback=put('fallback',{'private':True,'identity':{'sha256':base},'independent_verification':{'sha256':base,'bytes':size,'readback_location':'owner'},'verified_at':1})
    fallback_presence=put('fallback-presence',{'sha256':base,'bytes':size,'owner_access':True,'private_object_present':True,'observed_at':2,'receipt_sha256':fallback['sha256']})
    live=json.loads(Path(inv['path']).read_text());live['peers'][0].update(parent_custody='live-with-verified-seed-fallback',private_receipt=fallback,owner_presence_receipt=fallback_presence)
    args.inventory=put('live-inventory',live);args.out=tmp_path/'liveout';r=prepare(args)
    assert r['peers'][0]['parent_private_readback']=='pending' and r['peers'][0]['fallback_sha256']==base
    bad_fallback=json.loads(Path(fallback['path']).read_text());bad_fallback['independent_verification']['readback_location']='sandbox';live['peers'][0]['private_receipt']=put('bad-fallback',bad_fallback)
    args.inventory=put('bad-live-inventory',live);args.out=tmp_path/'badliveout'
    with pytest.raises(ValueError,match='private full readback missing'):prepare(args)
    invdoc=json.loads(Path(inv['path']).read_text());invdoc['peers'][2]['parent']['sha256']='c'*64;bad=put('bad-inventory',invdoc);args.inventory=bad;args.out=tmp_path/'badout'
    with pytest.raises(ValueError,match='pin receipt differs'):prepare(args)

    witnessdoc=json.loads(Path(witness['path']).read_text());witnessdoc['pid']=13;badw=put('bad-witness',witnessdoc)
    invdoc=json.loads(Path(inv['path']).read_text());invdoc['peer1_lineage_witness']=badw;args.inventory=put('bad-witness-inventory',invdoc);args.out=tmp_path/'bad-witness-out'
    with pytest.raises(ValueError,match='continuous fresh-writer custody'):prepare(args)
