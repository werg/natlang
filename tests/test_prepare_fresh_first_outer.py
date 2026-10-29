import argparse,hashlib,json,os
from pathlib import Path
import pytest
from scripts.coreweave.prepare_fresh_first_outer import prepare


def test_first_outer_builder_exact_receipts_and_common_identity(tmp_path):
    def put(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    base='51bacc47b54e937684e8e757dde12e1b01719c8f24e1d52d8f185b236753648f';size=12482163595;train='9d93141df15287e147afb7fa32b499ef4d59d54cdfe5de1755d24b879a821362';held='5053ab2b6754f2998294215e7892b5c18443d973f55e6c04239803ffac7b5e7f';peers=[]
    for rank in range(4):
        sha='a'*64 if rank==0 else base;step=3 if rank==0 else 2;parent={'path':f'/work/peer-{rank}.pt','sha256':sha,'bytes':size,'step':step}
        pin=put(f'pin-{rank}',{**parent,'source_sha256':train});private=put(f'private-{rank}',{'private':True,'identity':{'sha256':sha},'independent_verification':{'sha256':sha,'bytes':size,'readback_location':'sandbox'},'verified_at':1});meta=put(f'meta-{rank}',{'checkpoint_sha256':sha,'step':step,'train_sha256':train,'held_sha256':held,'source_hash':'b'*64,'full_state_verified':True,'optimizer_verified':True,'rng_verified':True,'exposure_verified':True});presence=put(f'presence-{rank}',{'sha256':sha,'bytes':size,'owner_access':True,'private_object_present':True,'observed_at':2});peers.append({'worker':rank,'parent':parent,'pin_receipt':pin,'private_receipt':private,'full_state_receipt':meta,'owner_presence_receipt':presence})
    inv=put('inventory',{'schema':'natlang.fresh-first-outer-inventory/1','seed':{'sha256':base,'bytes':size},'peers':peers})
    recipe=Path(os.environ.get('NATLANG_FRESH_RECIPE_FIXTURE','/nonexistent/fresh-recipe.json'))
    if not recipe.is_file():pytest.skip('production recipe fixture not available')
    args=argparse.Namespace(inventory=inv,recipe={'path':str(recipe),'sha256':hashlib.sha256(recipe.read_bytes()).hexdigest()},out=tmp_path/'out',seed_logical='/work/fresh/seed.pt',worker_out='/work/fresh/learner',coordinator_root='/work/fresh/coordinator',coordinator_url='https://example.invalid',run_id='test')
    result=prepare(args);assert result['quorum']==3 and result['settings']['local_steps']==8 and len(result['peer_declarations'])==4
    declarations=[json.loads(Path(x['path']).read_text()) for x in result['peer_declarations']]
    assert declarations[0]['rng_fork_seed'] is None and len({x['rng_fork_seed'] for x in declarations[1:]})==3
    assert all(x['owner_get']=='pending' for x in result['peers'])
    invdoc=json.loads(Path(inv['path']).read_text());invdoc['peers'][2]['parent']['sha256']='c'*64;bad=put('bad-inventory',invdoc);args.inventory=bad;args.out=tmp_path/'badout'
    with pytest.raises(ValueError,match='pin receipt differs'):prepare(args)
