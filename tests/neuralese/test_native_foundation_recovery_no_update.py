import copy
import json
from pathlib import Path

import torch

from test_native_model import tiny_source
from test_native_state_sft import graph, authenticated_pair, source_manifest
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.model.native_model import load_native_model
from natlang_neuralese.train import native_state_sft as trainer


def test_completed_in_place_recovery_persists_declaration_with_zero_optimizer_updates(tmp_path):
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    source_dir=tmp_path/'source';source_dir.mkdir()
    source=source_manifest(source_dir,[],[authenticated_pair('old',roles=roles.ids,fingerprint=config['tokenizer_fingerprint'])])
    value=json.loads(source.read_text())
    value['sources']=[item for item in value['sources'] if item['kind']=='semantic-pairs']
    source.write_text(json.dumps(value))
    held=tmp_path/'held.jsonl'
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=config['tokenizer_fingerprint']))+'\n')
    out=tmp_path/'learner'
    args=['--base',str(base),'--device','cpu','--dtype','float32','--rank','4',
          '--native-graphs',str(source),'--foundation-native-graphs',str(source),
          '--communication-foundation','--foundation-readiness-policy','on-shared-request',
          '--held-native-graphs',str(held),'--stage-id','full','--epochs','1',
          '--eval-every','100','--out',str(out),'--forward-atol','1e-4','--gradient-atol','1e-5']
    assert trainer.main(args)==0
    slot=out/'checkpoint.pt'
    original=torch.load(slot,weights_only=False)
    checkpoint={'sha256':sha256_file_hex(slot),'bytes':slot.stat().st_size}
    def write(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value))
        return {'path':str(path),'sha256':sha256_file_hex(path)}
    historical_recipe=write('historical-recipe.json',{'input_bindings':{'foundation':{'path':str(source),'sha256':sha256_file_hex(source)}},
        'stages':[{'id':'full','kind':'native_state_sft','inputs':{'foundation_native_graphs':'foundation'},
                   'parameters':{'communication_foundation':True,'epochs':1.}}]})
    historical_metadata=write('historical-metadata.json',{'checkpoint_identity':checkpoint,
        'model_config':original['model_config'],'training':{'global_step':original['training']['global_step'],
            'options':original['training']['options']}})
    verification=write('verification.json',{'passed':True,'checkpoint':checkpoint,'stage_id':'full',
        'architecture':config['architecture'],'global_step':original['training']['global_step'],
        'optimizer_present':True,'rng_present':True})
    proof=write('recovery.json',{'schema':'natlang.native-foundation-declaration-recovery/1','stage_id':'full',
        'historical_recipe':historical_recipe,'historical_training_metadata':historical_metadata,
        'historical_checkpoint_verification':verification,
        'foundation_sources':{'path':str(source),'sha256':sha256_file_hex(source)},'reason':'Recover declaration after unavailable header.'})
    # Simulate the metadata of a real same-lineage recovery continuation. Its
    # numeric state and already earned complete row counters remain untouched.
    parent=copy.deepcopy(original)
    for frame in [parent['training'],*parent['training']['sft_progress'].values()]:
        frame['options'].update(communication_foundation=False,foundation_native_graphs=None)
        frame['options']['native_snapshot_ancestors']=[checkpoint['sha256']]
        frame['completed_foundations']=[]
    torch.save(parent,slot)
    assert trainer.main([*args,'--foundation-declaration-recovery',proof['path']])==0
    saved=torch.load(slot,weights_only=False)
    assert saved['training']['global_step']==parent['training']['global_step']
    assert saved['training']['exposure_counts']==parent['training']['exposure_counts']
    from natlang_neuralese.train.native_continuation import collect_sft_progress
    assert collect_sft_progress(saved['training'])['full']['required_visits']==collect_sft_progress(parent['training'])['full']['required_visits']
    history=saved['training']['options'].get('foundation_declaration_recovery_history')
    assert history and len(history)==1 and history[0]['new_exposure']==0
    assert saved['training']['options']['communication_foundation'] is True
    expected_header={'path':str(source.resolve()),'sha256':sha256_file_hex(source),
        'bytes':source.stat().st_size}
    from natlang_neuralese.train.native_foundation_recovery import native_graph_bindings
    expected_header['dependencies']=[item for item in native_graph_bindings(expected_header)
        if item['path']!=expected_header['path']]
    assert saved['training']['options']['foundation_native_graphs_binding'] == expected_header
    assert saved['training']['options']['foundation_declaration_recovery_binding'] == {
        'path':str(Path(proof['path']).resolve()),'sha256':sha256_file_hex(proof['path']),
        'bytes':Path(proof['path']).stat().st_size}
    def equal(left,right):
        if isinstance(left,torch.Tensor):assert torch.equal(left,right)
        elif isinstance(left,dict):
            assert left.keys()==right.keys()
            for key in left:equal(left[key],right[key])
        elif isinstance(left,(list,tuple)):
            assert len(left)==len(right)
            for a,b in zip(left,right):equal(a,b)
        else:assert left==right
    for group in ('backbone_state','interface_state'):
        equal(saved[group],parent[group])
    equal(saved['training']['optimizer'],parent['training']['optimizer'])
    equal(saved['training']['rng'],parent['training']['rng'])
