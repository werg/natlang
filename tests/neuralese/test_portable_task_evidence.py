import copy, hashlib, json, shutil
from pathlib import Path
import pytest
from natlang_neuralese.eval import portable_task_evidence as portable
from natlang_neuralese.eval.task_packet import TASK_OUTCOMES

def write(path,value):path.write_text(json.dumps(value)+'\n')
def hashfile(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def fixture(tmp_path):
    packet=tmp_path/'packet.jsonl';packet.write_text('\n'.join(json.dumps({'id':x}) for x in ('a','b','c'))+'\n')
    root=tmp_path/'source';root.mkdir();spec=root/'spec.json';write(spec,{'packet':{'path':'/original/cases.jsonl','sha256':hashfile(packet)},
        'cases':[2,0],'neuralese_server':True,'prompt_rendering':'full'})
    identity={'checkpoint_sha256':'a'*64,'execution':'source-v1'}
    out=root/'evaluation';out.mkdir()
    run=out/'task-eval/step-0000067';run.mkdir(parents=True)
    counts={k:0 for k in TASK_OUTCOMES};counts['task_success']=1;counts['semantic_mismatch']=1
    summary={'run':'/work/evaluation/task-eval/step-0000067','step':67,'cases':2,'case_indices':[2,0], 'packet_sha256':hashfile(packet),
        'outcome_counts':counts,'task_successes':1,'eligible':True,'complete':True,'prompt_rendering':'full'}
    score={**summary,'cases_detail':[{'id':'c','outcome':'task_success'},{'id':'a','outcome':'semantic_mismatch'}]}
    write(run/'task-score.json',score)
    policy={'schema':'natlang.rl-policy-binding/2','checkpoint_sha256':identity['checkpoint_sha256'],
            'native_identity':identity,'inference_source':{'schema':'natlang.native-inference-source/1',
                'training_checkpoint_sha256':identity['checkpoint_sha256'],'step':67}}
    contract={'checkpoint_sha256':identity['checkpoint_sha256'],'spec_sha256':hashfile(spec),'step':67,
        'endpoint':'http://actor','model':'observed','native_identity':policy}
    write(out/'intent.json',contract)
    info={'rl_policy_binding':policy,'native_state':{'checkpoint_sha256':identity['checkpoint_sha256'],'enabled':True},'prompt_rendering':'full'}
    write(out/'completed.json',{'contract':contract,'actor_info':info,
        'status':'finished','summary':summary,'training_admission':False,'separate_held_outputs':True})
    write(root/'view.json',{'view':True});write(root/'runtime.json',{'runtime':True})
    write(root/'transport.json',{'source':{'sha256':identity['checkpoint_sha256']},'view':{'sha256':'d'*64},
        'view_receipt':{'sha256':hashfile(root/'view.json')},'scope':'diagnostic-local-pin'})
    write(root/'launch.json',{'checkpoint_sha256':identity['checkpoint_sha256'],'task_spec_sha256':hashfile(spec),
        'step':67,'container_id':'observed','image_id':'sha256:image',
        'runtime_receipt_sha256':hashfile(root/'runtime.json'),'transport_sha256':hashfile(root/'transport.json'),'inference_view_sha256':'d'*64})
    write(root/'actor-info.json',info)
    return out,spec,packet,identity

def seal(source,spec,packet,out):
    root=source.parent
    return portable.seal(source,spec,packet,out,source_root=root,container_root='/work',
        transport_path=root/'transport.json',view_receipt_path=root/'view.json',runtime_closure_path=root/'runtime.json')

def test_sealed_evidence_relocates_and_rejects_tampering_and_wrong_actor(tmp_path):
    source,spec,packet,identity=fixture(tmp_path)
    manifest=seal(source,spec,packet,tmp_path/'sealed')
    moved=tmp_path/'copied';shutil.copytree(manifest.parent,moved)
    ref={'path':str(moved/'evidence.json'),'sha256':hashfile(moved/'evidence.json')}
    score=portable.validate(ref,identity=identity,packet_path=packet)['score']
    assert score['run']=='/work/evaluation/task-eval/step-0000067' and score['task_successes']==1
    with pytest.raises(ValueError,match='actor'):
        portable.validate(ref,identity={**identity,'checkpoint_sha256':'b'*64},packet_path=packet)
    (moved/'task-score.json').write_bytes((moved/'task-score.json').read_bytes()+b' ')
    with pytest.raises(ValueError,match='binding'):
        portable.validate(ref,identity=identity,packet_path=packet)

def test_seal_requires_original_score_and_packet_case_identity(tmp_path):
    source,spec,packet,identity=fixture(tmp_path)
    scorepath=source/'task-eval/step-0000067/task-score.json'
    score=json.loads(scorepath.read_text());score['cases_detail'][0]['id']='a';write(scorepath,score)
    with pytest.raises(ValueError,match='packet cases'):
        seal(source,spec,packet,tmp_path/'bad')
    packet.write_text(packet.read_text()+'{}\n')
    with pytest.raises(ValueError,match='packet or case'):
        seal(source,spec,packet,tmp_path/'bad2')
    scorepath.unlink()
    with pytest.raises(FileNotFoundError):
        seal(source,spec,packet,tmp_path/'bad3')
