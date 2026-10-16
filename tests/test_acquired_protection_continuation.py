"""No numeric re-render is needed when exact admitted rows remain unaffected."""
import json
import shutil
from pathlib import Path
import pytest
from natlang_neuralese.common.hashing import sha256_file_hex, canonical_json_sha256_hex
from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as teacher
from scripts.neuralese_data.continue_acquired_protection import certify, verify_certificate
from scripts.neuralese_data.close_acquired_source_protection import project
from test_acquired_connected_protection import export


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value)+'\n');return path


def fixture(tmp_path,monkeypatch,groups=('SAFE',)):
    review=tmp_path/'reviews';review.mkdir();(review/'holds.jsonl').write_text('')
    monkeypatch.setattr(teacher.source_review,'REGISTRY',review)
    monkeypatch.setattr(teacher.source_review,'pending_unpinned_source_groups',lambda:set())
    old=tmp_path/'old'/'protection-manifest.json';files=[]
    for relative,values in [('held-union/seed.jsonl',['H']),('quality-groups.jsonl',['Q'])]:
        path=write(old.parent/relative,{'source_groups':values})
        files.append({'path':relative,'sha256':sha256_file_hex(path),'bytes':path.stat().st_size})
    binding=export(tmp_path/'sources'/'D12',[(['A','B'],'train'),(['B','H'],'test'),(['SAFE'],'train')])
    write(old,{'schema':'natlang.training-protection-projection/1','files':files,'held_groups':1,
        'quality_groups':1,'source_review_sha256':sha256_file_hex(review/'holds.jsonl'),
        'normalized_export_bindings':[binding]})
    new=project(old,tmp_path/'new')
    root=tmp_path/'cohort';out=root/'conversion';outputs=[]
    for split in ('train','test'):
        path=out/split/'native-graphs.jsonl.zst';path.parent.mkdir(parents=True);path.write_bytes(b'opaque exact numeric bytes '+split.encode())
        outputs.append({'path':str(path),'sha256':sha256_file_hex(path),'bytes':path.stat().st_size,'rows':int(split=='train')})
    resume=write(out/'resume-state.json',{});omission=write(out/'omissions.jsonl',{})
    proof=teacher.current_protection(old)['proof']
    receipt=write(out/'receipt.json',{'schema':'natlang.acquired-full-session-graphs/1','protection':proof,
        'decision_holds':[],'outputs':outputs,'resume_state_sha256':sha256_file_hex(resume),
        'omissions_sha256':sha256_file_hex(omission)})
    rows=root/'audit'/'row-proof.jsonl'
    write(rows,{'model':'mellum','id':'real-row','graph_sha256':'a'*64,'split':'train','source_groups':list(groups)})
    audit=write(rows.parent/'audit.json',{'schema':'natlang.acquired-session-admission-audit/1','status':'passed',
        'oracle_verified':False,'train_held_group_intersections':0,'protection':proof,'decision_holds':[],
        'pending_source_groups_sha256':canonical_json_sha256_hex([]),'row_proof_sha256':sha256_file_hex(rows),
        'conversions':[{'model':'mellum','receipt':{'path':str(receipt),'sha256':sha256_file_hex(receipt)},
            'outputs':[{**value,'split':Path(value['path']).parent.name} for value in outputs]}]})
    return audit,old,new,root/'partition'/'certificate.json'


def test_actual_closed_sources_allow_unchanged_numeric_bytes_and_portable_receipt(tmp_path,monkeypatch):
    audit,old,new,output=fixture(tmp_path,monkeypatch)
    result=certify(audit,old,new,output,decision_paths=[])
    assert result['held_groups_added']==2 and result['row_counts']=={'mellum':{'train':1,'test':0}}
    assert result['eligibility_only'] and not result['numeric_rerendered'] and not result['exposure_granted']
    # Aggregate bytes are deliberately not JSON; the verifier never reparses tokens.
    assert verify_certificate(output)['passed']
    with pytest.raises(ValueError,match='immutable'):certify(audit,old,new,output,decision_paths=[])
    moved=tmp_path/'moved';shutil.copytree(audit.parent.parent,moved);shutil.rmtree(audit.parent.parent)
    assert verify_certificate(moved/'partition'/'certificate.json')['passed']


@pytest.mark.parametrize('change',['affected','released','quality','pending','aggregate','receipt','row-proof','decisions','closure'])
def test_continuation_rejects_affected_or_tampered_eligibility(tmp_path,monkeypatch,change):
    audit,old,new,output=fixture(tmp_path,monkeypatch,groups=('A',) if change=='affected' else ('SAFE',))
    if change in {'released','quality'}:
        manifest=json.loads(new.read_text())
        relative='held-union/seed.jsonl' if change=='released' else 'quality-groups.jsonl'
        # Remove all inherited held files too, so this really releases protection.
        if change=='released':
            for item in manifest['files']:
                if item['path'].startswith('held-union/'):
                    path=write(new.parent/item['path'],{'source_groups':[]});item.update(sha256=sha256_file_hex(path),bytes=path.stat().st_size)
            manifest['held_groups']=0
        else:
            item=next(v for v in manifest['files'] if v['path']==relative)
            path=write(new.parent/relative,{'source_groups':['Q','SAFE']});item.update(sha256=sha256_file_hex(path),bytes=path.stat().st_size)
            manifest['quality_groups']=2
        write(new,manifest)
    elif change=='pending':monkeypatch.setattr(teacher.source_review,'pending_unpinned_source_groups',lambda:{'SAFE'})
    elif change=='aggregate':(audit.parent.parent/'conversion/train/native-graphs.jsonl.zst').write_bytes(b'tamper')
    elif change=='receipt':(audit.parent.parent/'conversion/receipt.json').write_text('{}')
    elif change=='row-proof':(audit.parent/'row-proof.jsonl').write_text('{}')
    elif change=='closure':
        source=new.parent.parent/'sources/D12/one/full-trajectories.jsonl.zst';source.write_bytes(b'tampered original source')
    decisions=[]
    if change=='decisions':decisions=[write(tmp_path/'new-holds.jsonl',{'id':'held','reason':'actual reviewed failure'})]
    with pytest.raises(ValueError):certify(audit,old,new,output,decision_paths=decisions)
    assert not output.exists()


def test_published_certificate_rejects_numeric_dependency_changes(tmp_path,monkeypatch):
    audit,old,new,output=fixture(tmp_path,monkeypatch)
    certify(audit,old,new,output,decision_paths=[])
    (audit.parent.parent/'conversion/train/native-graphs.jsonl.zst').write_bytes(b'tamper')
    with pytest.raises(ValueError,match='changed'):verify_certificate(output)
