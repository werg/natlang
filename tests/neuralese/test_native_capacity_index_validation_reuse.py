"""The normal strict verifier runs once per unchanged complete witness."""
import copy,json,os
from pathlib import Path
import pytest
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.common import row_index
from test_native_capacity_index_witness_union import witnesses


def test_29_before_use_checks_load_and_digest_each_actual_index_once(tmp_path,monkeypatch):
    values=witnesses(tmp_path,monkeypatch);loaded=[];original=row_index.load
    def load(path):loaded.append(str(path));return original(path)
    monkeypatch.setattr(row_index,'load',load)
    validation=resident.IndexWitnessValidation()
    assert resident.check_index_witness_union([copy.deepcopy(values)for _ in range(29)],validation)==len(values)
    for _ in range(29):validation(copy.deepcopy(values))
    assert len(loaded)==len(values)and validation.strict_witnesses==len(values)
    assert validation.strict_runs==1 and validation.unchanged_uses==29*len(values)


@pytest.mark.parametrize('field',['metadata','key','cache','bytes'])
def test_distinct_conflicting_same_cache_witness_is_reverified_not_reused(tmp_path,monkeypatch,field):
    values=witnesses(tmp_path,monkeypatch);validation=resident.IndexWitnessValidation();validation(values)
    bad=copy.deepcopy(values[0])
    if field=='metadata':bad['metadata_sha256']='0'*64
    elif field=='key':bad['key']['role_ids']=[123]
    elif field=='cache':bad['cache']['sha256']='0'*64
    else:bad['cache']['bytes']=Path(bad['cache']['path']).stat().st_size+1
    with pytest.raises(ValueError):validation([bad])
    assert len(validation.completed)==len(values)


@pytest.mark.parametrize('mutation',['mtime','inode','content','missing','cache-root'])
def test_actual_file_or_cache_root_changes_cannot_reuse_the_receipt(tmp_path,monkeypatch,mutation):
    values=witnesses(tmp_path,monkeypatch);calls=[]
    def strict(items):calls.append(len(items));resident._check_indexes(items)
    validation=resident.IndexWitnessValidation(strict);validation(values)
    path=Path(values[0]['cache']['path'])
    if mutation=='mtime':
        st=path.stat();os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns+1_000_000))
    elif mutation=='inode':
        temporary=path.with_suffix('.new');temporary.write_bytes(path.read_bytes());temporary.replace(path)
    elif mutation=='content':path.write_bytes(path.read_bytes()+b'changed')
    elif mutation=='missing':path.unlink()
    else:monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE',str(tmp_path/'other-cache'))
    if mutation in('mtime','inode'):
        validation(values);assert calls==[len(values),1]
    else:
        with pytest.raises((ValueError,FileNotFoundError)):validation(values)
    assert validation.unchanged_uses==(0 if mutation in('missing','cache-root')else len(values)-1)


def test_file_changed_during_strict_verification_never_publishes_receipt(tmp_path,monkeypatch):
    values=witnesses(tmp_path,monkeypatch)
    def strict(items):
        resident._check_indexes(items)
        path=Path(items[0]['cache']['path']);st=path.stat();os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns+1_000_000))
    validation=resident.IndexWitnessValidation(strict)
    with pytest.raises(ValueError,match='during strict validation'):validation(values)
    assert not validation.completed and validation.strict_runs==0


def test_matching_optional_byte_count_is_distinct_valid_whole_witness(tmp_path,monkeypatch):
    values=witnesses(tmp_path,monkeypatch);validation=resident.IndexWitnessValidation();validation(values)
    extra=copy.deepcopy(values[0]);extra['cache']['bytes']=Path(extra['cache']['path']).stat().st_size
    validation([extra]);assert validation.strict_runs==2 and len(validation.completed)==len(values)+1


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_actual_independent_capacity_cases_keep_numerics_and_state_with_one_index_check(tmp_path,monkeypatch,family):
    import torch
    from natlang_neuralese.eval import native_memory as memory
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from test_native_memory_capacity import prepared_fixture,assert_exact
    from test_native_state_sft import graph
    model,optimizer,train,roles,config,args=prepared_fixture(tmp_path,monkeypatch,family=family)
    source_path=tmp_path/'three-actual-cases.jsonl'
    source_path.write_text(''.join(json.dumps(graph('actual-'+str(i)))+'\n'for i in range(3)))
    source=NativeGraphSource(source_path,'tokenizer-sha',required_split='train',role_ids=roles.ids)
    template=json.loads(resident.bound(json.loads(args.memory_qualification)).read_text())
    template.update(source=resident.binding(source_path),indexes=resident._source_witness(source))
    requests=[]
    for index,entry in enumerate(source.entries):
        request={**template,'cases':[{'index':index,'id':entry['id'],'binding':entry['binding']}]}
        path=tmp_path/f'capacity-request-{index}.json';path.write_text(json.dumps(request));requests.append(resident.binding(path))
    args.memory_qualification=json.dumps(requests)
    weights=copy.deepcopy(model.state_dict());state=copy.deepcopy(optimizer.state_dict());rng=torch.get_rng_state().clone()
    calls=[];strict=resident._check_indexes
    def observed(witnesses):calls.append(copy.deepcopy(witnesses));return strict(witnesses)
    monkeypatch.setattr(resident,'_check_indexes',observed)
    result=memory.run_prepared(model,optimizer,train,roles,config,args,backend=None,checkpoint_sha256='a'*64)
    assert result['passed']and len(result['checks'])==3 and len(calls)==1
    assert result['index_witness_validation']=={'strict_runs':1,'strict_witnesses':len(template['indexes']),
        'unchanged_uses':3*len(template['indexes'])}
    assert [check['capacity']['cases'][0]['id']for check in result['checks']]==['actual-'+str(i)for i in range(3)]
    assert all(check['capacity']['passed']and check['optimizer_updates']==0 for check in result['checks'])
    assert_exact(weights,model.state_dict());assert_exact(state,optimizer.state_dict());assert torch.equal(rng,torch.get_rng_state())
