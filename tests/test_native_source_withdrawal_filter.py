import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest
from scripts.neuralese_data.withdraw_native_graph_targets import filter_source,continue_filtered_index
from natlang_neuralese.common.hashing import sha256_file_hex,canonical_json_sha256_hex as digest


def fixture(tmp_path):
    from natlang_neuralese.data.native_graph_schema import parse_graph, communication_classification
    from natlang_neuralese.data.native_sources import _row_bindings
    rows=[{'schema':'natlang.neuralese-native-graph/1','id':key,'split':'train',
           'source_groups':['source:'+key],'tokenizer_fingerprint':'pin',
           'nodes':[{'name':'text','token_ids':[1,2],'target_weights':[0,1]}],
           'conversion':{'role_interpretation':'ordinary-text'}}for key in ('keep','bad','last')]
    source=tmp_path/'source.jsonl';raw=[json.dumps(row).encode()+b'\n'for row in rows];source.write_bytes(b''.join(raw))
    parsed=[parse_graph(row,tokenizer_fingerprint='pin') for row in rows]
    bindings=_row_bindings(parsed)
    targets=[{'id':'bad','binding':bindings['bad'],'communication':False,'source_sha256':sha256_file_hex(source)}]
    key={'source_sha256':sha256_file_hex(source),'kind':'native-graphs','split':'train','policy_sha256':'exact-old-policy','tokenizer':'pin','role_ids':[1,2]}
    entries=[{'id':r.id,'binding':bindings[r.id],**communication_classification(r),'metadata':['untouched',i]}
             for i,r in enumerate(parsed)]
    index=tmp_path/'old.pkl';value={'schema':'natlang.row-index/1','key':key,'entries':entries,'metadata_sha256':digest(entries)};index.write_bytes(pickle.dumps(value))
    return source,raw,targets,index,value


def test_raw_filter_preserves_every_remaining_byte_and_full_index_value(tmp_path):
    source,raw,targets,index,old=fixture(tmp_path);before=source.read_bytes();original=index.read_bytes()
    proof=filter_source(source,tmp_path/'filtered.jsonl',targets,source_sha256=sha256_file_hex(source))
    assert (tmp_path/'filtered.jsonl').read_bytes()==raw[0]+raw[2]and source.read_bytes()==before
    out=continue_filtered_index(index,proof,index_sha256=sha256_file_hex(index),metadata_sha256=old['metadata_sha256'])
    new=pickle.loads(Path(out['path']).read_bytes())
    assert new['entries']==[old['entries'][0],old['entries'][2]]and index.read_bytes()==original
    assert new['key']=={**old['key'],'source_sha256':proof['output']['sha256']}
    assert continue_filtered_index(index,proof,index_sha256=sha256_file_hex(index),metadata_sha256=old['metadata_sha256'])==out
    assert filter_source(source,tmp_path/'filtered.jsonl',targets,source_sha256=sha256_file_hex(source))==proof


@pytest.mark.parametrize('fault',['missing','duplicate','wrong_source','communication'])
def test_filter_fails_closed_and_never_publishes_partial(tmp_path,fault):
    source,raw,targets,index,old=fixture(tmp_path)
    if fault=='missing':targets[0]['id']='absent'
    elif fault=='duplicate':targets.append(dict(targets[0]))
    elif fault=='wrong_source':targets[0]['source_sha256']='f'*64
    else:
        row=json.loads(raw[1]);row['conversion']['role_interpretation']='communication';raw[1]=json.dumps(row).encode()+b'\n';source.write_bytes(b''.join(raw));targets[0]['source_sha256']=sha256_file_hex(source)
    output=tmp_path/'filtered.jsonl'
    with pytest.raises(ValueError):filter_source(source,output,targets,source_sha256=sha256_file_hex(source))
    assert not output.exists()


@pytest.mark.parametrize('fault',['binding','class','missing_class','tokenizer','ids','policy_metadata','output','index_sha','existing_conflict'])
def test_index_projection_requires_exact_original_and_retained_custody(tmp_path,fault):
    source,raw,targets,index,old=fixture(tmp_path)
    proof=filter_source(source,tmp_path/'filtered.jsonl',targets,source_sha256=sha256_file_hex(source))
    sha=sha256_file_hex(index);metadata=old['metadata_sha256']
    if fault=='binding':proof['removed'][0]['binding']='x'*64
    elif fault=='class':proof['removed'][0]['communication']=True
    elif fault=='missing_class':del proof['removed'][0]['communication']
    elif fault=='tokenizer':proof['removed'][0]['tokenizer_fingerprint']='other'
    elif fault=='ids':proof['new_ids_sha256']='x'*64
    elif fault=='policy_metadata':metadata='x'*64
    elif fault=='output':Path(proof['output']['path']).write_bytes(b'changed')
    elif fault=='index_sha':sha='x'*64
    else:
        out=continue_filtered_index(index,proof,index_sha256=sha,metadata_sha256=metadata)
        Path(out['path']).write_bytes(pickle.dumps({**old,'key':old['key']}))
    with pytest.raises(ValueError):continue_filtered_index(index,proof,index_sha256=sha,metadata_sha256=metadata)


def test_existing_unrelated_destination_is_never_replaced(tmp_path):
    source,raw,targets,index,old=fixture(tmp_path);output=tmp_path/'filtered.jsonl';output.write_bytes(b'keep existing bytes')
    with pytest.raises(ValueError,match='conflicts'):filter_source(source,output,targets,source_sha256=sha256_file_hex(source))
    assert output.read_bytes()==b'keep existing bytes'


def test_real_compressed_source_readback_and_compressed_output(tmp_path):
    import subprocess
    source,raw,targets,index,old=fixture(tmp_path);compressed=tmp_path/'source.jsonl.zst'
    with compressed.open('wb')as target:subprocess.run(['zstd','-qc',str(source)],stdout=target,check=True)
    targets[0]['source_sha256']=sha256_file_hex(compressed)
    proof=filter_source(compressed,tmp_path/'filtered.jsonl.zst',targets,source_sha256=sha256_file_hex(compressed))
    assert subprocess.check_output(['zstd','-qdc',proof['output']['path']])==raw[0]+raw[2]


def test_index_projection_uses_explicit_actual_transport_path(tmp_path):
    import shutil
    source,raw,targets,index,old=fixture(tmp_path)
    proof=filter_source(source,tmp_path/'original-filtered.jsonl',targets,source_sha256=sha256_file_hex(source))
    local=tmp_path/'actual-sourcehost-copy.jsonl';shutil.copyfile(proof['output']['path'],local);Path(proof['output']['path']).unlink()
    out=continue_filtered_index(index,proof,index_sha256=sha256_file_hex(index),metadata_sha256=old['metadata_sha256'],output_path=local)
    assert out['rows']==2


def test_index_projection_rechecks_decompressed_payload_not_mutable_file_binding(tmp_path):
    source,raw,targets,index,old=fixture(tmp_path)
    proof=filter_source(source,tmp_path/'filtered.jsonl',targets,source_sha256=sha256_file_hex(source))
    output=Path(proof['output']['path']);output.write_bytes(output.read_bytes().replace(b'keep',b'fake'))
    proof['output'].update(sha256=sha256_file_hex(output),bytes=output.stat().st_size)
    with pytest.raises(ValueError,match='retained raw payload'):
        continue_filtered_index(index,proof,index_sha256=sha256_file_hex(index),metadata_sha256=old['metadata_sha256'])


def test_complete_communication_graph_withdrawal_uses_real_source_index(tmp_path, monkeypatch):
    from natlang_neuralese.data.native_sources import NativeGraphSource
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    def ordinary(ident):
        return {'schema':'natlang.neuralese-native-graph/1','id':ident,'split':'train',
                'source_groups':['group:'+ident],'tokenizer_fingerprint':'pin',
                'nodes':[{'name':'text','token_ids':[1,2], 'target_weights':[0,1]}],
                'conversion':{'role_interpretation':'ordinary-text'}}
    graph={'schema':'natlang.neuralese-native-graph/1','id':'conversation','split':'train',
           'source_groups':['comm'],'tokenizer_fingerprint':'pin','nodes':[
               {'name':'writer','token_ids':[1,2,3], 'target_weights':[0,1,1], 'producing':[[1,3]]},
               {'name':'reader','token_ids':[4,5,6], 'target_weights':[0,0,1],
                'imports':[{'at':1,'source':'writer','start':1,'stop':3}]}]}
    rows=[ordinary('first'),graph,ordinary('ordinary-remove'),ordinary('last')]
    # JSONL rows deliberately vary spacing while remaining one physical line each.
    raw=[json.dumps(row, separators=(',', ': ') if i==1 else (', ', ':')).encode()+b'\n'
         for i,row in enumerate(rows)]
    source=tmp_path/'mixed.jsonl';source.write_bytes(b''.join(raw))
    indexed=NativeGraphSource(source,'pin',required_split='train')
    index,=list((tmp_path/'cache').glob('native-graphs-*.pkl'))
    old=pickle.loads(index.read_bytes()); selected=indexed.entries[1]
    assert selected['communication'] is True and selected['producer_consumer'] is True
    target={**selected,'source_sha256':sha256_file_hex(source)}
    for change in ('missing_class','false_class','integer_class','wrong_binding'):
        bad=copy.deepcopy(target)
        if change=='missing_class':del bad['communication']
        elif change=='false_class':bad['communication']=False
        elif change=='integer_class':bad['communication']=1
        else:bad['binding']='e'*64
        output=tmp_path/(change+'.jsonl')
        with pytest.raises(ValueError):filter_source(source,output,[bad],source_sha256=sha256_file_hex(source))
        assert not output.exists()
    ordinary_target={**indexed.entries[2], 'source_sha256':sha256_file_hex(source)}
    assert ordinary_target['communication'] is False
    proof=filter_source(source,tmp_path/'remaining.jsonl',[target,ordinary_target],source_sha256=sha256_file_hex(source))
    assert (tmp_path/'remaining.jsonl').read_bytes()==raw[0]+raw[3]
    assert proof['removed'][0]['raw_row_sha256']==hashlib.sha256(raw[1]).hexdigest()
    projected=continue_filtered_index(index,proof,index_sha256=sha256_file_hex(index),metadata_sha256=old['metadata_sha256'])
    assert pickle.loads(Path(projected['path']).read_bytes())['entries']==[old['entries'][0],old['entries'][3]]
    tampered=copy.deepcopy(proof);tampered['removed'][0]['communication']=False
    with pytest.raises(ValueError,match='index target binding'):
        continue_filtered_index(index,tampered,index_sha256=sha256_file_hex(index),metadata_sha256=old['metadata_sha256'])
    bad_index=copy.deepcopy(old);bad_index['entries'][1]['communication']=False
    bad_index['metadata_sha256']=digest(bad_index['entries'])
    false_index=tmp_path/'false-index.pkl';false_index.write_bytes(pickle.dumps(bad_index))
    with pytest.raises(ValueError,match='index target binding'):
        continue_filtered_index(false_index,proof,index_sha256=sha256_file_hex(false_index),
                                metadata_sha256=bad_index['metadata_sha256'])


def test_invalid_communication_graph_is_never_partially_withdrawn(tmp_path):
    source, raw, targets, _, _=fixture(tmp_path)
    row=json.loads(raw[1])
    row['nodes'].append({'name':'reader','token_ids':[3,4], 'target_weights':[0,1],
                         'imports':[{'at':0,'source':'absent','start':0,'stop':1}]})
    raw[1]=json.dumps(row).encode()+b'\n';source.write_bytes(b''.join(raw))
    target={**targets[0], 'source_sha256':sha256_file_hex(source), 'communication':True}
    with pytest.raises(ValueError):
        filter_source(source,tmp_path/'invalid.jsonl',[target],source_sha256=sha256_file_hex(source))
    assert not (tmp_path/'invalid.jsonl').exists()
