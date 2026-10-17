import copy
import json
import pickle
from pathlib import Path

import pytest
from scripts.neuralese_data.withdraw_native_graph_targets import filter_source,continue_filtered_index
from natlang_neuralese.common.hashing import sha256_file_hex,canonical_json_sha256_hex as digest


def fixture(tmp_path):
    rows=[{'id':key,'split':'train','nodes':[{'name':'text','token_ids':[1,2]}],
           'conversion':{'role_interpretation':'ordinary-text'}}for key in ('keep','bad','last')]
    source=tmp_path/'source.jsonl';raw=[json.dumps(row).encode()+b'\n'for row in rows];source.write_bytes(b''.join(raw))
    targets=[{'id':'bad','binding':'b'*64,'source_sha256':sha256_file_hex(source)}]
    key={'source_sha256':sha256_file_hex(source),'kind':'native-graphs','split':'train','policy_sha256':'exact-old-policy','tokenizer':'pin','role_ids':[1,2]}
    entries=[{'id':r['id'],'binding':('b'if r['id']=='bad'else'c')*64,'communication':False,'metadata':['untouched',i]}for i,r in enumerate(rows)]
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


@pytest.mark.parametrize('fault',['binding','ids','policy_metadata','output','index_sha','existing_conflict'])
def test_index_projection_requires_exact_original_and_retained_custody(tmp_path,fault):
    source,raw,targets,index,old=fixture(tmp_path)
    proof=filter_source(source,tmp_path/'filtered.jsonl',targets,source_sha256=sha256_file_hex(source))
    sha=sha256_file_hex(index);metadata=old['metadata_sha256']
    if fault=='binding':proof['removed'][0]['binding']='x'*64
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
