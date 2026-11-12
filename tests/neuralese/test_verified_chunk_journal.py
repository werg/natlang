import json
import pytest
from scripts.neuralese_data.verified_chunk_journal import verify_chunks


def test_partial_failure_resume_validates_exact_prefix_and_never_publishes_failed_row(tmp_path):
    rows=[{'id':str(i),'payload':i} for i in range(5)];identity={'source_sha256':'a'*64,'actor':'actual','code_sha256':'b'*64,'split':'train'}
    def check(value):assert value==identity
    def fail(row):
        if row['id']=='3':raise ValueError('invalid source')
        return {'id':'graph:'+row['id'],'source':row['payload']}
    with pytest.raises(ValueError):verify_chunks(rows,out=tmp_path,identity=identity,verify=fail,validate_identity=check,chunk_rows=2)
    assert len(list(tmp_path.glob('*.receipt.json')))==1
    assert json.loads((tmp_path/'chunk-00000000.receipt.json').read_text())['rows']==2
    calls=[]
    def good(row):calls.append(row['id']);return {'id':'graph:'+row['id'],'source':row['payload']}
    result=verify_chunks(rows,out=tmp_path,identity=identity,verify=good,validate_identity=check,chunk_rows=2)
    assert result['rows']==5 and result['reused_rows']==2 and calls==['2','3','4']
    with pytest.raises(ValueError,match='prefix'):
        verify_chunks([{**rows[0],'payload':999},*rows[1:]],out=tmp_path,identity=identity,verify=good,validate_identity=check,chunk_rows=2)
    (tmp_path/'chunk-00000000.jsonl').write_text('tampered')
    with pytest.raises(ValueError,match='artifact'):
        verify_chunks(rows,out=tmp_path,identity=identity,verify=good,validate_identity=check,chunk_rows=2)


def test_orphan_chunk_is_reverified_before_recovering_receipt(tmp_path):
    from natlang_neuralese.common.jsonio import canonical_json_str
    rows=[{'id':'a','payload':1}];graph={'id':'graph:a','source':1};identity={'code':'pinned'}
    (tmp_path/'chunk-00000000.jsonl').write_text(canonical_json_str(graph)+'\n')
    result=verify_chunks(rows,out=tmp_path,identity=identity,verify=lambda _:graph,validate_identity=lambda _:None)
    assert result['rows']==1 and (tmp_path/'chunk-00000000.receipt.json').exists()
