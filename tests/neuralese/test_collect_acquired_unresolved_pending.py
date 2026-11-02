import hashlib
import json
import pytest
from scripts.neuralese_data.collect_acquired_unresolved_pending import collect


def ref(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def write(path,value):path.write_text(json.dumps(value));return ref(path)


def test_exact_remaining_omissions_stay_source_bound(tmp_path):
    omissions=tmp_path/'omissions.jsonl'
    originals=[{'id':'candidate','reason':'source-render-pending:tool-arguments'},
               {'id':'ambiguous','reason':'source-render-pending:tool-arguments'},
               {'id':'bad-name','reason':'source-render-pending:tool-call-unrepresentable'}]
    omissions.write_text(''.join(json.dumps(row)+'\n' for row in originals))
    converter=write(tmp_path/'converter.json',{'omissions_sha256':ref(omissions)['sha256']})
    source={'path':'/source','sha256':'a'*64};directory=tmp_path/'audit';directory.mkdir()
    candidates=write(directory/'candidates.json',{'rows':[{'id':'candidate'}]})
    audit=write(directory/'receipt.json',{'source':source,'converter_receipt':converter,'candidate_sha256':candidates['sha256'],
        'counts':{'ambiguous_or_unsupported':1,'not_both_json_failures':0,'unrepresentable_rows':1}})
    census=write(tmp_path/'census.json',{'schema':'natlang.acquired-render-pending-census/1','training_admission':False,
        'receipts':[audit], 'counts':{'ambiguous_or_unsupported':1,'not_both_json_failures':0,'unrepresentable_rows':1}})
    result=collect(census,tmp_path/'out')
    rows=[json.loads(line) for line in (tmp_path/'out'/'unresolved.jsonl').read_text().splitlines()]
    assert result['rows']==2 and {row['original_omission']['id'] for row in rows}=={'ambiguous','bad-name'}
    assert all(row['training_admission'] is False for row in rows)
    omissions.write_text('{}\n')
    with pytest.raises(ValueError,match='omission ledger changed'):collect(census,tmp_path/'out2')
