import json

from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import write_jsonl_text
from scripts.neuralese_data.extract_acquired_syntax_repair_candidates import extract
from test_acquired_session_syntax_repair import SCHEMA


def test_exact_pending_census_separates_safe_ambiguous_and_invalid_names(tmp_path):
    def value(ident,name,argument):
        return {'source_row':ident,'source_file':{'path':'original','sha256':'a'*64},'positive_candidate':True,
          'trajectory':{'id':ident,'tools':[{'function':{'name':'browser.search','parameters':SCHEMA}}],
            'messages':[{'role':'assistant','tool_calls':[{'function':{'name':name,'arguments':argument}}]}]}}
    rows=[value('repairable','browser.search','{"query": hello there, "topn": 10, "source": "news"}'),
          value('ambiguous','browser.search','{"query": hello, "topn": nope}'),
          value('invalid-name','assistant<|channel|>analysis','{"cursor": 1}')]
    source=tmp_path/'full-trajectories.jsonl.zst'
    with write_jsonl_text(source) as out:
        for row in rows:out.write(json.dumps(row)+'\n')
    omission=tmp_path/'omissions.jsonl'
    with omission.open('w') as out:
        for index,row in enumerate(rows):
            out.write(json.dumps({'id':row['trajectory']['id'],'normalized_source_sha256':canonical_json_sha256_hex(row),
                'reason':'source-render-pending:tool-call-unrepresentable' if index==2 else 'source-render-pending:tool-arguments'})+'\n')
    receipt=tmp_path/'receipt.json';receipt.write_text(json.dumps({'source':{'path':str(source),'sha256':sha256_file_hex(source)},
        'omissions_sha256':sha256_file_hex(omission)}))
    result=extract(receipt,tmp_path/'audit')
    assert result['counts']['pending_rows']==3 and result['counts']['fully_bounded_candidates']==1
    assert result['counts']['ambiguous_or_unsupported']==1 and result['counts']['unrepresentable_rows']==1
    assert result['unrepresentable_tool_name_rows']=={'assistant<|channel|>analysis':1}
    assert json.loads((tmp_path/'audit'/'candidates.json').read_text())['rows'][0]['id']=='repairable'
