import json
from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
from natlang_neuralese.common.jsonio import write_jsonl_text
from scripts.neuralese_data.audit_acquired_recovery_context import audit


def test_recovery_context_preserves_observed_neighbors_without_inferring_call(tmp_path):
    value={'trajectory':{'id':'episode','outcome':{'label':'source-reported'},'messages':[
        {'role':'user','content':'question'}, {'role':'assistant','tool_calls':[{'function':{'name':'assistant<|channel|>analysis','arguments':'{}'}}]},
        {'role':'tool','tool_call_id':'x','content':'tool reported an error'}, {'role':'assistant','content':'later answer'}]}}
    source=tmp_path/'source.jsonl.zst'
    with write_jsonl_text(source) as f:f.write(json.dumps(value)+'\n')
    sample={'sampled':[{'id':'episode','reason':'source-render-pending:tool-call-unrepresentable',
        'normalized_source_sha256':canonical_json_sha256_hex(value),'calls':[{'message_index':1,'name':'assistant<|channel|>analysis'}]}]}
    sample_path=tmp_path/'sample.json';sample_path.write_text(json.dumps(sample))
    converter=tmp_path/'receipt.json';converter.write_text(json.dumps({'source':{'path':str(source),'sha256':sha256_file_hex(source)}}))
    result=audit(sample_path,converter,tmp_path/'result.json')
    window=result['rows'][0]['windows'][0]
    assert window['next']['role']=='tool' and window['following']['role']=='assistant'
    assert 'no tool execution' in result['rows'][0]['status'] and result['training_admission'] is False
