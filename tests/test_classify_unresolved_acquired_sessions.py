import json
import zstandard
import pytest
from scripts.neuralese_data import classify_unresolved_acquired_sessions as audit
from natlang_neuralese.common.hashing import canonical_json_sha256_hex


def test_classify_preserves_invalid_name_and_outcome():
    value={'trajectory':{'messages':[{'role':'assistant','tool_calls':[{'function':{'name':'assistant<|channel|>analysis','arguments':'{}'}}]}],
                         'tools':[],'outcome':{'label':'teacher','checked':None}}}
    result=audit.classify(value)
    assert result['disposition']=='quality-excluded'
    assert result['errors'][0]['tool_name']=='assistant<|channel|>analysis'
    assert result['source_outcome']==value['trajectory']['outcome']
    assert result['training_admission'] is False


def test_one_pass_capture_resume_and_corruption(tmp_path, monkeypatch):
    value={'source_row':'source:1','source_file':{'path':'fixture'},'trajectory':{'id':'id-1','messages':[{'role':'assistant',
          'tool_calls':[{'function':{'name':'assistant<|channel|>analysis','arguments':'{}'}}]}], 'tools':[], 'outcome':{'label':'teacher'}}}
    source=tmp_path/'source.jsonl.zst';source.write_bytes(zstandard.ZstdCompressor().compress((json.dumps(value)+'\n').encode()))
    ref=audit.bind(source);wanted={'id-1':{'id':'id-1','normalized_source_sha256':canonical_json_sha256_hex(value)}}
    ledger=tmp_path/'ledger.json';ledger.write_text('{}');extra=tmp_path/'extra.json';extra.write_text('{}')
    grouped={(ref['path'],ref['sha256']):wanted};sources={next(iter(grouped)):ref}
    monkeypatch.setattr(audit,'inputs',lambda *_:(grouped,sources,audit.bind(ledger),audit.bind(extra)))
    output=tmp_path/'out';audit.build(ledger,extra,output)
    result=json.loads((output/'receipt.json').read_text());assert result['counts']=={'quality-excluded':1}
    receipt=json.loads(open(result['sources'][0]['path']).read());assert receipt['source']['sha256']==ref['sha256']
    monkeypatch.setattr(audit.zstandard,'ZstdDecompressor',lambda:pytest.fail('resume reread source'))
    audit.build(ledger,extra,output)
    with open(receipt['classification']['path'],'ab') as stream:stream.write(b' ')
    with pytest.raises(ValueError,match='bound evidence changed'):audit.build(ledger,extra,output)
