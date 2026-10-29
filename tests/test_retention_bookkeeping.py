import json
import pytest
from scripts.coreweave.retention_bookkeeping import own_bookkeeping,recheck_own_bookkeeping


def test_only_exact_self_records_are_exempt(tmp_path):
    d='a'*64;r={'role':'old','run':'r','name':'n','identity':{'sha256':d,'bytes':4}}
    phase=tmp_path/'phase.json';phase.write_text(json.dumps(r|{'upload':{'sha256':d,'bytes':4}}))
    cache=tmp_path/'cache.json';cache.write_text(json.dumps({'schema':'natlang.private-export-cache/1','identity':r['identity'],'receipt':r,'scope':'role old'}))
    review=tmp_path/'review.json';review.write_text(json.dumps({'schema':'natlang.standalone-retention-guard-review/1','candidates':[{'role':'old','sha256':d}]}))
    state=tmp_path/'state.json';state.write_text(json.dumps({'operations':[{'phase':'deleted','newer_role':'old','newer_sha256':d}]}))
    kw={'role':'old','digest':d,'receipt':r,'phase':phase,'cache':cache,'reviews':[review],'retirement_state':state};docs=own_bookkeeping(**kw);recheck_own_bookkeeping(docs,role='old',digest=d)
    value=json.loads(review.read_text());value['consumer']=d;review.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='another reference'):own_bookkeeping(**kw)
    with pytest.raises(ValueError,match='changed'):recheck_own_bookkeeping(docs,role='old',digest=d)
