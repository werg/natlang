import json
from types import SimpleNamespace
import pytest
from natlang_neuralese.rl.native_stage import jolt_privileged_source
from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.rl import jolt


class ExportedTokenizer:
    name_or_path='/no-base-config-in-saved-tokenizer'
    natlang_history_reasoning={'backbone':'lfm3','policy':'keep'}
    def convert_tokens_to_ids(self,values):
        if isinstance(values,list):return list(range(10,10+len(values)))
        return 3
    def __call__(self,text,**kwargs):return SimpleNamespace(input_ids=[1,2])


def test_jolt_replay_uses_authenticated_backbone_for_exported_tokenizer(tmp_path,monkeypatch):
    cases=tmp_path/'cases.jsonl';cases.write_text(json.dumps({'id':'case'})+'\n')
    monkeypatch.setattr(ProtectedWorlds,'load',lambda **kw:object())
    seen={};monkeypatch.setattr(jolt,'privileged_source',lambda rows,**kw:seen.update(kw) or 'source')
    roles=SimpleNamespace(produce_open=1,produce_close=2,consume_open=3,consume_close=4)
    tokenizer=ExportedTokenizer()
    assert jolt_privileged_source(rows=[],tokenizer=tokenizer,cases=cases,roles=roles,tokenizer_fingerprint='a'*64)=='source'
    assert seen['rendering'].template=='lfm3' and seen['rendering'].bos==3
    tokenizer.natlang_history_reasoning={'backbone':None}
    with pytest.raises(ValueError,match='authenticated tokenizer backbone'):
        jolt_privileged_source(rows=[],tokenizer=tokenizer,cases=cases,roles=roles,tokenizer_fingerprint='a'*64)
