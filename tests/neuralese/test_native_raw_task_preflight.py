"""Task admission and selector validation precede full parent deserialization."""
import json
import pytest
from natlang_neuralese.rl import native_raw_rounds as raw


@pytest.mark.parametrize('metadata',[
    {'level':0,'qualification':'live-48-accepted'},
    {'level':-1,'qualification':'qualified'},
    {'level':True,'qualification':'qualified'},
])
def test_bad_task_metadata_fails_before_handoff_or_cache(tmp_path,monkeypatch,metadata):
    cases=tmp_path/'cases.jsonl'
    cases.write_text(json.dumps({'id':'private','split':'train','curriculum':{'rl':metadata}})+'\n')
    directory=tmp_path/'model';directory.mkdir()
    # Even a purported existing config cannot bypass bound task validation.
    (directory/'config.json').write_text('{}')
    def heavy(*args,**kwargs):pytest.fail('large checkpoint verifier reached before task preflight')
    monkeypatch.setattr(raw,'verify_handoff',heavy)
    with pytest.raises(ValueError,match='task'):
        raw.configure_model(tmp_path/'plan.json',{'cases':raw.binding(cases)},None,directory)


def test_preflight_keeps_existing_qualified_and_ordinary_tasks(tmp_path):
    cases=tmp_path/'cases.jsonl'
    rows=[{'id':'qualified','split':'train','curriculum':{'rl':{'level':0,'qualification':'qualified'}}},
          {'id':'ordinary','split':'train'}]
    cases.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    assert raw._task_cases({'cases':raw.binding(cases)})==cases
