import json
import pytest
from natlang_neuralese.rl import local_cycles as lc


def test_round_publication_before_cursor_commit_is_not_repeated():
    assert lc.round_target(4,5)
    assert not lc.round_target(5,5)
    with pytest.raises(ValueError,match='unknown controller successor'):
        lc.round_target(6,5)


def test_supervised_publication_retries_exact_child_but_rejects_foreign_successor(tmp_path,monkeypatch):
    state=tmp_path/'state.json';parent=tmp_path/'parent.pt';child=tmp_path/'child.pt';foreign=tmp_path/'foreign.pt'
    for p in (parent,child,foreign):p.write_bytes(p.name.encode())
    config={'out':str(tmp_path)};calls=[]
    def publish(config,checkpoint):
        calls.append(checkpoint);state.write_text(json.dumps({'checkpoint':checkpoint['path']}))
    monkeypatch.setattr(lc,'_publish_checkpoint',publish)
    state.write_text(json.dumps({'checkpoint':str(parent)}))
    lc.publish_supervised(config,lc.binding(parent),lc.binding(child))
    lc.publish_supervised(config,lc.binding(parent),lc.binding(child))
    assert calls==[lc.binding(child)]
    state.write_text(json.dumps({'checkpoint':str(foreign)}))
    with pytest.raises(ValueError,match='unknown policy successor'):
        lc.publish_supervised(config,lc.binding(parent),lc.binding(child))


def test_failed_worker_can_yield_to_useful_work_only_without_saved_child(tmp_path):
    import subprocess
    parent=tmp_path/'parent.pt';parent.write_bytes(b'earned-full-state')
    recipe=tmp_path/'recipe.json';recipe.write_text('{}')
    (tmp_path/'state.json').write_text(json.dumps({'checkpoint':str(parent)}))
    config={'out':str(tmp_path),'production_recipe':str(recipe)}
    stage={'id':'dpo','parameters':{'epochs':1}}
    operation=tmp_path/'operation';operation.mkdir();cursor={'stage_index':1}
    error=subprocess.CalledProcessError(1,['actual-worker'])
    failure=lc.record_unstarted_stage_failure(cursor,stage,config,operation,lc.binding(parent),error,enabled=True)
    assert cursor['stage_index']==2 and not failure['completed']
    assert failure['signature']==lc.stage_failure_signature(stage,config)
    changed={**stage,'parameters':{'epochs':2}}
    assert failure['signature']!=lc.stage_failure_signature(changed,config)
    (operation/'learner').mkdir();(operation/'learner/checkpoint.pt').write_bytes(b'newer-earned-state')
    before=json.loads(json.dumps(cursor))
    with pytest.raises(subprocess.CalledProcessError):
        lc.record_unstarted_stage_failure(cursor,stage,config,operation,lc.binding(parent),error,enabled=True)
    assert cursor==before
