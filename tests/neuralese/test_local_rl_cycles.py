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
