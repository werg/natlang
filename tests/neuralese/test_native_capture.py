"""The in-context capture of a sampled reply (serve/engine.py native_capture; OPERATORS_RESTORE_20261117.md §3.1)."""
import pytest
import torch

from natlang_neuralese.serve.engine import RequestError
from test_native_guidance import THINK, actor


def test_capture_takes_the_reply_content_in_its_own_context(tmp_path):
    engine = actor(tmp_path, 'lfm', lexical=True, words=THINK)
    stop = sorted(engine.stop_ids)[0]
    reply = [11, 7, 12, 13, stop]
    result = engine.native_capture([4, 9, 10], reply, value_type='Neuralese<string>')
    capture = result['source_capture']
    ids = [t for segment in capture['segments'] for t in segment['token_ids']]
    start, stop_at = capture['payload_start'], capture['payload_stop']
    # The span is the content after the reasoning (``</think>`` = 7) and before the end of turn, in place.
    assert ids[:start] == [4, 9, 10, 11, 7, engine.role_tokens.produce_open]
    assert ids[start:stop_at] == [12, 13] and ids[stop_at] == engine.role_tokens.produce_close
    assert [segment['producing'] for segment in capture['segments']][1] == [True, True]
    assert result['native_state']['type'] == 'Neuralese<string>'
    block = engine.stored_native_state(result['id'])
    assert block is not None
    # The state depends on its context: the same reply after another prompt is another state. (The tiny model's state
    # sees its context only through its conv layer's 4-token window, so the change is made inside it.)
    near = engine.native_capture([4, 9, 10], [7, 12, 13, stop])
    other = engine.native_capture([4, 9, 5], [7, 12, 13, stop])
    assert other['id'] != near['id']


def test_capture_refuses_a_reply_without_content(tmp_path):
    engine = actor(tmp_path, 'lfm', lexical=True, words=THINK)
    stop = sorted(engine.stop_ids)[0]
    with pytest.raises(RequestError):
        engine.native_capture([4, 9], [11, 7, stop])
    with pytest.raises(RequestError):
        engine.native_capture([4, engine.role_tokens.produce_open], [12, stop])
