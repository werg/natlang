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


def test_selected_span_covers_the_note_as_the_tool_call_spells_it():
    from natlang_neuralese.serve.engine import selected_span
    pieces = ['[compact', '_history(', 'note="', 'Fixed', ' a\\n', 'b.', '")]']
    decode = lambda ids: ''.join(pieces[i] for i in ids)
    ids = list(range(len(pieces)))
    # The note "Fixed a\nb." is spelled escaped inside the call: the span is its whole tokens.
    assert selected_span(decode, ids, 'Fixed a\nb.') == (3, 6)
    with pytest.raises(RequestError):
        selected_span(decode, ids, 'absent')


def test_capture_select_takes_the_note_tokens_in_place(tmp_path):
    engine = actor(tmp_path, 'lfm', lexical=True, words=THINK)
    stop = sorted(engine.stop_ids)[0]
    reply = [7, 12, 13, 14, 15, stop]
    text = engine.tokenizer.decode([13, 14], skip_special_tokens=False)
    result = engine.native_capture([4, 9], reply, select=text)
    capture = result['source_capture']
    ids = [t for segment in capture['segments'] for t in segment['token_ids']]
    assert ids[capture['payload_start']:capture['payload_stop']] == [13, 14]
    assert ids[:capture['payload_start']] == [4, 9, 7, 12, engine.role_tokens.produce_open]


def test_capture_refuses_a_reply_that_ended_inside_its_reasoning(tmp_path):
    engine = actor(tmp_path, 'lfm', lexical=True, words=THINK)
    stop = sorted(engine.stop_ids)[0]
    # The prompt opens the reasoning (<think> = 6); a reply cut off before </think> has no content.
    with pytest.raises(RequestError, match='inside its reasoning'):
        engine.native_capture([4, 9, 6], [12, 13, 14])
    assert engine.native_capture([4, 9, 6], [12, 7, 13, stop])['source_capture']['payload_stop'] > 0


def test_capture_in_a_context_that_imported_a_state(tmp_path):
    """An operator call whose prompt imported a soft prompt state: the capture runs in that same context, and its
    source records the state where it stood (payload positions count its rows)."""
    engine = actor(tmp_path, 'lfm', lexical=True, words=THINK)
    stop = sorted(engine.stop_ids)[0]
    roles = engine.role_tokens
    soft = engine.native_write_text(engine.tokenizer.decode([20, 21, 22], skip_special_tokens=False))['id']
    other = engine.native_write_text(engine.tokenizer.decode([23, 24], skip_special_tokens=False))['id']
    count = len(engine.store.get_native_state(soft).block.token_ids)
    segments = lambda state: [{'kind': 'tokens', 'token_ids': [4, roles.consume_open]},
                              {'kind': 'native_state', 'id': state},
                              {'kind': 'tokens', 'token_ids': [roles.consume_close, 9]}]
    result = engine.native_capture(None, [7, 12, 13, stop], prompt_segments=segments(soft))
    capture = result['source_capture']
    kinds = [segment['kind'] for segment in capture['segments']]
    assert kinds == ['tokens', 'native_state', 'tokens', 'tokens', 'tokens']
    assert capture['segments'][1]['id'] == soft and capture['segments'][1]['token_count'] == count
    assert capture['segments'][0]['token_ids'] == [4, roles.consume_open]
    assert capture['segments'][2]['token_ids'] == [roles.consume_close, 9, 7, roles.produce_open]
    assert capture['payload_start'] == 4 + count + 2 and capture['payload_stop'] == capture['payload_start'] + 2
    # The imported state is context of the capture: another state there gives another captured state.
    assert engine.native_capture(None, [7, 12, 13, stop], prompt_segments=segments(other))['id'] != result['id']
    with pytest.raises(RequestError, match='consume_open'):
        engine.native_capture(None, [7, 12, stop], prompt_segments=[{'kind': 'native_state', 'id': soft},
                                                                     {'kind': 'tokens', 'token_ids': [roles.consume_close]}])
    with pytest.raises(RequestError, match='not both'):
        engine.native_capture([4], [7, 12, stop], prompt_segments=segments(soft))
    with pytest.raises(RequestError):
        engine.native_capture(None, [7, 12, stop], prompt_segments=segments('nzs1_' + 'a' * 52))
    # Role markers outside an import's own consume markers are still refused.
    with pytest.raises(RequestError, match='role markers'):
        engine.native_capture(None, [7, 12, stop], prompt_segments=[*segments(soft),
                                                                     {'kind': 'tokens', 'token_ids': [roles.consume_open]}])


def test_a_captured_reply_is_a_capture_and_splices_when_read(tmp_path, monkeypatch):
    """A reply captured in its own context (a compaction operator's reply, a note) is stored with producer_kind
    'capture' and is read spliced as stored, not as its tokens inline (DECISIONS 2026-11-25 "Learned compaction on by
    default": the spliced state beats the inline tokens in the use ablation). NATLANG_NEURALESE_CAPTURE_IN_CONTEXT=1
    reads it inline like a span. A value's text written after its producing call stays a 'span'."""
    engine = actor(tmp_path, 'lfm', lexical=True, words=THINK)
    stop = sorted(engine.stop_ids)[0]
    result = engine.native_capture([4, 9, 10], [11, 7, 12, 13, stop])
    block = engine.store.get_native_state(result['id']).block
    assert block.producer_kind == 'capture'
    assert not engine.reads_in_context(result['id'], block)
    monkeypatch.setenv('NATLANG_NEURALESE_CAPTURE_IN_CONTEXT', '1')
    assert engine.reads_in_context(result['id'], block)
    monkeypatch.delenv('NATLANG_NEURALESE_CAPTURE_IN_CONTEXT')
    written = engine._native_capture_impl([4, 9, 10], [11, 7, 12, stop], text='t6 t7')
    assert engine.store.get_native_state(written['id']).block.producer_kind == 'span'
