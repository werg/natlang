"""Teacher-text Neuralese blocks (a remote teacher's literal-text transport) are not native-state occurrences."""
import pytest

from natlang_neuralese.rl.episode_graph import validate_rollout_graph
from natlang_neuralese.rl.native_occurrences import capture_occurrences, graph_binding


def teacher_text_episode():
    """A producer turn returns a typed JSON result through the teacher-text writer; a consumer turn reads it."""
    block = 'nzt1_' + 'a' * 52
    write = {'node': 'w', 'kind': 'block_write', 'turn': 'p#turn1', 'call_id': 'p', 'block': block, 'inputs': [],
             'producer': 'teacher-text-emulation', 'format': 'teacher-text', 'source_kind': 'typed-json-result'}
    read = {'node': 'r', 'kind': 'block_read', 'turn': 'c#turn1', 'call_id': 'c', 'block': block,
            'format': 'teacher-text', 'producer': 'teacher-text-emulation', 'inputs': [{'node': 'w', 'block': block}]}
    turns = [{'node': 'p#turn1', 'kind': 'model_turn', 'call_id': 'p', 'inputs': []},
             {'node': 'c#turn1', 'kind': 'model_turn', 'call_id': 'c', 'inputs': []}]
    return turns + [write, read]


def test_teacher_text_writes_and_reads_bind_to_an_empty_native_capture():
    nodes = {event['node']: event for event in teacher_text_episode()}
    assert capture_occurrences('p#turn1', {}, nodes) == ([], [])
    assert capture_occurrences('c#turn1', {}, nodes) == ([], [])
    assert graph_binding('c#turn1', {}, nodes)['read_bindings'] == {}


def test_a_native_block_write_without_its_capture_is_still_refused():
    nodes = {event['node']: event for event in teacher_text_episode()}
    nodes['w'] = {**nodes['w'], 'block': 'nzs1_' + 'b' * 52, 'format': 'native-state', 'producer': 'model'}
    with pytest.raises(ValueError, match='state capture differs'):
        capture_occurrences('p#turn1', {}, nodes)


def test_teacher_text_rollout_validates():
    graph = teacher_text_episode()
    turns = [{'invocation_id': 'p', 'messages': [], 'target': {}}, {'invocation_id': 'c', 'messages': [], 'target': {}}]
    names, nodes = validate_rollout_graph({'turns': turns, 'execution_graph': graph})
    assert names == ['p#turn1', 'c#turn1'] and nodes['r']['kind'] == 'block_read'


def test_an_in_context_text_read_is_outside_the_native_inventory():
    """A text-written state read as its exact tokens in the reader's prompt (receipt ``in-context-text``) has no
    native-state segment in the capture: the read binds to nothing, and one recorded as native-state is refused."""
    block = 'nzs1_' + 'c' * 52
    turn = {'node': 'c#turn1', 'kind': 'model_turn', 'call_id': 'c', 'inputs': []}
    read = {'node': 'r', 'kind': 'block_read', 'turn': 'c#turn1', 'call_id': 'c', 'block': block,
            'format': 'in-context-text', 'inputs': [{'node': 'external', 'block': block}]}
    capture = {'prompt': [{'kind': 'tokens', 'token_ids': [1, 2, 3]}], 'writes': []}
    nodes = {'c#turn1': turn, 'r': read}
    assert capture_occurrences('c#turn1', capture, nodes) == ([], [])
    assert graph_binding('c#turn1', capture, nodes)['read_bindings'] == {}
    nodes['r'] = {**read, 'format': 'native-state'}
    with pytest.raises(ValueError, match='state capture differs'):
        capture_occurrences('c#turn1', capture, nodes)
