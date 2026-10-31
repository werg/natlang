from natlang_neuralese.rl.native_stage import _defer_text_result_state_groups


def _rollout(task, group, sample, *, host_write):
    turn = {'invocation_id': f'{task}/1', 'sampling': {'writes': [], 'prompt': []}}
    events = [{'kind': 'block_write', 'turn': f'{task}/1#turn1', 'block': f'nzs1_{task}{sample}', 'node': 'n1',
               'producer': None, 'source_kind': 'typed-text-result'}] if host_write else []
    return {'task_id': task, 'group': group, 'sample': sample, 'turns': [turn], 'execution_graph': events}


def test_uncaptured_text_result_state_defers_its_whole_group(tmp_path):
    rows = [_rollout('a', 0, 0, host_write=True), _rollout('a', 0, 1, host_write=False),
            _rollout('b', 0, 0, host_write=False)]
    kept, record = _defer_text_result_state_groups(rows, tmp_path)
    assert [row['task_id'] for row in kept] == ['b']
    assert [(e['task_id'], e['sample'], e['state_writes']) for e in record['episodes']] == [
        ('a', 0, ['nzs1_a0']), ('a', 1, [])]
    assert (tmp_path / 'deferred-text-result-state-graphs.jsonl').read_text().count('\n') == 2


def test_model_captured_writes_are_not_deferred(tmp_path):
    row = _rollout('a', 0, 0, host_write=True)
    row['turns'][0]['sampling']['writes'] = [{'state_id': 'nzs1_a0'}]
    row['execution_graph'][0]['producer'] = 'model'
    kept, record = _defer_text_result_state_groups([row], tmp_path)
    assert kept == [row] and record is None
