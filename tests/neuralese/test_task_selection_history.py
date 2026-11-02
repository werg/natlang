import json
from pathlib import Path
import pytest
from natlang_neuralese.rl.selection_history import snapshot
from natlang_neuralese.rl.production import initial_controller_state
from natlang_neuralese.common.hashing import sha256_file_hex_cached


def fixture(tmp_path):
    cases=tmp_path/'cases.jsonl';cases.write_text(json.dumps({'id':'world'})+'\n')
    state=tmp_path/'state.json';row={'task_id':'world','observed_policy_id':'a'*64,'scored':4,'mean':1.}
    state.write_text(json.dumps({'checkpoint':'old-actor.pt','trained_rounds':17,'offset':4,
        'task_curriculum_history':{'world':row,'withdrawn':{**row,'task_id':'withdrawn'}}}))
    ref={'path':str(state),'sha256':sha256_file_hex_cached(state)}
    history=snapshot(ref,tmp_path/'controls',cases)
    return {'checkpoint':'fresh-trained-parent.pt','cases':str(cases),'task_selection_history':history},state,cases


def test_history_survives_source_owner_advancement_without_policy_credit(tmp_path):
    config,source,_=fixture(tmp_path)
    source.write_text('{}')
    state=initial_controller_state(config)
    assert state['checkpoint']=='fresh-trained-parent.pt' and state['trained_rounds']==0
    assert state['trained_supervised_tokens']==state['trained_target_tokens']==0
    assert state['round']==0 and state['collected_task_ids']==[] and state['trained_task_ids']==[]
    assert state['task_curriculum_history']['world']['observed_policy_id']=='a'*64
    assert state['task_selection_history_excluded_ids']==['withdrawn']
    assert 'group_stats' not in state


@pytest.mark.parametrize('damage',['packet','packet-same-stat','snapshot','actor'])
def test_changed_inputs_and_unknown_actor_are_rejected(tmp_path,damage):
    config,_,cases=fixture(tmp_path)
    ref=config['task_selection_history'];path=Path(ref['path'])
    if damage.startswith('packet'):
        import os
        before=cases.stat()
        cases.write_text(json.dumps({'id':'other' if damage=='packet-same-stat' else 'another-world'})+'\n')
        if damage=='packet-same-stat':os.utime(cases,ns=(before.st_atime_ns,before.st_mtime_ns))
    else:
        record=json.loads(path.read_text());source=Path(record['source_controller']['path'])
        if damage=='snapshot':source.write_text('{}')
        else:
            state=json.loads(source.read_text());state['task_curriculum_history']['world'].pop('observed_policy_id')
            source.write_text(json.dumps(state));record['source_controller']['sha256']=sha256_file_hex_cached(source)
            path.write_text(json.dumps(record));ref['sha256']=sha256_file_hex_cached(path)
    with pytest.raises(ValueError):initial_controller_state(config)
