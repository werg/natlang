import copy
from collections import Counter
import pytest
from natlang_neuralese.rl.episode_order import mix_episodes, order_receipt


def fixture():
    graphs=[];rows={}
    for role,samples in [('student',4),('teacher',1)]:
        for task in 'abcd':
            for sample in range(samples):
                ident=f'{role}-{task}-{sample}'
                graphs.append({'id':ident,'nodes':[{'sample':sample}]})
                rows[ident]={'task_id':task,'jolt':{'role':role},'advantage':sample/100}
    return graphs,{'rows':rows}


def test_actual_twenty_episode_geometry_keeps_task_and_role_diversity():
    graphs,objective=fixture();references=list(graphs);original=copy.deepcopy(graphs);bound=copy.deepcopy(objective)
    mix_episodes(graphs,objective)
    assert objective==bound
    assert {g['id']:g for g in graphs}=={g['id']:g for g in original}
    assert {id(g) for g in graphs}=={id(g) for g in references}
    for start in range(0,20,4):
        rows=[objective['rows'][g['id']] for g in graphs[start:start+4]]
        assert len({r['task_id'] for r in rows})==4
        assert sum(r['jolt']['role']=='teacher' for r in rows)<=1
        assert sum(r['jolt']['role']=='student' for r in rows)>=3
    receipt=order_receipt(graphs,objective)
    assert receipt['adjacent_same_task']==0 and receipt['role_counts']=={'student':16,'teacher':4}
    again,_=fixture();mix_episodes(again,objective);assert again==graphs


def test_single_task_preserves_every_episode_without_inventing_diversity():
    graphs,objective=fixture()
    for row in objective['rows'].values():row['task_id']='one-world'
    mix_episodes(graphs,objective)
    receipt=order_receipt(graphs,objective)
    assert receipt['tasks']==1 and receipt['adjacent_same_task']==19


@pytest.mark.parametrize('damage',['duplicate','missing','unknown-task'])
def test_invalid_or_unknown_custody_fails_closed(damage):
    graphs,objective=fixture()
    if damage=='duplicate':graphs[1]=graphs[0]
    elif damage=='missing':graphs.pop()
    else:objective['rows'][graphs[0]['id']].pop('task_id')
    with pytest.raises(ValueError):mix_episodes(graphs,objective)


def test_actual_resume_seal_keeps_earned_prefix_and_reports_real_order(tmp_path):
    import json
    from natlang_neuralese.rl.policy_source_continuation import arrange_and_seal
    graphs,objective=fixture();prefix=copy.deepcopy(graphs[:4])
    (tmp_path/'checkpoint.pt').write_bytes(b'immutable checkpoint identity')
    (tmp_path/'rl-native-graphs.jsonl').write_text(''.join(json.dumps(g)+'\n' for g in prefix))
    (tmp_path/'rl-native-objective.json').write_text(json.dumps({'rows':{g['id']:objective['rows'][g['id']] for g in prefix}}))
    mix_episodes(graphs,objective)
    transition=arrange_and_seal(tmp_path,graphs,objective)
    assert graphs[:4]==prefix and transition['old_rows']==4 and transition['added_rows']==16
    assert len({g['id'] for g in graphs})==20
    assert order_receipt(graphs,objective)['adjacent_same_task']>=3
