import pytest
from natlang_neuralese.rl import production
from natlang_neuralese.rl.curriculum import DEFAULT


def cases():
    return [{'id':f'task-{index}','curriculum':{'rl':{'level':index//3,'qualification':'qualified'}}}
            for index in range(12)]


def test_practice_starts_easy_and_grows_from_actual_successes():
    rows=cases();budget={'rows':3,'group':4,'offset':9}
    chosen=production.select_tasks(rows,[],budget,curriculum=DEFAULT)
    assert [item['row'] for item in chosen]==[0,1,9]
    stats=[{'task_id':row['id'],'scored':4,'mean':1.} for row in rows[:3]]
    chosen=production.select_tasks(rows,stats,budget,curriculum=DEFAULT)
    assert [item['row'] for item in chosen]==[3,4,9]
    # Unscored infrastructure cannot masquerade as mastery or a scored failure.
    stats.append({'task_id':'task-3','scored':0,'mean':None})
    assert production.select_tasks(rows,stats,budget,curriculum=DEFAULT)==chosen


def test_rotating_breadth_reaches_every_task_even_when_solved_or_harder():
    rows=cases();stats=[{'task_id':row['id'],'scored':4,'mean':1.} for row in rows]
    offset=0;seen=set()
    for _ in rows:
        chosen=production.select_tasks(rows,stats,{'rows':3,'group':4,'offset':offset},curriculum=DEFAULT)
        seen.update(item['row'] for item in chosen);offset=(chosen[-1]['row']+1)%len(rows)
    assert seen==set(range(len(rows)))
    rows.append({'id':'new-hard','curriculum':{'rl':{'level':8}}})
    for _ in rows:
        chosen=production.select_tasks(rows,stats,{'rows':3,'group':4,'offset':offset},curriculum=DEFAULT)
        seen.update(item['row'] for item in chosen);offset=(chosen[-1]['row']+1)%len(rows)
    assert 12 in seen


def test_pending_scaffolds_are_not_silently_adopted():
    rows=cases();rows[0]['curriculum']['rl']['qualification']='pending-live-48'
    with pytest.raises(ValueError,match='qualification'):
        production.select_tasks(rows,[],{'rows':3,'group':4,'offset':0},curriculum=DEFAULT)


def test_nl_policy_receives_typed_facts_and_cannot_drop_breadth(monkeypatch):
    import json
    from types import SimpleNamespace
    captured=[]
    def command(argv,**kwargs):
        captured.append(json.loads(kwargs['input']))
        return SimpleNamespace(stdout=json.dumps([{'row':0,'samples':4},{'row':1,'samples':4}]))
    monkeypatch.setattr(production.subprocess,'run',command)
    with pytest.raises(ValueError,match='breadth'):
        production.select_tasks(cases(),[],{'rows':2,'group':4,'offset':9},
                                curriculum=DEFAULT,command=['natlang-selector'])
    assert captured[0]['tasks'][0]=={'row':0,'id':'task-0','level':0,'family':'unspecified'}
    assert captured[0]['curriculum']==DEFAULT


def test_declared_selection_allocation_is_verified(monkeypatch):
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(production.subprocess,'run',lambda *a,**kw:SimpleNamespace(
        stdout=json.dumps([{'row':i,'samples':4} for i in range(3)])))
    with pytest.raises(ValueError,match='allocation'):
        production.select_tasks(cases(),[],{'rows':2,'group':4,'offset':0},command=['selector'])


def test_mixed_outcome_practice_does_not_crowd_out_unvisited_families():
    rows=[{'id':f'qa-{i}','family':'qa','curriculum':{'rl':{'level':0}}} for i in range(8)]
    rows += [{'id':'new-program','family':'program-execution','curriculum':{'rl':{'level':1}}},
             {'id':'new-state','family':'stateful-control','curriculum':{'rl':{'level':0}}}]
    stats=[{'task_id':f'qa-{i}','scored':6,'mean':.5} for i in range(8)]
    budget={'rows':6,'group':6,'offset':0}
    selected=production.select_tasks(rows,stats,budget,curriculum=DEFAULT)
    ids={rows[item['row']]['id'] for item in selected}
    assert {'new-program','new-state'}<=ids
    assert any(value.startswith('qa-') for value in ids)
    assert len(selected)==6 and {item['samples'] for item in selected}=={6}
    assert [item['row'] for item in selected[-2:]]==[0,1]
    # A retired task's historical outcome is not current-source coverage.
    assert production.select_tasks(rows,[*stats,{'task_id':'retired','scored':1000,'mean':1.}],
                                   budget,curriculum=DEFAULT)==selected


def test_exploration_does_not_repeat_solved_singleton_instead_of_new_invocations():
    rows=[{'id':'done','family':'singleton','curriculum':{'rl':{'level':0}}},
          {'id':'fresh','family':'other','curriculum':{'rl':{'level':0}}}]
    stats=[{'task_id':'done','scored':6,'mean':1.}]
    selected=production.select_tasks(rows,stats,{'rows':2,'group':4,'offset':0},curriculum=DEFAULT)
    assert [item['row'] for item in selected]==[1,0]


@pytest.mark.parametrize('fraction',[float('nan'),float('inf'),-0.1,1.1,True])
def test_invalid_family_exploration_fraction_is_rejected(fraction):
    with pytest.raises(ValueError,match='family_exploration_fraction'):
        production.select_tasks(cases(),[],{'rows':3,'group':4,'offset':0},
                                curriculum={**DEFAULT,'family_exploration_fraction':fraction})


def test_family_metadata_is_not_fabricated_from_task_ids():
    from natlang_neuralese.rl.curriculum import task_descriptors
    rows=[{'id':'a'},{'id':'b','family':None}]
    assert [row['family'] for row in task_descriptors(rows)]==['unspecified','unspecified']
    with pytest.raises(ValueError,match='family'):
        task_descriptors([{'id':'bad','family':['not','an','identifier']}])
