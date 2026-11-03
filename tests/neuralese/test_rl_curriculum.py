import pytest
from natlang_neuralese.rl import production
from natlang_neuralese.rl.curriculum import DEFAULT_V1 as DEFAULT, DEFAULT as DEFAULT_V2


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


def test_mastered_family_advances_independently_of_unrelated_failures():
    rows=[{'id':'math-easy','family':'math','curriculum':{'rl':{'level':2}}},
          {'id':'math-hard','family':'math','curriculum':{'rl':{'level':3}}},
          {'id':'tools-easy','family':'tools','curriculum':{'rl':{'level':2}}},
          {'id':'tools-hard','family':'tools','curriculum':{'rl':{'level':3}}},
          {'id':'breadth','family':'other','curriculum':{'rl':{'level':0}}}]
    stats=[{'task_id':'math-easy','scored':4,'mean':1.},
           {'task_id':'tools-easy','scored':4,'mean':.5}]
    chosen=production.select_tasks(rows,stats,{'rows':3,'group':4,'offset':4},
                                  curriculum={**DEFAULT,'family_exploration_fraction':0.})
    ids=[rows[item['row']]['id'] for item in chosen]
    assert ids==['tools-easy','math-hard','breadth']
    assert all(item['samples']==4 for item in chosen)


def test_curriculum_cannot_silently_reintroduce_a_global_difficulty_gate():
    with pytest.raises(ValueError,match='frontier_scope'):
        production.select_tasks(cases(),[],{'rows':3,'group':4,'offset':0},
                                curriculum={**DEFAULT,'frontier_scope':'global'})


# -- policy /2: median start, spread-driven difficulty, recorded transition -----------------------------------------

def family_rows(family,levels,per_level=3):
    return [{'id':f'{family}-{level}-{index}','family':family,'curriculum':{'rl':{'level':level}}}
            for level in levels for index in range(per_level)]


def test_v2_new_families_start_at_their_median_level():
    rows=family_rows('math',[0,1,2,3])+family_rows('qa',[0,1,2])
    chosen=production.select_tasks(rows,[],{'rows':4,'group':8,'offset':0},curriculum=DEFAULT_V2)
    practice=[rows[item['row']] for item in chosen[:-1]]
    levels={row['family']:row['curriculum']['rl']['level'] for row in practice}
    assert levels=={'math':1,'qa':1}
    # The historical /1 selector starts both at their easiest level.
    chosen=production.select_tasks(rows,[],{'rows':4,'group':8,'offset':0},curriculum=DEFAULT)
    assert {rows[item['row']]['curriculum']['rl']['level'] for item in chosen[:2]}=={0}
    assert {item['samples'] for item in chosen}=={8}


def test_v2_skipped_easier_levels_count_as_passed_after_median_start():
    rows=family_rows('math',[0,1,2,3])
    stats=[{'task_id':'math-1-0','scored':8,'mean':1.}]
    chosen=production.select_tasks(rows,stats,{'rows':3,'group':8,'offset':0},
                                  curriculum={**DEFAULT_V2,'family_exploration_fraction':0.})
    assert rows[chosen[0]['row']]['curriculum']['rl']['level']==2


def outcomes(round_number,count,mean,spread,prefix='seen'):
    return [{'task_id':f'{prefix}-{round_number}-{index}','scored':8,'mean':mean,'spread':spread,
             'observed_round':round_number} for index in range(count)]


def test_v2_low_spread_rounds_raise_or_lower_practice_difficulty():
    from natlang_neuralese.rl.curriculum import difficulty_shift, policy_options
    config=policy_options(DEFAULT_V2)
    saturated=outcomes(0,4,1.,False)+outcomes(1,4,1.,False)
    assert difficulty_shift(saturated,config)==2
    # An informative newer round resets the shift; failures move practice down.
    assert difficulty_shift(saturated+outcomes(2,3,.5,True,'mixed')+outcomes(2,4,1.,False),config)==0
    assert difficulty_shift(outcomes(0,5,0.,False),config)==-1
    # Direction changes stop the walk; the limit bounds it.
    assert difficulty_shift(outcomes(0,4,0.,False)+outcomes(1,4,1.,False),config)==1
    many=[row for number in range(6) for row in outcomes(number,4,1.,False)]
    assert difficulty_shift(many,config)==3
    # Observations without round/spread telemetry do not move difficulty.
    assert difficulty_shift([{'task_id':'x','scored':4,'mean':1.}],config)==0


def test_v2_shift_moves_every_family_and_stays_within_its_levels():
    rows=family_rows('math',[0,1,2,3])+family_rows('qa',[0,1])
    stats=[{**row,'task_id':'math-1-0'} for row in outcomes(0,1,1.,False)]
    stats+=[{**row,'task_id':'qa-0-0'} for row in outcomes(0,1,1.,False)]
    chosen=production.select_tasks(rows,stats,{'rows':3,'group':8,'offset':0},
                                  curriculum={**DEFAULT_V2,'family_exploration_fraction':0.})
    practice=[rows[item['row']] for item in chosen[:-1]]
    assert {row['family']:row['curriculum']['rl']['level'] for row in practice}=={'math':3,'qa':1}


def test_v2_options_are_validated_and_v1_rejects_v2_fields():
    with pytest.raises(ValueError,match='spread_target'):
        production.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT_V2,'spread_target':2})
    with pytest.raises(ValueError,match='/1 does not accept'):
        production.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT,'spread_target':.3})
    with pytest.raises(ValueError,match='new_family_start'):
        production.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT_V2,'new_family_start':'hard'})


def test_v1_config_adopts_v2_through_one_recorded_transition(tmp_path):
    state={'round':3};path=tmp_path/'state.json'
    config={'task_curriculum':{'schema':'natlang.rl-task-curriculum/1','breadth_fraction':.25,
            'promotion_success_rate':.6},'rows_per_round':4,'group':4}
    adopted=production.adopt_task_curriculum_defaults(config,state,path)
    assert adopted['task_curriculum']==DEFAULT_V2
    assert (adopted['rows_per_round'],adopted['group'])==(16,8)
    [record]=state['task_curriculum_transitions']
    assert record['from']['rows_per_round']==4 and record['to']['group']==8 and record['round']==3
    assert production.adopt_task_curriculum_defaults(config,state,path)==adopted
    assert len(state['task_curriculum_transitions'])==1
    # Larger explicit budgets stay; a sealed in-flight selection and the opt-out keep the recorded policy.
    assert production.adopt_task_curriculum_defaults({**config,'rows_per_round':32},{},path)['rows_per_round']==32
    assert production.adopt_task_curriculum_defaults(config,{},path,sealed_selection=True)==config
    assert production.adopt_task_curriculum_defaults({**config,'task_curriculum_keep_v1':True},{},path)['group']==4
    assert production.adopt_task_curriculum_defaults({'rows_per_round':4},{},path)=={'rows_per_round':4}


def test_spread_yield_counts_only_scored_groups():
    from natlang_neuralese.rl.curriculum import spread_yield
    assert spread_yield([{'scored':4,'spread':True},{'scored':4,'spread':False},{'scored':0,'spread':True}])=={
        'scored_groups':2,'groups_with_spread':1,'spread_fraction':.5}
