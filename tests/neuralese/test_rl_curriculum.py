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


def test_class_quota_is_filled_by_the_selector_and_enforced_on_an_external_policy(monkeypatch):
    import json
    from types import SimpleNamespace
    rows=[{'id':f'q-{index}','curriculum':{'rl':{'level':0,'family':f'f{index%7}','qualification':'qualified',
           **({'mix_class':'effect_state'} if index%10==0 else {})}}} for index in range(100)]
    budget={'rows':16,'group':8,'offset':3}
    chosen=production.select_tasks(rows,[],budget,curriculum=DEFAULT_V2)
    effect=[item['row'] for item in chosen if rows[item['row']]['curriculum']['rl'].get('mix_class')]
    assert len(effect)>=6 and len({item['row'] for item in chosen})==16
    # A pool without tagged rows selects exactly as the quota-free policy.
    plain=[{**row,'curriculum':{'rl':{key:value for key,value in row['curriculum']['rl'].items() if key!='mix_class'}}} for row in rows]
    assert production.select_tasks(plain,[],budget,curriculum=DEFAULT_V2)==production.select_tasks(
        plain,[],budget,curriculum={**DEFAULT_V2,'class_quota':{}})
    # An external policy that ignores the quota is refused.
    ignoring=[item for item in chosen if item not in [c for c in chosen if rows[c['row']]['curriculum']['rl'].get('mix_class')]]
    ignoring=ignoring[:-4]+[c for c in chosen if rows[c['row']]['curriculum']['rl'].get('mix_class')][:0]+chosen[-4:]
    ignoring=list({item['row']:item for item in ignoring}.values())
    monkeypatch.setattr(production.subprocess,'run',lambda argv,**kwargs:SimpleNamespace(stdout=json.dumps(ignoring)))
    with pytest.raises(ValueError,match='class effect_state'):
        production.select_tasks(rows,[],budget,curriculum=DEFAULT_V2,command=['natlang-selector'])


def mixed_pool(count=60):
    return [{'id':f'v-{index}','curriculum':{'rl':{'level':1,'family':'f','qualification':'qualified',
             **({'mix_class':'effect_state'} if index%4==0 else {})}}} for index in range(count)]


def observed_mixed(rows,visits=None,rounds=None):
    """Every row observed once with a mixed outcome (the spread the selector seeks); ``visits`` overrides per row id."""
    return [{'task_id':row['id'],'scored':8,'mean':.5,'spread':True,'observed_round':10,
             'visits':(visits or {}).get(row['id'],1),'visit_rounds':(rounds or {}).get(row['id'],[10])} for row in rows]


def test_record_visit_counts_each_round_once_and_never_overcounts_untracked_history():
    from natlang_neuralese.rl.curriculum import record_visit,VISIT_ROUNDS_KEPT
    first=record_visit(None,4)
    assert first=={'visits':1,'visit_rounds':[4]}
    second=record_visit({**first,'observed_round':4},9)
    assert second=={'visits':2,'visit_rounds':[4,9]}
    assert record_visit({**second,'observed_round':9},9)==second  # a retried collection of the same round
    # An observation from before visit tracking is one visit, and its own round is not counted again.
    assert record_visit({'observed_round':3,'task_id':'x'},3)=={'visits':1,'visit_rounds':[3]}
    assert record_visit({'observed_round':3,'task_id':'x'},5)=={'visits':2,'visit_rounds':[5]}
    state=None
    for number in range(VISIT_ROUNDS_KEPT+10):state=record_visit(state,number)
    assert state['visits']==VISIT_ROUNDS_KEPT+10 and len(state['visit_rounds'])==VISIT_ROUNDS_KEPT


def test_visit_penalty_moves_an_overvisited_mixed_row_behind_fresh_rows_of_the_same_level():
    rows=mixed_pool();budget={'rows':8,'group':8,'offset':3}  # breadth holds rows 3-4; rotation order then starts at row 5
    off={**DEFAULT_V2,'visit_penalty':0,'visit_cap':0,'class_quota':{},'family_exploration_fraction':0.}
    on={**off,'visit_penalty':1.0}
    pick=lambda stats,config:[item['row'] for item in production.select_tasks(rows,stats,budget,curriculum=config)]
    heavy=observed_mixed(rows,visits={'v-5':40})
    # Without decay the mixed row is picked as soon as its rotation turn comes; with it, it waits behind fresher rows.
    assert 5 in pick(heavy,off) and 5 not in pick(heavy,on)
    # A pool where every row has the same visits has nothing to decay: the typical count is the row's own.
    even=observed_mixed(rows,visits={row['id']:40 for row in rows})
    assert pick(even,on)==pick(even,off)


def test_visit_cap_ranks_capped_rows_last_inside_the_window_and_releases_them_after_it():
    rows=mixed_pool();budget={'rows':8,'group':8,'offset':0}
    config={**DEFAULT_V2,'visit_penalty':0,'visit_cap':2,'visit_cap_window':50,'class_quota':{},'family_exploration_fraction':0.}
    hot=['v-5','v-6','v-7']
    stats=observed_mixed(rows,visits={r:2 for r in hot},rounds={r:[8,10] for r in hot})
    chosen={rows[item['row']]['id'] for item in production.select_tasks(rows,stats,budget,curriculum=config)}
    assert not chosen&set(hot)
    # The same visits older than the window no longer cap them.
    old=observed_mixed(rows,visits={r:2 for r in hot},rounds={r:[8,10] for r in hot})
    for item in old:item['observed_round']=200
    for item in old:
        if item['task_id'] not in hot:item['visit_rounds']=[200]
    released=production.select_tasks(rows,old,budget,curriculum=config)
    assert released==production.select_tasks(rows,old,budget,curriculum={**config,'visit_cap':0})


def test_visit_cap_still_fills_the_round_when_the_pool_has_too_few_uncapped_rows():
    rows=mixed_pool(10)
    config={**DEFAULT_V2,'visit_penalty':1.,'visit_cap':1,'visit_cap_window':50,'class_quota':{}}
    stats=observed_mixed(rows,visits={row['id']:5 for row in rows},rounds={row['id']:[9] for row in rows})
    chosen=production.select_tasks(rows,stats,{'rows':8,'group':8,'offset':2},curriculum=config)
    assert len(chosen)==8 and len({item['row'] for item in chosen})==8 and all(item['samples']==8 for item in chosen)


def test_visit_decay_keeps_class_quota_breadth_and_floor():
    rows=mixed_pool(200);budget={'rows':16,'group':8,'offset':20}
    hot={row['id']:30 for row in rows[:60]}
    stats=observed_mixed(rows,visits=hot,rounds={r:[7,8,9] for r in hot})
    chosen=production.select_tasks(rows,stats,budget,curriculum=DEFAULT_V2)
    from natlang_neuralese.rl.curriculum import breadth_rows
    assert len(chosen)==16 and {item['row'] for item in chosen}>=set(breadth_rows(rows,budget,DEFAULT_V2))
    effect=[item['row'] for item in chosen if rows[item['row']]['curriculum']['rl'].get('mix_class')]
    assert len(effect)>=6  # ceil(.34*16)
    assert not {item['row'] for item in chosen if rows[item['row']]['id'] in hot}-set(breadth_rows(rows,budget,DEFAULT_V2))


def test_untracked_observations_and_a_disabled_decay_select_as_before():
    rows=mixed_pool();budget={'rows':8,'group':8,'offset':5}
    legacy=[{'task_id':row['id'],'scored':8,'mean':.5,'spread':True,'observed_round':3} for row in rows]
    off={**DEFAULT_V2,'visit_penalty':0,'visit_cap':0}
    assert production.select_tasks(rows,legacy,budget,curriculum=DEFAULT_V2)==production.select_tasks(rows,legacy,budget,curriculum=off)


def test_repeated_selection_spreads_visits_across_the_pool():
    from natlang_neuralese.rl.curriculum import record_visit
    rows=mixed_pool(120)
    def run(config,rounds=150):
        history={};offset=0;visits={}
        for number in range(rounds):
            chosen=production.select_tasks(rows,list(history.values()),{'rows':16,'group':8,'offset':offset},curriculum=config)
            for item in chosen:
                row=rows[item['row']]
                history[row['id']]={'task_id':row['id'],'scored':8,'mean':.5,'spread':True,'observed_round':number,
                                     **record_visit(history.get(row['id']),number)}
                visits[row['id']]=visits.get(row['id'],0)+1
            offset=(chosen[-1]['row']+1)%len(rows)
        return visits
    off=run({**DEFAULT_V2,'visit_penalty':0,'visit_cap':0})
    on=run(DEFAULT_V2)
    assert len(on)>=len(off) and max(on.values())<max(off.values())
    assert max(on.values())<=.6*max(off.values())


def test_visit_decay_options_are_validated():
    for bad in ({'visit_penalty':-1},{'visit_cap':-1},{'visit_cap_window':1.5},{'visit_cap':65},{'visit_penalty':float('nan')}):
        with pytest.raises(ValueError,match='visit'):
            production.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT_V2,**bad})
    with pytest.raises(ValueError,match='/1 does not accept'):
        production.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT,'visit_cap':2})


def test_v2_config_without_visit_decay_adopts_it_through_its_own_recorded_transition(tmp_path):
    state={'round':9};path=tmp_path/'state.json'
    legacy={key:value for key,value in DEFAULT_V2.items() if not key.startswith('visit_')}
    config={'task_curriculum':legacy,'rows_per_round':32,'group':8}
    adopted=production.adopt_task_curriculum_defaults(config,state,path)
    assert adopted['task_curriculum']==DEFAULT_V2 and (adopted['rows_per_round'],adopted['group'])==(32,8)
    [record]=state['task_curriculum_transitions']
    assert record['transition']==production.TASK_CURRICULUM_VISIT_DECAY_TRANSITION=='task-curriculum-visit-decay-20261104'
    assert record['adopted_options']==['visit_cap','visit_cap_window','visit_penalty'] and record['round']==9
    assert production.adopt_task_curriculum_defaults(config,state,path)==adopted and len(state['task_curriculum_transitions'])==1
    # A sealed in-flight selection, the explicit opt-out and an already adopted config keep what they have.
    assert production.adopt_task_curriculum_defaults(config,{},path,sealed_selection=True)==config
    assert production.adopt_task_curriculum_defaults(adopted,{},path)==adopted
    # An explicit opt-out of the decay itself is a complete declaration, not a missing field.
    off={**DEFAULT_V2,'visit_penalty':0,'visit_cap':0}
    assert production.adopt_task_curriculum_defaults({**config,'task_curriculum':off},{},path)['task_curriculum']==off
    # A /2 config that also predates class_quota records the earlier transition for the combined default adoption.
    older={key:value for key,value in legacy.items() if key!='class_quota'}
    state={};production.adopt_task_curriculum_defaults({**config,'task_curriculum':older},state,path)
    assert state['task_curriculum_transitions'][0]['transition']==production.TASK_CURRICULUM_CLASS_QUOTA_TRANSITION
