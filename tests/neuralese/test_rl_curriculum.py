import json

import pytest
from natlang_neuralese.rl import curriculum
from natlang_neuralese.rl.curriculum import DEFAULT_V1 as DEFAULT, DEFAULT as DEFAULT_V2


def cases():
    return [{'id':f'task-{index}','curriculum':{'rl':{'level':index//3,'qualification':'qualified'}}}
            for index in range(12)]


def test_practice_starts_easy_and_grows_from_actual_successes():
    rows=cases();budget={'rows':3,'group':4,'offset':9}
    chosen=curriculum.select_tasks(rows,[],budget,curriculum=DEFAULT)
    assert [item['row'] for item in chosen]==[0,1,9]
    stats=[{'task_id':row['id'],'scored':4,'mean':1.} for row in rows[:3]]
    chosen=curriculum.select_tasks(rows,stats,budget,curriculum=DEFAULT)
    assert [item['row'] for item in chosen]==[3,4,9]
    # Unscored infrastructure cannot masquerade as mastery or a scored failure.
    stats.append({'task_id':'task-3','scored':0,'mean':None})
    assert curriculum.select_tasks(rows,stats,budget,curriculum=DEFAULT)==chosen


def test_rotating_breadth_reaches_every_task_even_when_solved_or_harder():
    rows=cases();stats=[{'task_id':row['id'],'scored':4,'mean':1.} for row in rows]
    offset=0;seen=set()
    for _ in rows:
        chosen=curriculum.select_tasks(rows,stats,{'rows':3,'group':4,'offset':offset},curriculum=DEFAULT)
        seen.update(item['row'] for item in chosen);offset=(chosen[-1]['row']+1)%len(rows)
    assert seen==set(range(len(rows)))
    rows.append({'id':'new-hard','curriculum':{'rl':{'level':8}}})
    for _ in rows:
        chosen=curriculum.select_tasks(rows,stats,{'rows':3,'group':4,'offset':offset},curriculum=DEFAULT)
        seen.update(item['row'] for item in chosen);offset=(chosen[-1]['row']+1)%len(rows)
    assert 12 in seen


def test_pending_scaffolds_are_not_silently_adopted():
    rows=cases();rows[0]['curriculum']['rl']['qualification']='pending-live-48'
    with pytest.raises(ValueError,match='qualification'):
        curriculum.select_tasks(rows,[],{'rows':3,'group':4,'offset':0},curriculum=DEFAULT)


def test_nl_policy_receives_typed_facts_and_cannot_drop_breadth(monkeypatch):
    import json
    from types import SimpleNamespace
    captured=[]
    def command(argv,**kwargs):
        captured.append(json.loads(kwargs['input']))
        return SimpleNamespace(stdout=json.dumps([{'row':0,'samples':4},{'row':1,'samples':4}]))
    monkeypatch.setattr(curriculum.subprocess,'run',command)
    with pytest.raises(ValueError,match='breadth'):
        curriculum.select_tasks(cases(),[],{'rows':2,'group':4,'offset':9},
                                curriculum=DEFAULT,command=['natlang-selector'])
    assert captured[0]['tasks'][0]=={'row':0,'id':'task-0','level':0,'family':'unspecified'}
    assert captured[0]['curriculum']==DEFAULT


def test_declared_selection_allocation_is_verified(monkeypatch):
    import json
    from types import SimpleNamespace
    monkeypatch.setattr(curriculum.subprocess,'run',lambda *a,**kw:SimpleNamespace(
        stdout=json.dumps([{'row':i,'samples':4} for i in range(3)])))
    with pytest.raises(ValueError,match='allocation'):
        curriculum.select_tasks(cases(),[],{'rows':2,'group':4,'offset':0},command=['selector'])


def test_mixed_outcome_practice_does_not_crowd_out_unvisited_families():
    rows=[{'id':f'qa-{i}','family':'qa','curriculum':{'rl':{'level':0}}} for i in range(8)]
    rows += [{'id':'new-program','family':'program-execution','curriculum':{'rl':{'level':1}}},
             {'id':'new-state','family':'stateful-control','curriculum':{'rl':{'level':0}}}]
    stats=[{'task_id':f'qa-{i}','scored':6,'mean':.5} for i in range(8)]
    budget={'rows':6,'group':6,'offset':0}
    selected=curriculum.select_tasks(rows,stats,budget,curriculum=DEFAULT)
    ids={rows[item['row']]['id'] for item in selected}
    assert {'new-program','new-state'}<=ids
    assert any(value.startswith('qa-') for value in ids)
    assert len(selected)==6 and {item['samples'] for item in selected}=={6}
    assert [item['row'] for item in selected[-2:]]==[0,1]
    # A retired task's historical outcome is not current-source coverage.
    assert curriculum.select_tasks(rows,[*stats,{'task_id':'retired','scored':1000,'mean':1.}],
                                   budget,curriculum=DEFAULT)==selected


def test_exploration_does_not_repeat_solved_singleton_instead_of_new_invocations():
    rows=[{'id':'done','family':'singleton','curriculum':{'rl':{'level':0}}},
          {'id':'fresh','family':'other','curriculum':{'rl':{'level':0}}}]
    stats=[{'task_id':'done','scored':6,'mean':1.}]
    selected=curriculum.select_tasks(rows,stats,{'rows':2,'group':4,'offset':0},curriculum=DEFAULT)
    assert [item['row'] for item in selected]==[1,0]


@pytest.mark.parametrize('fraction',[float('nan'),float('inf'),-0.1,1.1,True])
def test_invalid_family_exploration_fraction_is_rejected(fraction):
    with pytest.raises(ValueError,match='family_exploration_fraction'):
        curriculum.select_tasks(cases(),[],{'rows':3,'group':4,'offset':0},
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
    chosen=curriculum.select_tasks(rows,stats,{'rows':3,'group':4,'offset':4},
                                  curriculum={**DEFAULT,'family_exploration_fraction':0.})
    ids=[rows[item['row']]['id'] for item in chosen]
    assert ids==['tools-easy','math-hard','breadth']
    assert all(item['samples']==4 for item in chosen)


def test_curriculum_cannot_silently_reintroduce_a_global_difficulty_gate():
    with pytest.raises(ValueError,match='frontier_scope'):
        curriculum.select_tasks(cases(),[],{'rows':3,'group':4,'offset':0},
                                curriculum={**DEFAULT,'frontier_scope':'global'})


# -- policy /2: median start, spread-driven difficulty, recorded transition -----------------------------------------

def family_rows(family,levels,per_level=3):
    return [{'id':f'{family}-{level}-{index}','family':family,'curriculum':{'rl':{'level':level}}}
            for level in levels for index in range(per_level)]


def test_v2_new_families_start_at_their_median_level():
    rows=family_rows('math',[0,1,2,3])+family_rows('qa',[0,1,2])
    chosen=curriculum.select_tasks(rows,[],{'rows':4,'group':8,'offset':0},curriculum=DEFAULT_V2)
    practice=[rows[item['row']] for item in chosen[:-1]]
    levels={row['family']:row['curriculum']['rl']['level'] for row in practice}
    assert levels=={'math':1,'qa':1}
    # The historical /1 selector starts both at their easiest level.
    chosen=curriculum.select_tasks(rows,[],{'rows':4,'group':8,'offset':0},curriculum=DEFAULT)
    assert {rows[item['row']]['curriculum']['rl']['level'] for item in chosen[:2]}=={0}
    assert {item['samples'] for item in chosen}=={8}


def test_v2_skipped_easier_levels_count_as_passed_after_median_start():
    rows=family_rows('math',[0,1,2,3])
    stats=[{'task_id':'math-1-0','scored':8,'mean':1.}]
    chosen=curriculum.select_tasks(rows,stats,{'rows':3,'group':8,'offset':0},
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
    chosen=curriculum.select_tasks(rows,stats,{'rows':3,'group':8,'offset':0},
                                  curriculum={**DEFAULT_V2,'family_exploration_fraction':0.})
    practice=[rows[item['row']] for item in chosen[:-1]]
    assert {row['family']:row['curriculum']['rl']['level'] for row in practice}=={'math':3,'qa':1}


def test_v2_options_are_validated_and_v1_rejects_v2_fields():
    with pytest.raises(ValueError,match='spread_target'):
        curriculum.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT_V2,'spread_target':2})
    with pytest.raises(ValueError,match='/1 does not accept'):
        curriculum.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT,'spread_target':.3})
    with pytest.raises(ValueError,match='new_family_start'):
        curriculum.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT_V2,'new_family_start':'hard'})


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
    chosen=curriculum.select_tasks(rows,[],budget,curriculum=DEFAULT_V2)
    effect=[item['row'] for item in chosen if rows[item['row']]['curriculum']['rl'].get('mix_class')]
    assert len(effect)>=6 and len({item['row'] for item in chosen})==16
    # A pool without tagged rows selects exactly as the quota-free policy.
    plain=[{**row,'curriculum':{'rl':{key:value for key,value in row['curriculum']['rl'].items() if key!='mix_class'}}} for row in rows]
    assert curriculum.select_tasks(plain,[],budget,curriculum=DEFAULT_V2)==curriculum.select_tasks(
        plain,[],budget,curriculum={**DEFAULT_V2,'class_quota':{}})
    # An external policy that ignores the quota is refused.
    ignoring=[item for item in chosen if item not in [c for c in chosen if rows[c['row']]['curriculum']['rl'].get('mix_class')]]
    ignoring=ignoring[:-4]+[c for c in chosen if rows[c['row']]['curriculum']['rl'].get('mix_class')][:0]+chosen[-4:]
    ignoring=list({item['row']:item for item in ignoring}.values())
    monkeypatch.setattr(curriculum.subprocess,'run',lambda argv,**kwargs:SimpleNamespace(stdout=json.dumps(ignoring)))
    with pytest.raises(ValueError,match='class effect_state'):
        curriculum.select_tasks(rows,[],budget,curriculum=DEFAULT_V2,command=['natlang-selector'])


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
    pick=lambda stats,config:[item['row'] for item in curriculum.select_tasks(rows,stats,budget,curriculum=config)]
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
    chosen={rows[item['row']]['id'] for item in curriculum.select_tasks(rows,stats,budget,curriculum=config)}
    assert not chosen&set(hot)
    # The same visits older than the window no longer cap them.
    old=observed_mixed(rows,visits={r:2 for r in hot},rounds={r:[8,10] for r in hot})
    for item in old:item['observed_round']=200
    for item in old:
        if item['task_id'] not in hot:item['visit_rounds']=[200]
    released=curriculum.select_tasks(rows,old,budget,curriculum=config)
    assert released==curriculum.select_tasks(rows,old,budget,curriculum={**config,'visit_cap':0})


def test_visit_cap_still_fills_the_round_when_the_pool_has_too_few_uncapped_rows():
    rows=mixed_pool(10)
    config={**DEFAULT_V2,'visit_penalty':1.,'visit_cap':1,'visit_cap_window':50,'class_quota':{}}
    stats=observed_mixed(rows,visits={row['id']:5 for row in rows},rounds={row['id']:[9] for row in rows})
    chosen=curriculum.select_tasks(rows,stats,{'rows':8,'group':8,'offset':2},curriculum=config)
    assert len(chosen)==8 and len({item['row'] for item in chosen})==8 and all(item['samples']==8 for item in chosen)


def test_visit_decay_keeps_class_quota_breadth_and_floor():
    rows=mixed_pool(200);budget={'rows':16,'group':8,'offset':20}
    hot={row['id']:30 for row in rows[:60]}
    stats=observed_mixed(rows,visits=hot,rounds={r:[7,8,9] for r in hot})
    chosen=curriculum.select_tasks(rows,stats,budget,curriculum=DEFAULT_V2)
    from natlang_neuralese.rl.curriculum import breadth_rows
    assert len(chosen)==16 and {item['row'] for item in chosen}>=set(breadth_rows(rows,budget,DEFAULT_V2))
    effect=[item['row'] for item in chosen if rows[item['row']]['curriculum']['rl'].get('mix_class')]
    assert len(effect)>=6  # ceil(.34*16)
    assert not {item['row'] for item in chosen if rows[item['row']]['id'] in hot}-set(breadth_rows(rows,budget,DEFAULT_V2))


def test_untracked_observations_and_a_disabled_decay_select_as_before():
    rows=mixed_pool();budget={'rows':8,'group':8,'offset':5}
    legacy=[{'task_id':row['id'],'scored':8,'mean':.5,'spread':True,'observed_round':3} for row in rows]
    off={**DEFAULT_V2,'visit_penalty':0,'visit_cap':0}
    assert curriculum.select_tasks(rows,legacy,budget,curriculum=DEFAULT_V2)==curriculum.select_tasks(rows,legacy,budget,curriculum=off)


def test_repeated_selection_spreads_visits_across_the_pool():
    from natlang_neuralese.rl.curriculum import record_visit
    rows=mixed_pool(120)
    def run(config,rounds=150):
        history={};offset=0;visits={}
        for number in range(rounds):
            chosen=curriculum.select_tasks(rows,list(history.values()),{'rows':16,'group':8,'offset':offset},curriculum=config)
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
            curriculum.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT_V2,**bad})
    with pytest.raises(ValueError,match='/1 does not accept'):
        curriculum.select_tasks(cases(),[],{'rows':3,'group':8,'offset':0},curriculum={**DEFAULT,'visit_cap':2})


# ---- the held-out evaluation set (rl.held_eval) -------------------------------------------------------------------------
def held_pool(families=8, per=12):
    rows=[]
    for family in range(families):
        for index in range(per):
            rows.append({'id':f'f{family}-{index}','split':'train','family':f'fam{family}',
                         'curriculum':{'split_group':f'g{family}-{index//2}',
                                       'rl':{'level':index%4,'family':f'fam{family}','qualification':'qualified',
                                             **({'mix_class':'effect_state'} if family%3==0 else {})}}})
    return rows


def write_pool(tmp_path,rows):
    path=tmp_path/'pool.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return path


def test_held_manifest_is_stratified_deterministic_and_avoids_attempted_tasks(tmp_path):
    from natlang_neuralese.rl import held_eval
    pool=write_pool(tmp_path,held_pool())
    avoid={'f0-0','f1-5'}
    manifest,rows=held_eval.build_manifest(pool,avoid=avoid,tasks=12)
    again,_=held_eval.build_manifest(pool,avoid=avoid,tasks=12)
    assert manifest==again and len(rows)==12 and len({t['id'] for t in manifest['tasks']})==12
    assert not {t['id'] for t in manifest['tasks']}&avoid
    # A split-group sibling of an attempted task is not held out either (related worlds stay on one side).
    assert not {t['id'] for t in manifest['tasks']}&{'f0-1','f1-4'}
    assert manifest['effect_state']==4 and sorted(manifest['by_level'].values())==[3,3,3,3]
    assert len({t['split_group'] for t in manifest['tasks']})==12 and manifest['families']>=6
    with pytest.raises(ValueError,match='eligible held tasks'):
        held_eval.build_manifest(pool,avoid=avoid,tasks=500)


def test_a_larger_held_set_keeps_the_smaller_one_and_exposes_its_exclusion_ids(tmp_path):
    from natlang_neuralese.rl import held_eval
    pool=write_pool(tmp_path,held_pool())
    small,rows=held_eval.build_manifest(pool,avoid={'f0-0'},tasks=8)
    kept={t['id'] for t in small['tasks']}
    groups={t['split_group'] for t in small['tasks']}
    other=next(row for row in held_pool() if row['curriculum']['split_group'] not in groups)
    sibling=next(row['id'] for row in held_pool() if row['curriculum']['split_group']==other['curriculum']['split_group']
                 and row['id']!=other['id'])
    big,big_rows=held_eval.build_manifest(pool,avoid={other['id']},tasks=24,include=kept)
    ids={t['id'] for t in big['tasks']}
    assert kept<=ids and len(ids)==24 and not {other['id'],sibling}&ids and big['effect_state']==8
    held_eval.write_manifest(tmp_path/'held',big,big_rows)
    excluded=held_eval.exclusion_ids(tmp_path/'held'/'manifest.json')
    siblings={row['id'] for row in held_pool() if row['curriculum']['split_group'] in set(big['excluded_split_groups'])}
    assert excluded==siblings and ids<=excluded and len(excluded)==48
    with pytest.raises(ValueError,match='also to be avoided'):
        held_eval.build_manifest(pool,avoid=kept,tasks=24,include=kept)


def test_held_tasks_and_their_group_siblings_are_never_selected_from_the_adopting_round(tmp_path):
    from natlang_neuralese.rl import held_eval
    rows=held_pool();pool=write_pool(tmp_path,rows)
    state={'round':2,'attempted_task_ids':['f1-1']};state_path=tmp_path/'state.json'
    config={'cases':str(pool),'held_eval':{'tasks':8}}
    adopted=held_eval.adopt_defaults(config,state,state_path,out=tmp_path,sealed_selection=True)
    [record]=state['held_eval_transitions']
    assert record['transition']=='held-eval-exclusion-20261110' and record['from_round']==3
    manifest=json.loads((tmp_path/'held-eval'/'manifest.json').read_text())
    assert adopted['held_eval_binding']['sha256']==record['manifest_sha256'] and len(manifest['tasks'])==8
    assert held_eval.adopt_defaults(config,state,state_path,out=tmp_path,sealed_selection=True)==adopted
    assert len(state['held_eval_transitions'])==1
    held={t['id'] for t in manifest['tasks']};groups=set(manifest['excluded_split_groups'])
    assert held_eval.exclusion(adopted,rows,2)==frozenset()          # a sealed round keeps its selection
    excluded=held_eval.exclusion(adopted,rows,3)
    assert held<=excluded and all(r['curriculum']['split_group'] in groups for r in rows if r['id'] in excluded)
    offset=0;seen=set()
    for _ in range(40):
        chosen=curriculum.select_tasks(rows,[],{'rows':16,'group':8,'offset':offset},curriculum=DEFAULT_V2,exclude_ids=excluded)
        ids={rows[item['row']]['id'] for item in chosen}
        assert not ids&excluded and len(chosen)==16
        seen|=ids;offset=(chosen[-1]['row']+1)%len(rows)
    assert seen==({r['id'] for r in rows}-excluded)                  # breadth still reaches every remaining task
    # An external selector sees the pool without the held rows and its answer maps back to the full pool.
    from unittest import mock
    sub=[r for r in rows if r['id'] not in excluded]
    def fake(command,input,**kw):
        request=json.loads(input);assert len(request['tasks'])==len(sub)
        return mock.Mock(stdout=json.dumps([{'row':len(sub)-1,'samples':4}]))
    with mock.patch.object(curriculum.subprocess,'run',fake):
        chosen=curriculum.select_tasks(rows,[],{'rows':1,'group':4,'offset':0},command=['x'],exclude_ids=excluded)
    assert rows[chosen[0]['row']]['id']==sub[-1]['id']
    # Opt-out keeps the whole pool selectable and records nothing.
    off=held_eval.adopt_defaults({**config,'held_eval':False},{},state_path,out=tmp_path)
    assert 'held_eval_binding' not in off and held_eval.exclusion(off,rows,9)==frozenset()


def held_result(policy,started,outcomes,key='s1'):
    return {'schema':held_eval.RESULT_SCHEMA,'policy_id':policy,'started_at':started,'series':{'key':key},'complete':True,
            'ran':len(outcomes),'scored':sum(o is not None for o in outcomes.values()),
            'successes':sum(o==1.0 for o in outcomes.values()),'loop_rate':.1,'repeated_action_rate':.2,'outcome_counts':{},
            'samples':[{'task_id':t,'sample':s,'reward':r,'ran':True} for (t,s),r in outcomes.items()]}


from natlang_neuralese.rl import held_eval  # noqa: E402


def test_held_paired_comparison_is_exact_on_matched_scored_samples():
    tasks=[(f't{i}',s) for i in range(10) for s in (0,1)]
    a={k:1.0 if i<6 else 0.0 for i,k in enumerate(tasks)}
    b={k:1.0 if (i<6 or 10<=i<18) else 0.0 for i,k in enumerate(tasks)}
    b[tasks[0]]=None                      # unscored in one arm: not a pair
    result=held_eval.paired(held_result('a',1,a),held_result('b',2,b))
    assert result['pairs']==19 and result['only_first']==0 and result['only_second']==8
    assert result['mcnemar_exact_p']==round(2*0.5**8,4) and result['tasks_up']==4 and result['tasks_down']==0
    assert held_eval.binomial_two_sided(0,0)==1.0 and held_eval.binomial_two_sided(5,5)==1.0


def test_held_standalone_comparison_counts_timeouts_apart_and_bounds_the_change():
    tasks=[(f't{i}',s) for i in range(10) for s in (0,1)]
    a={k:1.0 if i<6 else 0.0 for i,k in enumerate(tasks)}
    b={k:1.0 if (i<6 or 10<=i<18) else 0.0 for i,k in enumerate(tasks)}
    b[tasks[0]]=None
    first,second=held_result('a',1,a),held_result('b',2,b)
    second['samples'][0]['timed_out']=True
    report=held_eval.compare(first,second)
    assert report['pairs_lost_to_timeout']==1 and report['second']['timeouts']==1 and report['first']['timeouts']==0
    assert report['paired']['pairs']==19 and report['success_rate_delta']==round(8/19,4)
    low,high=report['delta_ci95_task_bootstrap']
    assert 0<low<=report['success_rate_delta']<=high and report['loop_rate_delta']==0
    assert held_eval.compare(first,first)['success_rate_delta']==0


def test_held_trend_pairs_with_previous_and_first_within_one_series(tmp_path):
    tasks=[(f't{i}',s) for i in range(6) for s in (0,1)]
    def write(policy,started,successes,key='s1'):
        d=held_eval.policy_directory(tmp_path,policy);d.mkdir(parents=True)
        (d/'result.json').write_text(json.dumps(held_result(policy,started,{k:1.0 if i<successes else 0.0 for i,k in enumerate(tasks)},key)))
    write('a'*64,1,3);write('b'*64,2,5);write('c'*64,3,6);write('d'*64,4,4,key='s2')
    first,second,third,other=held_eval.trend(tmp_path)
    assert first['vs_previous'] is None and first['vs_first'] is None
    assert second['vs_previous']['only_second']==2 and second['vs_first'] is None    # the first is also the previous
    assert third['vs_previous']['only_second']==1 and third['vs_first']['only_second']==3
    assert other['vs_previous'] is None and other['vs_first'] is None                # a new series starts its own baseline
    (held_eval.policy_directory(tmp_path,'e'*64)).mkdir(parents=True)
    (held_eval.policy_directory(tmp_path,'e'*64)/'result.json').write_text(json.dumps({'schema':held_eval.RESULT_SCHEMA,'error':'x'}))
    assert len(held_eval.trend(tmp_path))==4                                         # a failed run is not a measurement


def test_held_trend_follows_the_lineage_after_a_rewind(tmp_path):
    tasks=[(f't{i}',s) for i in range(6) for s in (0,1)]
    def write(policy,started,successes):
        d=held_eval.policy_directory(tmp_path,policy);d.mkdir(parents=True)
        (d/'result.json').write_text(json.dumps(held_result(policy,started,{k:1.0 if i<successes else 0.0 for i,k in enumerate(tasks)})))
    a,b,c,d,e='a'*64,'b'*64,'c'*64,'d'*64,'e'*64     # a collected round 0; b,c = updates 0,1; d = window 0 (after round 1); e = rolled back
    write(d,1,5);write(b,2,3);write(e,3,2);write(c,4,6)             # measured out of lineage order; e was measured after d
    assert [r['policy'] for r in held_eval.trend(tmp_path)]==[d,b,e,c]                       # no state: by start time
    (tmp_path/'state.json').write_text(json.dumps({
        'task_curriculum_spread_yield':[{'round':0,'policy_id':a}],
        'completed_update_history':[{'round':0,'checkpoint':{'sha256':b}},{'round':1,'checkpoint':{'sha256':c}}],
        'interleave':{'windows':[{'controller_round':2,'checkpoint':{'sha256':d}}]},
        'rewind_history':[{'rolled_back_policies':[e,'f'*64]}]}))
    rows=held_eval.trend(tmp_path)
    assert [r['policy'] for r in rows]==[b,c,d]                                              # lineage order, e is not in it
    assert rows[1]['vs_previous']['only_second']==3 and rows[2]['vs_previous']['only_first']==1 and rows[2]['vs_first']['only_second']==2
    assert [r['policy'] for r in held_eval.rolled_back(tmp_path)]==[e] and 'vs_previous' not in held_eval.rolled_back(tmp_path)[0]


def test_held_measurement_never_blocks_the_round(tmp_path):
    logs=[]
    # Opted out, unadopted, or an actor that cannot be reached: no handle, a logged reason, no exception.
    assert held_eval.begin({'held_eval':False,'held_eval_binding':{}},out=tmp_path,policy_id='a'*64,endpoint='http://x',log=logs.append) is None
    assert held_eval.begin({},out=tmp_path,policy_id='a'*64,endpoint='http://x',log=logs.append) is None
    assert held_eval.begin({'held_eval_binding':{'manifest':'m','sha256':'0'}},out=tmp_path,policy_id='a'*64,
                           endpoint='http://127.0.0.1:9',log=logs.append) is None
    assert 'held eval did not start' in logs[-1]
    assert held_eval.Handle(tmp_path,{'pid':2**22+5,'starttime':1},1).finish() is None


def test_static_rows_related_to_a_held_task_are_recognised_with_or_without_the_source_prefix(tmp_path):
    from natlang_neuralese.rl import held_eval
    assert held_eval.static_relation(tmp_path) is None   # a controller without held tasks relates nothing
    directory = held_eval.directory_of(tmp_path)
    directory.mkdir()
    (directory / 'manifest.json').write_text(json.dumps({'excluded_split_groups': ['grp-a']}))
    (directory / 'cases.jsonl').write_text(json.dumps({'id': 't1', 'source_groups': ['2wiki-q:Q7'],
                                                       'curriculum': {'split_group': 'grp-a'}}) + '\n')
    related = held_eval.static_relation(tmp_path)
    assert related({'source_groups': ['grp-a']}) and related({'source_groups': ['2wiki:Q7']})
    assert not related({'source_groups': ['2wiki:Q8', 'other']}) and not related({})
