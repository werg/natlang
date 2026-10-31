"""Crisp adapter for the typed selectTrainingTasks natlang policy.

Historical outcomes choose work only; they are never behavior likelihoods or
current-policy qualification. Complete rotating breadth prevents a permanent
small curriculum endpoint, including when all observed tasks were successful.
"""
import math

DEFAULT={'schema':'natlang.rl-task-curriculum/1','breadth_fraction':.25,'promotion_success_rate':.6,
         'family_exploration_fraction':.5}


def metadata(row):
    value=(row.get('curriculum') or {}).get('rl') or {}
    level=value.get('level',3)
    if type(level) is not int or level<0:raise ValueError('task curriculum level must be a nonnegative integer')
    if value.get('qualification') not in (None,'qualified'):
        raise ValueError('task scaffold requires live model-facing qualification before production selection')
    family=value.get('family',row.get('family','unspecified'))
    if family is None or family=='':family='unspecified'
    if not isinstance(family,str):raise ValueError('task curriculum family must be a string or unknown')
    return {'level':level,'family':family}


def policy_options(value):
    config={**DEFAULT,**value}
    if config['schema']!=DEFAULT['schema']:raise ValueError('unsupported task curriculum policy schema')
    for key in ('breadth_fraction','promotion_success_rate'):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]) or not 0<config[key]<=1:
            raise ValueError('task curriculum '+key+' must be finite in (0,1]')
    fraction=config['family_exploration_fraction']
    if type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('task curriculum family_exploration_fraction must be finite in [0,1]')
    return config


def task_descriptors(cases):
    return [{'row':index,'id':row['id'],**metadata(row)} for index,row in enumerate(cases)]


def breadth_rows(cases,budget,config):
    count=min(len(cases),budget['rows'])
    if not count:return []
    size=min(count,max(1,math.ceil(count*config['breadth_fraction'])))
    return [(budget['offset']+index)%len(cases) for index in range(size)]


def select(cases,stats,budget,config):
    config=policy_options(config);tasks=task_descriptors(cases)
    if not tasks:return []
    observed={row['task_id']:row for row in stats if type(row.get('scored')) is int and row['scored']>0
              and type(row.get('mean')) in (int,float) and math.isfinite(row['mean']) and 0<=row['mean']<=1}
    levels=sorted({task['level'] for task in tasks});frontier=levels[-1]
    for level in levels:
        groups=[observed[task['id']] for task in tasks if task['level']==level and task['id'] in observed]
        scored=sum(group['scored'] for group in groups)
        if not scored or sum(group['mean']*group['scored'] for group in groups)/scored<config['promotion_success_rate']:
            frontier=level;break
    breadth=breadth_rows(cases,budget,config);reserved=set(breadth)
    def priority(index):
        task=tasks[index];group=observed.get(task['id']);mean=group['mean'] if group else None
        signal=0 if mean is not None and 0<mean<1 else 1 if mean is None else 2 if mean==0 else 3
        return (abs(task['level']-frontier),signal,(index-budget['offset'])%len(cases))
    practice=sorted((index for index in range(len(cases)) if index not in reserved),key=priority)
    count=min(budget['rows'],len(cases))
    slots=count-len(breadth)
    # The same few mixed-outcome tasks otherwise occupy almost all practice.
    # Reserve useful exploration by actual semantic family, without changing
    # physical breadth's cursor or treating historical outcomes as on-policy.
    served={}
    for task in tasks:
        if task['id'] in observed:
            served[task['family']]=served.get(task['family'],0)+1
    unknown={}
    for index in practice:
        task=tasks[index]
        if task['id'] not in observed:
            unknown.setdefault(task['family'],[]).append(index)
    representatives=[]
    for family,indices in unknown.items():
        # New families start at their easiest admitted task. Once exercised,
        # choose new invocations near the measured difficulty frontier.
        representative=min(indices,key=(lambda i:(tasks[i]['level'],priority(i)))
                           if not served.get(family) else priority)
        representatives.append(representative)
    representatives.sort(key=lambda i:(served.get(tasks[i]['family'],0),
        tasks[i]['level'] if not served.get(tasks[i]['family']) else abs(tasks[i]['level']-frontier),
        priority(i)))
    exploration=representatives[:math.ceil(slots*config['family_exploration_fraction'])]
    chosen_practice=list(exploration);selected=set(exploration)
    families={tasks[index]['family'] for index in chosen_practice}
    for index in practice:
        if len(chosen_practice)>=slots:break
        if index not in selected and tasks[index]['family'] not in families:
            chosen_practice.append(index);selected.add(index);families.add(tasks[index]['family'])
    for index in practice:
        if len(chosen_practice)>=slots:break
        if index not in selected:chosen_practice.append(index);selected.add(index)
    # Existing production offset advances from the last selected row. Keeping
    # breadth last makes its cursor advance exactly by the breadth allocation.
    chosen=chosen_practice+breadth
    return [{'row':index,'samples':budget['group']} for index in chosen]


def validate_breadth(cases,budget,config,chosen):
    expected=breadth_rows(cases,budget,policy_options(config))
    if expected and (not chosen or chosen[-1]['row']!=expected[-1]
                     or not set(expected)<={item['row'] for item in chosen}):
        raise ValueError('selectTrainingTasks must retain rotating breadth and finish with its final row')
