"""Crisp adapter for the typed selectTrainingTasks natlang policy.

Historical outcomes choose work only; they are never behavior likelihoods or
current-policy qualification. Complete rotating breadth prevents a permanent
small curriculum endpoint, including when all observed tasks were successful.
"""
import math

DEFAULT={'schema':'natlang.rl-task-curriculum/1','breadth_fraction':.25,'promotion_success_rate':.6}


def metadata(row):
    value=(row.get('curriculum') or {}).get('rl') or {}
    level=value.get('level',3)
    if type(level) is not int or level<0:raise ValueError('task curriculum level must be a nonnegative integer')
    if value.get('qualification') not in (None,'qualified'):
        raise ValueError('task scaffold requires live model-facing qualification before production selection')
    return {'level':level,'family':value.get('family',row.get('family','unspecified'))}


def policy_options(value):
    config={**DEFAULT,**value}
    if config['schema']!=DEFAULT['schema']:raise ValueError('unsupported task curriculum policy schema')
    for key in ('breadth_fraction','promotion_success_rate'):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]) or not 0<config[key]<=1:
            raise ValueError('task curriculum '+key+' must be finite in (0,1]')
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
    # Existing production offset advances from the last selected row. Keeping
    # breadth last makes its cursor advance exactly by the breadth allocation.
    chosen=practice[:count-len(breadth)]+breadth
    return [{'row':index,'samples':budget['group']} for index in chosen]


def validate_breadth(cases,budget,config,chosen):
    expected=breadth_rows(cases,budget,policy_options(config))
    if expected and (not chosen or chosen[-1]['row']!=expected[-1]
                     or not set(expected)<={item['row'] for item in chosen}):
        raise ValueError('selectTrainingTasks must retain rotating breadth and finish with its final row')
