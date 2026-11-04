"""Crisp adapter for the typed selectTrainingTasks natlang policy.

Historical outcomes choose work only; they are never behavior likelihoods or
current-policy qualification. Complete rotating breadth prevents a permanent
small curriculum endpoint, including when all observed tasks were successful.

Two recorded policy schemas exist. ``/1`` is the historical selector: frozen
selectors and bound collections keep it exactly. ``/2`` (2026-11-03 recipe
review) is the default: new families start at their median level, practice
targets a per-family frontier that treats skipped easier levels as passed, and
the observed outcome-spread fraction of recent rounds shifts every family's
practice level so groups stay informative. ``min_rows_per_round``/``min_group``
are the selection budget floor the controller adopts with the policy.

``class_quota`` (policy /2, 2026-11-03) enforces construct shares at selection
time instead of by dropping candidates from the pool: a row tagged
``curriculum.rl.mix_class`` (the pool assembler tags rows whose oracle verifies a
world after the run ``effect_state``) must make up at least the declared share of
every round's rows. Rotating breadth keeps its exact cursor; the shortfall is
filled from practice by the ordinary frontier, outcome and family-exploration
order, restricted to that class. Rows are never duplicated and untagged pools
(no row carries the class) select exactly as before.

Visit decay (policy /2, 2026-11-04): a task whose outcome stays mixed keeps the spread the selector seeks, so without
decay the same few rows are re-picked until they are memorised. Each observation carries the row's actual ``visits``
and the rounds of its recent visits (``visit_rounds``, written by the controller). Practice then (a) charges every row
``visit_penalty`` levels of frontier distance per doubling of its visits above the pool's typical (mean observed) visit
count, and (b) ranks a row behind every other candidate once it has had ``visit_cap`` visits within the last
``visit_cap_window`` rounds. The cap is a ranking, not a drop: a pool too small to fill the round still fills it, least
recently over-visited rows last. Rotating breadth (its own cursor), ``class_quota`` shares, per-family frontiers and the
``min_rows_per_round``/``min_group`` floor are untouched; unobserved rows have no visits.
"""
import math

V1_SCHEMA='natlang.rl-task-curriculum/1'
V2_SCHEMA='natlang.rl-task-curriculum/2'
DEFAULT_V1={'schema':V1_SCHEMA,'breadth_fraction':.25,'promotion_success_rate':.6,
            'family_exploration_fraction':.5,'frontier_scope':'semantic-family'}
DEFAULT={**DEFAULT_V1,'schema':V2_SCHEMA,'new_family_start':'median','spread_target':.3,
         'difficulty_shift_limit':3,'min_rows_per_round':16,'min_group':8,'class_quota':{'effect_state':.34},
         'visit_penalty':1.0,'visit_cap':2,'visit_cap_window':50}
VISIT_ROUNDS_KEPT=64  # observations retain this many most recent visit rounds; bounds visit_cap_window use
V2_FIELDS=frozenset(DEFAULT)


def metadata(row):
    value=(row.get('curriculum') or {}).get('rl') or {}
    level=value.get('level',3)
    if type(level) is not int or level<0:raise ValueError('task curriculum level must be a nonnegative integer')
    if value.get('qualification') not in (None,'qualified'):
        raise ValueError('task scaffold requires live model-facing qualification before production selection')
    family=value.get('family',row.get('family','unspecified'))
    if family is None or family=='':family='unspecified'
    if not isinstance(family,str):raise ValueError('task curriculum family must be a string or unknown')
    mix=value.get('mix_class')
    if mix is not None and (not isinstance(mix,str) or not mix):raise ValueError('task curriculum mix_class must be a nonempty string')
    return {'level':level,'family':family,**({'mix_class':mix} if mix is not None else {})}


def policy_options(value):
    schema=(value or {}).get('schema',V1_SCHEMA)
    if schema==V1_SCHEMA:
        config={**DEFAULT_V1,**value}
        if set(config)-set(DEFAULT_V1):
            raise ValueError('task curriculum /1 does not accept /2 options: '+', '.join(sorted(set(config)-set(DEFAULT_V1))))
    elif schema==V2_SCHEMA:
        config={**DEFAULT,**value}
        if set(config)-V2_FIELDS:raise ValueError('unknown task curriculum options: '+', '.join(sorted(set(config)-V2_FIELDS)))
    else:raise ValueError('unsupported task curriculum policy schema')
    if config['frontier_scope'] != 'semantic-family':
        raise ValueError('task curriculum frontier_scope must be semantic-family')
    for key in ('breadth_fraction','promotion_success_rate'):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]) or not 0<config[key]<=1:
            raise ValueError('task curriculum '+key+' must be finite in (0,1]')
    fraction=config['family_exploration_fraction']
    if type(fraction) not in (int,float) or not math.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('task curriculum family_exploration_fraction must be finite in [0,1]')
    if schema==V2_SCHEMA:
        if config['new_family_start'] not in ('median','easiest'):
            raise ValueError('task curriculum new_family_start must be median or easiest')
        target=config['spread_target']
        if type(target) not in (int,float) or not math.isfinite(target) or not 0<=target<=1:
            raise ValueError('task curriculum spread_target must be finite in [0,1]')
        for key,low in (('difficulty_shift_limit',0),('min_rows_per_round',1),('min_group',2)):
            if type(config[key]) is not int or config[key]<low:
                raise ValueError('task curriculum '+key+' must be an integer >= '+str(low))
        quota=config['class_quota']
        if not isinstance(quota,dict) or any(not isinstance(name,str) or not name or type(share) not in (int,float)
                or not math.isfinite(share) or not 0<=share<=1 for name,share in quota.items()) or sum(quota.values())>1:
            raise ValueError('task curriculum class_quota must map class names to shares in [0,1] summing to at most 1')
        penalty=config['visit_penalty']
        if type(penalty) not in (int,float) or not math.isfinite(penalty) or penalty<0:
            raise ValueError('task curriculum visit_penalty must be finite and >= 0')
        for key in ('visit_cap','visit_cap_window'):
            if type(config[key]) is not int or config[key]<0:
                raise ValueError('task curriculum '+key+' must be an integer >= 0')
        if config['visit_cap']>VISIT_ROUNDS_KEPT:
            raise ValueError('task curriculum visit_cap must not exceed the '+str(VISIT_ROUNDS_KEPT)+' retained visit rounds')
    return config


def task_descriptors(cases):
    return [{'row':index,'id':row['id'],**metadata(row)} for index,row in enumerate(cases)]


def breadth_rows(cases,budget,config):
    count=min(len(cases),budget['rows'])
    if not count:return []
    size=min(count,max(1,math.ceil(count*config['breadth_fraction'])))
    return [(budget['offset']+index)%len(cases) for index in range(size)]


def _observed(stats):
    return {row['task_id']:row for row in stats if type(row.get('scored')) is int and row['scored']>0
            and type(row.get('mean')) in (int,float) and math.isfinite(row['mean']) and 0<=row['mean']<=1}


def difficulty_shift(stats,config):
    """Signed practice-level shift from the outcome spread of the latest observed rounds.

    Walk back from the newest round with scored groups. Each consecutive round
    whose informative (outcome-spread) fraction is below ``spread_target`` moves
    practice one level: up when its spread-free groups are mostly solved, down
    when they are mostly all-fail. The walk stops at an informative round or a
    change of direction. Observations from earlier rounds than the newest are
    the latest per task, so this is telemetry, never on-policy qualification.
    """
    rounds={}
    for row in _observed(stats).values():
        if type(row.get('observed_round')) is int and type(row.get('spread')) is bool:
            rounds.setdefault(row['observed_round'],[]).append(row)
    shift=0
    for number in sorted(rounds,reverse=True):
        groups=rounds[number]
        if sum(group['spread'] for group in groups)>=config['spread_target']*len(groups):break
        solved=sum(1 for group in groups if not group['spread'] and group['mean']>=.5)
        failed=sum(1 for group in groups if not group['spread'] and group['mean']<.5)
        step=1 if solved>=failed else -1
        if shift and (step>0)!=(shift>0):break
        shift+=step
        if abs(shift)>=config['difficulty_shift_limit']:break
    return shift


def record_visit(prior,round_number,limit=VISIT_ROUNDS_KEPT):
    """Visit bookkeeping for one newly scored observation of a task, from its previous observation (or None).

    Idempotent per round: a retried collection of the same round does not count twice. A prior observation that
    predates visit tracking counts as one visit (a lower bound, never an over-count)."""
    prior=prior or {}
    visits=prior.get('visits') if type(prior.get('visits')) is int and prior['visits']>=0 else (1 if prior else 0)
    rounds=[r for r in (prior.get('visit_rounds') or []) if type(r) is int]
    if not prior:return {'visits':1,'visit_rounds':[round_number]}
    if round_number in rounds or (not rounds and prior.get('observed_round')==round_number):
        return {'visits':max(visits,1),'visit_rounds':rounds or [round_number]}
    return {'visits':visits+1,'visit_rounds':(rounds+[round_number])[-limit:]}


def _visits(group):
    """(total visits, recent visit rounds) of an observation; an untracked observation is one visit."""
    if not group:return 0,[]
    visits=group.get('visits')
    visits=visits if type(visits) is int and visits>=0 else 1
    return visits,[r for r in (group.get('visit_rounds') or []) if type(r) is int]


def _median_level(levels):
    ordered=sorted(level for level,tasks in levels.items() for _ in tasks)
    return ordered[(len(ordered)-1)//2]


def _v2_frontier(levels,observed,config):
    """A family's practice level before the spread shift (policy /2).

    Unobserved families start at the median of their admitted task levels. An
    observed family practices its lowest level whose scored success is below the
    promotion rate, else the first level above its highest promoted level; an
    easier level that was skipped by a median start counts as passed.
    """
    rates={}
    for level,tasks in levels.items():
        groups=[observed[task['id']] for task in tasks if task['id'] in observed]
        scored=sum(group['scored'] for group in groups)
        if scored:rates[level]=sum(group['mean']*group['scored'] for group in groups)/scored
    if not rates:
        return _median_level(levels) if config['new_family_start']=='median' else min(levels)
    failing=[level for level,rate in rates.items() if rate<config['promotion_success_rate']]
    promoted=[level for level,rate in rates.items() if rate>=config['promotion_success_rate']]
    above=[level for level in sorted(levels) if promoted and level>max(promoted)]
    return min(failing+([above[0]] if above else [max(levels)]))


def select(cases,stats,budget,config):
    config=policy_options(config);tasks=task_descriptors(cases)
    if not tasks:return []
    v2=config['schema']==V2_SCHEMA
    observed=_observed(stats)
    # A solved math level must not wait for unrelated tool/semantic tasks.
    # Missing family metadata stays one explicit unknown bucket.
    family_levels={}
    for task in tasks:
        family_levels.setdefault(task['family'],{}).setdefault(task['level'],[]).append(task)
    frontiers={}
    if v2:
        shift=difficulty_shift(stats,config)
        for family,levels in family_levels.items():
            base=_v2_frontier(levels,observed,config)
            frontiers[family]=min(max(levels),max(min(levels),base+shift))
    else:
        for family,levels in family_levels.items():
            frontiers[family]=max(levels)
            for level in sorted(levels):
                groups=[observed[task['id']] for task in levels[level] if task['id'] in observed]
                scored=sum(group['scored'] for group in groups)
                if not scored or sum(group['mean']*group['scored'] for group in groups)/scored<config['promotion_success_rate']:
                    frontiers[family]=level;break
    breadth=breadth_rows(cases,budget,config);reserved=set(breadth)
    penalty=config.get('visit_penalty',0) if v2 else 0
    cap=config.get('visit_cap',0) if v2 else 0
    window=config.get('visit_cap_window',0) if v2 else 0
    counts=[_visits(group)[0] for group in observed.values()]
    typical=max(1.,sum(counts)/len(counts)) if counts else 1.
    now=1+max([group['observed_round'] for group in observed.values() if type(group.get('observed_round')) is int],default=-1)
    def visit_cost(group):
        """(capped, level penalty) from an observation's visits relative to the pool's typical visit count."""
        visits,rounds=_visits(group)
        capped=bool(cap and window and sum(1 for r in rounds if r>now-window)>=cap)
        return capped,(penalty*math.log2(visits/typical) if penalty and visits>typical else 0.)
    memo={}
    def priority(index):
        if index in memo:return memo[index]
        task=tasks[index];group=observed.get(task['id']);mean=group['mean'] if group else None
        signal=0 if mean is not None and 0<mean<1 else 1 if mean is None else 2 if mean==0 else 3
        capped,extra=visit_cost(group)
        memo[index]=(capped,abs(task['level']-frontiers[task['family']])+extra,signal,(index-budget['offset'])%len(cases))
        return memo[index]
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
    def choose(candidates,slots):
        """Exploration of the least-covered families first, then distinct families, then priority order."""
        if slots<=0:return []
        unknown={}
        for index in candidates:
            task=tasks[index]
            if task['id'] not in observed:
                unknown.setdefault(task['family'],[]).append(index)
        representatives=[]
        for family,indices in unknown.items():
            # Policy /1: new families start at their easiest admitted task. Policy
            # /2: new families start at their (shifted) median level. Once
            # exercised, choose new invocations near the measured frontier.
            representative=min(indices,key=(lambda i:(tasks[i]['level'],priority(i)))
                               if not served.get(family) and not v2 else priority)
            representatives.append(representative)
        representatives.sort(key=lambda i:(served.get(tasks[i]['family'],0),
            tasks[i]['level'] if not served.get(tasks[i]['family']) and not v2
            else abs(tasks[i]['level']-frontiers[tasks[i]['family']]),
            priority(i)))
        exploration=representatives[:math.ceil(slots*config['family_exploration_fraction'])]
        picked=list(exploration);selected=set(exploration)
        families={tasks[index]['family'] for index in picked}
        # Rows at their visit cap rank behind every uncapped row, whatever their family.
        for tier in ([i for i in candidates if not priority(i)[0]],[i for i in candidates if priority(i)[0]]):
            for index in tier:
                if len(picked)>=slots:break
                if index not in selected and tasks[index]['family'] not in families:
                    picked.append(index);selected.add(index);families.add(tasks[index]['family'])
            for index in tier:
                if len(picked)>=slots:break
                if index not in selected:picked.append(index);selected.add(index)
        return picked
    chosen_practice=[]
    if v2:
        for name,share in sorted((config.get('class_quota') or {}).items()):
            members=[index for index in practice if tasks[index].get('mix_class')==name]
            have=sum(1 for index in breadth if tasks[index].get('mix_class')==name)
            have+=sum(1 for index in chosen_practice if tasks[index].get('mix_class')==name)
            need=min(len(members),slots-len(chosen_practice),max(0,math.ceil(share*count)-have))
            chosen_practice+=choose(members,need)
    taken=set(chosen_practice)
    chosen_practice+=choose([index for index in practice if index not in taken],slots-len(chosen_practice))
    # Existing production offset advances from the last selected row. Keeping
    # breadth last makes its cursor advance exactly by the breadth allocation.
    chosen=chosen_practice+breadth
    return [{'row':index,'samples':budget['group']} for index in chosen]


def spread_yield(groups):
    """Informative-group telemetry for one collection: scored groups and those with outcome spread."""
    scored=[group for group in groups if type(group.get('scored')) is int and group['scored']>0]
    spread=sum(1 for group in scored if group.get('spread') is True)
    return {'scored_groups':len(scored),'groups_with_spread':spread,
            'spread_fraction':spread/len(scored) if scored else None}


def validate_breadth(cases,budget,config,chosen):
    expected=breadth_rows(cases,budget,policy_options(config))
    if expected and (not chosen or chosen[-1]['row']!=expected[-1]
                     or not set(expected)<={item['row'] for item in chosen}):
        raise ValueError('selectTrainingTasks must retain rotating breadth and finish with its final row')


def validate_quota(cases,budget,config,chosen):
    """An external selector must honour ``class_quota`` as the crisp one does: each class gets at least its share of the
    round's rows, bounded by the class rows the pool has."""
    config=policy_options(config)
    if config['schema']!=V2_SCHEMA:return
    count=min(budget['rows'],len(cases))
    for name,share in (config.get('class_quota') or {}).items():
        members=sum(1 for row in cases if metadata(row).get('mix_class')==name)
        required=min(math.ceil(share*count),members)
        got=sum(1 for item in chosen if metadata(cases[item['row']]).get('mix_class')==name)
        if got<required:
            raise ValueError(f'selectTrainingTasks must give class {name} at least {required} of {count} rows (got {got})')
