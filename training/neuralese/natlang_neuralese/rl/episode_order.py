"""Deterministic whole-episode mixing before policy sources are sealed."""
from collections import Counter, deque
from ..common.hashing import canonical_json_sha256_hex


def _identities(graphs, objective):
    rows=objective['rows'];seen=set()
    for graph in graphs:
        ident=graph['id']
        if ident in seen or ident not in rows:
            raise ValueError('episode ordering needs distinct matching graph/objective identities')
        seen.add(ident);row=rows[ident];task=row.get('task_id')
        if not isinstance(task,str) or not task:
            raise ValueError('episode ordering needs the actual task identity')
        role=(row.get('jolt') or {}).get('role','policy')
        yield graph,task,role
    if seen!=set(rows):raise ValueError('episode ordering cannot omit objective rows')


def mix_episodes(graphs, objective):
    """Mix role mass proportionately and choose the least recently served task.

    Only graph references move. All actions, custody, teacher/student terms and
    objective bytes remain unchanged. Existing checkpoint prefixes are restored
    by arrange_and_seal after this proposal, so earned cursor meaning is intact.
    """
    lanes={};counts=Counter()
    for graph,task,role in _identities(graphs,objective):
        lanes.setdefault(role,{}).setdefault(task,deque()).append(graph);counts[role]+=1
    emitted=Counter();last={};output=[];total=len(graphs)
    for position in range(total):
        role=max((role for role in lanes if emitted[role]<counts[role]),
                 key=lambda role:(position+1)*counts[role]-emitted[role]*total)
        tasks=lanes[role]
        task=min((task for task in tasks if tasks[task]),key=lambda task:last.get(task,-1))
        output.append(tasks[task].popleft());last[task]=position;emitted[role]+=1
    graphs[:]=output


def order_receipt(graphs, objective):
    """Describe the actual sealed order, including any preserved resume prefix."""
    identities=[(graph['id'],task,role) for graph,task,role in _identities(graphs,objective)]
    return {'schema':'natlang.native-policy-episode-order/1',
            'policy':'proportional-role-and-least-recent-task/1',
            'ordered_identities_sha256':canonical_json_sha256_hex(identities),
            'episodes':len(identities),'tasks':len({task for _,task,_ in identities}),
            'role_counts':dict(Counter(role for _,_,role in identities)),
            'adjacent_same_task':sum(a[1]==b[1] for a,b in zip(identities,identities[1:])),
            'scope':'materialized source order; actual packed/update grouping remains separately measured'}
