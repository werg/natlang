"""Exact held coverage for the owner's ordinary/communication monitoring policy.

1. Preserve visits to every previously checked row, including legacy rotation.
2. Check at least one whole graph from each available class in a normal check.
3. Alternate the classes within the existing budget, preferring less visited rows.
4. Check every row at final qualification. Never truncate a graph.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy

from ..common.hashing import canonical_json_sha256_hex


def commit_held_observations(entries, coverage, planned, selection, observed_ids):
    """Count only completed whole graphs when an operational stop interrupts evaluation."""
    observed=list(observed_ids)
    if len(set(observed))!=len(observed):raise ValueError('held observations contain duplicate rows')
    requested={index for row in selection['classes'].values()for index in row['selected_indices']}
    by_id={entry['id']:index for index,entry in enumerate(entries)}
    if any(key not in by_id or by_id[key]not in requested for key in observed):
        raise ValueError('held observation differs from the selected source rows')
    actual={by_id[key]for key in observed};result=deepcopy(planned);proof=deepcopy(selection)
    for name,record in result['classes'].items():
        pool=[i for i,entry in enumerate(entries)if ('communication'if entry.get('communication')else'ordinary')==name]
        local={index:j for j,index in enumerate(pool)}
        chosen=selection['classes'][name]['selected_indices'];done=[i for i in chosen if i in actual]
        for index in chosen:
            if index not in actual:record['visits'][local[index]]-=1
        record['cursor']=((local[done[-1]]+1)%len(pool)if done else selection['classes'][name]['cursor_before'])
        proof['classes'][name].update(selected_indices=done,selected_ids=[entries[i]['id']for i in done],cursor_after=record['cursor'])
    counts=[value for record in result['classes'].values()for value in record['visits']]
    result.update(cursor=result['classes']['ordinary']['cursor']if result['classes']['ordinary']['rows']else result['classes']['communication']['cursor'],
        graph_visits=sum(counts),completed_sweeps=min(counts),
        forward_token_visits=int((coverage or {}).get('forward_token_visits',0))+sum(entries[i]['forward_tokens']for i in actual))
    proof.update(requested_graphs=len(requested),observed_graphs=len(actual),operationally_interrupted=len(actual)!=len(requested))
    return result,proof


def held_selection(entries, coverage=None, *, source_sha256, token_budget=0,
                   graph_budget=0, final=False, previous_result=None):
    if not entries or min(token_budget, graph_budget) < 0:
        raise ValueError('held selection needs nonempty entries and nonnegative budgets')
    coverage = coverage or {}
    pools = {name: [i for i, e in enumerate(entries)
                    if bool(e.get('communication')) == communication]
             for name, communication in [('communication', True), ('ordinary', False)]}
    saved = coverage.get('classes')
    legacy = None
    if saved is None:
        old_size = int((previous_result or {}).get('source_graphs', len(entries)))
        cursor = int(coverage.get('cursor', 0))
        sweeps = int(coverage.get('completed_sweeps', 0))
        visits = int(coverage.get('graph_visits', sweeps * old_size + cursor))
        if not 0 < old_size <= len(entries) or not 0 <= cursor < old_size:
            raise ValueError('legacy held coverage has invalid source size/cursor')
        if visits != sweeps * old_size + cursor:
            raise ValueError('legacy held visits need an exact historical rotation binding')
        legacy = [sweeps + int(i < cursor) if i < old_size else 0
                  for i in range(len(entries))]
    state, before = {}, {}
    for name, pool in pools.items():
        rows = [[entries[i].get('id', str(i)), entries[i].get('binding')] for i in pool]
        prior = (saved or {}).get(name)
        if prior is not None:
            old_count = prior['rows']
            if old_count > len(pool) or canonical_json_sha256_hex(rows[:old_count]) != prior['row_prefix_sha256']:
                raise ValueError('held class continuation changed an existing row identity/binding')
            counts = list(prior['visits'])
            if len(counts) != old_count or any(type(n) is not int or n < 0 for n in counts):
                raise ValueError('invalid held class visit counts')
            counts.extend([0] * (len(pool)-old_count))
            cursor = int(prior['cursor'])
            if not 0 <= cursor < max(1, old_count):
                raise ValueError('invalid held class cursor')
        else:
            counts = [legacy[i] if legacy is not None else 0 for i in pool]
            # The next unvisited legacy row in this class retains its turn.
            cursor = next((j for j, i in enumerate(pool)
                           if legacy is not None and i >= int(coverage.get('cursor', 0))), 0)
        state[name] = {'rows': len(pool), 'row_prefix_sha256': canonical_json_sha256_hex(rows),
                       'visits': counts, 'cursor': cursor}
        before[name] = cursor
    selected, selected_by_class, tokens = [], {name: [] for name in pools}, 0
    if final:
        for name, pool in pools.items():
            for local, index in enumerate(pool):
                state[name]['visits'][local] += 1
                selected_by_class[name].append(index)
            state[name]['cursor'] = 0
        selected = list(range(len(entries)))
        tokens = sum(e['forward_tokens'] for e in entries)
    else:
        candidates = {}
        for name, pool in pools.items():
            cursor = state[name]['cursor']
            candidates[name] = deque(sorted(range(len(pool)), key=lambda j:
                (state[name]['visits'][j], (j-cursor) % max(1, len(pool)))))
        while any(candidates.values()):
            advanced = False
            for name, pool in pools.items():
                if not candidates[name]:
                    continue
                local = candidates[name][0]
                index = pool[local]
                size = entries[index]['forward_tokens']
                required = not selected_by_class[name]
                if not required and ((graph_budget and len(selected) >= graph_budget)
                                     or (token_budget and tokens+size > token_budget)):
                    continue
                candidates[name].popleft()
                selected.append(index)
                selected_by_class[name].append(index)
                tokens += size
                state[name]['visits'][local] += 1
                state[name]['cursor'] = (local+1) % len(pool)
                advanced = True
            if not advanced:
                break
    # Evaluation is a sum over independent graphs. Ascending physical order
    # avoids reopening/decompressing a leaf at every class alternation.
    selected.sort()
    all_counts = [count for value in state.values() for count in value['visits']]
    next_coverage = {'schema': 'natlang.native-held-coverage/2',
        'source_sha256': source_sha256, 'source_graphs': len(entries), 'classes': state,
        'cursor': (state['ordinary']['cursor'] if pools['ordinary'] else state['communication']['cursor']),
        'graph_visits': sum(all_counts),
        'forward_token_visits': int(coverage.get('forward_token_visits', 0))+tokens,
        'completed_sweeps': min(all_counts)}
    selection = {'source_sha256': source_sha256,
        'classes': {name: {'rows': len(pool), 'cursor_before': before[name],
            'cursor_after': state[name]['cursor'],
            'selected_ids': [entries[i].get('id', str(i)) for i in selected_by_class[name]],
            'selected_indices': selected_by_class[name]} for name, pool in pools.items()},
        'token_budget': token_budget, 'graph_budget': graph_budget,
        'whole_graph_class_minimum_over_budget': bool(
            (token_budget and tokens > token_budget) or
            (graph_budget and len(selected) > graph_budget)),
        'legacy_coverage_migrated': saved is None}
    return selected, next_coverage, selection
