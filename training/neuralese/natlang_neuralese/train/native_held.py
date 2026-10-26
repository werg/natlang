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


def rebind_held_coverage_after_exact_correction(previous_entries, current_entries, coverage,
                                                *, source_sha256):
    """Retain visits while rebinding coverage to a proved same-membership target repair.

    The caller must first validate the correction ledger against both exact
    held receipts, row bindings, and source groups. This function independently
    checks that the correction did not move a row between ordinary and
    communication classes, and that the old coverage prefix matches the
    predecessor entries before updating only its binding digest.
    """
    if not isinstance(coverage, dict) or not isinstance(coverage.get('classes'), dict):
        return coverage
    result=deepcopy(coverage)
    retired=[]
    retired_tokens=0
    previous_by_id={entry.get('id',str(i)):entry for i,entry in enumerate(previous_entries)}
    for name, communication in [('communication', True), ('ordinary', False)]:
        previous=[[e.get('id', str(i)), e.get('binding')] for i,e in enumerate(previous_entries)
                  if bool(e.get('communication')) is communication]
        current=[[e.get('id', str(i)), e.get('binding')] for i,e in enumerate(current_entries)
                 if bool(e.get('communication')) is communication]
        if [row[0] for row in previous] != [row[0] for row in current]:
            raise ValueError('protected held correction changed class membership/order')
        state=result['classes'].get(name)
        if state is None:
            if previous:
                raise ValueError('protected held correction lacks prior class coverage')
            continue
        old_count=state.get('rows')
        if type(old_count) is not int or old_count != len(previous):
            raise ValueError('protected held correction coverage row count differs from exact predecessor')
        if canonical_json_sha256_hex(previous) != state.get('row_prefix_sha256'):
            raise ValueError('protected held correction predecessor differs from saved coverage binding')
        visits=state.get('visits')
        if not isinstance(visits,list) or len(visits)!=len(previous):
            raise ValueError('protected held correction coverage lacks exact per-row observations')
        for local, (old_row,new_row) in enumerate(zip(previous,current)):
            if old_row[1] == new_row[1]:
                continue
            count=visits[local]
            if type(count) is not int or count < 0:
                raise ValueError('protected held correction has invalid prior target observations')
            old_entry=previous_by_id[old_row[0]]
            tokens=old_entry.get('forward_tokens')
            if count and (type(tokens) is not int or tokens < 1):
                raise ValueError('cannot retire held target visits without their exact token counts')
            retired.append({'id':old_row[0],'class':name,
                'old_binding_sha256':canonical_json_sha256_hex(old_row[1]),
                'visits':count,'forward_tokens_per_visit':tokens or 0,
                'retired_forward_token_visits':count * (tokens or 0),
                'reason':'held target binding changed'})
            retired_tokens += count * (tokens or 0)
            visits[local]=0
        state['row_prefix_sha256']=canonical_json_sha256_hex(current)
        state['rows']=len(current)
    result['source_sha256']=source_sha256
    if retired:
        result['retired_target_visit_history']=list(result.get('retired_target_visit_history') or [])+retired
        prior_tokens=result.get('forward_token_visits')
        if retired_tokens and (type(prior_tokens) is not int or prior_tokens < retired_tokens):
            raise ValueError('protected held correction token history cannot cover retired target visits')
        if type(prior_tokens) is int:
            result['forward_token_visits']=prior_tokens-retired_tokens
    if retired:
        counts=[count for state in result['classes'].values() for count in state.get('visits',[])]
        result['graph_visits']=sum(counts)
        result['completed_sweeps']=min(counts,default=0)
    return result


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


def bind_held_quality_exclusions(entries, coverage, excluded_ids, *, source_sha256,
                                allow_transition=False):
    """Reset held coverage only for rows explicitly held from target scoring."""
    excluded = set(excluded_ids)
    ids = [entry['id'] for entry in entries]
    if not excluded.issubset(set(ids)):
        raise ValueError('held quality hold names rows outside the exact held source')
    result = deepcopy(coverage) if isinstance(coverage, dict) else None
    if result is None or not isinstance(result.get('classes'), dict):
        prior_ids = set((result or {}).get('quality_excluded_ids') or ())
        if result is not None and prior_ids != excluded and not allow_transition:
            raise ValueError('held quality exclusion set changed without an explicit held correction')
        if excluded:
            # The first held evaluation will initialize a fresh complete row
            # ledger. Preserve the exact exclusion identity in its selector.
            return {'schema': 'natlang.native-held-coverage/2',
                'source_sha256': source_sha256, 'source_graphs': len(entries),
                'classes': None, 'quality_excluded_ids': sorted(excluded),
                'cursor': 0, 'graph_visits': 0, 'forward_token_visits': 0,
                'completed_sweeps': 0}
        return coverage
    result_ids = result.get('quality_excluded_ids')
    if result_ids is None and result.get('classes') is not None:
        if excluded and not allow_transition:
            raise ValueError('held quality exclusions require an explicit source-correction transition')
    elif result_ids is not None and set(result_ids) != excluded and not allow_transition:
        raise ValueError('held quality exclusion set changed without an explicit held correction')
    retired=[]
    retired_tokens=0
    previously_excluded=set(result.get('quality_excluded_ids') or ())
    newly_excluded=excluded-previously_excluded
    for name, communication in [('communication', True), ('ordinary', False)]:
        pool = [entry for entry in entries if bool(entry.get('communication')) is communication]
        rows = [[entry['id'], entry.get('binding')] for entry in pool]
        state = result['classes'].get(name)
        if state is None:
            if rows:
                raise ValueError('held quality hold lacks prior class coverage')
            continue
        if (state.get('rows') != len(rows)
                or state.get('row_prefix_sha256') != canonical_json_sha256_hex(rows)
                or len(state.get('visits') or []) != len(rows)):
            raise ValueError('held quality hold coverage differs from exact current row bindings')
        for index, entry in enumerate(pool):
            if entry['id'] in excluded:
                count=state['visits'][index]
                if count and entry['id'] in newly_excluded:
                    tokens=entry.get('forward_tokens')
                    if type(tokens) is not int or tokens < 1:
                        raise ValueError('cannot retire held quality visits without exact token counts')
                    retired.append({'id':entry['id'],'class':name,
                        'old_binding_sha256':canonical_json_sha256_hex(entry.get('binding')),
                        'visits':count,'forward_tokens_per_visit':tokens,
                        'retired_forward_token_visits':count * tokens,
                        'reason':'held target excluded from loss scoring'})
                    retired_tokens += count*tokens
                state['visits'][index] = 0
            elif entry['id'] in previously_excluded and state['visits'][index] != 0:
                raise ValueError('previously excluded held row has saved observations')
    result['source_sha256'] = source_sha256
    result['source_graphs'] = len(entries)
    result['quality_excluded_ids'] = sorted(excluded)
    if retired:
        result['retired_target_visit_history']=list(result.get('retired_target_visit_history') or [])+retired
        prior_tokens=result.get('forward_token_visits')
        if type(prior_tokens) is not int or prior_tokens < retired_tokens:
            raise ValueError('held quality exclusion token history cannot cover retired target visits')
        result['forward_token_visits']=prior_tokens-retired_tokens
    counts = [visit for state in result['classes'].values() for visit in state.get('visits', [])]
    result['graph_visits'] = sum(counts)
    result['completed_sweeps'] = min(counts, default=0)
    return result


def held_selection(entries, coverage=None, *, source_sha256, token_budget=0,
                   graph_budget=0, final=False, previous_result=None, excluded_ids=()):
    if not entries or min(token_budget, graph_budget) < 0:
        raise ValueError('held selection needs nonempty entries and nonnegative budgets')
    coverage = coverage or {}
    excluded = set(excluded_ids)
    ids = {entry.get('id', str(i)) for i, entry in enumerate(entries)}
    if not excluded.issubset(ids):
        raise ValueError('held quality exclusion names a row outside the exact held source')
    saved_excluded = set(coverage.get('quality_excluded_ids') or ())
    if coverage and saved_excluded != excluded:
        raise ValueError('held quality exclusion set differs from saved coverage')
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
                if entries[index].get('id', str(index)) in excluded:
                    continue
                state[name]['visits'][local] += 1
                selected_by_class[name].append(index)
            state[name]['cursor'] = 0
        selected = [i for i, entry in enumerate(entries) if entry.get('id', str(i)) not in excluded]
        tokens = sum(e['forward_tokens'] for e in entries if e.get('id') not in excluded)
    else:
        candidates = {}
        for name, pool in pools.items():
            cursor = state[name]['cursor']
            candidates[name] = deque(sorted((j for j, index in enumerate(pool)
                if entries[index].get('id', str(index)) not in excluded), key=lambda j:
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
        'quality_excluded_ids': sorted(excluded),
        'cursor': (state['ordinary']['cursor'] if pools['ordinary'] else state['communication']['cursor']),
        'graph_visits': sum(all_counts),
        'forward_token_visits': int(coverage.get('forward_token_visits', 0))+tokens,
        'completed_sweeps': min(all_counts)}
    selection = {'source_sha256': source_sha256,
        'quality_excluded_ids': sorted(excluded),
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
