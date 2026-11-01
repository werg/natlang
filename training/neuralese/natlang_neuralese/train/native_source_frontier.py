"""One-visit priority for authenticated newly appended SFT rows."""
from __future__ import annotations
import copy


def extend(history, added_ids, bindings, *, source_sha256, global_step):
    """Record exact appended identities without granting any exposure."""
    result=copy.deepcopy(list(history or []));known={row['id'] for record in result for row in record['rows']}
    rows=[]
    for ident in added_ids:
        if ident in known or ident not in bindings:raise ValueError('fresh source frontier changed or repeated an identity')
        known.add(ident);rows.append({'id':ident,'binding':bindings[ident]})
    if rows:
        result.append({'schema':'natlang.native-source-first-visit-frontier/1','source_sha256':source_sha256,
                       'at_global_step':int(global_step),'rows':rows,'earned_visits_granted':0})
    return result


def pending(history, bindings, exposure, withdrawal_history=()):
    """Return only still-unvisited additions; authenticated withdrawals remain history."""
    withdrawn={(r['id'],r['binding']) for item in withdrawal_history for r in item.get('rows',[])}
    result=set();seen=set()
    for record in history or []:
        if record.get('schema')!='natlang.native-source-first-visit-frontier/1' or record.get('earned_visits_granted')!=0:
            raise ValueError('invalid fresh source frontier history')
        for row in record.get('rows',[]):
            ident,old=row['id'],row['binding']
            if ident in seen:raise ValueError('fresh source frontier repeats an identity')
            seen.add(ident)
            if bindings.get(ident)!=old:
                if (ident,old) in withdrawn:continue
                raise ValueError('fresh source frontier lost its exact row binding')
            count=exposure.get(ident,0)
            if type(count) is not int or count<0:raise ValueError('fresh source frontier has invalid earned visits')
            if count==0:result.add(ident)
    return result


def interleave_batches(batches, entries, frontier_ids, *, schedule_offset, period=5, priority_ids=()):
    """Give fresh rows regular early visits without bypassing the joint schedule.

    ``batches`` are already complete optimizer batches from the communication,
    urgent and ordinary fairness scheduler. This only reorders whole batches:
    one fresh-frontier batch receives each ``period`` slots while preserving
    order within both lanes and keeping the phase across checkpoint resumes.
    If the established schedule is exhausted, the remaining frontier is still
    drained, so this changes no visit quota or source coverage.
    """
    if type(schedule_offset) is not int or schedule_offset < 0:
        raise ValueError('fresh frontier interleave needs a nonnegative optimizer offset')
    if type(period) is not int or period < 2:
        raise ValueError('fresh frontier interleave period must be at least two')
    frontier_ids=set(frontier_ids or ())
    if not frontier_ids or not batches:return list(batches)
    fresh=[];established=[];priority=[]
    priority_ids=set(priority_ids or ())
    for batch in batches:
        if (not isinstance(batch,tuple) or len(batch)!=3 or not isinstance(batch[1],list)
                or any(type(index)is not int or not 0<=index<len(entries) for index in batch[1])):
            raise ValueError('fresh frontier scheduling requires intact indexed batches')
        row_ids={entries[index]['id'] for index in batch[1]}
        if row_ids & priority_ids:
            priority.append(batch)
        elif row_ids & frontier_ids:
            fresh.append(batch)
        else:
            established.append(batch)
    if not fresh or not established:return list(batches)
    # Explicit user/policy focus is stronger than first-visit freshness. A
    # mixed batch containing either priority rows stays in the priority lane.
    result=list(priority);fresh_index=established_index=0;phase=schedule_offset%period
    while fresh_index<len(fresh) or established_index<len(established):
        take_fresh=(phase==period-1 and fresh_index<len(fresh))
        if take_fresh:
            result.append(fresh[fresh_index]);fresh_index+=1
        elif established_index<len(established):
            result.append(established[established_index]);established_index+=1
        else:
            result.append(fresh[fresh_index]);fresh_index+=1
        phase=(phase+1)%period
    if len(result)!=len(batches) or sorted(map(id,result))!=sorted(map(id,batches)):
        raise ValueError('fresh frontier interleave changed batch custody')
    return result
