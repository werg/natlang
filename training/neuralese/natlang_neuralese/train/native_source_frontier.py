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
