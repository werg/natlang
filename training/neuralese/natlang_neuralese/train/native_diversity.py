"""Bounded, custody-preserving source-window diversity for native graph batches."""
from __future__ import annotations
from collections import Counter
import re
import os
import resource
from .native_source_streams import scheduling_rank

_PROGRAM = re.compile(r'(?:^|:)teacher-program:([^:]+)')

def program_key(entry):
    match = _PROGRAM.search(entry['id'])
    return match.group(1) if match else None

def task_key(entry):
    for group in entry.get('groups',()):
        fields=group.split(':')
        if len(fields)>1 and fields[0]=='inline-curriculum':return fields[1]
    return None

def stream_windows(source, grouped, *, priority_ids=(), cohort_ids=None, rotation_index=None,
                   focus_ids=(), urgent_cohort_fairness=True, ordinary_cohort_fairness=True,
                   communication_cohort_fairness=True):
    """Read each source cursor monotonically; buffer only one prepared window."""
    priority_ids=set(priority_ids or ());focus_ids=set(focus_ids or ());entries=source.entries
    # Planning and consumption traverse the same windows, including callers
    # that supply a one-shot iterator.
    grouped=list(grouped)
    if cohort_ids is not None and len(cohort_ids)!=len(entries):raise ValueError('cohort count differs')
    def key(visit,index,focused):
        e=entries[index];foundation=e['id'] in priority_ids
        # Reader identity follows physical source custody, independently of
        # whether scheduling fairness for that lane is enabled. Merging these
        # cursors can rewind a source after cross-cohort packing.
        cohort=cohort_ids[index] if cohort_ids is not None else None
        return visit,scheduling_rank(e),foundation,cohort,bool(rotation_index is not None and index<rotation_index),focused
    requests={};last={}
    for window in grouped:
        parts={}
        for visit,indices,_ in window:
            if not indices:raise ValueError('empty diversity batch')
            focused=any(entries[index]['id'] in focus_ids for index in indices)
            for index in indices:parts.setdefault(key(visit,index,focused),[]).append(index)
        for k,indices in parts.items():
            indices.sort()
            if len(indices)!=len(set(indices)) or (k in last and indices[0]<=last[k]):
                raise ValueError(f'diverse source windows are not monotone and distinct: {k!r}, last={last.get(k)}, next={indices[:2]}')
            requests.setdefault(k,[]).extend(indices);last[k]=indices[-1]
    streams={};remaining={k:len(indices) for k,indices in requests.items()}
    try:
        for window in grouped:
            parts={};values={}
            for visit,indices,_ in window:
                focused=any(entries[index]['id'] in focus_ids for index in indices)
                for index in indices:parts.setdefault(key(visit,index,focused),[]).append(index)
            for k,indices in parts.items():
                if k not in streams:
                    soft,_=resource.getrlimit(resource.RLIMIT_NOFILE)
                    if soft!=resource.RLIM_INFINITY and len(os.listdir('/proc/self/fd'))+8>=soft:
                        raise ValueError('native diversity source readers exceed available file descriptors')
                    streams[k]=source.iter_indices(requests[k])
                stream=streams[k]
                for index in sorted(indices):
                    try:item=next(stream)
                    except StopIteration as exc:raise ValueError('diverse source ended early') from exc
                    if item.id!=entries[index]['id']:raise ValueError('diverse source row identity differs')
                    values[(k,index)]=item
                    remaining[k]-=1
                if remaining[k]==0:
                    try:next(stream)
                    except StopIteration:pass
                    else:raise ValueError('diverse source yielded extra rows')
                    stream.close();del streams[k]
            for visit,indices,end in window:
                focused=any(entries[index]['id'] in focus_ids for index in indices)
                yield visit,indices,end,[values.pop((key(visit,index,focused),index)) for index in indices]
            if values:raise ValueError('unused diverse source rows')
        if any(remaining.values()):raise ValueError('diverse source rows remain unread')
    finally:
        for stream in streams.values():stream.close()

def cohort_head_merge(entries,batches,cohorts,*,priority_ids=(),focus_ids=(),batch_tokens=0,batch_graphs=1,lookahead=256,schedule_offset=0):
    """Pack ordered cohort heads, bounded by metadata lookahead, preserving all visits.

    Existing focus batches stay exact. Each other scheduling-class queue retains
    source order and is served on the existing lane slots; residual work cycles
    through those slots. No graph is decoded while constructing the schedule.
    """
    from collections import deque
    if len(entries)!=len(cohorts) or type(lookahead) is not int or lookahead<1 or batch_graphs<1 or batch_tokens<0 or type(schedule_offset) is not int or schedule_offset<0:
        raise ValueError('invalid cohort-head merge inputs')
    priorities=set(priority_ids or ());focus=set(focus_ids or ());groups={};slots=[]
    def key(visit,indices):
        row=entries[indices[0]]
        return visit,scheduling_rank(row),bool(any(entries[i]['id'] in priorities for i in indices))
    for position,(visit,indices,end) in enumerate(batches):
        if any(entries[i]['id'] in focus for i in indices):
            slots.append((position,visit,None,indices,end));continue
        k=key(visit,indices);slots.append((position,visit,k,None,end));queues=groups.setdefault(k,{})
        for index in indices:queues.setdefault(cohorts[index],deque()).append((position,index))
    def pack(queues,position):
        chosen=[];programs=set();tasks=set();size=0
        while batch_tokens or len(chosen)<batch_graphs:
            candidates=[]
            for cohort,queue in queues.items():
                if not queue:continue
                origin,index=queue[0];row=entries[index];tokens=row['forward_tokens']
                if origin>position+lookahead:continue
                if chosen and batch_tokens and size+tokens>batch_tokens:continue
                program,task=program_key(row),task_key(row)
                candidates.append(((int(bool(task and task in tasks)),int(bool(program and program in programs)),origin,index),cohort,index,tokens,program,task))
            if not candidates:break
            _,cohort,index,tokens,program,task=min(candidates,key=lambda x:x[0]);queues[cohort].popleft();chosen.append(index);size+=tokens
            if program:programs.add(program)
            if task:tasks.add(task)
            if batch_tokens and size>=batch_tokens:break
        chosen.sort();return chosen
    result=[]
    for position,visit,k,focused,end in slots:
        if focused is not None:result.append((visit,focused,end));continue
        selected=pack(groups[k],position)
        if selected:result.append((visit,selected,False))
    # If changed packing created residual work, cycle the same lane slots and
    # advance their metadata horizon. Every non-empty emission consumes rows.
    pending=lambda:any(queue for queues in groups.values() for queue in queues.values())
    sweep=0
    while pending():
        consumed=0;sweep+=1
        for position,visit,k,focused,_ in slots:
            if k is None:continue
            selected=pack(groups[k],position+sweep*len(slots))
            if selected:result.append((visit,selected,False));consumed+=len(selected)
        if not consumed:raise ValueError('cohort-head merge cannot make progress')
    # Reapply the existing four-fast/one-debt cycle after repacking changes
    # the number of batches. Focus remains an explicit leading obligation.
    focused=[b for b in result if any(entries[i]['id'] in focus for i in b[1])]
    other=[b for b in result if not any(entries[i]['id'] in focus for i in b[1])]
    fast=[b for b in other if entries[b[1][0]].get('communication') or entries[b[1][0]].get('training_priority',0)>0]
    debt=[b for b in other if not entries[b[1][0]].get('communication') and entries[b[1][0]].get('training_priority',0)==0]
    if fast and debt:
        interleaved=[];i=j=0;cursor=schedule_offset%5
        while i<len(fast) or j<len(debt):
            if cursor!=4 and i<len(fast):interleaved.append(fast[i]);i+=1
            elif cursor==4 and j<len(debt):interleaved.append(debt[j]);j+=1
            elif i<len(fast):interleaved.append(fast[i]);i+=1
            else:interleaved.append(debt[j]);j+=1
            cursor=(cursor+1)%5
        result=focused+interleaved
    else:result=focused+other
    last={visit:i for i,(visit,_,_) in enumerate(result)}
    return [(visit,indices,last[visit]==i) for i,(visit,indices,_) in enumerate(result)]
