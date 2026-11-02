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


def monotone_windows(grouped, entries, key, *, max_buffer_tokens=1048576):
    """Join only overlapping reader windows, retaining the exact batch order.

    A mixed packed batch can include an early row from one source and a later
    row from another. Promoting whole batches can therefore cross a source's
    next requested row. Read that minimal connected window in source order,
    then deliver its original batches from the bounded window buffer.
    """
    grouped = list(grouped)
    if type(max_buffer_tokens) is not int or max_buffer_tokens < 1:
        raise ValueError('source window token bound must be positive')
    readers = {}
    largest = 0
    for position, window in enumerate(grouped):
        tokens = 0
        for visit, indices, _ in window:
            for index in indices:
                readers.setdefault(key(visit, index, indices), []).append((index, position))
                tokens += entries[index]['forward_tokens']
        largest = max(largest, tokens)
    # Every inverted adjacent pair in a reader's sorted index sequence forbids
    # a cut between its two windows. Union across readers gives minimal joins.
    changes = [0] * (len(grouped) + 1)
    for rows in readers.values():
        rows.sort()
        for (left, a), (right, b) in zip(rows, rows[1:]):
            if left == right:
                raise ValueError('diverse source repeats an index in one reader')
            if a > b:
                changes[b] += 1
                changes[a] -= 1
    result = []
    active = 0
    pending = []
    tokens = 0
    limit = max(max_buffer_tokens, largest)
    for position, window in enumerate(grouped):
        pending.extend(window)
        tokens += sum(entries[index]['forward_tokens'] for _, indices, _ in window for index in indices)
        if tokens > limit:
            raise ValueError(f'monotone source window exceeds its token buffer bound: {tokens} > {limit}')
        active += changes[position]
        if active == 0:
            result.append(pending)
            pending = []
            tokens = 0
    if pending or active:
        raise ValueError('incomplete monotone source window plan')
    return result

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
    focus_visits = {}
    for window in grouped:
        for visit, indices, _ in window:
            for index in indices:
                if entries[index]['id'] in focus_ids:
                    focus_visits[index] = min(visit, focus_visits.get(index, visit))
    # Promotion is a one-time visit, not every future repetition of that row.
    # Rows with different earned counts can have different first pending visits.
    focused_batches = {(visit, id(indices)): any(focus_visits.get(index) == visit for index in indices)
                       for window in grouped for visit, indices, _ in window}
    def batch_key(visit, index, indices):
        return key(visit, index, focused_batches[(visit, id(indices))])
    grouped = monotone_windows(grouped, entries, batch_key)
    requests={};last={}
    for window in grouped:
        parts={}
        for visit,indices,_ in window:
            if not indices:raise ValueError('empty diversity batch')
            focused=focused_batches[(visit, id(indices))]
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
                focused=focused_batches[(visit, id(indices))]
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
                focused=focused_batches[(visit, id(indices))]
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
    focused_seen=set()
    for position,(visit,indices,end) in enumerate(batches):
        present={entries[i]['id'] for i in indices} & focus
        if present-focused_seen:
            focused_seen.update(present)
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
    focused=[];other=[];focused_seen=set()
    for batch in result:
        present={entries[i]['id'] for i in batch[1]} & focus
        if present-focused_seen:
            focused.append(batch);focused_seen.update(present)
        else:
            other.append(batch)
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


def sustained_communication_interleave(entries, batches, *, focus_ids=(), schedule_offset=0):
    """Spread repeated communication visits across the remaining ordinary epoch.

    Reorders complete packed batches only; each (visit, row) remains exactly
    once. Within ordinary work the existing four-urgent/one-debt service rule
    is preserved. No randomness or graph contents are consumed.
    """
    focus = set(focus_ids or ())
    focused, comm, urgent, debt = [], [], [], []
    focused_seen=set()
    for batch in batches:
        indices = batch[1]
        if not indices:
            raise ValueError('sustained interleave received an empty batch')
        present={entries[i]['id'] for i in indices} & focus
        if present-focused_seen:
            focused.append(batch);focused_seen.update(present)
        elif entries[indices[0]].get('communication'):
            if not all(entries[i].get('communication') for i in indices):
                raise ValueError('sustained interleave requires homogeneous communication batches')
            comm.append(batch)
        elif entries[indices[0]].get('training_priority', 0) > 0:
            urgent.append(batch)
        else:
            debt.append(batch)
    ordinary=[];u=d=0;cursor=schedule_offset%5
    while u<len(urgent) or d<len(debt):
        if cursor!=4 and u<len(urgent):ordinary.append(urgent[u]);u+=1
        elif cursor==4 and d<len(debt):ordinary.append(debt[d]);d+=1
        elif u<len(urgent):ordinary.append(urgent[u]);u+=1
        else:ordinary.append(debt[d]);d+=1
        cursor=(cursor+1)%5
    result=list(focused);c=o=0
    while c<len(comm) or o<len(ordinary):
        if c<len(comm) and (o==len(ordinary) or c*len(ordinary)<=o*len(comm)):
            result.append(comm[c]);c+=1
        else:
            result.append(ordinary[o]);o+=1
    last={visit:i for i,(visit,_,_) in enumerate(result)}
    return [(visit,indices,last[visit]==i) for i,(visit,indices,_) in enumerate(result)]
