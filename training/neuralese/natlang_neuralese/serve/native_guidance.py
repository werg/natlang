"""Exact native cache rollback for the existing Guide policy.

Only a prompt anchor is retained. Rejected reply prefixes are replayed through
native_step with their actual roles; training still uses the single graph pass.
"""
from __future__ import annotations
from .chat import RequestError

_CONTROL_FIELDS=('native_write_start_sample','native_write_close_sample','native_write_start_own','native_write_open_sample')
_REASONING_START='<think>'


def reasoning_opened(engine,seq):
    """Whether the generation prompt ends inside a reasoning block the reply continues (thinking templates, LFM3)."""
    if seq.items or seq.native_forced_tokens or seq.request.template or seq.request.forced_message is not None:
        return False
    plan=seq.prompt_plan or ()
    if not plan or plan[-1][0]!='tokens' or not plan[-1][1]:
        return False
    tail=engine.tokenizer.decode(list(plan[-1][1][-16:]),skip_special_tokens=False)
    return tail.rstrip().endswith(_REASONING_START)


def initialize(engine,seq):
    from dataclasses import is_dataclass,replace
    from .guidance import Guide
    seq.guide=Guide(seq.request.guidance)
    settings=getattr(seq.guide,'settings',None)
    if is_dataclass(settings) and hasattr(settings,'reasoning_open') and not seq.guide.forced_prefix():
        # Reasoning the generation prompt opened is not an action (fork a6401bbd3): unchecked, and it streams.
        seq.guide.settings=replace(settings,reasoning_open=reasoning_opened(engine,seq))
    seq.native_guidance_anchor={'cache':seq.cache,'logits':seq.logits,'own':seq.native_own_position,
        'generated':seq.generated_positions,'inputs':len(seq.native_rl_inputs),
        'open':seq.native_capture_open,'chunks':list(seq.native_capture_chunks),'ids':list(seq.native_capture_ids),
        'suffix':list(seq.native_template_suffix),**{key:getattr(seq,key) for key in _CONTROL_FIELDS}}
    prefix=seq.guide.forced_prefix()
    if prefix and not seq.request.template:
        seq.native_forced_tokens=engine._tokens(prefix)
    reply,_,_=render(engine,seq)
    if seq.guide.check(reply) is not None:
        raise RequestError('neuralese-guidance-template-conflict','the forced template violates guidance; choose a compatible call template')


def retained_source_offsets(tokenizer, text, ids):
    """Character offsets of actual sampled IDs in their decoded text, including byte fragments and noncanonical BPE
    splits; returns (offsets, policy)."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) == list(ids):
        return [list(pair) for pair in encoded['offset_mapping']], 'exact-tokenizer-offsets'
    from difflib import SequenceMatcher
    def decode(values):
        return tokenizer.decode(values, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    decoded = decode(ids)
    boundaries = [0]
    for stop in range(1, len(ids)+1):
        prefix = decode(ids[:stop])
        common = 0
        for left, right in zip(prefix, decoded):
            if left != right:
                break
            common += 1
        boundaries.append(max(boundaries[-1], common))
    boundaries[-1] = len(decoded)
    mapping = list(range(len(decoded)+1))
    policy = 'actual-token-decoder-alignment'
    if decoded != text:
        policy = 'actual-token-decoder-text-alignment'
        for _, a, b, c, d in SequenceMatcher(None, decoded, text, autojunk=False).get_opcodes():
            if a == b:
                mapping[a] = d
            else:
                for index in range(a, b+1):
                    mapping[index] = c + (index-a)*(d-c)//(b-a)
    return [[mapping[a], mapping[b]] for a,b in zip(boundaries, boundaries[1:])], policy


def render(engine,seq):
    text,starts,stops='',[],[]
    run=[]
    def flush():
        nonlocal text,run
        if not run:return
        value=engine.tokenizer.decode(run,skip_special_tokens=False)
        offsets,_=retained_source_offsets(engine.tokenizer,value,run)
        starts.extend(len(text)+a for a,b in offsets)
        stops.extend(len(text)+b for a,b in offsets)
        text+=value;run=[]
    for item in seq.items:
        if isinstance(item,int):run.append(item)
        else:
            flush();starts.append(len(text));text+='__nz';stops.append(len(text))
    flush()
    return text,starts,stops


def check(engine,seq):
    reply,starts,stops=render(engine,seq)
    verdict=seq.guide.check(reply)
    if verdict is None:
        stream_reasoning(seq,reply,stops)
        return False
    reason,at=verdict
    target=next((i for i,stop in enumerate(stops) if stop>at),len(stops)-1)
    while target>0 and starts[target-1]==starts[target]:target-=1
    if target<seq.native_guidance_streamed:
        # Streamed reasoning cannot be retracted; guidance never rejects inside it (Guide.check blanks it).
        seq.guide.accept(at);seq.guide.rejections.append({'reason':reason,'offset':at,'accepted':True})
        stream_reasoning(seq,reply,stops)
        return False
    own=seq.native_item_positions[target]
    anchor=seq.native_guidance_anchor
    position=anchor['inputs']+own-anchor['own']
    if own<anchor['own'] or position>=len(seq.native_rl_inputs) or seq.native_rl_inputs[position]['sample_index'] is None:
        raise RequestError('neuralese-guidance-template-conflict','guidance rejected forced template text; choose a compatible call template')
    tries=seq.retries.get(at,0)
    if tries>=seq.guide.settings.retries:
        seq.guide.accept(at);seq.guide.rejections.append({'reason':reason,'offset':at,'accepted':True})
        return False
    seq.retries[at]=tries+1
    seq.guide.rejections.append({'reason':reason,'offset':at})
    seq.bans.setdefault(own,set()).add(seq.native_rl_inputs[position]['token_id'])
    retained=seq.native_rl_inputs[anchor['inputs']:position]
    seq.cache,seq.logits=anchor['cache'],anchor['logits']
    seq.native_capture_open=anchor['open']
    seq.native_capture_chunks=list(anchor['chunks']);seq.native_capture_ids=list(anchor['ids'])
    for key in _CONTROL_FIELDS:setattr(seq,key,anchor[key])
    seq.native_own_position=anchor['own']
    # Batch ordinary runs, but preserve every producer boundary/role exactly.
    cursor=0
    while cursor<len(retained):
        entry=retained[cursor];token=entry['token_id'];producing=entry['input_role']=='produce'
        special=token in (engine.role_tokens.produce_open,engine.role_tokens.produce_close)
        end=cursor+1
        if not special:
            while end<len(retained) and (retained[end]['input_role']=='produce')==producing and retained[end]['token_id'] not in (engine.role_tokens.produce_open,engine.role_tokens.produce_close):end+=1
        ids=[entry['token_id'] for entry in retained[cursor:end]]
        output=engine._guidance_replay(seq,ids,producing)
        if token==engine.role_tokens.produce_open:
            seq.native_capture_open=True;seq.native_capture_chunks=[];seq.native_capture_ids=[]
            seq.native_write_start_own=seq.native_own_position
            seq.native_write_open_sample=entry['sample_index'];seq.native_write_start_sample=entry['sample_index']
        elif token==engine.role_tokens.produce_close:
            seq.native_capture_open=False;seq.native_capture_chunks=[];seq.native_capture_ids=[]
            for key in _CONTROL_FIELDS:setattr(seq,key,None)
        elif producing:
            if output is not None:seq.native_capture_chunks.append(output)
            seq.native_capture_ids.extend(ids)
        seq.native_own_position+=len(ids);cursor=end
    seq.generated_positions=anchor['generated']+len(retained)
    seq.native_rl_inputs=seq.native_rl_inputs[:position]
    seq.rl_samples=[sample for sample in seq.rl_samples if sample['query_own_position']<own-1]
    seq.native_rl_writes=[write for write in seq.native_rl_writes if write['own_end']<=own]
    del seq.items[target:];del seq.native_item_positions[target:]
    seq.blocks=[item for item in seq.items if not isinstance(item,int)]
    seq.native_forced_tokens=[]
    seq.native_template_suffix=list(anchor['suffix']) if seq.native_capture_open else []
    for point in list(seq.bans):
        if point>own:del seq.bans[point]
    seq.guide.rewind(at)
    return True


def _reasoning_bounds(seq,reply):
    """(stream limit, closing tag) of the reply's open reasoning: what precedes the limit is reasoning for good."""
    end=seq.guide.reasoning_end(reply)
    if end is not None:return end[0],end
    # Hold back a tail that may be the start of the closing tag.
    tags=[tag for tag in seq.guide.settings.reasoning_end if tag]
    hold=max((n for tag in tags for n in range(1,len(tag)) if reply.endswith(tag[:n])),default=0)
    return len(reply)-hold,None


def stream_reasoning(seq,reply,stops):
    """Streams the reply's open reasoning as it is generated (fork a6401bbd3): guidance neither checks nor rolls it
    back. Only a contiguous run of tokens from the start of the reply streams; the rest waits for `flush`."""
    listener=seq.request.on_delta
    settings=getattr(seq.guide,'settings',None)
    if listener is None or not getattr(settings,'reasoning_open',False) or not hasattr(seq.guide,'reasoning_end'):return
    limit,end=_reasoning_bounds(seq,reply)
    count=seq.native_guidance_streamed
    while (count<len(seq.items) and isinstance(seq.items[count],int) and stops[count]<=limit):count+=1
    while count>seq.native_guidance_streamed and reply[:stops[count-1]].endswith('\ufffd'):count-=1
    if count<=seq.native_guidance_streamed:return
    text=reply[seq.native_guidance_streamed_text:stops[count-1]]
    seq.native_guidance_streamed,seq.native_guidance_streamed_text=count,stops[count-1]
    if text:listener({'reasoning':text})


def flush(engine,seq):
    """Publish a guided reply once it is final: a stream has no retraction event. Reasoning already streamed is left
    out; the rest of it and what follows its closing tag go first, as reasoning and text."""
    if seq.request.on_delta is None:return
    start=seq.native_guidance_streamed
    if start:
        reply,_,stops=render(engine,seq)
        limit,end=_reasoning_bounds(seq,reply)
        after=len(seq.items)
        if end is not None:
            after=next((i+1 for i in range(start,len(seq.items)) if stops[i]>=end[1]),len(seq.items))
        if all(isinstance(item,int) for item in seq.items[start:after]):
            stop=stops[after-1] if after>start else seq.native_guidance_streamed_text
            tail=reply[seq.native_guidance_streamed_text:end[0] if end is not None else stop]
            if tail:seq.request.on_delta({'reasoning':tail})
            if end is not None and stop>end[1]:seq.request.on_delta({'text':reply[end[1]:stop]})
            start=after
    seq.native_guidance_flushing=True
    try:
        for item in seq.items[start:]:
            if isinstance(item,int):engine._emit(seq,token=item)
            else:seq.request.on_delta({'native_state':item.meta()})
    finally:seq.native_guidance_flushing=False
