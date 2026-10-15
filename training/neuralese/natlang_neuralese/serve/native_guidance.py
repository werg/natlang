"""Exact native cache rollback for the existing Guide policy.

Only a prompt anchor is retained. Rejected reply prefixes are replayed through
native_step with their actual roles; training still uses the single graph pass.
"""
from __future__ import annotations
from .chat import RequestError

_CONTROL_FIELDS=('native_write_start_sample','native_write_close_sample','native_write_start_own','native_write_open_sample')


def initialize(engine,seq):
    from .guidance import Guide
    seq.guide=Guide(seq.request.guidance)
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


def render(engine,seq):
    from ..model.semantic_message import retained_source_offsets
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
    if verdict is None:return False
    reason,at=verdict
    target=next((i for i,stop in enumerate(stops) if stop>at),len(stops)-1)
    while target>0 and starts[target-1]==starts[target]:target-=1
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
        output=engine._native_forward(ids,seq.cache,producing=producing)
        seq.cache,seq.logits=output['cache'],engine._native_logits(output['h_final'])
        if token==engine.role_tokens.produce_open:
            seq.native_capture_open=True;seq.native_capture_chunks=[];seq.native_capture_ids=[]
            seq.native_write_start_own=seq.native_own_position
            seq.native_write_open_sample=entry['sample_index'];seq.native_write_start_sample=entry['sample_index']
        elif token==engine.role_tokens.produce_close:
            seq.native_capture_open=False;seq.native_capture_chunks=[];seq.native_capture_ids=[]
            for key in _CONTROL_FIELDS:setattr(seq,key,None)
        elif producing:
            seq.native_capture_chunks.append(output);seq.native_capture_ids.extend(ids)
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


def flush(engine,seq):
    # Guided stream has no retraction event: publish only committed output.
    if seq.request.on_delta is None:return
    seq.native_guidance_flushing=True
    try:
        for item in seq.items:
            if isinstance(item,int):engine._emit(seq,token=item)
            else:seq.request.on_delta({'native_state':item.meta()})
    finally:seq.native_guidance_flushing=False
