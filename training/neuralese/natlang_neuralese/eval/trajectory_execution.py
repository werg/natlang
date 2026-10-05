"""Free decoded final values after recorded tool execution, with recurrent return ablations.

This measures conditional final-decision correctness, not autonomous end-to-end task success.
Tool prefixes are recorded teacher executions. Writers regenerate every dependent return
from producer context; no gold payload or length is supplied. All arms use the same typed
return envelope, then decode the value freely. Whole-task execution is evaluated separately.
"""
from __future__ import annotations
import argparse, hashlib, json, time, re
from pathlib import Path
import torch
from ..serve import load_engine
from ..serve.engine import GenerationRequest
from ..serve.grad import encode_text, embed_text
from ..serve.store import make_block
from ..serve.chat import call_reply
from ..train.trajectories import crisp_messages, render, reads, target_write, handover_notes, write_site

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()
def returned(row):
    for call in row.get('target',{}).get('tool_calls',[]):
        if call['function']['name']=='return_result':
            args=json.loads(call['function']['arguments'])
            if args.get('status')=='success' and 'value' in args and not (isinstance(args['value'],dict) and '$write' in args['value']):return True,args['value']
    return False,None

def decoded(response):
    message=response['choices'][0]['message']
    for call in message.get('tool_calls',[]):
        if call['function']['name']=='return_result':
            args=call['function']['arguments']
            try:
                args=json.loads(args) if isinstance(args,str) else args
            except (json.JSONDecodeError, TypeError):
                return False,None
            if not isinstance(args,dict):return False,None
            if args.get('status')=='success' and 'value' in args:return True,args['value']
    return False,None

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--pieces',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--limit',type=int,default=12);p.add_argument('--max-output',type=int,default=1024)
    p.add_argument('--arms',default='crisp,written,shuffled,zero,removed');p.add_argument('--depth',type=int,default=8)
    a=p.parse_args(argv);torch.set_num_threads(a.threads)
    arms=a.arms.split(','); allowed={'crisp','written','shuffled','zero','removed','embedded','encoded','embedded-transparent','encoded-transparent'}
    if not set(arms)<=allowed or len(set(arms))!=len(arms):raise ValueError('invalid arms')
    if a.out.exists():raise ValueError('fresh immutable output required')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if state.get('schema')!='natlang.neuralese_recurrence_checkpoint/1':raise ValueError('requires complete recurrence checkpoint')
    options=state['identity']['options']
    engine=load_engine(options.get('base'),heads_checkpoint=options.get('heads'),device=a.device)
    engine.heads.load_state_dict(state['heads']); engine.backbone.ffn_chunk_tokens=2048
    if state.get('control_rows') is not None:
        with torch.no_grad():engine.backbone.control_rows.copy_(state['control_rows'].to(engine.backbone.control_rows))
    # A mixed stage may have extended the parent's adapter coverage.
    # Recreate exactly that coverage before restoring its trained values.
    adapter_state=state.get('lora',{})
    ranks={int(value.shape[0]) for name,value in adapter_state.items() if '.lora_A.' in name}
    layers=sorted({int(match[1]) for name in adapter_state for match in [re.search(r'model\.layers\.(\d+)\.',name)] if match})
    if len(ranks)>1:raise ValueError('mixed adapter ranks require explicit deployment mapping')
    if layers and ranks:
        from ..train.adapters import inject_lora
        rank=next(iter(ranks))
        inject_lora(engine.backbone,layers,rank=rank,alpha=2*rank)
    parameters=dict(engine.backbone.hf.named_parameters())
    with torch.no_grad():
        for name,value in state.get('lora',{}).items():
            if name not in parameters or parameters[name].shape!=value.shape:raise ValueError('backbone adapter mismatch: '+name)
            parameters[name].copy_(value.to(parameters[name]))
    texts={r['name']:r['text'] for r in map(json.loads,a.pieces.open())}
    all_rows=[json.loads(line) for line in a.records.open()]
    producers={}
    for row in all_rows:
        name=target_write(row)
        if name:
            if name in producers:raise ValueError('ambiguous producer '+name)
            producers[name]=row
    # The assembler provides reviewed test-only rows; no train data contributes to pass counts.
    selected=[]
    for row in all_rows:
        valid,_=returned(row)
        if row.get('split')=='test' and row.get('training_admission',{}).get('approved') is True and valid and reads(row):selected.append(row)
    # Spread across families and chain depths rather than taking one contiguous fixture block.
    selected=sorted(selected,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest())[:a.limit]
    if not selected:raise ValueError('no held-out final-value readers')
    a.out.mkdir(parents=True)
    pins={str(path):sha(path) for path in [a.checkpoint,a.records,a.pieces]}
    # Learned parameters are selected by name only when their original text is identical.
    training_piece_path=Path(options['pieces'])
    if sha(training_piece_path)!=state['identity']['files'][str(training_piece_path.resolve())]:raise ValueError('checkpoint initialization texts changed')
    trained_texts={r['name']:r['text'] for r in map(json.loads,training_piece_path.open())}
    soft_ids={}; payloads={}; memo={}; soft_initializations={}
    def soft(name):
        if name not in soft_ids:
            if name in state['params'] and trained_texts.get(name)==texts[name]:
                block=engine.store.put(make_block(state['params'][name].to(a.device),engine.dialect));soft_initializations[name]='trained'
            else:block=encode_text(engine,texts[name]);soft_initializations[name]='unseen-text-initialized'
            soft_ids[name]=block.id
        return {'type':'neuralese','id':soft_ids[name]}
    def write(name,visiting=()):
        if name in memo:return memo[name]
        if name in visiting:raise ValueError('cycle')
        if len(visiting)>=a.depth:raise ValueError('depth would require a gold fallback; raise --depth')
        producer=producers[name]; own=target_write(producer)
        children={n:write(n,visiting+(name,)) for n in reads(producer) if n!=own}
        messages=render(producer['messages'],soft,handover_notes(producer),children)
        tool,before,argument,_=write_site(producer)
        prefix=call_reply(lambda m,g:engine.tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=g),tool,before,argument)[0]
        response=engine.generate(GenerationRequest(messages=messages,tools=producer.get('tools'),max_tokens=engine.max_block+len(engine._tokens(prefix))+8,
          forced=[prefix,{'neuralese':'write'}],temperature=0,neuralese_temperature=0))
        blocks=response.get('neuralese',{}).get('blocks',[])
        if len(blocks)!=1:raise ValueError('writer did not produce exactly one block')
        block=engine.store.get(blocks[0]['id']);memo[name]=block.id;payloads[name]=block.payload
        print(json.dumps({'writer_complete':name,'vectors':block.payload.shape[0],'writers_completed':len(memo)}),flush=True)
        return block.id
    result_rows=[]; summaries={}
    if any(arm in {'written','shuffled','zero','removed'} for arm in arms):
        for row in selected:
            for name in reads(row):write(name)
    original_prompt_embeddings=engine.prompt_embeddings
    for arm in arms:
        engine.prompt_embeddings=lambda messages,tools:original_prompt_embeddings(messages,tools,block_mode='transparent' if arm.endswith('-transparent') else 'port')
        passed=0;started=time.time()
        for i,row in enumerate(selected):
            expected=returned(row)[1]
            if arm in {'embedded','encoded','embedded-transparent','encoded-transparent'}:
                initializer=embed_text if arm.startswith('embedded') else encode_text
                mapping={n:initializer(engine,handover_notes(producers[n])[n]).id for n in reads(row)}
            else:
                mapping={n:memo[n] for n in reads(row)} if arm!='crisp' else {}
            if arm=='shuffled':
                other=selected[(i+1)%len(selected)]
                donor=[payloads[n] for n in sorted(reads(other))]
                if len(selected)<2 or not donor:raise ValueError('shuffle requires another reader')
                for k,name in enumerate(sorted(mapping)):
                    n=payloads[name].shape[0];d=donor[k%len(donor)];d=d.repeat((n+d.shape[0]-1)//d.shape[0],1)[:n]
                    mapping[name]=engine.store.put(make_block(d,engine.dialect)).id
            elif arm=='zero':
                mapping={n:engine.store.put(make_block(torch.zeros_like(payloads[n]),engine.dialect)).id for n in mapping}
            messages=crisp_messages(row['messages'],texts,handover_notes(row)) if arm=='crisp' else render(row['messages'],(lambda name:{'type':'text','text':texts[name]}) if arm in {'embedded','encoded','embedded-transparent','encoded-transparent'} else soft,handover_notes(row),mapping)
            if arm=='removed':
                for message in messages:
                    if isinstance(message.get('content'),list):
                        message['content']=[part for part in message['content'] if part.get('id') not in set(mapping.values())]
            response=engine.generate(GenerationRequest(messages=messages,tools=row.get('tools'),max_tokens=a.max_output,temperature=0,seed=0,
              template={'call':'return_result','arguments':{'status':'success'},'value':'decode'}))
            valid,value=decoded(response);ok=valid and value==expected;passed+=ok
            identity=None
            if arm=='embedded-transparent':
                plain=original_prompt_embeddings(crisp_messages(row['messages'],texts,handover_notes(row)),row.get('tools'))
                transported=engine.prompt_embeddings(messages,row.get('tools'))
                identity={'plain_positions':plain.shape[1],'transport_positions':transported.shape[1],'max_abs_error':float((plain-transported).abs().max()) if plain.shape==transported.shape else None}
            result={'id':row['id'],'arm':arm,'family':row.get('task_family'),'expected':expected,'decoded':value,'valid_return':valid,'passed':ok,'response':response,'input_identity':identity}
            result_rows.append(result)
            with (a.out/'results.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            print(json.dumps({'arm':arm,'completed':i+1,'passed':passed}),flush=True)
        summaries[arm]={'n':len(selected),'passed':passed,'seconds':round(time.time()-started,2)}
    report={'schema':'natlang.conditional-return-execution/1','checkpoint_step':state['step'],'pins':pins,'arms':summaries,
      'selected':[r['id'] for r in selected],'soft_initializations':soft_initializations,'writer_blocks':len(memo),
      'scope':'free decoded final values after recorded teacher tool prefixes; not autonomous whole-task success',
      'gold_writer_inputs':False,'gold_source_control_arms':[arm for arm in arms if arm in {'embedded','encoded','embedded-transparent','encoded-transparent'}],'gold_length_hint':False,'forced_envelope':'return_result(status=success,value=<free decoding>)'}
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries))
if __name__=='__main__':main()
