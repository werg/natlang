"""Free decoded final values after recorded tool execution, with recurrent return ablations.

This measures conditional final-decision correctness, not autonomous end-to-end task success.
Tool prefixes are recorded teacher executions. Writers regenerate every dependent return
from producer context; no gold payload or length is supplied. All arms use the same typed
return envelope, then decode the value freely. Whole-task execution is evaluated separately.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch
from ..serve.engine import GenerationRequest
from ..serve.grad import encode_text, embed_text
from ..serve.store import make_block, encode_block
from ..serve.chat import call_reply
from ..train.trajectory_probe import select_held, select_paired_held, source_groups
from ..train.trajectories import (crisp_messages, render, reads, target_write, handover_notes,
                                  write_site, write_value_type, write_value_path, write_site_arguments)
from ..common.hashing import sha256_file_hex as sha

def returned(row):
    def contains_write(value):
        if isinstance(value,dict):
            return '$write' in value or any(contains_write(item) for item in value.values())
        if isinstance(value,list):return any(contains_write(item) for item in value)
        return False
    for call in row.get('target',{}).get('tool_calls',[]):
        if call['function']['name']=='return_result':
            args=json.loads(call['function']['arguments'])
            if args.get('status')=='success' and 'value' in args and not contains_write(args['value']):return True,args['value']
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
    p.add_argument('--dtype',choices=['auto','float32','bfloat16'],default='auto',
                   help='model arithmetic for matched-device diagnostics; auto uses the loader default')
    p.add_argument('--limit',type=int,default=12);p.add_argument('--max-output',type=int,default=1024)
    p.add_argument('--arms',default='crisp,written,shuffled,zero,removed');p.add_argument('--depth',type=int,default=8)
    p.add_argument('--prompt-parameters',choices=['trained','initial','encoded','crisp'],default='trained',
                   help='instruction embedding intervention for generated-writer arms; source-control arms use crisp instructions')
    p.add_argument('--inspect-writer-text', action='store_true', help='diagnostic exact embedding token trace; no approximate decoding or gold generation inputs')
    p.add_argument('--content-projection',choices=['trained','identity'],default='trained',
                   help='diagnostic raw content residual intervention; identity is not the trained checkpoint')
    a=p.parse_args(argv);torch.set_num_threads(a.threads)
    arms=a.arms.split(','); allowed={'crisp','written','shuffled','zero','removed','embedded','encoded','embedded-transparent','encoded-transparent'}
    if not set(arms)<=allowed or len(set(arms))!=len(arms):raise ValueError('invalid arms')
    if a.out.exists():raise ValueError('fresh immutable output required')
    dtype = None if a.dtype == 'auto' else getattr(torch, a.dtype)
    from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
    engine, state = load_recurrence_checkpoint(a.checkpoint, device=a.device, dtype=dtype)
    options=state['identity']['options']
    # Keep this evaluator's established execution geometry for historical comparisons.
    engine.backbone.ffn_chunk_tokens=2048
    if a.content_projection == 'identity':
        if engine.heads.profile != 'raw-token-v1':
            raise ValueError('content identity diagnostic requires raw-token-v1')
        with torch.no_grad():
            engine.heads.set_content_transport('raw-identity')
            engine.heads.content.proj.weight.zero_()
            engine.heads.content.proj.bias.zero_()
    piece_rows=list(map(json.loads,a.pieces.open()))
    texts={r['name']:r['text'] for r in piece_rows}
    piece_kinds={r['name']:r.get('kind') for r in piece_rows}
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
    # Share the trainer's typed, factual-disjoint reciprocal donor proof.
    # Free stopping remains distinct from its forced-native-length CE probe.
    pool=selected
    if 'shuffled' in arms:
        selected,selection=select_paired_held(pool,a.limit,producers,piece_kinds)
    else:
        selected=select_held(pool,a.limit)
        selection={'policy':'factual-group-round-robin-without-donor-arm',
                   'input_rows':len(pool),'selected_ids':[r['id'] for r in selected]}
    if not selected:raise ValueError('no eligible held-out final-value readers; donor exclusions require review')
    a.out.mkdir(parents=True)
    (a.out/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    pins={str(path):sha(path) for path in [a.checkpoint,a.records,a.pieces]}
    # Learned parameters are selected by name only when their original text is identical.
    training_piece_path=Path(options['pieces'])
    if sha(training_piece_path)!=state['identity']['files'][str(training_piece_path.resolve())]:raise ValueError('checkpoint initialization texts changed')
    trained_texts={r['name']:r['text'] for r in map(json.loads,training_piece_path.open())}
    soft_ids={}; payloads={}; memo={}; soft_initializations={}
    def soft(name):
        if a.prompt_parameters == 'crisp':
            return {'type': 'text', 'text': texts[name]}
        if name not in soft_ids:
            bank = state['params'] if a.prompt_parameters == 'trained' else state.get('init', {})
            if a.prompt_parameters != 'encoded' and name in bank and trained_texts.get(name)==texts[name]:
                block=engine.store.put(make_block(bank[name].to(a.device),engine.dialect));soft_initializations[name]=a.prompt_parameters
            else:
                from ..text_init import instruction_rows
                block=instruction_rows(engine,texts[name],'Neuralese<SystemPrompt>')[0];soft_initializations[name]='unseen-text-initialized-in-context'
            soft_ids[name]=block.id
        return {'type':'neuralese','id':soft_ids[name]}
    writer_rows=[]; writer_trace=None
    if a.inspect_writer_text:
        if engine.heads.profile != 'raw-token-v1' or engine.heads.content.transport != 'raw-identity':
            raise ValueError('exact writer trace requires raw-token identity transport')
        from .raw_writer_trace import RawWriterTrace
        writer_trace=RawWriterTrace(engine.heads.feedback.embedding)
    def write(name,visiting=()):
        if name in memo:return memo[name]
        if name in visiting:raise ValueError('cycle')
        if len(visiting)>=a.depth:raise ValueError('depth would require a gold fallback; raise --depth')
        producer=producers[name]; own=target_write(producer)
        children={n:write(n,visiting+(name,)) for n in reads(producer) if n!=own}
        messages=render(producer['messages'],soft,handover_notes(producer),children)
        tool,before,argument,_=write_site(producer)
        value_path=write_value_path(producer,name)
        arguments=write_site_arguments(producer,name) if len(value_path)>1 else before
        argument_path=list(value_path) if len(value_path)>1 else None
        prefix=call_reply(lambda m,g:engine.tokenizer.apply_chat_template(m,tokenize=False,add_generation_prompt=g),
                          tool,arguments,argument,argument_path=argument_path)[0]
        response=engine.generate(GenerationRequest(messages=messages,tools=producer.get('tools'),max_tokens=engine.max_block+len(engine._tokens(prefix))+8,
          template={'call':tool,'arguments':arguments,'argument':argument,'argument_path':argument_path,
                    'value':'write','value_type':write_value_type(producer)},temperature=0,neuralese_temperature=0))
        blocks=response.get('neuralese',{}).get('blocks',[])
        if len(blocks)!=1:raise ValueError('writer did not produce exactly one block')
        block=engine.store.get(blocks[0]['id']);memo[name]=block.id;payloads[name]=block.payload
        writer_row={'name':name, 'producer_record_id':producer['id'], 'depth':len(visiting)+1,
                    'truncated':block.truncated, 'vectors':block.payload.shape[0],
                    'max_block':engine.max_block, 'dependency_names':sorted(children)}
        stop_logits=block.producer.get('stop_logits',[])
        writer_row['stop_log_odds']={'count':len(stop_logits),
            'max':max(stop_logits) if stop_logits else None,
            'last':stop_logits[-1] if stop_logits else None}
        writer_row['source_groups']=sorted(source_groups(producer))
        # Preserve costly generated returns for diagnosis even if later reader
        # execution fails. These are diagnostic artifacts, never SFT admission.
        block_dir=a.out/'writer-blocks';block_dir.mkdir(exist_ok=True)
        block_path=block_dir/(hashlib.sha256(name.encode()).hexdigest()+'.safetensors')
        block_path.write_bytes(encode_block(block))
        writer_row.update(block_id=block.id,block_file=str(block_path.relative_to(a.out)),
                          block_sha256=sha(block_path))
        if writer_trace is not None:
            writer_row.update(writer_trace.decode(block.payload,engine.tokenizer))
        writer_rows.append(writer_row)
        with (a.out/'writers.jsonl').open('a') as writer_log:
            writer_log.write(json.dumps(writer_row)+'\n')
        print(json.dumps({'writer_complete':name,'vectors':block.payload.shape[0],
                          'truncated':block.truncated,'writers_completed':len(memo)}),flush=True)
        return block.id
    result_rows=[]; summaries={}
    if any(arm in {'written','shuffled','zero','removed'} for arm in arms):
        for row in selected:
            for name in reads(row):write(name)
    original_prompt_embeddings=engine.prompt_embeddings
    for arm in arms:
        def arm_prompt_embeddings(messages, tools, *, prepared=None, block_mode='port'):
            return original_prompt_embeddings(messages, tools, prepared=prepared,
                block_mode='transparent' if arm.endswith('-transparent') else 'port')
        engine.prompt_embeddings=arm_prompt_embeddings
        passed=0;started=time.time()
        for i,row in enumerate(selected):
            expected=returned(row)[1]
            if arm in {'embedded','encoded','embedded-transparent','encoded-transparent'}:
                initializer=embed_text if arm.startswith('embedded') else encode_text
                mapping={n:initializer(engine,handover_notes(producers[n])[n]).id for n in reads(row)}
            else:
                mapping={n:memo[n] for n in reads(row)} if arm!='crisp' else {}
            if arm=='shuffled':
                proof=selection['selected_mappings'][row['id']]
                slots=proof['donor_payload_to_recipient_payload']
                if set(slots.values())!=set(mapping):raise ValueError('proven donor slot mapping changed')
                # Preserve the entire observed donor block: no repetition, clipping or gold length hint.
                mapping={recipient:engine.store.put(make_block(payloads[donor],engine.dialect)).id
                         for donor,recipient in slots.items()}
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
            result={'id':row['id'],'arm':arm,'family':row.get('task_family'),'expected':expected,'decoded':value,'valid_return':valid,'passed':ok,'response':response,'input_identity':identity,'source_groups':sorted(source_groups(row)),
                    'donor_proof':selection.get('selected_mappings',{}).get(row['id']) if arm=='shuffled' else None,
                    'slot_lengths':{name:engine.store.get(block_id).payload.shape[0] for name,block_id in mapping.items()}}
            result_rows.append(result)
            with (a.out/'results.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
            print(json.dumps({'arm':arm,'completed':i+1,'passed':passed}),flush=True)
        summaries[arm]={'n':len(selected),'passed':passed,'seconds':round(time.time()-started,2)}
    report={'schema':'natlang.conditional-return-execution/1','checkpoint_step':state['step'],'pins':pins,'arms':summaries,
      'prompt_parameters':a.prompt_parameters, 'device':a.device, 'model_dtype':str(engine.backbone.embedding_weight.dtype),
      'content_projection':a.content_projection, 'content_transport':engine.heads.content.transport, 'inspect_writer_text':a.inspect_writer_text, 'checkpoint_weights_unmodified':a.content_projection == 'trained',
      'instruction_control_arms':[arm for arm in arms if arm in {'crisp','embedded','encoded','embedded-transparent','encoded-transparent'}],
      'selection_policy':'typed-factual-reciprocal-readers-v1' if 'shuffled' in arms else selection['policy'],
      'selection':selection,'donor_length_policy':'whole observed donor; no padding/repetition/clipping; possible length confound recorded',
      'selected':[r['id'] for r in selected],'soft_initializations':soft_initializations,'writer_blocks':len(memo),
      'writer_stopping':{'generated_blocks':len(writer_rows),
                         'truncated_blocks':sum(bool(r['truncated']) for r in writer_rows),
                         'max_generated_vectors':max((r['vectors'] for r in writer_rows),default=0)},
      'scope':'free decoded final values after recorded teacher tool prefixes; not autonomous whole-task success',
      'gold_writer_inputs':False,'gold_source_control_arms':[arm for arm in arms if arm in {'embedded','encoded','embedded-transparent','encoded-transparent'}],'gold_length_hint':False,'forced_envelope':'return_result(status=success,value=<free decoding>)'}
    (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summaries))
if __name__=='__main__':main()
