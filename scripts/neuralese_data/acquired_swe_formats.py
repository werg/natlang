"""Exact released SWE raw/compact protocol readers; no text-based role guessing."""
from __future__ import annotations

import json
from .common import Reject
from .trajectory import tool_call


def compact_context(records,session_id):
    from .acquired_granted import complete_messages
    original=[];rows=[]
    for index,event in records:
        role=event.get('type')
        if role not in ('user','assistant','system','tool'):
            continue
        content=event.get('content')
        if not isinstance(content,list):raise Reject('unsupported SWE compact content inventory')
        for block in content:
            if not isinstance(block,dict):raise Reject('unsupported SWE compact content block')
            if block.get('type')=='tool_use':
                dispatch={key:value for key,value in block.items() if key!='result'}
                original.append({'role':role,'content':[dispatch]});rows.append(index)
                if 'result' in block:
                    # Compact v1 puts actual result bytes on its dispatch. Keep
                    # every result field and move it to observed context.
                    original.append({'role':'user','content':[{'type':'tool_result',
                        'tool_use_id':block['id'],'content':block['result']}]});rows.append(index)
            else:
                if 'text' in block and 'type' not in block:block={**block,'type':'text'}
                original.append({'role':role,'content':[block]});rows.append(index)
    messages=complete_messages(original)
    if not any(message['role']=='assistant' for message in messages):
        raise Reject('compact SWE source lacks original assistant targets')
    return records[-1][0],messages,{'session_id':session_id,'leaf_uuid':'compact-source',
        'original_message_rows':rows,'missing_ancestry':{},'original_models':[],
        'source_format':'swe-compact-v1','raw_context_not_normalized_table':True}


def codex_context(records,session_id):
    from .acquired_granted import complete_messages
    messages=[];rows=[];redacted=[];models=set();seen=set()
    for index,event in records:
        payload=event.get('payload',event.get('item'))
        if event.get('type')=='session_meta' and isinstance(payload,dict):
            instruction=payload.get('base_instructions')
            if isinstance(instruction,dict):instruction=instruction.get('text')
            if isinstance(instruction,str) and instruction:
                messages.append({'role':'system','content':instruction});rows.append(index)
            session_id=payload.get('session_id') or payload.get('id') or session_id
        if event.get('type')=='turn_context' and isinstance(payload,dict) and payload.get('model'):
            models.add(str(payload['model']))
        if event.get('type')!='response_item':continue
        if not isinstance(payload,dict):raise Reject('missing SWE Codex original response item')
        identifier=payload.get('id')
        if identifier:
            if identifier in seen:raise Reject('duplicate SWE Codex response item identity')
            seen.add(identifier)
        kind=payload.get('type');message=None
        if kind=='message':
            message={'role':payload['role'],'content':payload.get('content')}
        elif kind=='agent_message':
            message={'role':'user','content':payload.get('content')}
        elif kind in ('function_call','custom_tool_call'):
            name=payload.get('name');call_id=payload.get('call_id')
            if not isinstance(name,str) or not name or not isinstance(call_id,str) or not call_id:
                raise Reject('SWE Codex call lacks original name/call identity')
            arguments=payload.get('arguments') if kind=='function_call' else json.dumps({'input':payload.get('input')},ensure_ascii=False)
            if arguments is None:raise Reject('SWE Codex call lacks original arguments')
            message={'role':'assistant','content':None,'tool_calls':[tool_call(name,arguments,call_id)]}
        elif kind in ('function_call_output','custom_tool_call_output'):
            if not payload.get('call_id'):raise Reject('SWE Codex result lacks original call identity')
            message={'role':'tool','tool_call_id':payload['call_id'],'content':payload.get('output')}
        elif kind=='reasoning':
            summaries=payload.get('summary') or []
            texts=[item['text'] for item in summaries if isinstance(item,dict) and isinstance(item.get('text'),str)]
            if texts:message={'role':'assistant','content':'','reasoning':'\n'.join(texts)}
            if payload.get('encrypted_content') or payload.get('content'):redacted.append(index)
        else:raise Reject('unsupported original SWE Codex response item: '+str(kind))
        if message is not None:
            content=message.get('content')
            if isinstance(content,list) and any(isinstance(block,dict) and block.get('type')=='encrypted_content' for block in content):
                redacted.append(index)
            messages.append(message);rows.append(index)
    messages=complete_messages(messages)
    if not any(message['role']=='user' for message in messages) or not any(message['role']=='assistant' for message in messages):
        raise Reject('SWE Codex source lacks original request/assistant context')
    return records[-1][0],messages,{'session_id':session_id,'leaf_uuid':'codex-source',
        'original_message_rows':rows,'missing_ancestry':{},'original_models':sorted(models),
        'redacted_context_rows':redacted,'source_format':'codex-response-items',
        'raw_context_not_normalized_table':True,'custom_tool_argument_adaptation':'exact-input-field'}


def gemini_session_context(records):
    """Read one released pretty-printed Gemini session as a full trajectory.

    The filename suffix is misleading for this source family: each file is a
    single JSON session document. Keep source turn order, reasoning, tool
    dispatches, and the result nested on each original dispatch.
    """
    from .acquired_granted import complete_messages
    if len(records)!=1:
        raise Reject('Gemini session document must have one source row')
    row_index,document=records[0]
    if not isinstance(document,dict) or not isinstance(document.get('sessionId'),str) or not isinstance(document.get('messages'),list):
        raise Reject('unsupported pretty-printed SWE session document')
    raw_messages=document['messages']
    normalized=[];models=set();warning_rows=[];turn_rows=[];call_bindings=[]
    for index,original in enumerate(raw_messages):
        if not isinstance(original,dict):
            raise Reject('original Gemini session turn is not an object')
        kind=original.get('type')
        if kind=='user':
            content=original.get('content')
            if not isinstance(content,list):
                raise Reject('original Gemini user turn lacks structured content')
            # Normalize only the released text-only user block shape. Other
            # blocks remain addressable JSON through the common block reader.
            blocks=[]
            for block in content:
                if isinstance(block,dict) and isinstance(block.get('text'),str) and set(block)<= {'text'}:
                    blocks.append({'type':'text','text':block['text']})
                else:
                    blocks.append(block)
            normalized.append({'role':'user','content':blocks})
        elif kind=='gemini':
            content=original.get('content')
            if content is not None and not isinstance(content,str):
                raise Reject('original Gemini assistant content is not text')
            reasoning=original.get('thoughts') or []
            if not isinstance(reasoning,list):
                raise Reject('original Gemini thoughts are not a list')
            reasoning_parts=[]
            for thought in reasoning:
                if not isinstance(thought,dict) or not isinstance(thought.get('description'),str):
                    raise Reject('original Gemini thought lacks exact description text')
                subject=thought.get('subject')
                reasoning_parts.append((str(subject)+'\n' if isinstance(subject,str) and subject else '')+thought['description'])
            calls=[];results=[]
            tool_calls=original.get('toolCalls') or []
            if not isinstance(tool_calls,list):
                raise Reject('original Gemini toolCalls is not a list')
            for call in tool_calls:
                if not isinstance(call,dict) or not isinstance(call.get('id'),str) or not call['id'] or not isinstance(call.get('name'),str) or not call['name']:
                    raise Reject('original Gemini tool call lacks exact id or name')
                if 'args' not in call:
                    raise Reject('original Gemini tool call lacks original arguments')
                calls.append({'id':call['id'],'function':{'name':call['name'],'arguments':call['args']}})
                call_bindings.append({'message_index':index,'call_id':call['id'],'name':call['name'],
                    'arguments_sha256':__import__('hashlib').sha256(__import__('json').dumps(call['args'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                    'status':call.get('status'),'result_sha256':__import__('hashlib').sha256(__import__('json').dumps(call.get('result'),ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest() if 'result' in call else None})
                # The released result is explicitly nested under this exact
                # call, so retain the complete object as observed tool output.
                if 'result' in call and call['result'] is not None:
                    results.append({'role':'tool','tool_call_id':call['id'],'name':call['name'],
                                    'content':__import__('json').dumps(call['result'],ensure_ascii=False,separators=(',',':'))})
            assistant={'role':'assistant','content':content}
            if reasoning_parts:assistant['reasoning']='\n\n'.join(reasoning_parts)
            if calls:assistant['tool_calls']=calls
            normalized.append(assistant);normalized.extend(results)
            if original.get('model'):models.add(str(original['model']))
        elif kind=='warning':
            content=original.get('content')
            if not isinstance(content,str):
                raise Reject('original Gemini warning lacks exact text')
            # Warnings are system context, never assistant targets.
            normalized.append({'role':'system','content':content});warning_rows.append(index)
        else:
            raise Reject('unsupported original Gemini session turn type: '+str(kind))
        turn_rows.append(index)
    messages=complete_messages(normalized)
    if not any(message['role']=='user' for message in messages) or not any(message['role']=='assistant' for message in messages):
        raise Reject('Gemini source lacks original request/assistant context')
    from .acquired_granted import _digest
    session_id=document['sessionId']
    return row_index,messages,{'session_id':session_id,'leaf_uuid':session_id,
        'original_message_rows':turn_rows,'missing_ancestry':{},'original_models':sorted(models),
        'source_format':'gemini-session-document','raw_context_not_normalized_table':True,
        'source_project_hash':document.get('projectHash'),'source_document_kind':document.get('kind'),
        'source_document_sha256':_digest(document),'warning_message_rows':warning_rows,
        'gemini_tool_call_bindings':call_bindings,'reasoning_source_field':'thoughts',
        'full_source_turn_count':len(raw_messages)}
