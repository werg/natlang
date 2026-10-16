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
