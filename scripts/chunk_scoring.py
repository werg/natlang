"""Parsed native assistant chunks and exact token-offset likelihood aggregation."""
import ast
import json
import math
import re


def _position(source, node, end=False):
    lines=source.splitlines(keepends=True)
    line=(node.end_lineno if end else node.lineno)-1
    column=node.end_col_offset if end else node.col_offset
    return sum(len(s) for s in lines[:line])+len(lines[line].encode()[:column].decode())


def _string_map(raw, value):
    # Map decoded characters to spans in a single Python string literal.
    # Reject unsupported literal spellings rather than inventing token attribution.
    if raw[:1] not in ("'", '"') or raw[-1:]!=raw[:1] or raw[:3] in ("'''", '"""'):
        raise ValueError('unsupported string spelling')
    positions=[];decoded='';i=1
    while i<len(raw)-1:
        start=i
        if raw[i]=='\\':
            i+=2
            if raw[start+1]=='u':i=start+6
            elif raw[start+1]=='U':i=start+10
            elif raw[start+1]=='x':i=start+4
            elif raw[start+1] in '01234567':
                while i<min(start+4,len(raw)-1) and raw[i] in '01234567':i+=1
        else:i+=1
        fragment=raw[start:i]
        char=ast.literal_eval(raw[0]+fragment+raw[0])
        decoded+=char;positions.extend([(start,i)]*len(char))
    if decoded!=value:raise ValueError('string offset reconstruction mismatch')
    return positions


def parsed_chunks(text, assistant):
    chunks=[]
    start=text.find('<|tool_call_start|>');end=text.find('<|tool_call_end|>')
    calls=assistant.get('tool_calls') or []
    if calls and start>=0 and end>start:
        offset=start+len('<|tool_call_start|>');source=text[offset:end]
        tree=ast.parse(source,mode='eval')
        nodes=tree.body.elts if isinstance(tree.body,(ast.List,ast.Tuple)) else [tree.body]
        if len(nodes)!=len(calls):raise ValueError('rendered call count mismatch')
        for index,(node,call) in enumerate(zip(nodes,calls)):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name):raise ValueError('unsupported call syntax')
            function=call['function'];arguments=function.get('arguments') or {}
            if isinstance(arguments,str):arguments=json.loads(arguments)
            if node.func.id!=function['name']:raise ValueError('rendered tool name mismatch')
            def add(kind,value,a,b,**extra):
                chunks.append(dict(kind=kind,value=value,char_start=offset+a,char_end=offset+b,call_index=index,**extra))
            add('tool_name',node.func.id,_position(source,node.func),_position(source,node.func,True))
            for keyword in node.keywords:
                if keyword.arg not in arguments:raise ValueError('rendered argument mismatch')
                value=arguments[keyword.arg];a=_position(source,keyword.value);b=_position(source,keyword.value,True)
                if keyword.arg=='code' and isinstance(value,str) and '\n' in value:
                    mapping=_string_map(source[a:b],value);cursor=0
                    for line in value.splitlines(keepends=True):
                        length=len(line)
                        if line.strip():add('code_line',line,a+mapping[cursor][0],a+mapping[cursor+length-1][1],argument_name=keyword.arg,value_start=cursor,value_end=cursor+length)
                        cursor+=length
                else:add('argument',value,a,b,argument_name=keyword.arg)
    elif not calls:
        content=assistant.get('content') or ''
        # Locate exact content, not reasoning/system text or formatting markers.
        offset=text.find(content) if content else -1
        if offset>=0:
            for match in re.finditer(r'[^.!?\n]+(?:[.!?]+|\n|$)',content):
                if match.group().strip():chunks.append(dict(kind='sentence',value=match.group(),value_start=match.start(),value_end=match.end(),char_start=offset+match.start(),char_end=offset+match.end()))
    if not chunks:chunks=[dict(kind='action',value=assistant,char_start=0,char_end=len(text))]
    return chunks


def score_chunks(tokenizer, ids, assistant, logprobs):
    text=tokenizer.decode(ids,skip_special_tokens=False)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=ids:raise ValueError('canonical completion offset tokenization mismatch')
    offsets=encoded['offset_mapping'];chunks=parsed_chunks(text,assistant)
    for chunk in chunks:
        indices=[i for i,(a,b) in enumerate(offsets) if b>a and a<chunk['char_end'] and b>chunk['char_start']]
        if not indices:raise ValueError('parsed chunk has no scored tokens')
        losses=[-logprobs[i] for i in indices]
        chunk.update(token_indices=indices,mean_nll=math.fsum(losses)/len(losses),max_nll=max(losses))
    return {'completion_text':text,'chunks':chunks,'chunk_scoring':'parsed-native-assistant-offsets/1'}
