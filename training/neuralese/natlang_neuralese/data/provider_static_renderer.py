"""Model-free exact teacher target renderer, mirrored from serving."""
import json
from ..serve.chat import RequestError

def render_forced_message_plan(self,message,tools,*,opening,spans):
    """Render authenticated targets without running a model or inventing captures."""
    if not isinstance(message,dict) or message.get('role')!='assistant':
        raise RequestError('forced-message','the replay target must be one assistant message')
    import copy,re
    target=copy.deepcopy(message);markers={}
    vocab=getattr(self.tokenizer,'get_vocab',lambda:{})()
    def marker_for(body):
        supplied=json.dumps({'target':target,'opening':opening,'tools':tools},ensure_ascii=False,sort_keys=True)
        special_ids=set(getattr(self.tokenizer,'all_special_ids',()))
        candidates=sorted(vocab.items(),key=lambda item:(item[1],item[0]))
        for token,token_id in candidates:
            if (not isinstance(token,str) or not re.fullmatch(r'[A-Za-z0-9_]{2,}',token) or
                    token in self.specials or token_id in special_ids or token_id in self.stop_ids or
                    token in supplied or token in body or token in markers):continue
            ids=self.tokenizer(token,add_special_tokens=False,split_special_tokens=True)['input_ids']
            if ids==[token_id]:
                markers[token]=body
                return token
        raise RequestError('forced-message','could not allocate a collision-free exact teacher-span marker')
    if not isinstance(spans,list):raise RequestError('forced-message','teacher native spans need explicit typed occurrence paths')
    seen_spans=set();parsed_arguments={}
    for span in spans:
        if (not isinstance(span,dict) or not isinstance(span.get('path'),list) or
                not isinstance(span.get('marker'),str) or not span['marker'] or
                not isinstance(span.get('text'),str) or not span['text']):
            raise RequestError('forced-message','teacher native span path, marker, or typed body is malformed')
        path=span['path'];marker=span['marker'];body=span['text']
        if marker in seen_spans or json.dumps(target,ensure_ascii=False).count(marker)!=1:
            raise RequestError('forced-message','teacher span marker is duplicate or ambiguous in its exact assistant target')
        seen_spans.add(marker)
        if path==['content']:
            value=target.get('content')
            if not isinstance(value,str) or value.count(marker)!=1:
                raise RequestError('forced-message','teacher content span does not resolve to one exact typed occurrence')
            target['content']=value.replace(marker,marker_for(body),1)
            continue
        if len(path)<4 or path[0]!='tool_calls' or type(path[1]) is not int or path[2]!='arguments':
            raise RequestError('forced-message','teacher tool span must name an exact tool argument field')
        index=path[1];calls=target.get('tool_calls') or []
        if index<0 or index>=len(calls):raise RequestError('forced-message','teacher tool span call ordinal is outside the target')
        function=calls[index].get('function') or {};name=function.get('name')
        argument_path=path[3:]
        if not ((name=='eval' and argument_path and argument_path[0]=='code') or
                (name=='return_result' and argument_path and argument_path[0]=='value')):
            raise RequestError('forced-message','typed teacher span is not at a supported eval-code or typed return-result occurrence')
        if index not in parsed_arguments:
            try:parsed_arguments[index]=json.loads(function.get('arguments'))
            except (TypeError,json.JSONDecodeError) as error:
                raise RequestError('forced-message','teacher tool arguments are not exact structured JSON') from error
        value=parsed_arguments[index]
        for part in argument_path[:-1]:
            if isinstance(value,dict) and isinstance(part,str) and part in value:value=value[part]
            elif isinstance(value,list) and type(part) is int and 0<=part<len(value):value=value[part]
            else:raise RequestError('forced-message','teacher typed span path is missing in structured tool arguments')
        leaf=argument_path[-1]
        if isinstance(value,dict) and isinstance(leaf,str) and isinstance(value.get(leaf),str) and value[leaf].count(marker)==1:
            value[leaf]=value[leaf].replace(marker,marker_for(body),1)
        elif isinstance(value,list) and type(leaf) is int and 0<=leaf<len(value) and isinstance(value[leaf],str) and value[leaf].count(marker)==1:
            value[leaf]=value[leaf].replace(marker,marker_for(body),1)
        else:
            raise RequestError('forced-message','teacher tool span does not resolve to one exact typed string occurrence')
    for index,arguments in parsed_arguments.items():
        target['tool_calls'][index]['function']['arguments']=json.dumps(arguments,ensure_ascii=False,separators=(',',':'))
    from ..serve.chat import assistant_reply_segments,normalize_tool_call_arguments
    target=normalize_tool_call_arguments([target])[0][0]
    def apply(messages,generation):
        # Prefix comparison validates the assistant continuation, not a
        # serialized rendering of native KV. The actual prompt was already
        # rendered by render_messages into token/native-state segments.
        # Give template prefix comparison a stable textual stand-in for
        # each native reference so input-state parts never stringify as
        # Python dictionaries and perturb the assistant continuation.
        import copy
        prepared=copy.deepcopy(messages)
        def stand_in(value):
            if isinstance(value,list):return [stand_in(item) for item in value]
            if isinstance(value,dict):
                if isinstance(value.get('role'),str) and isinstance(value.get('content'),list):
                    parts=[]
                    for part in value['content']:
                        if isinstance(part,dict) and part.get('type')=='native_state':
                            parts.append(f'[native state {part.get("id")}]')
                        elif isinstance(part,dict) and part.get('type')=='text' and isinstance(part.get('text'),str):
                            parts.append(part['text'])
                        else:
                            raise RequestError('forced-message','native-state prefix validation encountered an unsupported content part')
                    return {**{key:stand_in(item) for key,item in value.items() if key!='content'},
                        'content':''.join(parts)}
                if value.get('type')=='native_state':
                    ident=value.get('id')
                    return {'type':'text','text':f'[native state {ident}]'}
                return {key:stand_in(item) for key,item in value.items()}
            return value
        prepared=stand_in(prepared)
        # Provider-captured Natlang history uses OpenAI JSON-text tool
        # arguments. Chat-template renderers consume structured objects;
        # render_request performs this same normalization before prefix
        # validation and native prefill.
        from ..serve.chat import normalize_tool_call_arguments
        prepared=normalize_tool_call_arguments(prepared)[0]
        return self.tokenizer.apply_chat_template(prepared,tools=tools or None,
            tokenize=False,add_generation_prompt=generation)
    segments=assistant_reply_segments(apply,target,self.specials,opening=opening,
        call_format=getattr(self,"tool_call_format",None),tools=tools)
    if segments is None:raise RequestError('forced-message','assistant response does not continue this exact actual prompt')
    plan=[];found={marker:0 for marker in markers}
    def append_text(text,escaped):
        if text:
            plan.extend(self.tokenizer(text,add_special_tokens=False,split_special_tokens=True)['input_ids']
                if escaped else self._template_tokens(text))
    for text,escaped in segments:
        cursor=0
        while cursor<len(text):
            positions=[(text.find(marker,cursor),marker,body) for marker,body in markers.items()
                if text.find(marker,cursor)>=0]
            if not positions:
                append_text(text[cursor:],escaped);break
            start,marker,body=min(positions,key=lambda item:item[0])
            append_text(text[cursor:start],escaped)
            body_ids=self.tokenizer(body,add_special_tokens=False,split_special_tokens=True)['input_ids']
            if not body_ids or any(token in self.stop_ids for token in body_ids):
                raise RequestError('forced-message','native teacher text is empty or contains an exact model stop token')
            plan.extend([self.role_tokens.produce_open,*body_ids,self.role_tokens.produce_close])
            found[marker]+=1;cursor=start+len(marker)
    if any(count!=1 for count in found.values()):
        raise RequestError('forced-message','native teacher text marker does not resolve to one exact token aligned occurrence')
    return plan

