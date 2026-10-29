"""Authenticate assistant-content XML actions without inventing provider tool calls."""
from __future__ import annotations
import copy, hashlib, json, math, re
from ..serve.chat import parse_xml_calls

POLICY_SHA256 = '8e7ef9bf5fe166b706b301170e86e72a51917016426ff54f820d6acd697ab058'
_ANNOTATIONS={'description','title','default','examples','$schema','deprecated','readOnly','writeOnly'}
_SUPPORTED={'type','properties','required','additionalProperties','items','enum','const','anyOf','oneOf','allOf','minimum','maximum','exclusiveMinimum','exclusiveMaximum','minLength','maxLength','pattern','minItems','maxItems','uniqueItems'}|_ANNOTATIONS

def _matches(kind,value):
    return {'null':value is None,'object':isinstance(value,dict),'array':isinstance(value,list),
            'string':isinstance(value,str),'boolean':type(value) is bool,
            'integer':type(value) in (int,float) and math.isfinite(value) and value==int(value) and abs(value)<=9007199254740991,
            'number':type(value) in (int,float) and math.isfinite(value)}.get(kind,False)

def _valid(schema,value):
    if not isinstance(schema,dict) or any(k not in _SUPPORTED and not k.startswith('x-') for k in schema):return False
    kind=schema.get('type');kinds=kind if isinstance(kind,list) else [kind]
    if kind is not None and not any(_matches(k,value) for k in kinds):return False
    if 'enum' in schema and not any((type(value)==type(x) or type(value) in (int,float) and type(x) in (int,float)) and value==x for x in schema['enum']):return False
    if 'const' in schema and not ((type(value)==type(schema['const']) or type(value) in (int,float) and type(schema['const']) in (int,float)) and value==schema['const']):return False
    if 'anyOf' in schema and not any(_valid(s,value) for s in schema['anyOf']):return False
    if 'oneOf' in schema and sum(_valid(s,value) for s in schema['oneOf'])!=1:return False
    if 'allOf' in schema and not all(_valid(s,value) for s in schema['allOf']):return False
    if type(value) in (int,float):
        for key,cmp in (('minimum',lambda a,b:a>=b),('maximum',lambda a,b:a<=b),('exclusiveMinimum',lambda a,b:a>b),('exclusiveMaximum',lambda a,b:a<b)):
            if key in schema and not cmp(value,schema[key]):return False
    if isinstance(value,str):
        if len(value)<schema.get('minLength',0) or len(value)>schema.get('maxLength',float('inf')):return False
        if 'pattern' in schema and not re.search(schema['pattern'],value):return False
    if isinstance(value,list):
        if len(value)<schema.get('minItems',0) or len(value)>schema.get('maxItems',float('inf')):return False
        if schema.get('uniqueItems') and len({json.dumps(x,sort_keys=True,ensure_ascii=False) for x in value})!=len(value):return False
        if 'items' in schema and not all(_valid(schema['items'],x) for x in value):return False
    if isinstance(value,dict):
        props=schema.get('properties') or {}
        if any(k not in value for k in schema.get('required',[])):return False
        for key,item in value.items():
            sub=props.get(key,schema.get('additionalProperties',True))
            if sub is False or (isinstance(sub,dict) and not _valid(sub,item)):return False
    return True

def authenticate(turn):
    """Return exact normalized actions; all custody violations raise ValueError."""
    try:
        response=turn['model_response'];proof=response['tool_parse'];request=turn['model_request'];raw=response['raw_response']
        if (proof['schema']!='natlang.native-xml-tool-parse/1' or proof['source']!='assistant-content' or
            proof['schema_custody']!='response_tools' or proof['policy_sha256']!=POLICY_SHA256):raise ValueError('native XML parser policy differs')
        reqtext=turn['model_request_canonical'];rawtext=response['raw_response_canonical']
        if hashlib.sha256(reqtext.encode()).hexdigest()!=turn['request_sha256'] or hashlib.sha256(rawtext.encode()).hexdigest()!=turn['raw_response_sha256']:raise ValueError('native XML request or response digest differs')
        without={k:v for k,v in request.items() if k!='invocation_id'}
        if (json.loads(reqtext)!=without or json.loads(rawtext)!=raw or request.get('invocation_id')!=turn.get('invocation_id') or
            request['messages']!=turn['context'] or request['tools']!=turn['tools_offered'] or request['tools'] or
            request['response_tools']!=turn['response_tools'] or response['raw_calls']):raise ValueError('native XML request or provider-call custody differs')
        aliases=proof['wire_tool_aliases'];tools=copy.deepcopy(request['response_tools']);reverse={}
        if not isinstance(aliases,dict):raise ValueError('invalid aliases')
        for tool in tools:
            fn=tool['function'];name=fn['name'];wire=aliases.get(name,name)
            if not isinstance(wire,str) or wire in reverse:raise ValueError('ambiguous tool alias')
            reverse[wire]=name;fn['name']=wire
        choice=raw['choices'][0];message=choice['message']
        if choice.get('finish_reason')=='length' or not isinstance(message.get('content'),str) or message.get('tool_calls'):raise ValueError('native XML response incomplete')
        parsed=parse_xml_calls(message['content'],tools);schema_by={x['function']['name']:x['function']['parameters'] for x in tools}
        if not parsed or any(name not in schema_by or not _valid(schema_by[name],args) for name,args in parsed):raise ValueError('native XML action violates allowed schema')
        calls=[[reverse[name],args] for name,args in parsed];assistant=turn['assistant']
        if response.get('text')!='' or assistant.get('content')!='' or response['calls']!=calls or len(assistant['calls'])!=len(calls):raise ValueError('native XML recorded response differs')
        for (name,args),entry in zip(calls,assistant['calls']):
            if entry.get('tool')!=name or entry.get('source_tool')!=name or entry.get('arguments')!=args:raise ValueError('native XML recorded action differs')
        reason=message.get('reasoning_content',message.get('reasoning'))
        if (reason if isinstance(reason,str) and reason.strip() else None)!=(assistant.get('reasoning') or None):raise ValueError('native XML reasoning differs')
        return calls
    except (KeyError,TypeError,IndexError,AttributeError,json.JSONDecodeError) as error:raise ValueError('invalid native XML custody') from error

def derived_tool_calls(turn):
    """A rendering projection, explicitly distinct from the provider's raw_calls."""
    return [{'type':'function','function':{'name':name,'arguments':json.dumps(args,ensure_ascii=False,separators=(',',':'))}} for name,args in authenticate(turn)]


def render_packet(tokenizer, turn):
    """Render the actual captured prompt and verify the exact sampled XML suffix."""
    from .text_corpus import native_gold_packet
    calls=derived_tool_calls(turn);assistant=turn['assistant']
    target={'role':'assistant','content':'','tool_calls':calls}
    if assistant.get('reasoning'):target['reasoning_content']=assistant['reasoning']
    packet=native_gold_packet(tokenizer,turn['context'],target,turn['tools_offered'],
                              prompt_rendering='full',response_tools=turn['response_tools'])
    raw=turn['model_response']['raw_response']['choices'][0]['message']['content']
    end=tokenizer.decode([tokenizer.convert_tokens_to_ids('<|im_end|>')])
    if not packet[0].endswith(raw+end+'\n'):
        raise ValueError('native XML target rendering changes actual sampled assistant content')
    return packet


def validate_replay_turn(turn):
    """Bind a replay/rollout target to its original parsed source without changing prompt tools."""
    source=turn['native_xml_turn'];calls=authenticate(source);target=turn.get('target',turn.get('reply'))
    if (turn['messages']!=source['context'] or (turn.get('tools') or [])!=source['tools_offered'] or
            turn.get('response_tools')!=source['response_tools'] or not isinstance(target,dict)):
        raise ValueError('native XML replay prompt or schema custody differs')
    if (target.get('reasoning_content') or None)!=(source['assistant'].get('reasoning') or None):
        raise ValueError('native XML replay reasoning differs')
    if target.get('tool_calls') is not None:
        actual=[]
        for item in target['tool_calls']:
            fn=item['function'];args=fn['arguments'];actual.append([fn['name'],json.loads(args) if isinstance(args,str) else args])
        if actual!=calls or (target.get('content') or ''):raise ValueError('native XML replay projected calls differ')
    elif target.get('content')!=source['model_response']['raw_response']['choices'][0]['message']['content']:
        raise ValueError('native XML replay raw content differs')
    return calls


def verify_projection(turn, messages, tools, target):
    """Require a materialized target and prompt to be the exact authenticated action projection."""
    calls=authenticate(turn)
    if messages!=turn['context'] or (tools or [])!=turn['tools_offered'] or not isinstance(target,dict) or (target.get('content') or ''):
        raise ValueError('native XML materialized prompt differs from exact sampled request')
    projected=[]
    for item in target.get('tool_calls') or []:
        fn=item['function'];value=fn['arguments'];projected.append([fn['name'],json.loads(value) if isinstance(value,str) else value])
    if projected!=calls:raise ValueError('native XML materialized target differs from exact sampled action')
    return calls
