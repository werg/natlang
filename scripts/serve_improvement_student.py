#!/usr/bin/env python3
"""Serve a pinned CPU student and optional LoRA checkpoint without altering other model servers."""
import argparse,ast,json,re,time,uuid,contextlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
if __package__:
    from .render_training_corpus import _call_template, RENDERER_VERSION
else:
    from render_training_corpus import _call_template, RENDERER_VERSION

def literal(node):
    if isinstance(node,ast.Name) and node.id in ('true','false','null'):
        return {'true':True,'false':False,'null':None}[node.id]
    if isinstance(node,ast.List):return [literal(x) for x in node.elts]
    if isinstance(node,ast.Dict):return {literal(k):literal(v) for k,v in zip(node.keys,node.values)}
    return ast.literal_eval(node)

def tool_calls(text):
    match=re.search(r'<\|tool_call_start\|>(.*?)<\|tool_call_end\|>',text,re.S)
    if not match:return []
    nodes=ast.parse(match.group(1),mode='eval').body
    if not isinstance(nodes,ast.List):raise ValueError('tool calls must form a list')
    calls=[]
    for node in nodes.elts:
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.args or any(k.arg is None for k in node.keywords):raise ValueError('invalid tool-call syntax')
        calls.append({'id':'call_'+uuid.uuid4().hex,'type':'function','function':{'name':node.func.id,'arguments':json.dumps({k.arg:literal(k.value) for k in node.keywords})}})
    return calls

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='LiquidAI/LFM2.5-350M');parser.add_argument('--revision',required=True);parser.add_argument('--adapter');parser.add_argument('--checkpoint-map');parser.add_argument('--port',type=int,default=8082);parser.add_argument('--max-context',type=int,default=8192);parser.add_argument('--threads',type=int,default=2);args=parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,dtype=torch.float32,local_files_only=True)
    if args.adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,args.adapter)
    selected={'name':'starting' if not args.adapter else 'default'};checkpoints={}
    if args.checkpoint_map:
        checkpoints=json.load(open(args.checkpoint_map))
        from peft import PeftModel
        adapters=[(name,path) for name,path in checkpoints.items() if path]
        for index,(name,path) in enumerate(adapters):
            if index==0 and not args.adapter:model=PeftModel.from_pretrained(model,path,adapter_name=name)
            else:model.load_adapter(path,adapter_name=name)
        selected['name']='starting'
    model.eval()
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,data):
            payload=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
        def do_GET(self):self.send(200,{'data':[{'id':args.adapter or args.model,'object':'model'}]})
        def do_POST(self):
            try:
                request=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path=='/select':
                    name=request['checkpoint']
                    if name not in checkpoints:raise ValueError('checkpoint is not in the frozen experiment map')
                    if checkpoints[name]:model.set_adapter(name)
                    selected['name']=name;self.send(200,{'checkpoint':name});return
                messages=request['messages']
                prompt=_call_template(tokenizer,messages,request.get('tools') or [],True)
                inputs=tokenizer(prompt,return_tensors='pt');length=inputs.input_ids.shape[1]
                output_limit=min(request.get('max_tokens',request.get('max_completion_tokens',2048)),2048)
                if length+output_limit>args.max_context:raise ValueError('context allowance exceeded; input was not truncated')
                with torch.inference_mode(),(model.disable_adapter() if checkpoints and selected['name']=='starting' else contextlib.nullcontext()):output=model.generate(**inputs,max_new_tokens=output_limit,do_sample=False,pad_token_id=tokenizer.eos_token_id)
                tokens=output[0,length:];text=tokenizer.decode(tokens,skip_special_tokens=False)
                try:calls=tool_calls(text)
                except (ValueError,SyntaxError):calls=[]
                content=re.sub(r'<\|.*?\|>','',re.sub(r'<\|tool_call_start\|>.*?<\|tool_call_end\|>','',text,flags=re.S)).strip()
                self.send(200,{'id':'student_'+uuid.uuid4().hex,'object':'chat.completion','created':int(time.time()),'model':args.adapter or args.model,'choices':[{'index':0,'message':{'role':'assistant','content':content or None,**({'tool_calls':calls} if calls else {})},'finish_reason':'length' if len(tokens)>=output_limit else 'tool_calls' if calls else 'stop'}],'usage':{'prompt_tokens':length,'completion_tokens':len(tokens),'total_tokens':length+len(tokens)}})
            except Exception as error:self.send(400,{'error':{'message':str(error)}})
    print(json.dumps({'ready':True,'renderer':RENDERER_VERSION,'port':args.port,'model':args.model,'revision':args.revision,'adapter':args.adapter}),flush=True)
    HTTPServer(('0.0.0.0',args.port),Handler).serve_forever()
if __name__=='__main__':main()
