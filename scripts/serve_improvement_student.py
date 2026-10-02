#!/usr/bin/env python3
"""Serve a pinned CPU student and optional LoRA checkpoint without altering other model servers."""
import argparse,json,time,uuid,contextlib
from http.server import BaseHTTPRequestHandler,HTTPServer
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
if __package__:
    from .render_training_corpus import _call_template, RENDERER_VERSION
    from .model_response import parse_response, tool_calls
    from .student_serving import assistant_end_token_id, strip_final_assistant_terminator, response_finish_reason, bounded_output_limit
else:
    from render_training_corpus import _call_template, RENDERER_VERSION
    from model_response import parse_response, tool_calls
    from student_serving import assistant_end_token_id, strip_final_assistant_terminator, response_finish_reason, bounded_output_limit

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='LiquidAI/LFM2.5-350M');parser.add_argument('--revision',required=True);parser.add_argument('--adapter');parser.add_argument('--checkpoint-map');parser.add_argument('--port',type=int,default=8082);parser.add_argument('--max-context',type=int,default=8192);parser.add_argument('--max-output-tokens',type=int,default=2048,help='hard server-side cap on generated tokens, regardless of request');parser.add_argument('--threads',type=int,default=2);parser.add_argument('--device',choices=['cpu','cuda'],default='cpu');parser.add_argument('--load-in-4bit',action='store_true');parser.add_argument('--cuda-memory-fraction',type=float,default=1.0);args=parser.parse_args()
    torch.set_num_threads(args.threads)
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,local_files_only=True)
    assistant_eos_id=assistant_end_token_id(tokenizer)
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else assistant_eos_id
    if args.load_in_4bit and args.device!='cuda':raise ValueError('4-bit loading requires CUDA')
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
    options={'dtype':torch.bfloat16 if args.device=='cuda' else torch.float32}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        options.update(device_map={'':0},quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16))
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,local_files_only=True,**options)
    if not args.load_in_4bit:model=model.to(args.device)
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
                inputs=tokenizer(prompt,return_tensors='pt').to(model.device);length=inputs.input_ids.shape[1]
                output_limit=bounded_output_limit(request,args.max_output_tokens)
                if length+output_limit>args.max_context:raise ValueError('context allowance exceeded; input was not truncated')
                with torch.inference_mode(),(model.disable_adapter() if checkpoints and selected['name']=='starting' else contextlib.nullcontext()):output=model.generate(**inputs,max_new_tokens=output_limit,do_sample=False,pad_token_id=pad_token_id,eos_token_id=assistant_eos_id)
                tokens=output[0,length:];completion_token_count=len(tokens)
                response_tokens,terminated=strip_final_assistant_terminator(tokens,assistant_eos_id)
                text=tokenizer.decode(response_tokens,skip_special_tokens=False)
                try:decoded=parse_response(text,request.get('tools') or [])
                except (ValueError,SyntaxError):decoded={'content':text,'tool_calls':[],'reasoning_content':None}
                calls=decoded['tool_calls'];content=decoded['content']
                self.send(200,{'id':'student_'+uuid.uuid4().hex,'object':'chat.completion','created':int(time.time()),'model':args.adapter or args.model,'choices':[{'index':0,'message':{'role':'assistant','content':content or None,**({'reasoning_content':decoded['reasoning_content']} if decoded['reasoning_content'] else {}),**({'tool_calls':calls} if calls else {})},'finish_reason':response_finish_reason(terminated=terminated,token_count=completion_token_count,output_limit=output_limit,has_tool_calls=bool(calls))}],'usage':{'prompt_tokens':length,'completion_tokens':completion_token_count,'total_tokens':length+completion_token_count}})
            except Exception as error:self.send(400,{'error':{'message':str(error)}})
    print(json.dumps({'ready':True,'renderer':RENDERER_VERSION,'port':args.port,'model':args.model,'revision':args.revision,'adapter':args.adapter,'max_output_tokens':args.max_output_tokens}),flush=True)
    HTTPServer(('0.0.0.0',args.port),Handler).serve_forever()
if __name__=='__main__':main()
