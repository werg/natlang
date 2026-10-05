#!/usr/bin/env node
/** Local provider wire adapter. Preserve native runtime and save exact wire evidence. */
import {createServer} from 'node:http';import {mkdir,writeFile} from 'node:fs/promises';import {resolve,join} from 'node:path';import {pathToFileURL} from 'node:url';import {createHash} from 'node:crypto';
export function reasoningAliases(body){
 const result=structuredClone(body);
 for(const message of result.messages??[]){
  if(message.role!=='assistant')continue;
  const a=message.reasoning_content,b=message.reasoning;
  if(typeof a==='string'&&typeof b==='string'&&a!==b)throw Error('conflicting reasoning history aliases');
  const reasoning=typeof a==='string'?a:b;
  if(typeof reasoning==='string'){message.reasoning=reasoning;message.reasoning_content=reasoning;}
 }
 return result;
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href){
 const [port,upstream,output]=process.argv.slice(2);
 if(!/^http:\/\/127\.0\.0\.1:[0-9]+$/.test(upstream)||!/^\d+$/.test(port))throw Error('local endpoints required');
 await mkdir(output,{recursive:true});let ordinal=0;
 const digest=b=>createHash('sha256').update(b).digest('hex');
 const server=createServer(async(req,res)=>{
  const index=ordinal++,started=Date.now(),controller=new AbortController();let bytes=[],responseBytes=[],body;
  res.on('close',()=>{if(!res.writableEnded)controller.abort();});
  try{
   for await(const chunk of req)bytes.push(chunk);
   const raw=Buffer.concat(bytes);
   if(raw.length){body=JSON.parse(raw);if(['/tokenize','/v1/chat/completions'].includes(req.url))body=reasoningAliases(body);}
   const requestBytes=body?Buffer.from(JSON.stringify(body)):raw;
   await writeFile(join(output,`${index}-request.json`),JSON.stringify({method:req.method,path:req.url,body:body??null,wire_sha256:digest(requestBytes)})+'\n',{flag:'wx'});
   const response=await fetch(upstream+req.url,{method:req.method,signal:controller.signal,headers:{'content-type':'application/json'},...(requestBytes.length?{body:requestBytes}:{})});
   res.writeHead(response.status,{'content-type':response.headers.get('content-type')??'application/json'});
   for await(const chunk of response.body){responseBytes.push(Buffer.from(chunk));if(!res.write(chunk))await new Promise((ok,fail)=>{res.once('drain',ok);res.once('error',fail);});}
   res.end();const returned=Buffer.concat(responseBytes);
   await writeFile(join(output,`${index}-response.txt`),returned,{flag:'wx'});
   await writeFile(join(output,`${index}-receipt.json`),JSON.stringify({status:response.status,elapsed_ms:Date.now()-started,response_sha256:digest(returned),response_bytes:returned.length})+'\n',{flag:'wx'});
  }catch(error){if(!res.headersSent)res.writeHead(502,{'content-type':'application/json'});res.end(JSON.stringify({error:{message:String(error.message)}}));await writeFile(join(output,`${index}-error.json`),JSON.stringify({error:String(error.message),elapsed_ms:Date.now()-started})+'\n',{flag:'wx'});}
 });
 server.listen(Number(port),'127.0.0.1');
 for(const signal of ['SIGINT','SIGTERM'])process.once(signal,()=>server.close(()=>process.exit(0)));
}
