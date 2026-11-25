import assert from 'node:assert/strict';
import test from 'node:test';
import {mkdtemp,readFile,readdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createPiModelBackend} from '../dist/model/pi-provider.js';
import {collectBatch,nativeJobRunner} from '../dist/teacher/collector.js';
import {providerFailureEvidence} from '../dist/model/operational-error.js';

test('standalone Pi progress preserves actual thrown causes and original error identity without payloads',async()=>{
 const failure=new TypeError('fetch failed',{cause:Object.assign(new Error('secret request'),{code:'EAI_AGAIN',syscall:'getaddrinfo'})});
 const progress=[];const models={getProvider:()=>({}),getModel:()=>({provider:'mock',id:'fixture',api:'openai-completions'}),stream:()=>({result:()=>Promise.reject(failure),async *[Symbol.asyncIterator](){yield {type:'text_delta',delta:'observed'};throw failure;}})};
 const backend=createPiModelBackend('mock','fixture',{},undefined,undefined,undefined,undefined,'native',undefined,models);
 try{await assert.rejects(backend.turn({messages:[{role:'user',content:'fixture'}],tools:[],seed:null,max_tokens:20},undefined,{onProgress:p=>progress.push(p)}),error=>error===failure);
 const end=progress.at(-1);assert.equal(end.status,'failed');assert.equal(end.deltaBytes,8);assert.deepEqual(end.error,{name:'TypeError',cause:{name:'Error',code:'EAI_AGAIN',syscall:'getaddrinfo'}});assert.ok(!JSON.stringify(end).includes('secret'));}finally{backend.close();}
});

test('actual collector journals original direct-HTTP operational causes before reporting failure',async()=>{
 const dir=await mkdtemp(join(tmpdir(),'provider-cause-'));const prior=globalThis.fetch;
 const failure=new TypeError('fetch failed',{cause:Object.assign(new Error('private URL/body'),{code:'ENETUNREACH',syscall:'connect',address:'192.0.2.3',port:443})});let sends=0;
 globalThis.fetch=async()=>{sends++;throw failure;};
 try{
 const item={index:0,record:{version:'natlang.program/2',id:'provider-error-evidence',kind:'lambda_source',source:'fixture',split:'test',source_ids:['provider-error-evidence'],source_groups:['provider-error-evidence'],license:'test',semantics:{root:'one.nl',files:{'one.nl':'---\nargs: {}\nreturns: number\n---\nReturn one.\n'},inputs:{},expected:1,operation:'exact'}}};
 const config={jobs:join(dir,'jobs'),output:join(dir,'out.jsonl'),workers:1,endpoint:'http://127.0.0.1:9',modelId:'fixture',rootSeed:7,systemPrompt:'fixture',contextTokens:16384,toolSurfaceSha256:'fixture',transportRetries:0,requestRetries:0,execution:{identity:'provider-error-fixture/1',run:async(_record,driver)=>{await driver({messages:[{role:'user',content:'fixture'}],tools:[],invocation_id:'actual-root'});return {trace:[]};}}};
 const result=await collectBatch([item],config,nativeJobRunner(config));assert.equal(result.completed,0);assert.equal(sends,1);
 const terminal=JSON.parse(await readFile(join(config.jobs,'000000.error.json'),'utf8'));assert.equal(terminal.diagnostics.cause.cause.code,'ENETUNREACH');
 const names=await readdir(config.jobs);const file=names.find(name=>name.includes('.evidence-')&&name.endsWith('.jsonl'));assert.ok(file,'original journal exists');
 const rows=(await readFile(join(config.jobs,file),'utf8')).trim().split('\n').map(JSON.parse);const exchange=rows.find(row=>row.payload.kind==='provider_exchange').payload;
 assert.equal(exchange.request.invocation_id,'actual-root');assert.equal(exchange.request_ordinal,1);assert.deepEqual(exchange.error.diagnostics,{name:'Error',cause:{name:'TypeError',cause:{name:'Error',code:'ENETUNREACH',syscall:'connect',address:'192.0.2.3',port:443}}});assert.equal(exchange.response,undefined);assert.ok(!JSON.stringify(exchange.error.diagnostics).includes('private URL/body'));
 }finally{globalThis.fetch=prior;await rm(dir,{recursive:true,force:true});}
});

test('private failure capture observes no accessor or conversion hooks and preserves falsy failures',()=>{
 let reads=0;const error={};for(const key of ['name','message','code','status','cause','errors','providerCode','providerRetryable','retry_after_ms'])Object.defineProperty(error,key,{get(){reads++;throw Error('observer hook');}});
 Object.defineProperty(error,'toString',{value(){reads++;throw Error('no conversion');}});
 assert.deepEqual(providerFailureEvidence(error),{name:'Error',message:'Provider request failed'});assert.equal(reads,0);assert.equal(providerFailureEvidence(undefined).name,'Error');
});
