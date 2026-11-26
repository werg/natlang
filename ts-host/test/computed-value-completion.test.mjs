import assert from 'node:assert/strict';
import test from 'node:test';
import {execFileSync} from 'node:child_process';
import * as core from '../dist/index.js';
import {TOOLS_PROMPT,SEMANTIC_RESULT_PROMPT} from '../dist/native/prompt.js';
import {workerdPlatform} from '../dist/core/workerd.js';

const receipt={digest:'8b4126aba06879b7251820eea1139722d31173a8336cfe6f54c575ed9f752f2a',allowed:false,count:0,note:''};
for(const mode of ['node','workerd']) test(`${mode} computed completion passes actual native values through the available typed helper`,async()=>{
 let reads=0;const requests=[];
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const declarations={ports:'export type Receipt={digest:string;allowed:boolean;count:number;note:string}; export function current():Promise<Receipt>;'};
 const runtime=core.createNatlangRuntime({platform,calls:false,services:{ports:{async current(){reads++;return receipt;}}},serviceDeclarations:declarations,serviceScopes:{ports:['root/readReceipt.ts']},model:async request=>{requests.push(request);return {calls:[['eval',{code:'return await readReceipt();',finish:true}]]};}});
 try{
 const compiled=core.compileVirtualProject({root:'/computed-completion',files:{
 'main.ts':'import root from "./root.nl"; export default async function main(){return root();}',
 'root.nl':'---\nargs: {}\nreturns: {digest:string;allowed:boolean;count:number;note:string}\n---\nReturn the current native receipt unchanged.\n',
 'root/readReceipt.ts':'import {ports} from "natlang:services"; export default async function readReceipt():Promise<{digest:string;allowed:boolean;count:number;note:string}>{return ports.current();}',
 }},core,{platform,programId:'computed-completion',declarations});
 assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
 assert.deepEqual(await runtime.run(()=>compiled.require('main.ts').default()),receipt);
 assert.equal(reads,1);assert.match(requests[0].messages[0].content,/Both are completion tools/);
 }finally{runtime.close();}
});
test('direct semantic boolean remains a valid completion',async()=>{
 const fn=core.defineNatlang('---\nargs: {}\nreturns: boolean\n---\nReturn false.\n');
 const runtime=core.createNatlangRuntime({model:async()=>({calls:[['return_result',{status:'success',value:false}]]})});
 try{assert.equal(await runtime.run(()=>fn()),false);}finally{runtime.close();}
});
test('full prompt explains both completions without changing semantic-only or Neuralese boundaries',()=>{
 assert.match(TOOLS_PROMPT,/^Complete this function/);assert.match(TOOLS_PROMPT,/direct semantic answer/);
 assert.match(SEMANTIC_RESULT_PROMPT,/^Answer with the return_result tool/);
 assert.equal(TOOLS_PROMPT.split('\n\n')[1],SEMANTIC_RESULT_PROMPT.split('\n\n')[1]);
});
test('existing model-facing opt-out and legacy revision preserve their completion lead',()=>{
 for(const env of [{NATLANG_MODEL_FACING:'none'},{NATLANG_PROMPT_REVISION:'2026-11-22'}]){
 const value=execFileSync(process.execPath,['--input-type=module','-e',"import{TOOLS_PROMPT}from'./dist/native/prompt.js';process.stdout.write(TOOLS_PROMPT)"],{cwd:new URL('..',import.meta.url),env:{...process.env,...env},encoding:'utf8'});
 assert.doesNotMatch(value,/^Complete this function/);
 }
});
