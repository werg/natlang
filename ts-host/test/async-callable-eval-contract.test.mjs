import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {checkEvalFields,scopeDeclarations} from '../dist/compiler/eval-check.js';
import {parseType,formatType} from '../dist/native/types.js';

const scope={types:{Snapshot:'{id:string;value:number}',Reader:'(id:string)=>Snapshot|null'},inputs:[{name:'readSlot',type:'Reader'}],locals:[],captures:[],imports:[]};
test('semantic callable identity remains normalized, while actual eval declarations are async',()=>{
 assert.deepEqual(parseType('(id:string)=>Promise<Snapshot|null>'),parseType('(id:string)=>Snapshot|null'));
 assert.equal(formatType(parseType('(id:string)=>Promise<Snapshot|null>')),'(id: string) => Snapshot | null');
 assert.match(scopeDeclarations(scope),/type Reader = \(id: string\) => Promise<Snapshot \| null>/);
 for(const code of ['const snapshot=readSlot("owned");return snapshot.id;','return readSlot("owned")["id"];','const values=Promise.resolve({id:"actual"});return values.id;'])
  assert.ok(checkEvalFields(code,scope).some(d=>d.code==='undeclared-field'&&/unresolved Promise/.test(d.message)),code);
 for(const code of ['const snapshot=await readSlot("owned");return snapshot?.id;','return readSlot("owned").then(row=>row?.id);','const f=(id:string)=>({id});return f("sync").id;'])
  assert.deepEqual(checkEvalFields(code,scope),[],code);
 assert.deepEqual(checkEvalFields('const snapshot=readSlot("owned");return snapshot.id;',{...scope,checkAwaitedFields:false}),[]);
});
for(const mode of ['node','workerd'])for(const checkAwaitedFields of [true,false])test(`${mode} async typed callback field guard=${checkAwaitedFields} precedes effects and awaited values remain exact`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 let reads=0,seen=0;
 const runtime=core.createNatlangRuntime({platform,calls:false,checkAwaitedFields,agent:async session=>{
  seen++;
  const bad=await session.applyAsync('eval',{code:'const snapshot=readSlot("owned");const value=snapshot.id;',finish:false});
  assert.equal(bad.kind,checkAwaitedFields?'rejected':'ok',bad.text);
  assert.equal(reads,checkAwaitedFields?0:1,'the missing-await snippet must be refused before its callback effect');
  if(checkAwaitedFields)assert.match(bad.text,/unresolved Promise; await it before reading id/);
  const good=await session.applyAsync('eval',{code:'const observed=await readSlot("owned");const sync=(id:string)=>({id});return observed.id+":"+String(observed.value)+":"+sync("local").id;',finish:true});
  assert.equal(good.kind,'completed',good.text);
 }});
 try{
  const project=core.compileVirtualProject({root:'/async-callback',files:{
   'entry.ts':'import observe from "./observe.nl";export default async function entry(readSlot:(id:string)=>Promise<{id:string;value:number}|null>):Promise<string>{return observe(readSlot);}',
   'observe.nl':'---\nargs:\n  readSlot: "(id:string)=>Promise<{id:string;value:number}|null>"\nreturns: string\n---\nRead the actual owned callback value and return its fields.'
  }},core,{platform});assert.equal(project.ok,true,JSON.stringify(project.diagnostics));
  const value=await runtime.run(()=>project.require('entry.ts').default(async id=>{reads++;assert.equal(id,'owned');return{id:'actual-native',value:7};}));
  assert.equal(value,'actual-native:7:local');assert.equal(reads,checkAwaitedFields?1:2);assert.equal(seen,1);
 }finally{runtime.close();}
});
for(const mode of ['node','workerd'])test(`${mode} model-visible typed callback declarations preserve asynchronous result`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});let opening;
 const runtime=core.createNatlangRuntime({platform,calls:false,model:async request=>{opening=request;return{calls:[['return_result',{status:'success',value:'done'}]]};}});
 try{const project=core.compileVirtualProject({root:'/async-opening',files:{'entry.ts':'import observe from "./observe.nl";export default async function entry(readSlot:(id:string)=>Promise<{id:string}|null>):Promise<string>{return observe(readSlot);}','observe.nl':'---\nargs:\n  readSlot: "(id:string)=>Promise<{id:string}|null>"\nreturns: string\n---\nReturn done.'}},core,{platform});assert.equal(project.ok,true,JSON.stringify(project.diagnostics));assert.equal(await runtime.run(()=>project.require('entry.ts').default(async id=>({id}))),'done');assert.match(JSON.stringify(opening),/readSlot: \(id: string\) => Promise</);}finally{runtime.close();}
});
for(const mode of ['node','workerd'])for(const asynchronous of [false,true])test(`${mode} lexical ${asynchronous?'async':'synchronous'} JavaScript captures preserve the actual callable contract`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});let seen=0;
 const runtime=core.createNatlangRuntime({platform,calls:false,agent:async session=>{
  seen++;
  if(asynchronous){const bad=await session.applyAsync('eval',{code:'return callback("captured").id;',finish:false});assert.equal(bad.kind,'rejected',bad.text);assert.match(bad.text,/unresolved Promise/);}
  const result=await session.applyAsync('eval',{code:asynchronous?'return (await callback("captured")).id;':'return callback("captured").id;',finish:true});assert.equal(result.kind,'completed',result.text);
 }});
 try{const project=core.compileVirtualProject({root:'/lexical-callback',files:{'entry.ts':`import {nl} from "@natlang/node";export default async function entry():Promise<string>{const callback=${asynchronous?'async ':''}(id:string)${asynchronous?':Promise<{id:string}>':':{id:string}'}=>({id});return await nl.with<string>({callback})\`Use the actual supplied callback.\`();}`}},core,{platform});assert.equal(project.ok,true,JSON.stringify(project.diagnostics));assert.equal(await runtime.run(()=>project.require('entry.ts').default()),'captured');assert.equal(seen,1);}finally{runtime.close();}
});
for(const mode of ['node','workerd'])test(`${mode} imported TS helper retains its synchronous callback alias`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const runtime=core.createNatlangRuntime({platform,calls:false,agent:async session=>{const result=await session.applyAsync('eval',{code:'const callback=makeReader();return callback("native-sync").id;',finish:true});assert.equal(result.kind,'completed',result.text);}});
 try{const project=core.compileVirtualProject({root:'/imported-sync-callback',files:{'entry.ts':'import owner from "./owner.nl";export default async function entry():Promise<string>{return owner();}','owner.nl':'---\nargs: {}\nreturns: string\n---\nUse the exact synchronous TS callback.','owner/makeReader.ts':'export type Reader=(id:string)=>{id:string};export default function makeReader():Reader{return id=>({id});}'}},core,{platform});assert.equal(project.ok,true,JSON.stringify(project.diagnostics));assert.equal(await runtime.run(()=>project.require('entry.ts').default()),'native-sync');}finally{runtime.close();}
});
test('ordinary TS import aliases keep their native contract in eval ambient declarations',()=>{
 const text=scopeDeclarations({...scope,inputs:[],typeScriptAliases:['Reader'],imports:[{name:'makeReader',params:[],returns:'Reader',async:false,kind:'TypeScript',children:[]}]});assert.match(text,/type Reader = \(id:string\)=>Snapshot\|null;/);assert.doesNotMatch(text,/type Reader[^\n]*Promise/);
});
