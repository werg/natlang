/** TEST ONLY local scripted executor; no model/provider or teacher registration. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {currentFrame,runInFrame} from '../dist/runtime/context.js';
import {itemValue} from '../dist/runtime/callable.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';
import {ONCE_EFFECTS} from '../dist/native/effects.js';
const header=(args,returns,body)=>`---\nargs: ${args}\nreturns: '${returns}'\n---\n${body}`;
function store(){const records=new Map(),blobs=new Map();return{records,blobs,settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:(r,values)=>{records.set(r.call_id,r);for(const[k,v]of values)blobs.set(k,v);},version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};}
const platformOf=mode=>mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
for(const mode of ['node','workerd'])for(const variant of ['callback','mixed-parallel','once','escaped','refinement'])
test(`${mode} ${variant} native function retains lexical authority and observes the live semantic caller`,async()=>{
 const platform=platformOf(mode),calls=store();let physical=0,forbidden=0,judges=0,saved;
 const own={...(variant==='once'?{[ONCE_EFFECTS]:['read']}:{}),async read(id){physical++;await Promise.resolve();return id==='bad'?-1:id.length;}};
 const runtime=core.createNatlangRuntime({platform,calls,services:{own,forbidden:{read(){forbidden++;return 99;}}},
  serviceDeclarations:{own:'export function read(id:string):Promise<number>;',forbidden:'export function read():number;'},
  ...(variant==='refinement'?{refinements:{services:{'own.read':'Is<number, "positive owned value">'},crisp:{'positive owned value':value=>{judges++;return value>0;}}}}:{}),
  agent:async session=>{
   const name=session.lam.functionName;
   const code=name==='root'?variant==='escaped'?'return await make();':'return await observe(readOwned);':name==='make'?'return nl`Use the captured native readOwned on id and return the owned result.`;':name==='consumer'?'return await fn("later");':name==='observe'?variant==='mixed-parallel'?'const values=await Promise.all([read("captured"),own.read("direct")]);await Promise.resolve();values.push(await read("after"));return values.reduce((a,b)=>a+b,0);':variant==='once'?'const a=await read("same");await Promise.resolve();return a+await read("same");':variant==='refinement'?'let rejected=false;try{await read("bad");}catch{rejected=true;}if(!rejected)throw Error("missing current result check");return await read("good");':'const a=await read("before");await Promise.resolve();return a+await read("after");':'return await readOwned(id);';
   const result=await session.applyAsync('eval',{code,finish:true});assert.equal(result.kind,'completed',result.text);
  }});
 const files={
  'root.nl':header('{}',variant==='escaped'?'(id:string) => number':'number','Call the owned helpers.'),
  'root/observe.nl':header('{read: "(id:string) => number"}','number','Inspect only through the passed read function.'),
  'root/readOwned.ts':'import {own} from "natlang:services";export default async function readOwned(id:string):Promise<number>{return await own.read(id);}',
  'root/make.nl':header('{}','(id:string) => number','Retain the exact native read callback.'),
  'root/make/readOwned.ts':'import {own} from "natlang:services";export default async function readOwned(id:string):Promise<number>{return await own.read(id);}',
  'consumer.nl':header('{fn:"(id:string) => number"}','number','Invoke the retained callback.'),
 };
 const root=core.loadVirtualNatlang(files,'root.nl','/test-only-native-callback',platform),consumer=core.loadVirtualNatlang(files,'consumer.nl','/test-only-native-callback',platform);
 try{
  await runtime.run(async()=>{
   // A captured helper sees this exact narrower authority; no ambient forbidden port.
   const frame=currentFrame();
   const value=await runInFrame({...frame,services:{own}},()=>root());
   if(variant==='escaped'){
    saved=value;const creator=[...calls.records.values()].find(r=>r.definition.name==='make');assert.equal(frame.task.isInvocationActive(creator.call_id),false);
    assert.equal(await saved('alone'),5); // canonical independent NL root, owner TASK remains live
    assert.equal(await consumer(saved),5);
    const rows=[...calls.records.values()],parent=rows.find(r=>r.definition.name==='consumer');
    const invoked=rows.filter(r=>r.definition.name!=='root'&&r.definition.name!=='make'&&r.definition.name!=='consumer');
    assert.equal(invoked.at(-2).parent_call_id,null);assert.equal(invoked.at(-1).parent_call_id,parent.call_id);
    assert.equal(invoked.at(-1).effects.length,1);assert.equal(invoked.at(-1).effects[0].method,'read');
    assert.equal(creator.effects.length,0);assert.equal(rows.find(r=>r.definition.name==='root').effects.length,0);
   }else{
    assert.equal(value,variant==='mixed-parallel'?19:variant==='once'?8:variant==='refinement'?4:11);
    const rows=[...calls.records.values()],outer=rows.find(r=>r.definition.name==='root'),observer=rows.find(r=>r.definition.name==='observe');
    assert.equal(observer.parent_call_id,outer.call_id);assert.equal(outer.effects.length,0);
    assert.equal(observer.effects.length,variant==='mixed-parallel'?3:2);
    assert.deepEqual(observer.effects.map(e=>JSON.parse(calls.blobs.get(e.args.hash))[0]),variant==='mixed-parallel'?['captured','direct','after']:variant==='once'?['same','same']:variant==='refinement'?['bad','good']:['before','after']);
    assert.equal(physical,variant==='once'?1:variant==='mixed-parallel'?3:2);
    if(variant==='refinement')assert.equal(judges,2);
   }
   assert.equal(forbidden,0);assert.ok([...calls.records.values()].every(r=>r.cost.model_requests===0));
  });
  if(saved)await assert.rejects(saved('closed'),/closed|cancelled/);
 }finally{runtime.close();}
});

for(const mode of ['node','workerd'])test(`${mode} returned native method routes same-owner cross-task effects without rebasing grants and rejects foreign recorders or expired owners`,async()=>{
 const platform=platformOf(mode),calls=store();let native=0,ambient=0,fn;
 const agent=async session=>{const code=session.lam.functionName==='creator'?'return own.read;':'return await fn("asked");';const result=await session.applyAsync('eval',{code,finish:true});assert.equal(result.kind,'completed',result.text);};
 const owner=core.createNatlangRuntime({platform,calls,services:{own:{async read(id){native++;await Promise.resolve();return 10+id.length;}}},serviceDeclarations:{own:'export function read(id:string):Promise<number>;'},agent});
 const caller=core.createNatlangRuntime({platform,calls,services:{own:{read(){ambient++;return 99;}}},agent});
 const other=core.createNatlangRuntime({platform,calls:store(),agent});
 const creator=core.loadVirtualNatlang({'creator.nl':header('{}','(id:string) => number','Return the owned native reader.')},'creator.nl','/native-method-value',platform);
 const consumer=core.loadVirtualNatlang({'consumer.nl':header('{fn:"(id:string) => number"}','number','Use the retained capability.')},'consumer.nl','/native-method-value',platform);
 try{
  await owner.run(async()=>{
   fn=await creator();const original=[...calls.records.values()].find(r=>r.definition.name==='creator');assert.equal(currentFrame().task.isInvocationActive(original.call_id),false);
   assert.equal(await caller.run(()=>consumer(fn)),15);
   const observed=[...calls.records.values()].find(r=>r.definition.name==='consumer');assert.equal(observed.effects.length,1);assert.equal(observed.effects[0].service,'own');assert.equal(original.effects.length,0);
   assert.equal(ambient,0);assert.equal(native,1);
   await assert.rejects(other.run(()=>consumer(fn)),/different recorder|belongs to|completed/);assert.equal(native,1);
   currentFrame().task.cancel(Error('owner cancelled'));
   await assert.rejects(caller.run(()=>consumer(fn)),/closed|cancelled|completed/);assert.equal(native,1);
  });
  await assert.rejects(caller.run(()=>consumer(fn)),/closed|cancelled|completed/);assert.equal(native,1);
 }finally{owner.close();caller.close();other.close();}
});

test('existing std span owns native helper effects without inventing helper invocation records',async()=>{
 const platform=platformOf('node'),calls=store();let physical=0;
 const runtime=core.createNatlangRuntime({platform,calls,services:{own:{read(id){physical++;return id.length;}}},agent:async session=>{const result=await session.applyAsync('eval',{code:'const values=await semanticMap(["one","four"],readOwned);return values.reduce((a,b)=>a+b,0);',finish:true});assert.equal(result.kind,'completed',result.text);}});
 const root=core.loadVirtualNatlang({'root.nl':header('{}','number','Read the owned values.'),'root/readOwned.ts':'import {own} from "natlang:services";export default function readOwned(id:string):number{return own.read(id);}'},'root.nl','/std-native-callback',platform);
 try{assert.equal(await runtime.run(()=>root()),7);const rows=[...calls.records.values()],span=rows.find(r=>r.definition.id==='std:semanticMap'),parent=rows.find(r=>r.definition.name==='root');assert(span);assert.equal(span.parent_call_id,parent.call_id);assert.equal(span.effects.length,2);assert.equal(parent.effects.length,0);assert.equal(physical,2);}finally{runtime.close();}
});
