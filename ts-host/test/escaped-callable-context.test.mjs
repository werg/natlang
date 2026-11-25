import assert from 'node:assert/strict';
import test from 'node:test';
import {createHash} from 'node:crypto';
import {snapshot} from '../dist/calls/snapshot.js';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {currentFrame,runInFrame} from '../dist/runtime/context.js';
import {resolveBoundFrame} from '../dist/runtime/runtime.js';
import {loadVirtualNatlang} from '../dist/runtime/virtual-project.js';
import {namedCallable,recordOf,makeCallable} from '../dist/runtime/callable.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';
function store(){const records=new Map();return{records,settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:r=>records.set(r.call_id,r),version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};}
for(const mode of ['node','workerd'])test(`${mode} escaped functions retain lexical grants with live caller ancestry`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const calls=store();let bound,callback,owner,inline;const captures={token:'lexical'};let child;
 const lexical={read:n=>n+7};
 const runtime=core.createNatlangRuntime({platform,calls,services:{bridge:{save(){
  owner={...currentFrame(),services:{lexical}};
  bound=namedCallable('child',recordOf(child),owner,platform);
  callback=runInFrame(owner,()=>runtime.bind(()=>bound(3)));
  inline=runInFrame(owner,()=>makeCallable({kind:'inline',created:currentFrame(),definition:{},captures,invoke:(args,frame)=>{assert.equal(frame.services.lexical,lexical);return Promise.resolve(captures.token);}}));
  return 1;
 },use(){return bound(3);}},lexical},serviceDeclarations:{bridge:'export function save():number;export function use():Promise<number>;',lexical:'export function read(value:number):number;'},agent:async session=>{
  const name=session.lam.functionName;
  if(name==='child'){assert.deepEqual(Object.keys(session.availableServices()),['lexical']);assert.deepEqual(Object.keys(session.availableServiceDeclarations()),['lexical']);assert(!session.searchCode('bridge').some(row=>row.name==='bridge'));}
  const code=name==='creator'?'return bridge.save();':name==='consumer'?'return await bridge.use();':'return lexical.read(value);';
  const result=await session.applyAsync('eval',{code,finish:true});assert.equal(result.kind,'completed',result.text);
 }});
 const load=(name,args)=>loadVirtualNatlang({...(name==='child'?{'child/constants.ts':'export const answer:number=17;'}:{}),[name+'.nl']:`---\nargs: ${args}\nreturns: number\n---\nUse the assigned service.`},name+'.nl','/escaped',platform);
 child=load('child','{value: number}');const creator=load('creator','{}'),caller=load('consumer','{}');
 try{
  await runtime.run(async()=>{
   await creator();assert.equal(owner.task.isInvocationActive(owner.parentCallId),false);
   assert.equal(await bound(3),10);assert.equal(await callback(),10);assert.equal(await bound.partial(3)(),10);assert.equal(await inline(),'lexical');assert.equal(await bound.constants.answer,17);
   assert.equal(await caller(),10);
   const rows=[...calls.records.values()],parent=rows.find(r=>r.definition.name==='consumer');
   const children=rows.filter(r=>r.definition.name==='child');
   assert.deepEqual(children.slice(0,2).map(r=>r.parent_call_id),[null,null]);
   assert.equal(children.at(-1).parent_call_id,parent.call_id);
   assert.deepEqual(parent.graph_custody.child_call_ids,[children.at(-1).call_id]);
   const scalar=children.at(-1);const hash=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
   assert.deepEqual(scalar.inputs.value,{complete:true,hash:hash(3),bytes:1});
   assert.deepEqual(scalar.output,{complete:true,hash:hash(10),bytes:2});
   assert(scalar.effects.some(effect=>effect.service==='lexical'&&effect.method==='read'));
   assert.equal(scalar.cost.model_requests,0);
   const opaque=snapshot(bound,4096);assert.equal(opaque.ref.complete,false);assert.equal(opaque.ref.reason,'nonportable');

   const foreign=core.createNatlangRuntime({platform,calls});
   try{await foreign.run(async()=>{
    const frame=currentFrame();frame.task.beginInvocation('other-live',true);
    try{await runInFrame({...frame,parentCallId:'other-live'},async()=>{
     assert.equal(await bound(3),10);
     const last=[...calls.records.values()].at(-1);assert.equal(last.parent_call_id,'other-live');assert.equal(last.task_id,owner.task.id);
     assert.deepEqual(frame.task.graphChildren('other-live'),[last.call_id]);
     assert.equal(frame.task.reserveExecutionSiblingOrdinal('other-live','shared'),0);
     assert.equal(owner.task.reserveExecutionSiblingOrdinal('other-live','shared'),1);
    });}finally{frame.task.endInvocation('other-live');}
    await Promise.all(['parallel-a','parallel-b'].map(async id=>{
     frame.task.beginInvocation(id,true);
     try{await runInFrame({...frame,parentCallId:id},async()=>{
      assert.equal(await bound(3),10);
      const ids=frame.task.graphChildren(id);assert.equal(ids.length,1);
      assert.equal(calls.records.get(ids[0]).parent_call_id,id);
     });}finally{frame.task.endInvocation(id);}
    }));
   });}finally{foreign.close();}
   const other=core.createNatlangRuntime({platform,calls:store()});
   try{await other.run(async()=>{const frame=currentFrame();frame.task.beginInvocation('wrong-owner',true);try{await runInFrame({...frame,parentCallId:'wrong-owner'},async()=>{assert.equal(await bound(3),10);assert.equal([...calls.records.values()].at(-1).parent_call_id,null);});}finally{frame.task.endInvocation('wrong-owner');}});}finally{other.close();}
   const resolved=runInFrame(owner,()=>resolveBoundFrame(owner));assert.equal(resolved.parentCallId,undefined);assert.equal(resolved.services,owner.services);
   const callerSignal=new AbortController();
   const running=runInFrame({...owner,parentCallId:undefined,signal:callerSignal.signal},()=>resolveBoundFrame(owner));
   owner.task.cancel(Error('owner cancelled'));assert.equal(owner.task.isClosed,false);assert.equal(running.signal.aborted,true);await assert.rejects(bound(3),/cancelled/);
  });
  await assert.rejects(bound(3),/task.*closed|cancelled/);assert.throws(()=>bound.constants.answer,/task.*closed|cancelled/);
 }finally{runtime.close();}
});
