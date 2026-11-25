import assert from'node:assert/strict';import test from'node:test';
import * as core from'../dist/index.js';import{workerdPlatform}from'../dist/core/workerd.js';
const aliases=`export type Review={journal:"matched"|"absent"|"mismatch"|"incomplete";effect:"confirmed"|"missing"|"different"|"unchecked";readback:{digest:string;observedDigest:string|null;size:number|null}|null};
export type Confirmed={review:Review & {journal:"matched";effect:"confirmed";readback:{digest:string;observedDigest:string;size:number}}};`;
for(const mode of ['node','workerd'])test(`${mode} a confirmed record reaches its model boundary; unconfirmed alternatives never do`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args)}}});
 let entered=0;
 const rt=core.createNatlangRuntime({platform,calls:false,agent:async session=>{
  entered++;
  assert.equal(session.lam.args.result.review.journal,'matched');
  const result=await session.applyAsync('eval',{code:'return result.review.readback.size > 0;',finish:true});
  assert.equal(result.kind,'completed',result.text);
 }});
 try{
  const project=core.compileVirtualProject({root:'/confirmed',files:{
   'types.ts':aliases,
   'entry.ts':'import shouldStage from "./shouldStage.nl"; import type {Confirmed} from "./types.js"; export default async function run(result:Confirmed):Promise<boolean>{return await shouldStage(result);}',
   'shouldStage.nl':'---\nargs:\n  result: Confirmed\nreturns: boolean\n---\nReturn whether the confirmed readback contains data.',
  }},core,{platform});
  assert.equal(project.ok,true,JSON.stringify(project.diagnostics));
  const value={review:{journal:'matched',effect:'confirmed',readback:{digest:'original',observedDigest:'original',size:115}}};
  const run=project.require('entry.ts').default;
  assert.equal(await rt.run(()=>run(value)),true);assert.equal(entered,1);
  for(const change of [r=>r.review.journal='incomplete',r=>r.review.effect='missing',r=>r.review.readback=null,r=>r.review.readback.size=null,r=>delete r.review.readback]){
   const bad=structuredClone(value);change(bad);await assert.rejects(rt.run(()=>run(bad)));
  }
  assert.equal(entered,1,'invalid evidence rejected before any agent is entered');
 }finally{rt.close()}
});
