import assert from'node:assert/strict';import test from'node:test';
import{createNatlangRuntime,loadVirtualNatlang}from'../dist/index.js';
test('an impossible finite return fails before calling either generation or scoring',async()=>{
 let requests=0;
 const model=Object.assign(async()=>{requests++;throw Error('no generation should start')},{decide:async()=>{requests++;throw Error('no scoring should start')}});
 const runtime=createNatlangRuntime({calls:false,model:{driver:model,decisionReadout:"finite-returns"}});
 try{
  const fn=loadVirtualNatlang({'empty.nl':'---\nargs: {}\nreturns: ("matched" & "absent")\n---\nReturn the declared result.'},'empty.nl');
  await assert.rejects(runtime.run(()=>fn()),/decision-empty-domain/);
  assert.equal(requests,0);
 }finally{runtime.close()}
});
