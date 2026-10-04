import test from 'node:test';import assert from 'node:assert/strict';
import {mhDecision,randomStream,assertReplayContext} from '../scripts/projection-search.mjs';

test('MH uses summed target and forward/reverse proposal probabilities',()=>{
 const a=[{target:-8,proposal:-2}],b=[{target:-6,proposal:-1}];
 assert.equal(mhDecision(a,b,0,.9).accepted,true);
 assert.equal(mhDecision(b,a,0,.9).log_acceptance,-1);
 assert.equal(mhDecision(b,a,0,.9).accepted,false);
 assert.equal(mhDecision(b,a,0,.1).accepted,true);
 const prefix={target:-100,proposal:-20};assert.equal(mhDecision([prefix,...a],[prefix,...b],1,.9).target_delta,2);
 assert.throws(()=>mhDecision([{target:NaN,proposal:0}],b,0,.5));
});
test('asymmetric independent proposals converge to the intended finite target',()=>{
 const rng=randomStream(456),states=[{target:Math.log(.8),proposal:Math.log(.25)},{target:Math.log(.2),proposal:Math.log(.75)}];
 let current=0,count=0;
 for(let i=0;i<20000;i++){const proposed=rng()<.25?0:1;if(mhDecision([states[current]],[states[proposed]],0,rng()).accepted)current=proposed;if(i>=1000)count+=current===0;}
 assert.ok(Math.abs(count/19000-.8)<.025);
});
test('reference initialization may change prompt, but MH prefix may not',()=>{
 const x={tools:'same',referenceTools:'same',request:'new system prompt',referenceRequest:'old system prompt'};
 assert.doesNotThrow(()=>assertReplayContext({initial:true,...x}));
 assert.throws(()=>assertReplayContext({initial:false,...x}),/prefix observation/);
 assert.doesNotThrow(()=>assertReplayContext({initial:true,...x,tools:'changed'}));
});
