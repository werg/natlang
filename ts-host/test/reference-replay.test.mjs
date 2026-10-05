import test from 'node:test';import assert from 'node:assert/strict';
import {referenceObservationKey,selectReferenceTurn} from '../scripts/reference-replay.mjs';
const request=(value,system='old')=>({messages:[{role:'system',content:system},{role:'user',content:'Label this item'},{role:'assistant',tool_calls:[{id:'scope_0',function:{name:'eval',arguments:system}}]},{role:'tool',content:value}]});
test('harness changes preserve association while actual inputs must match',()=>{
 assert.equal(referenceObservationKey(request('a').messages),referenceObservationKey(request('a','new').messages));
 assert.notEqual(referenceObservationKey(request('a').messages),referenceObservationKey(request('b').messages));
 const a={request:request('a')},b={request:request('b')},used=new Set();
 assert.equal(selectReferenceTurn([a,b],used,request('b','new')),b);used.add(b);
 assert.equal(selectReferenceTurn([a,b],used,request('b')),undefined);
});
test('non-bootstrap actions and task instructions remain exact',()=>{
 const a=request('a'),b=request('a');b.messages[1].content='Different task';assert.notEqual(referenceObservationKey(a.messages),referenceObservationKey(b.messages));
 b.messages[1]=a.messages[1];a.messages[2].tool_calls[0].id='model-call';b.messages[2].tool_calls[0].id='another-call';assert.equal(referenceObservationKey(a.messages),referenceObservationKey(b.messages));b.messages[2].tool_calls[0].function.arguments='different code';assert.notEqual(referenceObservationKey(a.messages),referenceObservationKey(b.messages));
});
