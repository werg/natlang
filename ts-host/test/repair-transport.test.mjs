import test from 'node:test';
import assert from 'node:assert/strict';
import {postInferenceJson} from '../scripts/repair-transport.mjs';

test('socket failures retry identical inference requests with exponential delays',async()=>{
 const sent=[],delays=[],events=[];
 const result=await postInferenceJson('http://local/inference',{seed:17},{
  fetchImpl:async(url,request)=>{sent.push(request.body);if(sent.length<3)throw Object.assign(new TypeError('fetch failed'),{cause:{code:'UND_ERR_SOCKET'}});return new Response('{"ok":true}');},
  wait:async ms=>delays.push(ms),onRetry:e=>events.push(e)});
 assert.deepEqual(result,{ok:true});assert.deepEqual(delays,[5000,10000]);
 assert.equal(new Set(sent).size,1);assert.equal(events.length,2);
});
test('rate limit respects Retry-After and exhausted retries are failures',async()=>{
 const delays=[];let calls=0;
 await assert.rejects(postInferenceJson('http://local',{}, {retries:1,
  fetchImpl:async()=>{calls++;return new Response('{}',{status:429,headers:{'retry-after':'19'}});},wait:async ms=>delays.push(ms)}),/HTTP 429/);
 assert.equal(calls,2);assert.deepEqual(delays,[19000]);
});
test('request errors and malformed successful JSON are not retried',async()=>{
 for(const response of [()=>new Response('{}',{status:400}),()=>new Response('invalid')]){
  let calls=0;await assert.rejects(postInferenceJson('http://local',{}, {fetchImpl:async()=>{calls++;return response();}}));assert.equal(calls,1);
 }
});
test('cancellation interrupts backoff without another request',async()=>{
 const abort=new AbortController();let calls=0;
 await assert.rejects(postInferenceJson('http://local',{}, {signal:abort.signal,
  fetchImpl:async()=>{calls++;return new Response('{}',{status:503});},
  onRetry:()=>abort.abort()}),{name:'AbortError'});
 assert.equal(calls,1);
});
