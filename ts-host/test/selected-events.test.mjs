import assert from 'node:assert/strict';
import test from 'node:test';
import {SelectedEventQueue,foldSelectedEvents,mapSelectedEvents} from '../dist/runtime/selected-events.js';

test('event snapshots reject getters without invoking them and bound delivery history',async()=>{
  const queue=new SelectedEventQueue('bounded',1,()=>{},2);let reads=0;
  assert.throws(()=>queue.post('bad',{get value(){reads++;return 1;}}),/accessors/);assert.equal(reads,0);
  queue.post('one',{value:1});await queue.next();queue.post('two',{value:2});await queue.next();
  assert.throws(()=>queue.post('three',{value:3}),/history is full/);
  assert.equal(queue.post('one',{value:1}).disposition,'duplicate');
  const checkpoint=queue.checkpoint();assert.equal(checkpoint.historyCapacity,2);
  const restored=SelectedEventQueue.restore(checkpoint);
  assert.throws(()=>restored.post('three',{value:3}),/history is full/);restored.close();queue.close();
});

const deferred=()=>{let resolve;return {promise:new Promise(yes=>{resolve=yes;}),resolve:value=>resolve(value)};};

test('Fold waits in host, completes a step before consuming the next event, and observes closure',async()=>{
  const q=new SelectedEventQueue('counter',2),gate=deferred(),started=deferred(),seen=[];
  q.post('first',2);q.post('second',3);q.close();
  const work=foldSelectedEvents(q,0,async(state,value,event)=>{
    seen.push(event.id);if(event.id==='first'){started.resolve();await gate.promise;}return state+value;
  });
  await started.promise;assert.deepEqual(seen,['first']);assert.equal(q.pending,1);
  gate.resolve();assert.equal(await work,5);assert.deepEqual(seen,['first','second']);
});

test('Map emits completed work while an open source is below its capacity',{timeout:1000},async()=>{
  const q=new SelectedEventQueue('open',3),output=mapSelectedEvents(q,value=>value*2,3);
  const first=output.next();q.post('one',4);
  assert.deepEqual(await first,{value:{source:'open',id:'one',position:0,value:8},done:false});
  await output.return();assert.equal(q.status,'cancelled');
});

test('bounded Map retains input identity and order when later work completes first',async()=>{
  const q=new SelectedEventQueue('ordered',3),a=deferred(),b=deferred(),started=deferred(),active=[];
  q.post('a',1);q.post('b',2);q.post('c',3);q.close();
  const output=mapSelectedEvents(q,async(value,event)=>{
    active.push(event.id);if(active.length===2)started.resolve();
    if(event.id==='a')await a.promise;if(event.id==='b')await b.promise;return value+10;
  },2);
  const first=output.next();await started.promise;assert.deepEqual(active,['a','b']);assert.equal(q.pending,1);
  b.resolve();a.resolve();assert.equal((await first).value.id,'a');
  assert.equal((await output.next()).value.id,'b');assert.equal((await output.next()).value.id,'c');
  assert.equal((await output.next()).done,true);
});

test('ingress backpressure, duplicate identity, overflow and cancellation are explicit',async()=>{
  const receipts=[],q=new SelectedEventQueue('delivery',1,receipt=>receipts.push(receipt));
  q.post('a',{n:1});assert.throws(()=>q.post('b',2),/overflow/);
  const waiting=q.push('b',2);assert.equal(q.pending,1);
  assert.equal((await q.next()).value.id,'a');await waiting;
  q.close();assert.equal(q.post('a',{n:1}).disposition,'duplicate');
  assert.throws(()=>q.post('a',{n:9}),/another payload/);
  assert.throws(()=>q.post('late',1),/closed/);
  assert.equal((await q.next()).value.id,'b');assert.equal((await q.next()).done,true);
  assert(receipts.some(receipt=>receipt.disposition==='overflow'));
});

test('a pausable producer keeps waiting when another source claims the released slot',async()=>{
  const q=new SelectedEventQueue('competing',1);q.post('a',1);
  const waiting=q.push('b',2);
  const first=q.next();q.post('c',3);
  assert.equal((await first).value.id,'a');
  // The released producer resumes after the synchronous unpausable arrival.
  // It must continue backpressure instead of failing with a spurious overflow.
  await Promise.resolve();assert.equal((await q.next()).value.id,'c');
  await waiting;assert.equal((await q.next()).value.id,'b');q.close();
});

test('checkpoint restores the cursor, buffered input and duplicate ledger without replaying consumed input',async()=>{
  const q=new SelectedEventQueue('durable',2);q.post('a',1);q.post('b',2);await q.next();
  const snapshot=q.checkpoint(),restored=SelectedEventQueue.restore(snapshot);
  assert.equal(restored.post('a',1).disposition,'duplicate');assert.equal((await restored.next()).value.position,1);
  assert.equal(restored.post('c',3).position,2);
  const corrupt=structuredClone(snapshot);corrupt.buffered[0].value=999;
  assert.throws(()=>SelectedEventQueue.restore(corrupt),/payload or order/);
});

test('source failure and cancelled children cannot report normal completion',async()=>{
  const q=new SelectedEventQueue('failed',2);q.post('a',1);q.fail(new Error('source lost'));
  assert.equal((await q.next()).value.value,1);await assert.rejects(q.next(),/source lost/);
  const input=new SelectedEventQueue('child',2);input.post('a',1);input.post('b',2);
  const output=mapSelectedEvents(input,()=>{throw new Error('child failed');},1);
  await assert.rejects(output.next(),/child failed/);assert.equal(input.status,'cancelled');
});
