import test from 'node:test';
import assert from 'node:assert/strict';
import { nativeCaptureGraphBinding } from '../dist/native/model-recording.js';

test('same artifact independent writer/use occurrences bind completely rather than last-writer wins',()=>{
  const id='nzs1_'+'a'.repeat(52);
  const nodes=new Map([
    ['first',{kind:'block_write',node:'first',turn:'producer',block:id}],
    ['second',{kind:'block_write',node:'second',turn:'producer',block:id}],
    ['read-first',{kind:'block_read',node:'read-first',turn:'reader',block:id,stop_gradient:true,inputs:[{node:'first',block:id}]}],
    ['read-second',{kind:'block_read',node:'read-second',turn:'reader',block:id,inputs:[{node:'second',block:id}]}],
  ]);
  const writer=nativeCaptureGraphBinding('producer',{writes:[{state_id:id},{state_id:id}]},nodes);
  assert.deepEqual(writer.written_nodes,{});
  assert.deepEqual(writer.written_occurrences,[{index:0,id,node:'first'},{index:1,id,node:'second'}]);
  const capture={prompt:[{kind:'native_state',id,source_occurrence:'first',stop_gradient:true},
    {kind:'native_state',id,source_occurrence:'second'}]};
  const reader=nativeCaptureGraphBinding('reader',capture,nodes);
  assert.deepEqual(reader.read_bindings,{});
  assert.deepEqual(reader.read_occurrences,[{index:0,id,source_state_id:id,node:'read-first',source:'first',stop_gradient:true},
    {index:1,id,source_state_id:id,node:'read-second',source:'second'}]);
  const wrong=structuredClone(capture);wrong.prompt[0].source_occurrence='second';
  assert.throws(()=>nativeCaptureGraphBinding('reader',wrong,nodes),/changed source occurrence/);
  const ambiguousNodes=new Map([...nodes].map(([key,event])=>[key,{...event}]));
  delete ambiguousNodes.get('read-first').stop_gradient;
  assert.throws(()=>nativeCaptureGraphBinding('reader',{prompt:[{kind:'native_state',id},{kind:'native_state',id}]},ambiguousNodes),/ambiguous/);
});

test('a unique external native-state root retains its differentiable read occurrence',()=>{
  const id='nzs1_'+'b'.repeat(52);
  const nodes=new Map([['external-read',{kind:'block_read',node:'external-read',turn:'reader',block:id,
    inputs:[{node:'external',block:id}]}]]);
  const binding=nativeCaptureGraphBinding('reader',{prompt:[{kind:'native_state',id}]},nodes);
  assert.deepEqual(binding.read_bindings,{});
  assert.deepEqual(binding.read_occurrences,[{index:0,id,source_state_id:id,node:'external-read',source:'external'}]);
});
