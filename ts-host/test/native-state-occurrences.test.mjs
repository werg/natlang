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

test('a read the engine fed as the state\'s own tokens (in-context-text) binds no native input',()=>{
  const id='nzs1_'+'c'.repeat(52);
  const nodes=new Map([
    ['w',{kind:'block_write',node:'w',turn:'producer',block:id}],
    ['read-inline',{kind:'block_read',node:'read-inline',turn:'reader',block:id,format:'in-context-text',inputs:[{node:'w',block:id}]}],
    ['read-spliced',{kind:'block_read',node:'read-spliced',turn:'reader',block:id,stop_gradient:true,inputs:[{node:'w',block:id}]}],
  ]);
  // one occurrence read in context (tokens in the prompt, its receipt only), one spliced
  const capture={prompt:[{kind:'tokens',token_ids:[5,6]},{kind:'native_state',id,source_occurrence:'w',stop_gradient:true}],
    native_state_imports:[{id,mode:'in-context-text',observed_native_state_id:id,source_occurrence:'w'},
      {id,mode:'exact-native-state',observed_native_state_id:id,source_occurrence:'w',stop_gradient:true}]};
  const reader=nativeCaptureGraphBinding('reader',capture,nodes);
  assert.deepEqual(reader.read_bindings,{[id]:'w'});
  // every read in context: nothing to bind, and no inventory mismatch
  const inline={prompt:[{kind:'tokens',token_ids:[5,6]}],native_state_imports:[{id,mode:'in-context-text',observed_native_state_id:id}]};
  const only=new Map([...nodes].filter(([key])=>key!=='read-spliced'));
  assert.deepEqual(nativeCaptureGraphBinding('reader',inline,only).read_bindings,{});
});

test('a unique external native-state root retains its differentiable read occurrence',()=>{
  const id='nzs1_'+'b'.repeat(52);
  const nodes=new Map([['external-read',{kind:'block_read',node:'external-read',turn:'reader',block:id,
    inputs:[{node:'external',block:id}]}]]);
  const binding=nativeCaptureGraphBinding('reader',{prompt:[{kind:'native_state',id}]},nodes);
  assert.deepEqual(binding.read_bindings,{});
  assert.deepEqual(binding.read_occurrences,[{index:0,id,source_state_id:id,node:'external-read',source:'external'}]);
});

test('a recorded request names only writers its trajectory holds; a foreign writer occurrence becomes an external use', async () => {
  const { localNativeOccurrences, nativeStateSentinel, runtimeTextToParts } = await import('../dist/native/neuralese.js');
  const id = 'nzs1_' + 'b'.repeat(52);
  const known = (_block, occurrence) => occurrence === 'task-2/1#3';
  const messages = [
    { role: 'user', content: `skill ${nativeStateSentinel(id, false, 'task-1/1#6')} and ${nativeStateSentinel(id, true, 'task-2/1#3')}` },
    { role: 'user', content: [{ type: 'native_state', id, source_occurrence: 'task-1/1#6' }, { type: 'native_state', id, stop_gradient: true }] },
    { role: 'assistant', content: null, tool_calls: [{ function: { name: 'eval', arguments: JSON.stringify({ code: nativeStateSentinel(id, false, 'task-1/1#6') }) } }] },
  ];
  const local = localNativeOccurrences(messages, known);
  assert.deepEqual(runtimeTextToParts(local[0].content).filter(part => part.type === 'native_state'),
    [{ type: 'native_state', id }, { type: 'native_state', id, stop_gradient: true, source_occurrence: 'task-2/1#3' }]);
  assert.deepEqual(local[1].content, [{ type: 'native_state', id }, { type: 'native_state', id, stop_gradient: true }]);
  assert.deepEqual(runtimeTextToParts(JSON.parse(local[2].tool_calls[0].function.arguments).code), [{ type: 'native_state', id }]);
  assert.equal(messages[1].content[0].source_occurrence, 'task-1/1#6', 'the input is not modified');
});

test('an arrow-function model driver records the actor id "model", not an empty name', async () => {
  const { createNatlangRuntime, defineNatlang } = await import('../dist/index.js');
  const traces = [];
  const runtime = createNatlangRuntime({ model: () => ({ calls: [['return_result', { status: 'success', value: 'ok' }]] }), trace: trace => traces.push(trace) });
  assert.equal(await runtime.run(() => defineNatlang('---\nreturns: string\n---\nSay ok.\n', { name: 'say' })()), 'ok');
  const turns = traces.flatMap(trace => trace.events).filter(event => event.kind === 'model_turn');
  assert.ok(turns.length && turns.every(event => event.actor?.id === 'model'), JSON.stringify(turns.map(event => event.actor)));
});
