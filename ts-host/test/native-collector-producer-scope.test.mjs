import test from 'node:test';
import assert from 'node:assert/strict';
import {executeProgram} from '../dist/teacher/collector.js';
import {MemoryNeuraleseStore} from '../dist/native/neuralese-store.js';
import {createNativeStateFixturePort} from './fixtures/native-state-port.mjs';

test('standalone collector retains a nested native writer for its parent model turn', async () => {
  const store=new MemoryNeuraleseStore();
  const neuralese={store,port:createNativeStateFixturePort(store),dialect:'native-state/fixture'};
  const record={version:'natlang.program/2',id:'collector-parent-scope',kind:'lambda_source',split:'train',
    semantics:{root:'solve.nl',files:{'solve.nl':'---\nargs: {}\nreturns: string\n---\nCall a child to write a memo, then finish.\n'},inputs:{},expected:'done'}};
  let index=0;
  const driver=Object.assign(async()=>{
    index++;
    if(index===1)return {calls:[['eval',{code:'const memo = await nl<Neuralese<string>>`Write a concise memo.`(); memo;',finish:false}]]};
    if(index===2)return {calls:[['return_result',{status:'success',value:'child memo'}]]};
    return {calls:[['return_result',{status:'success',value:'done'}]]};
  },{neuralese:true});
  const result=await executeProgram(record,driver,{systemPrompt:'Use the provided tools.',contextTokens:8192,
    maxTurns:8,rootSeed:19,runId:'collector-parent-scope-run',neuralese});
  assert.equal(result.outcome.accepted,true);
  const graph=result.outcome.execution_graph;
  const writer=graph.find(event=>event.kind==='block_write'&&event.format==='native-state');
  const read=graph.find(event=>event.kind==='block_read'&&event.block===writer?.block);
  assert.ok(writer);assert.ok(read);
  assert.ok(read.inputs.some(input=>input.node===writer.node),'actual parent read binds child writer');
});
