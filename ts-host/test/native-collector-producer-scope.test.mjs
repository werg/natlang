import test from 'node:test';
import assert from 'node:assert/strict';
import {executeProgram} from '../dist/teacher/collector.js';
import {MemoryNeuraleseStore} from '../dist/native/neuralese-store.js';
import {createNativeStateFixturePort} from './fixtures/native-state-port.mjs';

test('standalone collector retains a nested native writer for its parent model turn', async () => {
  const store=new MemoryNeuraleseStore();
  const neuralese={store,port:createNativeStateFixturePort(store),dialect:'native-state/fixture'};
  const record={version:'natlang.program/2',id:'collector-parent-scope',kind:'lambda_source',split:'train',
    semantics:{root:'solve.nl',files:{'solve.nl':'---\nargs: {}\nreturns: string\n---\nCall check_assumption to write a memo, then finish.\n', 'solve/check_assumption.nl':'---\nargs: {}\nreturns: Neuralese<string>\n---\nWrite a concise memo.\n'},inputs:{},expected:'done'}};
  let index=0; const requests=[];
  const driver=Object.assign(async(request)=>{
    requests.push(request); index++;
    if(index===1)return {calls:[['eval',{code:'const memo = await check_assumption(); memo;',finish:false}]]};
    if(index===2)return {calls:[['return_result',{status:'success',value:'child memo'}]]};
    return {calls:[['return_result',{status:'success',value:'done'}]]};
  },{neuralese:true});
  const result=await executeProgram(record,driver,{systemPrompt:'Use the provided tools.',contextTokens:8192,
    maxTurns:8,rootSeed:19,runId:'collector-parent-scope-run',neuralese});
  assert.equal(result.outcome.accepted,true);
  assert.equal(requests.length,3);
  assert.notEqual(requests[1].invocation_id,requests[0].invocation_id,'nested provider turn uses the instrumented collector driver');
  const graph=result.outcome.execution_graph;
  const childTrace=result.invocationTraces.find(trace=>trace.parentCallId==='collector-parent-scope-run');
  assert.ok(childTrace,'nested call is attached to the collector root');
  const manifest=childTrace.events.find(event=>event.kind==='manifest');
  assert.equal(manifest.seed_policy.root,19,'child inherits the collector seed policy');
  const writer=graph.find(event=>event.kind==='block_write'&&event.format==='native-state');
  const read=graph.find(event=>event.kind==='block_read'&&event.block===writer?.block);
  assert.ok(writer);assert.ok(read);
  assert.ok(read.inputs.some(input=>input.node===writer.node),'actual parent read binds child writer');
});
