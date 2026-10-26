import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyEffectfulTrainingCase,decodeWitness,SelectedExecutionHarness}
  from 'tracing-js-self/training';

test('Natlang dependency emits only source-grounded canonical semantic execution',async()=>{
  const fixture={source:'"use strict";function run(current){const next=current+1;return next;}',
    entry:'run',args:[7],family:'natlang-canonical-semantic',fixture:{calls:[]}};
  const result=await verifyEffectfulTrainingCase(fixture);
  assert.equal(result.status,'locally_verified',result.reason);
  assert.equal(result.exported.public.schema,'semantic-execution/5');
  assert.equal(result.exported.public.traceFormat,'semantic-trace/5');
  assert.equal(result.exported.public.contextFormat,'semantic-context/5');
  assert.equal(result.exported.sidecar.representation,'compact');
  assert.equal(result.exported.public.groups,undefined,'obsolete grouped views are not supported');
  const records=result.exported.public.timeline.filter(row=>row.kind==='semantic')
    .map(row=>decodeWitness(row.line));
  assert(records.some(row=>row.kind==='compute' && row.fields.bind==='next' &&
    row.fields.expr.op==='binary' && row.fields.expr.left.op==='local' &&
    row.fields.expr.left.name==='current' && row.fields.result[1]===8));
  const replay=SelectedExecutionHarness.replay(result.exported);
  const controller=replay.invocations.get('invocation1').controller;
  assert.equal(controller.completion.value[1],8);
  assert.equal(controller.segmentPacket().randomState,null,
    'unseeded contexts expose the required null RNG state in bounded replay history');
  assert.equal(replay.selectionContext('invocation1').selectedState.randomState,null,
    'the public model selection context exposes required RNG state');
  const historicalV4=structuredClone(result.exported);
  historicalV4.public.schema='semantic-execution/4';
  historicalV4.public.traceFormat='semantic-trace/4';
  historicalV4.public.contextFormat='semantic-context/4';
  for (const row of historicalV4.public.timeline) {
    if (row.schema) row.schema='semantic-execution/4';
    if (row.traceFormat) row.traceFormat='semantic-trace/4';
  }
  assert.throws(()=>SelectedExecutionHarness.replay(historicalV4),
  /Unsupported public replay contract/, 'the current selected runtime explicitly refuses frozen v4 executions');
  const obsolete=await verifyEffectfulTrainingCase({...fixture,projectionOptions:{representation:'explicit'}});
  assert.notEqual(obsolete.status,'locally_verified','old learner representation must refuse');
});
