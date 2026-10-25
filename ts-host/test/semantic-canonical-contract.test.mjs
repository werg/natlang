import test from 'node:test';
import assert from 'node:assert/strict';
import {verifyEffectfulTrainingCase,decodeWitness,SelectedExecutionHarness}
  from 'tracing-js-self/training';

test('Natlang dependency emits only source-grounded canonical semantic execution',async()=>{
  const fixture={source:'"use strict";function run(current){const next=current+1;return next;}',
    entry:'run',args:[7],family:'natlang-canonical-semantic',fixture:{calls:[]}};
  const result=await verifyEffectfulTrainingCase(fixture);
  assert.equal(result.status,'locally_verified',result.reason);
  assert.equal(result.exported.public.schema,'semantic-execution/4');
  assert.equal(result.exported.public.traceFormat,'semantic-trace/4');
  assert.equal(result.exported.sidecar.representation,'compact');
  assert.equal(result.exported.public.groups,undefined,'obsolete grouped views are not supported');
  const records=result.exported.public.timeline.filter(row=>row.kind==='semantic')
    .map(row=>decodeWitness(row.line));
  assert(records.some(row=>row.kind==='compute' && row.fields.bind==='next' &&
    row.fields.expr.op==='binary' && row.fields.expr.left.op==='local' &&
    row.fields.expr.left.name==='current' && row.fields.result[1]===8));
  assert.equal(SelectedExecutionHarness.replay(result.exported).invocations.get('invocation1')
    .controller.completion.value[1],8);
  const obsolete=await verifyEffectfulTrainingCase({...fixture,projectionOptions:{representation:'explicit'}});
  assert.notEqual(obsolete.status,'locally_verified','old learner representation must refuse');
});
