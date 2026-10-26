import assert from 'node:assert/strict';
import test from 'node:test';
import { runWorkflowPilot } from '../scripts/workflow-selected-pilot.mjs';
import { runSelectedPolicyOrder, controlledSourceProjectionSelector } from '../scripts/workflow-selected-pilot.mjs';

test('controlled order pilot traverses production service transitions and local simulated effects',async()=>{
  const report=await runWorkflowPilot();
  assert.equal(report.schema,'natlang-workflow-pilot/1');
  assert.equal(report.origin,'controlled');
  assert.equal(report.journeys.success.phase,'shipped');
  assert.equal(report.journeys.rejection.phase,'released');
  assert.equal(report.journeys.lostAck.phase,'reserved');
  assert.equal(report.journeys.lostAck.pending,'');
  assert.equal(report.journeys.cancel.phase,'released');
  assert.equal(report.journeys.refundRelease.phase,'released');
  assert.deepEqual(report.journeys.refundRelease.history.filter(row=>row.status==='done').map(row=>row.action),['reserve','charge','refund','release']);
  assert.notEqual(report.journeys.pendingRestart.pending,'');
  assert.equal(report.journeys.pendingRestart.phase,'new');
  assert.equal(report.journeys.repeated.phase,'shipped');
  assert.equal(report.journeys.repeated.history.filter(row=>row.status==='done').length,3);
  assert(report.journal.some(row=>row.action==='reconcile'));
  assert(report.trace.serviceEvents.some(row=>row.operation==='workflow.response-lost'));
  assert(report.effects.every(effect=>effect.status==='done'));
  assert.equal(report.cleanup.temporaryDirectoryRemoved,true);
});

test('model mode never falls back when selected checkpoint identity is absent',async()=>{
  const {spawnSync}=await import('node:child_process');
  const {fileURLToPath}=await import('node:url');
  const result=spawnSync(process.execPath,[fileURLToPath(new URL('../scripts/workflow-selected-pilot.mjs',import.meta.url)),'--mode','model'],{encoding:'utf8'});
  assert.notEqual(result.status,0);
  assert.match(result.stderr,/Model process executable is required|--model-command/);
});

test('selected policy rejects a source override that differs from the application policy',async()=>{
  await assert.rejects(runSelectedPolicyOrder({select:async()=>{throw new Error('should not initialize');},
    origin:'controlled',policySource:'export default function crisp(){ return {action:"ship"}; }'}),
  /differs from the application crisp\.ts source/);
});

test('owned-tracer certified selected policy drives every order journey through WorkflowService',async()=>{
  const selector=controlledSourceProjectionSelector();
  const result=await runSelectedPolicyOrder({origin:'controlled',seed:29,select:selector});
  assert.equal(result.outcome,'applied',JSON.stringify(result.journeys));
  assert.equal(result.journeys['ordinary-success'].state.phase,'shipped');
  assert.equal(result.journeys['rejected-operation'].state.phase,'released');
  assert.equal(result.journeys['lost-ack'].state.phase,'reserved');
  assert.equal(result.journeys['pending-restart'].state.pending,'pending-restart:reserve');
  assert.equal(result.journeys.cancel.state.phase,'released');
  assert.deepEqual(result.journeys['refund-release'].state.history.filter(row=>row.status==='done').map(row=>row.action),
    ['reserve','charge','refund','release']);
  assert.equal(result.journeys.repeated.state.history.filter(row=>row.status==='done').length,3);
  assert(result.controlledProjections.length>10);
  assert(result.controlledProjections.every(item=>item.originalCrispSourceDigest&&
    item.projectionDefinitionDigest&&item.snapshotLimitsDigest&&item.independentExpected&&
    JSON.stringify(item.expected)===JSON.stringify(item.independentExpected)&&
    item.trustedPolicyBinding==='exact application crisp.ts bytes enforced before selected compilation'&&
    item.observedEntrySourceDigest===item.sourceDigest&&
    item.compiledSelectedSourceDigest===item.sourceDigest&&
    (item.verification==='compiled-source-v8'||(item.publicDigest&&item.generatedRecords>0))));
  assert(result.selectedTrace.every(item=>item.selectedRecord));
});

test('model refusal during reconciliation stops later model journeys without selected fallback',async()=>{
  const base=controlledSourceProjectionSelector();let refused=false,callsAfterRefusal=0;
  const select=async prefix=>{
    if(refused)callsAfterRefusal++;
    if(prefix.entry==='crisp'&&JSON.stringify(prefix.selectedState).includes('"reconcile"')) {
      refused=true;throw new Error('fixture selected refusal during reconcile');
    }
    return base(prefix);
  };
  const report=await runSelectedPolicyOrder({origin:'model',seed:31,select});
  assert.equal(refused,true);
  assert.equal(report.outcome,'rejected');
  assert.equal(report.journeys['lost-ack'].outcome,'rejected');
  assert.equal(report.journeys['pending-restart'].outcome,'not-run-after-model-rejection');
  assert.equal(report.journeys.cancel.outcome,'not-run-after-model-rejection');
  assert.equal(callsAfterRefusal,0,'no later selected requests after model rejection');
  assert(report.selectedTrace.at(-1).failure.message.includes('fixture selected refusal'));
  assert(!report.effects.some(effect=>['pending-restart','cancel','refund-release','repeated'].includes(effect.order_id)));
});

