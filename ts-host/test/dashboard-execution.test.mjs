import test from 'node:test';
import assert from 'node:assert/strict';
import { createExecutionModel, lambdaName, lambdaStatus, executionSummary } from '../dashboard/execution.js';

const event = (callId, data, parentCallId = null) => ({ callId, parentCallId, name: callId, at: 1000, ...data });
const trace = (callId, data, parent = null) => event(callId, { type: 'trace', event: data }, parent);

test('the execution tree retains concurrent children, including children observed before their parent', () => {
  const model = createExecutionModel();
  model.add(event('child-a', { type: 'call_start' }, 'parent'));
  model.add(event('parent', { type: 'call_start' }));
  model.add(event('child-b', { type: 'call_start' }, 'parent'));
  assert.deepEqual(model.roots.map(call => call.id), ['parent']);
  assert.deepEqual(model.roots[0].children.map(call => call.id), ['child-a', 'child-b']);
  assert.match(lambdaStatus(model.roots[0]), /Waiting for 2 child calls/);
  assert.equal(executionSummary(model).active.length, 3);
  model.add(event('child-a', { type: 'call_end', outcome: 'done', detail: '', at: 1200 }, 'parent'));
  assert.match(lambdaStatus(model.roots[0]), /Waiting for 1 child call/);
  assert.equal(executionSummary(model).complete, 1);
  assert.equal(model.calls.get('child-a').ended - model.calls.get('child-a').started, 200);
});

test('streamed turns show generation and execution phases, and retries replace their abandoned draft', () => {
  const model = createExecutionModel();
  model.add(trace('answer', { kind: 'model_request', phase: 'start', turn: 1 }));
  assert.equal(lambdaStatus(model.roots[0]), 'Reading the input');
  model.add(event('answer', { type: 'model_delta', turn: 1, delta: { type: 'reasoning', text: 'Inspecting the input.' } }));
  assert.equal(lambdaStatus(model.roots[0]), 'Working through the instructions');
  model.add(event('answer', { type: 'model_delta', turn: 1, delta: { type: 'text', text: 'Abandoned draft' } }));
  model.add(event('answer', { type: 'model_delta', turn: 1, delta: { type: 'reset' } }));
  assert.equal(model.roots[0].turns.get(1).text, '');
  assert.equal(model.roots[0].turns.get(1).retries, 1);
  assert.equal(lambdaStatus(model.roots[0]), 'Retrying the model request');
  model.add(event('answer', { type: 'model_delta', turn: 1, delta: { type: 'text', text: 'New draft' } }));
  model.add(trace('answer', { kind: 'proposal', phase: 'generated', turn: 1, calls: [['eval', { code: 'return 7', finish: true }]] }));
  assert.equal(lambdaStatus(model.roots[0]), 'Running a step');
  model.add(trace('answer', { kind: 'action', name: 'eval', arguments: { code: 'return 7', finish: true }, outcome: 'applied', result_text: '7' }));
  const turn = model.roots[0].turns.get(1);
  assert.equal(turn.tools.length, 1); assert.equal(turn.tools[0].name, 'eval'); assert.deepEqual(turn.results, ['7']);
  assert.equal(turn.text, 'New draft');
  model.add(event('answer', { type: 'call_end', outcome: 'done', detail: '' }));
  assert.equal(lambdaStatus(model.roots[0]), 'Finished');
});

test('recovered step issues stay expandable without marking a successful lambda as failed', () => {
  const model = createExecutionModel();
  model.add(trace('answer', { kind: 'model_request', phase: 'start', turn: 1 }));
  model.add(trace('answer', { kind: 'action', name: 'eval', arguments: {}, outcome: 'rejected', result_text: 'Wrong result type.' }));
  assert.match(lambdaStatus(model.roots[0]), /Recovering/);
  model.add(event('answer', { type: 'call_end', outcome: 'done', detail: '' }));
  assert.equal(executionSummary(model).problems, 0);
  assert.deepEqual(model.roots[0].turns.get(1).failures, ['Wrong result type.']);
  model.add(event('failed', { type: 'call_start' }));
  model.add(event('failed', { type: 'call_end', outcome: 'failed', detail: 'Could not finish.' }));
  assert.equal(executionSummary(model).problems, 1);
});

test('a stopped run does not leave lambdas apparently running, and does not fabricate a result', () => {
  const model = createExecutionModel(); model.add(event('active', { type: 'call_start' }));
  const job = { status: 'stopped', drained: true };
  assert.equal(lambdaStatus(model.roots[0], job), 'Interrupted');
  assert.equal(executionSummary(model, job).active.length, 0);
  assert.equal(executionSummary(model, job).unfinished, 1);
  assert.equal(executionSummary(model, job).complete, 0);
  assert.equal(lambdaName({ name: 'nl@main.ts:12' }), 'Inline λ · main.ts:12');
  model.add(event('active', { type: 'call_end', outcome: 'quiesced', detail: 'Stopped by caller.' }));
  assert.equal(executionSummary(model, job).stopped, 1);
  assert.equal(executionSummary(model, job).problems, 0, 'an intentional stop is distinct from a failed lambda');
});

test('recorded turns work without streamed deltas, including empty action results', () => {
  const model = createExecutionModel();
  model.add(trace('answer', { kind: 'proposal', phase: 'generated', turn: 1, text: 'The response.', calls: [['eval', { code: 'const x = 1' }], ['read_code', { name: 'helper' }]] }));
  model.add(trace('answer', { kind: 'action', name: 'eval', arguments: { code: 'const x = 1' }, outcome: 'applied', result_text: '' }));
  model.add(trace('answer', { kind: 'action', name: 'read_code', arguments: { name: 'helper' }, outcome: 'applied', result_text: 'helper code' }));
  assert.deepEqual(model.roots[0].turns.get(1).results, ['', 'helper code']);
  assert.equal(model.roots[0].turns.get(1).tools.length, 2);
});
