import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { CONTEXT_WEIGHT, FEEDBACK_WEIGHT, REFINE_REPAIR_VERSION, collectRefineRepairs, recordTurns } from '../dist/teacher/refine-repair.js';

const POLITE = 'a reply that is polite';
const REPLY = `---\nargs: { complaint: string }\nreturns: 'Is<string, "${POLITE}">'\n---\nAnswer the complaint.\n`;

/** A scripted executor (one return_result per turn) and a scripted judge; no model is loaded. */
function scenario({ answers, truth, repairs }) {
  let turn = 0;
  const base = async () => ({ calls: [['return_result', { status: 'success', value: answers[Math.min(turn++, answers.length - 1)] }]] });
  const recorded = recordTurns(base);
  const driver = Object.assign(recorded.driver, {
    decide: async request => {
      const value = /<<<value\n([^]*?)\nvalue>>>/.exec(String(request.messages.at(-1).content))[1];
      const p = truth(value) ? 0.97 : 0.04;
      return { log_probs: [Math.log(p), Math.log(1 - p)] };
    },
  });
  const traces = [];
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, calls: false, trace: trace => traces.push(trace),
    refinements: repairs === undefined ? {} : { repairs } });
  const fn = loadVirtualNatlang({ 'reply.nl': REPLY }, 'reply.nl');
  return { frames: recorded.frames, events: () => traces.flatMap(trace => trace.events),
    run: () => runtime.run(() => fn('late parcel')) };
}

test('a rejected refined return that the executor repairs becomes a whole-trajectory record', async () => {
  const s = scenario({ answers: ['This is your own fault.', 'I am sorry about the delay.'], truth: value => !/fault/.test(value) });
  assert.equal(await s.run(), 'I am sorry about the delay.');
  assert.equal(s.frames.length, 2);
  const [record, ...rest] = collectRefineRepairs({ id: 'row-1', trajectory: s.frames }, { trace: s.events() });
  assert.equal(rest.length, 0);
  assert.equal(record.schema, REFINE_REPAIR_VERSION);
  assert.equal(record.source_row, 'row-1');
  assert.equal(record.refinement.confirmed_by_trace, true);
  assert.deepEqual(record.refinement.predicates, [POLITE]);
  const [rejection] = record.refinement.rejections;
  assert.equal(rejection.code, 'refinement-unsatisfied');
  assert.equal(rejection.path, 'return');
  assert.equal(rejection.attempted, 'This is your own fault.');
  assert.match(rejection.feedback, /^rejected\nrefinement-unsatisfied: Revise the value at return so that it is "a reply that is polite"/);
  assert.equal(record.refinement.repaired, 'I am sorry about the delay.');
  assert.deepEqual(record.weights, { context: CONTEXT_WEIGHT, feedback: FEEDBACK_WEIGHT, earlier_assistant: 0, rejected_attempt_target: 0 });
  assert.equal(FEEDBACK_WEIGHT, 0.25);
  // The rejected attempt keeps its place but is not taught as gold; the repair turn is supervised and sees the feedback.
  assert.deepEqual(record.turns.map(turn => [turn.role, turn.target_weight, turn.training_admission.approved]),
    [['rejected-attempt', 0, false], ['repair', 1, true]]);
  const repairContext = JSON.stringify(record.turns[1].messages);
  assert.ok(repairContext.includes('This is your own fault.') && repairContext.includes('refinement-unsatisfied'));
  assert.equal(record.turns[1].target.calls[0].tool, 'return_result');
});

test('two rejections then a repair give one episode with both rejections', async () => {
  const s = scenario({ answers: ['your fault', 'you caused this', 'We are sorry and will fix it.'], truth: value => /sorry/.test(value) });
  await s.run();
  const records = collectRefineRepairs({ trajectory: s.frames });
  assert.equal(records.length, 1);
  assert.deepEqual(records[0].refinement.rejections.map(item => item.attempted), ['your fault', 'you caused this']);
  assert.deepEqual(records[0].turns.map(turn => turn.role), ['rejected-attempt', 'rejected-attempt', 'repair']);
});

test('a first-try pass and an unrepaired failure produce no record', async () => {
  const pass = scenario({ answers: ['Thank you for telling us.'], truth: () => true });
  await pass.run();
  assert.deepEqual(collectRefineRepairs({ trajectory: pass.frames }), []);
  const fail = scenario({ answers: ['your fault'], truth: () => false, repairs: 1 });
  await assert.rejects(fail.run());
  // With the outcome of the row (the call failed), nothing was repaired; so no record even though a rejection exists.
  assert.deepEqual(collectRefineRepairs({ trajectory: fail.frames, outcome: { accepted: false } }), []);
  // The trace is the evidence that a final attempt stood: its last check failed, so it is not confirmed.
  assert.deepEqual(collectRefineRepairs({ trajectory: fail.frames }, { trace: fail.events() }), []);
});

test('rows that the collector rejected are skipped, and invocations are kept apart', async () => {
  const s = scenario({ answers: ['your fault', 'Sorry about that.'], truth: value => /Sorry/.test(value) });
  await s.run();
  assert.equal(collectRefineRepairs({ id: 'x', trajectory: s.frames, outcome: { accepted: false } }).length, 0);
  assert.equal(collectRefineRepairs({ id: 'x', trajectory: s.frames, outcome: { accepted: true } }).length, 1);
  // Frames of two invocations interleaved: a repair in one is not confused with the other's turns.
  const tagged = [{ ...s.frames[0], invocation_id: 'a' }, { ...s.frames[0], invocation_id: 'b', context: s.frames[0].context }, { ...s.frames[1], invocation_id: 'a' }];
  const records = collectRefineRepairs({ trajectory: tagged });
  assert.equal(records.length, 1);
  assert.equal(records[0].invocation_id, 'a');
  assert.deepEqual(records[0].turns.map(turn => turn.index), [0, 2]);
});
