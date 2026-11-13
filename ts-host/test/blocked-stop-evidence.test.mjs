// An expected blocked/failed root stop earns its reward only after the obstacle was observed (2026-10-29).
// Before this check every expected-blocked curriculum case accepted an immediate give-up with no work: a reward the RL
// policy could take without attempting the task (AGENTS.md "Task contracts and reward validity").
// Cases come from training seeds only.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { replayReference, verifyCases } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { FAMILIES } from '../scripts/inline-curriculum/families.mjs';

for (const family of ['event_retry', 'child_sufficiency', 'actor_route_replan', 'actor_greenhouse', 'iterate_schedule_repair']) {
  test(`${family}: blocked stop needs the decisive observation`, async () => {
    const record = FAMILIES[family].build(7, 0, 'train').find(item => item.semantics.operation === 'blocked');
    assert.ok(record?.curriculum.decisive.length, 'blocked variant declares its decisive observation');
    const reference = (await replayReference(record, TOOLS_PROMPT)).run.outcome;
    assert.equal(reference.accepted, true, JSON.stringify(reference.rejection_reasons));
    assert.equal(reference.checks.stop_evidence, true);
    const giveUp = structuredClone(record);
    giveUp.curriculum.reference.root = [['return_result', { status: 'failed', reason: 'I cannot complete this task.' }]];
    const outcome = (await replayReference(giveUp, TOOLS_PROMPT)).run.outcome;
    assert.equal(outcome.checks.honest_stop, true, 'the give-up is a well-formed stop');
    assert.equal(outcome.checks.stop_evidence, false);
    assert.equal(outcome.accepted, false);
  });
}

test('a case expected to finish carries no stop-evidence check', async () => {
  const record = FAMILIES.event_retry.build(7, 0, 'train').find(item => item.semantics.operation !== 'blocked');
  const outcome = (await replayReference(record, TOOLS_PROMPT)).run.outcome;
  assert.equal(outcome.accepted, true);
  assert.equal('stop_evidence' in outcome.checks, false);
});

// Numeric service readings and iteration cycles have no fixed text: their markers are the runtime's own graph records,
// which curriculum admission checks in the execution graph as the collector does (2026-10-30).
for (const [family, partial] of [
  // Reads the greenhouse twice and gives up before the third (still falling) reading.
  ['actor_greenhouse', ['eval', { code: 'greenhouse.set({ heater: true, vent: false });\ngreenhouse.read();\ngreenhouse.read().temp' }]],
  // Runs the repair for fewer steps than it takes to repeat a state.
  ['iterate_schedule_repair', ['eval', { code: 'const fixed = await resolve.iterateOn(schedule).withLimit({maxSteps: 5}).until(s => conflicts(s) === 0);\nfixed' }]],
]) {
  test(`${family}: runtime_graph marker admits the reference and rejects a stop before the observation`, async () => {
    const record = FAMILIES[family].build(7, 0, 'train').find(item => item.semantics.operation === 'blocked');
    assert.equal(record.curriculum.decisive[0].source, 'runtime_graph');
    const [verified] = await verifyCases([record], TOOLS_PROMPT);
    assert.equal(verified.ok, true, JSON.stringify(verified.problems));
    const early = structuredClone(record);
    early.curriculum.reference.root = [partial, record.curriculum.reference.root.at(-1)];
    const outcome = (await replayReference(early, TOOLS_PROMPT)).run.outcome;
    assert.equal(outcome.checks.honest_stop, true);
    assert.equal(outcome.checks.stop_evidence, false);
    assert.equal(outcome.accepted, false);
  });
}

// A plain loop over the program's real step function is evidence equivalent to an iterateOn step marker when the run's
// own calls produced the marked (revisited) state at least twice (2026-11-13, held-v2 row 30/1).
test('iterate_schedule_repair: a bounded loop over resolve that sees the state recur counts as the decisive observation', async () => {
  const record = FAMILIES.iterate_schedule_repair.build(7, 0, 'train').find(item => item.semantics.operation === 'blocked');
  const loop = (steps) => ['eval', { code: `let cur = schedule;\nfor (let i = 0; i < ${steps}; i++) { if (conflicts(cur) === 0) break; cur = resolve(cur); }\nconflicts(cur)` }];
  const looped = structuredClone(record);
  looped.curriculum.reference.root = [loop(40), record.curriculum.reference.root.at(-1)];
  const outcome = (await replayReference(looped, TOOLS_PROMPT)).run.outcome;
  assert.equal(outcome.checks.stop_evidence, true, JSON.stringify(outcome.rejection_reasons));
  assert.equal(outcome.accepted, true);
  const short = structuredClone(record);
  short.curriculum.reference.root = [loop(2), record.curriculum.reference.root.at(-1)];
  assert.equal((await replayReference(short, TOOLS_PROMPT)).run.outcome.checks.stop_evidence, false);
});
