// An expected blocked/failed root stop earns its reward only after the obstacle was observed (2026-10-29).
// Before this check every expected-blocked curriculum case accepted an immediate give-up with no work: a reward the RL
// policy could take without attempting the task (AGENTS.md "Task contracts and reward validity").
// Cases come from training seeds only.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { replayReference } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { FAMILIES } from '../scripts/inline-curriculum/families.mjs';

for (const family of ['event_retry', 'child_sufficiency', 'actor_route_replan']) {
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
