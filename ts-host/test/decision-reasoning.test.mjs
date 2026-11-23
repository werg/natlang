// Reasoned decision readouts (DECISIONS 2026-11-23 "reasoned decisions"): `readout: decision` samples the model's own
// reasoning on the readout's request, then scores the options after each reasoning; its probabilities are the mean of
// those conditionals. `readout: instant` scores right away. One client for every engine (model/decision-scoring.ts).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, decide, loadVirtualNatlang } from '../dist/index.js';
import { DECISION_SAMPLES, decisionSamples } from '../dist/native/decision.js';
import { batchScoreItem, promptLogprobRequest, scoresFromBatchResult } from '../dist/model/decision-scoring.js';
import { trajectoryTurn } from '../dist/runtime/trajectory.js';

const source = readout => `---\nargs: { invite: string }\nreturns: '"accept" | "decline"'\nreadout: ${readout}\n---\nAnswer the invitation.\n`;

/** A driver that reasons (sample i writes "think i") and scores each option after the reasoning it is given. */
function reasoningDriver(conditional) {
  const turns = [], scored = [];
  const driver = Object.assign(async request => {
    turns.push(request);
    return { text: '"accept"', reasoning: `think ${request.decision_reasoning?.index}`, completion_tokens: 5 };
  }, { decide: async request => {
    scored.push(request);
    return { log_probs: conditional(request.reasoning).map(Math.log), tokens: [2, 2] };
  } });
  return { driver, turns, scored };
}

test('readout: decision reasons first, scores the options after each reasoning and averages the conditionals', async () => {
  const conditionals = { 'think 0': [0.9, 0.1], 'think 1': [0.2, 0.8], 'think 2': [0.1, 0.9], 'think 3': [0.3, 0.7] };
  const { driver, turns, scored } = reasoningDriver(reasoning => conditionals[reasoning]);
  const traces = [];
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, trace: trace => traces.push(trace) });
  const fn = loadVirtualNatlang({ 'invite.nl': source('decision') }, 'invite.nl');
  const decision = await runtime.run(() => decide(fn, 'Thu 10:00'));
  assert.equal(turns.length, DECISION_SAMPLES);
  // the reasoning turns are generations on the readout's own messages, no tools, each marked and seeded apart
  assert.ok(turns.every(request => request.tools.length === 0 && request.messages === turns[0].messages));
  assert.deepEqual(turns.map(request => request.decision_reasoning), [0, 1, 2, 3].map(index => ({ index, samples: 4 })));
  assert.match(turns[0].messages.at(-1).content, /exactly one of these JSON values/);
  assert.deepEqual(scored.map(request => request.reasoning).sort(), ['think 0', 'think 1', 'think 2', 'think 3']);
  assert.ok(scored.every(request => request.messages === scored[0].messages && request.reasoning_sample.samples === 4));
  assert.equal(scored.filter(request => request.reasoning_sample.action).length, 1, 'exactly one action sample');
  assert.equal(decision.value, 'decline');
  assert.equal(decision.reasoned, 4);
  assert.deepEqual(decision.probabilities.map(item => +item.probability.toFixed(6)), [0.375, 0.625]);
  const event = traces.flatMap(trace => trace.events).find(item => item.kind === 'decision_readout' && item.phase === 'scored');
  assert.deepEqual(event.reasoning.map(item => item.text), ['think 0', 'think 1', 'think 2', 'think 3']);
  assert.equal(event.reasoned.samples, 4);
  assert.deepEqual(event.reasoned.conditionals.map(row => row.map(p => +p.toFixed(6))), Object.values(conditionals));
  // log_probs stay consistent with the reported probabilities for every consumer that renormalizes them
  assert.deepEqual(event.log_probs.map(Math.exp).map(p => +p.toFixed(6)), [0.375, 0.625]);
});

test('readout: instant, decisionSamples 0 and temperature 0 bound the reasoning', async () => {
  const fn = loadVirtualNatlang({ 'invite.nl': source('instant') }, 'invite.nl');
  let probe = reasoningDriver(() => [0.4, 0.6]);
  let decision = await createNatlangRuntime({ model: { driver: probe.driver }, seed: { mode: 'backend' } }).run(() => decide(fn, 'x'));
  assert.equal(probe.turns.length, 0); assert.equal(probe.scored.length, 1); assert.equal(probe.scored[0].reasoning, undefined);
  assert.equal(decision.reasoned, undefined); assert.equal(decision.value, 'decline');
  const reasoned = loadVirtualNatlang({ 'invite.nl': source('decision') }, 'invite.nl');
  probe = reasoningDriver(() => [0.4, 0.6]);
  await createNatlangRuntime({ model: { driver: probe.driver, decisionSamples: 0 }, seed: { mode: 'backend' } }).run(() => decide(reasoned, 'x'));
  assert.equal(probe.turns.length, 0);
  probe = reasoningDriver(() => [0.4, 0.6]);
  decision = await createNatlangRuntime({ model: { driver: probe.driver, temperature: 0 }, seed: { mode: 'backend' } }).run(() => decide(reasoned, 'x'));
  assert.equal(probe.turns.length, 1); assert.equal(decision.reasoned, 1);
  assert.equal(decisionSamples(undefined, undefined), DECISION_SAMPLES);
  assert.equal(decisionSamples(2, 0.7), 2);
  assert.throws(() => decisionSamples(-1), /whole number of reasoning samples/);
});

test('a sampling scorer chooses on the action sample only; that choice is the answer', async () => {
  const { driver, scored } = reasoningDriver(() => [0.9, 0.1]);
  const decideScores = driver.decide;
  driver.decide = async request => ({ ...await decideScores(request), ...(request.reasoning_sample.action ? { chosen: 1 } : {}) });
  const fn = loadVirtualNatlang({ 'invite.nl': source('decision') }, 'invite.nl');
  const decision = await createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' } }).run(() => decide(fn, 'x'));
  assert.equal(decision.value, 'decline', 'the sampled choice of the action sample, not the mixture argmax');
  assert.equal(scored.length, 4);
});

test('a scorer that ignores reasoning is refused, and the function reads out instantly from then on', async () => {
  const turns = [], scored = [];
  const driver = Object.assign(async request => { turns.push(request); return { text: '"accept"', reasoning: 'r' }; }, {
    decide: async request => {
      scored.push(request);
      return scoresFromBatchResult({ log_probs: [-1, -2] }, request);
    } });
  const fn = loadVirtualNatlang({ 'invite.nl': source('decision') }, 'invite.nl');
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' } });
  const first = await runtime.run(() => decide(fn, 'x'));
  assert.equal(first.value, 'accept'); assert.equal(first.reasoned, undefined);
  const before = turns.length;
  await runtime.run(() => decide(fn, 'y'));
  assert.equal(turns.length, before, 'no reasoning is sampled for a scorer that cannot use it');
});

test('every engine route carries the reasoning; the batch result must echo it', () => {
  assert.equal(batchScoreItem({ messages: [], options: ['"a"'], reasoning: 'R', reasoning_sample: { index: 0, samples: 1, action: true } }).reasoning, 'R');
  assert.equal('reasoning_sample' in batchScoreItem({ messages: [], options: ['"a"'], reasoning_sample: { index: 0, samples: 1, action: true } }), false);
  assert.equal('reasoning' in batchScoreItem({ messages: [], options: ['"a"'] }), false);
  const body = promptLogprobRequest([{ role: 'user', content: 'q' }], '"a"', {}, false, 'R');
  assert.deepEqual(body.messages.at(-1), { role: 'assistant', content: '"a"', reasoning: 'R', reasoning_content: 'R' });
  assert.throws(() => scoresFromBatchResult({ log_probs: [-1, -2] }, { options: ['a', 'b'], reasoning: 'R' }), /^Error: decision-reasoning-unsupported/);
  assert.deepEqual(scoresFromBatchResult({ log_probs: [-1, -2], reasoning: true }, { options: ['a', 'b'], reasoning: 'R' }).log_probs, [-1, -2]);
});

test('a reasoning turn is recorded as decision reasoning, never as an action', () => {
  const frame = trajectoryTurn({ messages: [], tools: [], seed: null, max_tokens: 10, decision_reasoning: { index: 1, samples: 4 } },
    { text: '"a"', reasoning: 'R' });
  assert.equal(frame.phase, 'decision_reasoning');
  assert.deepEqual(frame.decision_reasoning, { index: 1, samples: 4 });
  assert.equal(trajectoryTurn({ messages: [], tools: [], seed: null, max_tokens: 10 }, { text: 'x' }).phase, 'action');
});

test('a reasoning cut off by its token limit before it closed is still the reasoning the options are scored after', async () => {
  const scored = [];
  const driver = Object.assign(async () => ({ text: 'The calendar is blocked, so', truncated: true }), {
    decide: async request => { scored.push(request.reasoning); return { log_probs: [-1, -2] }; } });
  const fn = loadVirtualNatlang({ 'invite.nl': source('decision') }, 'invite.nl');
  await createNatlangRuntime({ model: { driver, decisionSamples: 1 }, seed: { mode: 'backend' } }).run(() => decide(fn, 'x'));
  assert.deepEqual(scored, ['The calendar is blocked, so']);
});
