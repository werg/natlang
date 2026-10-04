import test from 'node:test';
import assert from 'node:assert/strict';
import { playMatch, replayMatch } from '../dist/self-play/arena.js';
import { IllegalGameAction } from '../dist/self-play/types.js';
const game = { id: 'fixture', revision: '1', seats: ['a', 'b'], rules: 'Pick the secret you know.',
  initialize: scenario => ({ secret: scenario, turn: 0, done: false }), actor: () => 'a',
  observe: state => ({ answer: state.secret }), legalActions: () => [{ guess: 7 }],
  apply(state, seat, action) {
    if (action.guess !== state.secret) throw new IllegalGameAction('wrong guess');
    return { ...state, done: true };
  }, outcome: state => state.done ? { scores: { a: 1, b: 0 }, reason: 'correct guess' } : null };
const policies = { a: { id: 'p-a', decide: async view => ({ guess: view.observation.answer }) },
  b: { id: 'p-b', decide: async () => ({ guess: 0 }) } };

test('host replays every view, action, state and exact score without model calls', async () => {
  const match = await playMatch({ game, scenario: 7, seed: 1, policies, maxDecisions: 5 });
  assert.equal(match.disposition, 'completed');
  assert.deepEqual((await replayMatch(game, match)), match);
  for (const mutate of [m => { m.frames[0].view.observation.answer = 8; },
    m => { m.outcome.scores.a = .2; }, m => { m.policies.a = 'forged'; },
    m => { m.identity = 'forged'; }]) {
    const bad = structuredClone(match); mutate(bad);
    await assert.rejects(replayMatch(game, bad), /replay/);
  }
});

test('incomplete matches and engine errors never become draws or model wins', async () => {
  const noEnd = { ...game, apply: state => ({ ...state, turn: state.turn + 1 }), outcome: () => null };
  const capped = await playMatch({ game: noEnd, scenario: 7, seed: 1, policies, maxDecisions: 2 });
  assert.equal(capped.disposition, 'incomplete'); assert.equal(capped.outcome, null);
  const broken = await playMatch({ game: { ...game, apply: () => { throw Error('engine bug'); } },
    scenario: 7, seed: 1, policies, maxDecisions: 2 });
  assert.equal(broken.disposition, 'incomplete'); assert.match(broken.error, /engine bug/);
  await assert.rejects(replayMatch(game, capped), /complete legal matches/);
});

test('illegal moves retain their exact private evidence without invented teammate scores', async () => {
  const match = await playMatch({ game, scenario: 7, seed: 1, maxDecisions: 2,
    policies: { ...policies, a: { id: 'bad', decide: async () => ({ guess: 0 }) } } });
  assert.equal(match.disposition, 'illegal-action'); assert.equal(match.outcome, null);
  assert.deepEqual(match.rejected.action, { guess: 0 });
  assert.equal(match.rejected.seat, 'a');
});

test('a policy cannot mutate host state or saved observation through its view', async () => {
  const match = await playMatch({ game, scenario: 7, seed: 1, maxDecisions: 2,
    policies: { ...policies, a: { id: 'mutator', decide: async view => {
      view.observation.answer = 42; return { guess: 7 };
    } } } });
  assert.equal(match.disposition, 'completed');
  assert.equal(match.frames[0].view.observation.answer, 7);
  assert.equal(match.initial.secret, 7);
});
