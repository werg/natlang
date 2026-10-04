import assert from 'node:assert/strict';
import test from 'node:test';
import { wordGames, wordScenarioCases } from '../dist/self-play/word-games.js';
import { IllegalGameAction } from '../dist/self-play/types.js';

const games = new Map(wordGames.map(game => [game.id, game]));
const cases = wordScenarioCases();
const family = id => cases.filter(row => row.family === id);
const json = value => JSON.stringify(value);

test('word-game catalog has two meaning-centered games and four isolated scenario groups each', () => {
  assert.deepEqual(wordGames.map(game => game.id), ['near-synonym', 'taboo-clue']);
  assert.equal(new Set(cases.map(row => row.group)).size, cases.length);
  for (const game of wordGames) assert.equal(family(game.id).length, 4);
});

test('near-synonym challenge keeps the target private and scores the exact selected meaning', () => {
  const game = games.get('near-synonym'), scenario = family(game.id)[0].scenario;
  const state = game.initialize(scenario, 83), before = json(state);
  const setter = game.observe(state, 'challenger'), solver = game.observe(state, 'interpreter');
  assert.ok(setter.challengeChoices.length > 0);
  assert.equal(solver.challengeChoices, undefined);
  assert.equal(solver.targetChoices, undefined);
  assert.ok(solver.choices.length >= 3);
  assert.deepEqual(game.initialize(scenario, 83), state, 'seeded setup is deterministic');
  const otherSeed = game.initialize(scenario, 84);
  assert.notDeepEqual(state.slots.map(slot => slot.word), otherSeed.slots.map(slot => slot.word));

  const chosen = setter.challengeChoices.find(option => option.targetWord === 'skeptical');
  assert.ok(chosen);
  let next = game.apply(state, 'challenger', { type: 'challenge', challengeId: chosen.challengeId });
  assert.equal(game.actor(next), 'interpreter');
  const guessView = game.observe(next, 'interpreter');
  assert.equal(guessView.clue, chosen.clue);
  assert.equal(guessView.targetWord, undefined);
  const answer = guessView.choices.find(option => option.word === chosen.targetWord);
  next = game.apply(next, 'interpreter', { type: 'guess', optionId: answer.optionId });
  assert.deepEqual(game.outcome(next), { scores: { challenger: 0, interpreter: 1 }, reason: 'target interpreted correctly' });
  assert.equal(json(state), before, 'transition leaves input state unchanged');
});

test('taboo clue game exposes forbidden terms but never exposes the secret target to the interpreter', () => {
  const game = games.get('taboo-clue'), scenario = family(game.id)[1].scenario;
  let state = game.initialize(scenario, 29);
  const setterView = game.observe(state, 'cluegiver');
  const solverView = game.observe(state, 'interpreter');
  assert.deepEqual(setterView.forbiddenWords, scenario.taboo);
  assert.deepEqual(solverView.forbiddenWords, scenario.taboo);
  assert.equal(solverView.challengeChoices, undefined);
  assert.equal(solverView.secret, undefined);
  const selected = setterView.challengeChoices.find(item => item.targetWord === 'lend');
  assert.ok(selected);
  state = game.apply(state, 'cluegiver', { type: 'challenge', challengeId: selected.challengeId });
  const view = game.observe(state, 'interpreter');
  assert.equal(view.clue, selected.clue);
  assert.equal(view.targetWord, undefined);
  const targetOption = view.choices.find(choice => choice.word === 'lend');
  state = game.apply(state, 'interpreter', { type: 'guess', optionId: targetOption.optionId });
  assert.equal(game.outcome(state).scores.interpreter, 1);
  assert.equal(game.outcome(state).scores.cluegiver, 0);
});

test('taboo scenarios fail closed if an authored clue contains a forbidden word', () => {
  const game = games.get('taboo-clue'), scenario = structuredClone(family(game.id)[0].scenario);
  scenario.clues[0].text += ' The answer is not a repair.';
  assert.throws(() => game.initialize(scenario, 0), /forbidden word/);
});

test('word games reject invalid actions and keep states immutable and JSON serializable', () => {
  for (const game of wordGames) for (const row of family(game.id)) {
    const first = game.initialize(row.scenario, 17), second = game.initialize(row.scenario, 17);
    const before = json(first);
    assert.deepEqual(first, second);
    assert.doesNotThrow(() => JSON.stringify(first));
    assert.throws(() => game.apply(first, game.seats[1], { type: 'challenge', challengeId: 'missing' }), IllegalGameAction);
    const choice = game.observe(first, game.seats[0]).challengeChoices[0];
    const next = game.apply(first, game.seats[0], { type: 'challenge', challengeId: choice.challengeId });
    assert.equal(json(first), before);
    assert.equal(game.outcome(next), null);
    assert.throws(() => game.apply(next, game.seats[0], { type: 'guess', optionId: 'missing' }), IllegalGameAction);
    const guess = game.legalActions(next, game.seats[1])[0];
    const done = game.apply(next, game.seats[1], guess);
    const outcome = game.outcome(done);
    assert.ok(outcome);
    assert.ok(Object.values(outcome.scores).every(score => score === 0 || score === 1));
  }
});
