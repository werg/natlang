import test from 'node:test';
import assert from 'node:assert/strict';
import { createCrosswordEpisodes } from '../scripts/skills/build-crossword-episodes.mjs';
import { enumerateCrosswordSolutions, scoreCrosswordObjective, verifyCrosswordReference } from '../src/skills/crossword-objective.ts';

const card = (family, clue, length, constraints = []) => ({ schema: 'natlang.crossword-csp/1', family, rows: 1, cols: 1,
  instructions: 'Solve the clue.', slots: [{ id: 'x', clue, length }], constraints });

test('project episodes have held-out clue styles/groups and independently verified references', () => {
  const { episodes, audit } = createCrosswordEpisodes();
  assert.equal(episodes.length, 9);
  assert.deepEqual(new Set(audit.map(x => x.family)), new Set(['crossword', 'mini-cryptic', 'word-lattice']));
  for (const episode of episodes) {
    assert.notEqual(episode.support.cases[0].group, episode.query.cases[0].group);
    const instance = JSON.parse(episode.query.cases[0].args[0]);
    const solutions = enumerateCrosswordSolutions(instance);
    assert.equal(solutions.length, 1);
    assert.equal(episode.query.cases[0].expected.solutionCount, 1);
    assert.equal(scoreCrosswordObjective(instance, { fills: solutions[0] }, episode.query.cases[0].expected).quality, 1);
  }
});

test('crossword scoring gates contradictions and malformed/unrecognized answers to zero', () => {
  const episode = createCrosswordEpisodes().episodes.find(e => e.family === 'word-constraints-crossword');
  const instance = JSON.parse(episode.query.cases[0].args[0]);
  const expected = episode.query.cases[0].expected;
  const partial = scoreCrosswordObjective(instance, { fills: { 'across-1': 'DOG' } }, expected);
  assert.equal(partial.gates.feasible, true);
  assert.equal(partial.quality, 0.75 / instance.slots.length);
  assert.equal(scoreCrosswordObjective(instance, { fills: { 'across-1': 'DOG', 'down-1': 'ORE' } }, expected).quality, 0);
  assert.equal(scoreCrosswordObjective(instance, { fills: { 'across-1': 'CAT' } }, expected).quality, 0);
  assert.equal(scoreCrosswordObjective(instance, { fills: { 'not-a-slot': 'DOG' } }, expected).quality, 0);
});

test('ambiguous clue domains are explicitly enumerated and every feasible completion is accepted', () => {
  const instance = card('word-lattice', 'An adult sheep', 3);
  const reference = verifyCrosswordReference(instance);
  assert.equal(reference.solutionCount, 2);
  assert.deepEqual(new Set(reference.accepted.map(row => row.x)), new Set(['RAM', 'EWE']));
  assert.equal(scoreCrosswordObjective(instance, { fills: { x: 'RAM' } }, reference).quality, 1);
  assert.equal(scoreCrosswordObjective(instance, { fills: { x: 'EWE' } }, reference).quality, 1);
});

test('mini-cryptic answers must match authored letter operations, not only a definition', () => {
  const valid = card('mini-cryptic', 'Add C before AT to name a feline', 3);
  const ref = verifyCrosswordReference(valid);
  assert.deepEqual(ref.accepted, [{ x: 'CAT' }]);
  assert.equal(scoreCrosswordObjective(valid, { fills: { x: 'CAT' } }, ref).quality, 1);
  assert.equal(scoreCrosswordObjective(valid, { fills: { x: 'TAC' } }, ref).quality, 0);
  const unsupported = card('mini-cryptic', 'Put C after AT to name a feline', 3);
  assert.equal(scoreCrosswordObjective(unsupported, { fills: { x: 'ATC' } }, ref).quality, 0);
});

test('crossword structure must represent every physical intersection as a crossing', () => {
  const episode = createCrosswordEpisodes().episodes.find(e => e.family === 'word-constraints-crossword');
  const instance = JSON.parse(episode.query.cases[0].args[0]);
  instance.constraints.pop();
  assert.equal(scoreCrosswordObjective(instance, { fills: {} }, episode.query.cases[0].expected).quality, 0);
});
