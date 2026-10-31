import assert from 'node:assert/strict';
import {test} from 'node:test';
import {checkOracle} from '../dist/teacher/oracle.js';

const solves = async (actual, instance, puzzle) =>
  (await checkOracle(actual, instance, {level: 'normalized', normalization: 'puzzle-solution', puzzle})).accepted;

test('countdown accepts any exact expression over the given numbers', async () => {
  const instance = {numbers: [8, 5, 10, 6], target: 24, use_all: true};
  assert.equal(await solves('10*(8 - 5) - 6', instance, 'countdown'), true);
  assert.equal(await solves('(10 - 6) * (8 - 5) * 2', instance, 'countdown'), false); // 2 is not given
  assert.equal(await solves('6 / (5 - 8 / 10) * ... ', instance, 'countdown'), false);
  assert.equal(await solves('8 * (6 / (10 - 5) * 5) - 10 * 1.6', instance, 'countdown'), false); // repeats, decimal
  assert.equal(await solves('8 * 6 / (10 - 5) + 5', instance, 'countdown'), false); // 14.6, and 5 twice
  assert.equal(await solves('10 * (8 - 5) - 6 = 24', instance, 'countdown'), true);
  assert.equal(await solves('6 * (8 - 5) + 10 - 4', instance, 'countdown'), false);
  // exact rationals: 8 / (3 - 8 / 3) = 24
  assert.equal(await solves('8 / (3 - 8 / 3)', {numbers: [3, 3, 8, 8], target: 24, use_all: true}, 'countdown'), true);
  assert.equal(await solves('10 * 3', {numbers: [3, 10, 7], target: 30}, 'countdown'), true); // at most once
  assert.equal(await solves('10 * 3', {numbers: [3, 10, 7], target: 30, use_all: true}, 'countdown'), false);
  assert.equal(await solves('6 / (5 - 5)', {numbers: [6, 5, 5], target: 1}, 'countdown'), false);
});

test('n-queens accepts any complete non-attacking placement that keeps the fixed queens', async () => {
  const instance = {n: 4, fixed: [[0, 1]]};
  assert.equal(await solves([{row: 0, col: 1}, {row: 1, col: 3}, {row: 2, col: 0}, {row: 3, col: 2}], instance, 'n-queens'), true);
  assert.equal(await solves([[1, 3], [2, 0], [3, 2]], instance, 'n-queens'), true);
  assert.equal(await solves([[0, 2], [1, 0], [2, 3], [3, 1]], instance, 'n-queens'), false); // drops the fixed queen
  assert.equal(await solves([[1, 3], [2, 0]], instance, 'n-queens'), false);
  assert.equal(await solves([[1, 2], [2, 0], [3, 3]], instance, 'n-queens'), false);
});

test('sokoban simulates the moves and requires every box on a goal', async () => {
  const instance = {grid: ['#####', '#@$.#', '#####']};
  assert.equal(await solves('R', instance, 'sokoban'), true);
  assert.equal(await solves('r', instance, 'sokoban'), true);
  assert.equal(await solves('', instance, 'sokoban'), false);
  assert.equal(await solves('RR', instance, 'sokoban'), false); // pushes the box into the wall
  assert.equal(await solves('L', instance, 'sokoban'), false);
  const two = {grid: ['######', '#@$ .#', '#  $ #', '#   .#', '######']};
  assert.equal(await solves('RRDLD R', two, 'sokoban'), true);
  assert.equal(await solves('RRDLDL', two, 'sokoban'), false);
  assert.equal(await solves('RR', two, 'sokoban'), false); // second box still off its goal
});
