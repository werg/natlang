/**
 * One GEPA selection implementation, tested once (src/gepa), with both callers held to it.
 *
 * The component-search engine (src/optimization) adapts its candidates to `src/gepa`; the program-improver application
 * reaches the same functions as the platform module `natlang:gepa`, because callable folders may import only their
 * siblings and packages (runtime/modules.ts). This file holds the spec the old twin test held, now to the single module:
 *   1. A member is on the frontier when it has the highest quality on at least one case (a missing result counts as -1).
 *   2. The frontier lists its winners sorted, a member once per case it wins.
 *   3. The parent draw picks one of those winners, deterministically for a seed.
 *   4. Pruning keeps the protected members, then the frontier, then higher quality, within the limit.
 * And it checks there is no second copy: the application bundle holds no frontier, draw or prune code of its own.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as gepa from '../dist/gepa/index.js';
import { frontierParents, memberOf, meanBetter } from '../dist/optimization/strategies/gepa.js';
import { AUTHORED_IMPROVER } from '../dist/improvement/authored-source.js';
import { builtinModule } from '../dist/runtime/modules.js';
import '../dist/runtime/node.js';

const populations = [
  { cases: ['a', 'b', 'c'], quality: { base: [0.2, 0.5, 0.1], x: [0.9, 0.1, 0.1], y: [0.9, 0.5, 0.0], z: [0.1, 0.1, 0.1] } },
  { cases: ['a', 'b'], quality: { base: [0.5, 0.5], x: [0.5, 0.5], y: [0.5, 0.4] } },
  { cases: ['a'], quality: { base: [0.3], x: [0.7], y: [0.7], z: [0.7] } },
  { cases: ['a', 'b', 'c', 'd'], quality: { base: [0, 0, 0, 0], p: [1, 0, 0, 0], q: [0, 1, 0, 0], r: [0, 0, 1, 0], s: [0, 0, 0, 0.5] } },
];
const members = ({ cases, quality }) => Object.entries(quality).map(([id, scores]) =>
  ({ id, quality: scores.reduce((sum, value) => sum + value, 0) / scores.length, scores: cases.map((caseId, at) => ({ caseId, quality: scores[at] })) }));
const candidates = ({ cases, quality }) => Object.entries(quality).map(([id, scores]) =>
  ({ id, parents: id === 'base' ? [] : ['base'], validation: { quality: scores.reduce((sum, value) => sum + value, 0) / scores.length,
    results: cases.map((caseId, at) => ({ caseId, quality: scores[at] })) } }));
const winnersByRule = ({ cases, quality }) => {
  const winners = [];
  cases.forEach((_, index) => {
    const best = Math.max(...Object.values(quality).map(scores => scores[index]));
    for (const [id, scores] of Object.entries(quality)) if (scores[index] === best) winners.push(id);
  });
  return winners.sort();
};

test('the frontier lists the per-case winners, and the engine adapter returns the same list', () => {
  for (const world of populations) {
    assert.deepEqual(gepa.frontier(members(world)), winnersByRule(world));
    assert.deepEqual(frontierParents(candidates(world)), winnersByRule(world));
    assert.deepEqual(candidates(world).map(memberOf).map(item => item.id), Object.keys(world.quality));
  }
});

test('the parent draw picks a per-case winner for any seed, deterministically', () => {
  for (const world of populations) {
    const winners = new Set(winnersByRule(world));
    for (let seed = 0; seed < 60; seed++) {
      const first = gepa.drawParent(members(world), seed), again = gepa.drawParent(members(world), seed);
      assert.ok(winners.has(first.id), `seed ${seed} picked ${first.id}, which is not a winner of ${[...winners]}`);
      assert.deepEqual(first, again);
    }
  }
});

test('draw is the xorshift32 step the engine always used, and a chain of draws is reproducible', () => {
  const reference = seed => { let x = seed >>> 0; x ^= x << 13; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return { value: x / 4294967296, next: x }; };
  let seed = 44;
  for (let step = 0; step < 20; step++) {
    const got = gepa.draw(seed), want = reference(seed);
    assert.deepEqual(got, want);
    assert.ok(got.value >= 0 && got.value < 1);
    seed = got.next;
  }
  assert.equal(gepa.pick(['a', 'b', 'c'], 0.99), 'c');
  assert.equal(gepa.pick(['a', 'b', 'c'], 0), 'a');
  assert.throws(() => gepa.pick([], 0.5), /nothing to choose from/);
});

test('prune keeps the protected members, then the frontier, then higher quality, and keeps member order', () => {
  for (const world of populations) {
    const all = members(world), winners = new Set(winnersByRule(world));
    for (let limit = 1; limit <= all.length; limit++) {
      const kept = gepa.prune(all, ['base'], limit);
      assert.ok(kept.length <= Math.max(limit, 1));
      assert.ok(kept.some(item => item.id === 'base'));
      assert.deepEqual(kept.map(item => item.id), all.filter(item => kept.includes(item)).map(item => item.id), 'order is kept');
      const rest = kept.filter(item => item.id !== 'base').map(item => winners.has(item.id));
      const dropped = all.filter(item => !kept.includes(item)).map(item => winners.has(item.id));
      if (dropped.some(Boolean)) assert.ok(rest.every(Boolean), 'a frontier member is dropped only when every kept member is on the frontier');
    }
  }
  assert.deepEqual(gepa.prune(members(populations[0]), ['base', 'z'], 2).map(item => item.id), ['base', 'z']);
  assert.throws(() => gepa.prune([], [], 0), /positive integer/);
});

test('better, leaders and the crisp incumbent follow quality, then the objective measure, then the current incumbent', () => {
  const a = { quality: 0.5, cost: 100, modelCalls: 3 }, b = { quality: 0.5, cost: 80, modelCalls: 4 }, c = { quality: 0.6, cost: 500, modelCalls: 9 };
  assert.equal(gepa.better(c, a), true);
  assert.equal(gepa.better(a, c), false);
  assert.equal(gepa.better(b, a), false, 'equal quality is not better without a measure');
  assert.equal(gepa.better(b, a, { measure: gepa.objectiveMeasure('source-size') }), true);
  assert.equal(gepa.better(a, b, { measure: gepa.objectiveMeasure('source-size') }), false);
  assert.equal(gepa.better(a, b, { measure: gepa.objectiveMeasure('model-calls') }), true);
  assert.equal(gepa.better({ quality: 0.5 }, { quality: 0.5, cost: 1 }, { measure: gepa.objectiveMeasure('source-size') }), false, 'an unknown measure is not better');
  assert.equal(gepa.better(a, c, { eligible: item => item.quality < 0.6 }), true, 'eligibility comes first');
  const pool = [{ id: 'm1', quality: 0.5, cost: 100, scores: [] }, { id: 'm2', quality: 0.5, cost: 80, scores: [] }, { id: 'm3', quality: 0.4, cost: 1, scores: [] }];
  assert.deepEqual(gepa.leaders(pool), ['m1', 'm2']);
  assert.deepEqual(gepa.leaders(pool, 'source-size'), ['m2']);
  assert.equal(gepa.chooseIncumbent(pool, 'm1'), 'm1');
  assert.equal(gepa.chooseIncumbent(pool, 'm3'), 'm1');
  assert.equal(gepa.chooseIncumbent(pool, 'm1', 'source-size'), 'm2');
});

test('the engine comparison is the shared one: eligibility, quality, then the declared tie-break', () => {
  const candidate = (id, quality, calls) => ({ id, parents: [], validation: { gatesPassed: true, quality,
    results: [{ caseId: 'a', quality, latencyMs: 1, usage: { modelCalls: calls, cost: 0 } }] } });
  const fewer = candidate('fewer', 0.5, 1), more = candidate('more', 0.5, 3), best = candidate('best', 0.9, 9);
  assert.equal(meanBetter(best, more, {}), true);
  assert.equal(meanBetter(fewer, more, {}), false);
  assert.equal(meanBetter(fewer, more, { tieBreak: 'modelCalls' }), true);
  assert.equal(meanBetter(more, fewer, { tieBreak: 'modelCalls' }), false);
  assert.equal(meanBetter(best, more, { maxModelCalls: 5 }), false, 'an over-limit candidate is not eligible');
});

test('the application reaches the module as natlang:gepa and holds no second copy', () => {
  assert.equal(builtinModule('natlang:gepa'), gepa.gepaModule);
  for (const name of ['improveStep/parents.ts', 'improveStep/selection.ts', 'improveStep/rewriteProgram.nl', 'improveStep/rewriteProgram/bookkeeping.ts'])
    assert.equal(AUTHORED_IMPROVER[name], undefined, `${name} is gone`);
  const typescript = Object.entries(AUTHORED_IMPROVER).filter(([name]) => name.endsWith('.ts'));
  for (const [name, text] of typescript) {
    assert.doesNotMatch(text, /<<\s*13|>>>\s*17/, `${name} holds no xorshift draw of its own`);
    assert.doesNotMatch(text, /function\s+(frontierParent|prune|best)\b/, `${name} holds no selection function of its own`);
  }
  const users = typescript.filter(([, text]) => /from\s+'natlang:gepa'/.test(text)).map(([name]) => name).sort();
  assert.deepEqual(users, ['componentSearchStep/rewriteComponents.ts', 'improveStep/crisp.ts', 'improveStep/experiment.ts', 'improveStep/population.ts']);
});
