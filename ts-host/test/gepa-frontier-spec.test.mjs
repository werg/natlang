/**
 * One spec for the GEPA per-case frontier, held by both implementations.
 *
 * The runtime has `frontierParents` (src/optimization/strategies/gepa.ts). The program-improver application has a crisp twin
 * (applications/program-improver/improveStep/parents.ts `frontierParent`, selection.ts `prune`). The twin cannot import the
 * runtime: callable-folder code may import only its sibling items and packages (runtime/modules.ts), and the improver is
 * itself a program the runtime improves, so it carries its own copy. This test runs both on the same populations and
 * holds them to the spec:
 *   1. A candidate is on the frontier when it has the highest quality on at least one case (a missing result counts as -1).
 *   2. The runtime lists the frontier's winners, sorted (a candidate once per case it wins).
 *   3. The twin picks one of those winners, deterministically for a seed, and `prune` keeps the frontier first.
 * The two differ in one declared way: the runtime reads the case list from the first candidate, the twin from the union of
 * all members; they agree whenever all candidates are evaluated on the same cases, which a search guarantees.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { frontierParents } from '../dist/optimization/strategies/gepa.js';

const load = async name => {
  const source = readFileSync(new URL(`../../applications/program-improver/improveStep/${name}.ts`, import.meta.url), 'utf8');
  const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
  return import(`data:text/javascript;base64,${Buffer.from(js).toString('base64')}`);
};
const { frontierParent } = await load('parents');
const { prune } = await load('selection');

const populations = [
  { cases: ['a', 'b', 'c'], quality: { base: [0.2, 0.5, 0.1], x: [0.9, 0.1, 0.1], y: [0.9, 0.5, 0.0], z: [0.1, 0.1, 0.1] } },
  { cases: ['a', 'b'], quality: { base: [0.5, 0.5], x: [0.5, 0.5], y: [0.5, 0.4] } },
  { cases: ['a'], quality: { base: [0.3], x: [0.7], y: [0.7], z: [0.7] } },
  { cases: ['a', 'b', 'c', 'd'], quality: { base: [0, 0, 0, 0], p: [1, 0, 0, 0], q: [0, 1, 0, 0], r: [0, 0, 1, 0], s: [0, 0, 0, 0.5] } },
];
const runtimePopulation = ({ cases, quality }) => Object.entries(quality).map(([id, scores]) =>
  ({ id, validation: { results: cases.map((caseId, index) => ({ caseId, quality: scores[index] })) } }));
const appPopulation = ({ cases, quality }) => Object.entries(quality).map(([source, scores], index) =>
  ({ source, parent: index === 0 ? '' : 'base', quality: scores.reduce((sum, value) => sum + value, 0) / scores.length,
    scores: cases.map((caseId, at) => ({ caseId, quality: scores[at] })) }));
const winnersByRule = ({ cases, quality }) => {
  const winners = [];
  cases.forEach((_, index) => {
    const best = Math.max(...Object.values(quality).map(scores => scores[index]));
    for (const [id, scores] of Object.entries(quality)) if (scores[index] === best) winners.push(id);
  });
  return winners.sort();
};

test('the runtime frontier lists the per-case winners', () => {
  for (const world of populations) assert.deepEqual(frontierParents(runtimePopulation(world)), winnersByRule(world));
});

test('the improver twin picks a per-case winner for any seed and prune keeps the frontier first', () => {
  for (const world of populations) {
    const winners = new Set(winnersByRule(world));
    const members = appPopulation(world);
    for (let seed = 0; seed < 40; seed++) {
      const picked = frontierParent(members, seed);
      assert.ok(winners.has(picked), `seed ${seed} picked ${picked}, which is not a winner of ${[...winners]}`);
      assert.equal(frontierParent(members, seed), picked, 'the pick is deterministic for a seed');
    }
    const kept = prune(members, 'base', members.length).map(member => member.source);
    const rest = kept.filter(source => source !== 'base');
    const frontierRank = rest.map(source => winners.has(source));
    assert.deepEqual(frontierRank, [...frontierRank].sort((left, right) => Number(right) - Number(left)), 'frontier members precede the others');
  }
});
