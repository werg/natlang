import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import ts from 'typescript';
import { ComponentSelector } from '../dist/optimization/vendor/ax-gepa/gepaSelection.js';
import { getUpdateGroup } from '../dist/optimization/vendor/ax-gepa/gepaDependencies.js';
import * as adaptedPareto from '../dist/optimization/vendor/ax-gepa/paretoUtils.js';

const require = createRequire(import.meta.url);
const root = new URL('../../', import.meta.url);
function upstream(relativePath) {
  const path = new URL(relativePath, root);
  const source = readFileSync(path, 'utf8');
  const js = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText;
  const module = { exports: {} };
  new Function('exports', 'require', 'module', js)(module.exports, require, module);
  return module.exports;
}
const oldSelector = upstream('vendor/ax-gepa/src/ax/dsp/optimizers/gepaSelection.ts');
const oldDependencies = upstream('vendor/ax-gepa/src/ax/dsp/optimizers/gepaDependencies.ts');
const oldPareto = upstream('vendor/ax-gepa/src/ax/dsp/optimizers/paretoUtils.ts');

function scriptedRandom(values) {
  let index = 0;
  return () => values[index++ % values.length];
}
function sortedMap(map) { return Object.fromEntries([...map].sort(([a], [b]) => a - b)); }

test('native component bandit matches pinned Ax selections and state under scripted RNG', () => {
  const oldTargets = [{ id: 'alpha' }, { id: 'beta' }, { id: 'gamma' }];
  const targets = oldTargets.map(({ id }) => ({ key: id }));
  const initial = { alpha: { proposals: 4, accepts: 1, lastAcceptIter: 2, stagnation: 2 },
    beta: { proposals: 1, accepts: 1, lastAcceptIter: 5, stagnation: 0 } };
  const old = new oldSelector.AxGEPAComponentSelector(oldTargets, initial);
  const current = new ComponentSelector(targets, initial);
  const oldRandom = scriptedRandom([0.7, 0.31, 0.02, 0.5, 0.99, 0.18, 0.42]);
  const newRandom = scriptedRandom([0.7, 0.31, 0.02, 0.5, 0.99, 0.18, 0.42]);
  for (let iteration = 0; iteration < 24; iteration++) {
    const before = old.pick(iteration, oldRandom).id;
    const after = current.pick(iteration, newRandom).key;
    assert.equal(after, before, `selection at iteration ${iteration}`);
    old.recordProposal(before); current.recordProposal(after);
    const accepted = iteration % 4 === 1;
    old.recordResult(before, accepted, iteration); current.recordResult(after, accepted, iteration);
    assert.deepEqual(current.snapshot(), old.snapshot());
  }
});

test('native dependency grouping matches pinned Ax depth first order and cycle handling', () => {
  const oldTargets = [{ id: 'root', dependsOn: ['left', 'missing', 'right'] },
    { id: 'left', dependsOn: ['shared'] }, { id: 'shared', dependsOn: ['root'] },
    { id: 'right', dependsOn: ['shared'] }, { id: 'unused' }];
  const targets = oldTargets.map(({ id, dependsOn }) => ({ key: id, dependsOn }));
  const before = oldDependencies.getGEPAUpdateGroup(oldTargets[0], oldTargets).map(item => item.id);
  const after = getUpdateGroup(targets[0], targets).map(item => item.key);
  assert.deepEqual(after, before);
  assert.deepEqual(after, ['root', 'left', 'shared', 'right']);
});

test('native Pareto frontier and utilities match the pinned pure operations', () => {
  const items = [
    { idx: 0, scores: { quality: 0.6, cost: 0.7 } },
    { idx: 1, scores: { quality: 0.8, cost: 0.4 } },
    { idx: 2, scores: { quality: 0.5, cost: 0.5 } },
    { idx: 3, scores: { quality: 0.8, cost: 0.4 } },
    { idx: 4, scores: { quality: 0.7, cost: 0.6 } },
  ];
  assert.deepEqual(adaptedPareto.buildParetoFront(items), oldPareto.buildParetoFront(items));
  for (const [a, b, epsilon] of [
    [{ quality: 0.9, cost: 0.7 }, { quality: 0.8, cost: 0.6 }, 0],
    [{ quality: 0.9, cost: 0.7 }, { quality: 0.9, cost: 0.7 }, 0],
    [{ quality: 0.901, cost: 0.69 }, { quality: 0.9, cost: 0.7 }, 0.02],
  ]) {
    assert.equal(adaptedPareto.dominatesVector(a, b), oldPareto.dominatesVector(a, b));
    assert.equal(adaptedPareto.dominatesVectorEps(a, b, epsilon), oldPareto.dominatesVectorEps(a, b, epsilon));
  }
  const front = oldPareto.buildParetoFront(items).map(({ idx, scores }) => ({ idx, scores }));
  assert.deepEqual(sortedMap(adaptedPareto.computeCrowdingDistances(front)), sortedMap(oldPareto.computeCrowdingDistances(front)));
  const vectors = front.map(item => item.scores);
  assert.equal(adaptedPareto.hypervolume2D(vectors), oldPareto.hypervolume2D(vectors));
});
