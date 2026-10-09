/**
 * The pure parts of the search policies: the crisp defaults and verifiers of the program improver (improveStep/crisp.ts, run
 * as the authored source with `natlang:gepa`), and of the component-search engine (src/optimization/policies.ts).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { AUTHORED_IMPROVER } from '../dist/improvement/authored-source.js';
import { compileVirtualProject } from '../dist/runtime/virtual-project.js';
import * as runtime from '../dist/runtime/node.js';
import { gepaModule } from '../dist/gepa/index.js';
import { crispMove, verifyMove, verifyComponents, verifyParent, closeUnderDependencies, parentChoices, moveFacts, componentFactsOf, checkPolicySettings }
  from '../dist/optimization/policies.js';

const relative = name => AUTHORED_IMPROVER['improveStep/' + name + '.ts'].replace("'../types'", "'./types'");
const compiled = compileVirtualProject({ files: { 'main.ts': relative('crisp'), 'types.ts': AUTHORED_IMPROVER['types.ts'] } }, runtime,
  { constrained: true, target: 'node', modules: { 'natlang:gepa': gepaModule } });
assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
const crisp = compiled.require('main.ts');

const member = (source, quality, extra = {}) => ({ source, quality, parent: source === 'base' ? '' : 'base', ...extra });
const scores = values => values.map((quality, index) => ({ caseId: 'c' + index, quality }));

test('the crisp opportunity is the classification the improver always used, and a verifier bounds the natural-language one', () => {
  const facts = (objective, rows) => ({ objective, rows });
  const pass = (modelCalls = 1) => ({ caseId: 'a', passed: true, modelCalls });
  assert.equal(crisp.crispOpportunity(facts('quality', [pass(), { caseId: 'b', passed: false, failureKind: 'fixture' }])).kind, 'fixture');
  assert.equal(crisp.crispOpportunity(facts('quality', [pass(), { caseId: 'b', passed: false }])).kind, 'quality');
  assert.equal(crisp.crispOpportunity(facts('model-calls', [pass(3)])).kind, 'efficiency');
  assert.equal(crisp.crispOpportunity(facts('model-calls', [pass(1)])).kind, 'none');
  assert.equal(crisp.crispOpportunity(facts('source-size', [pass()])).kind, 'source-size');
  assert.equal(crisp.crispOpportunity(facts('quality', [pass()])).kind, 'none');
  const rows = [pass(), { caseId: 'b', passed: false }];
  assert.throws(() => crisp.verifyOpportunity({ kind: 'fixture', reason: 'x' }, facts('quality', rows)), /exactly when/);
  assert.throws(() => crisp.verifyOpportunity({ kind: 'quality', reason: 'x' }, facts('quality', [pass()])), /failed training row/);
  assert.throws(() => crisp.verifyOpportunity({ kind: 'efficiency', reason: 'x' }, facts('model-calls', [{ caseId: 'b', passed: false }])), /passing/);
  assert.equal(crisp.verifyOpportunity({ kind: 'quality', reason: 'x' }, facts('quality', rows)).kind, 'quality');
});

test('the crisp parent is a frontier draw and the verifier requires a population member', () => {
  const population = [member('base', 0.3, { scores: scores([0.2, 0.5, 0.1]) }), member('x', 0.4, { scores: scores([0.9, 0.1, 0.1]) }), member('z', 0.1, { scores: scores([0.1, 0.1, 0.1]) })];
  for (let seed = 0; seed < 30; seed++) {
    const picked = crisp.crispParent(population, seed);
    assert.ok(['base', 'x'].includes(picked), 'z wins no case, so it is never drawn');
    assert.equal(picked, crisp.crispParent(population, seed));
  }
  const choices = crisp.parentChoices(population, [{ source: 'x', parent: 'base' }, { source: 'y', parent: 'base' }]);
  assert.deepEqual(choices.map(item => [item.id, item.timesParent]), [['base', 2], ['x', 0], ['z', 0]]);
  assert.deepEqual(choices.find(item => item.id === 'x').wonCases, ['c0', 'c2']);
  assert.equal(crisp.verifyParent('x', population), 'x');
  assert.throws(() => crisp.verifyParent('q', population), /population member; choose one of: base, x, z/);
});

test('the crisp incumbent keeps a leading incumbent, ties to the smallest id, and the verifier names the leaders', () => {
  const population = [member('base', 0.5, { cost: 100, modelCalls: 5 }), member('b', 0.5, { cost: 80, modelCalls: 4 }), member('a', 0.5, { cost: 80, modelCalls: 6 }), member('low', 0.2, { cost: 1, modelCalls: 1 })];
  assert.equal(crisp.crispIncumbent(population, 'base', 'quality'), 'base');
  assert.equal(crisp.crispIncumbent(population, 'low', 'quality'), 'a');
  assert.equal(crisp.crispIncumbent(population, 'base', 'source-size'), 'a');
  assert.equal(crisp.crispIncumbent(population, 'base', 'model-calls'), 'b');
  assert.deepEqual(crisp.incumbentChoices(population, 'source-size').map(item => item.id), ['b', 'a']);
  assert.equal(crisp.verifyIncumbent('a', population, 'source-size'), 'a');
  assert.throws(() => crisp.verifyIncumbent('base', population, 'source-size'), /best-quality members: a, b/);
  assert.throws(() => crisp.verifyIncumbent('low', population, 'quality'), /best-quality/);
});

test('the crisp stop follows the declared order and the verifier forbids continuing past the experiment limit', () => {
  const facts = (extra = {}) => ({ disposition: 'rejected', reason: 'r', selectedQuality: 0.5, objective: 'quality', iteration: 0, maxExperiments: 3, recent: [], ...extra });
  assert.deepEqual(crisp.crispStop(facts({ disposition: 'fixture-error', reason: 'broken fixture', selectedQuality: 1 })), { stop: true, reason: 'broken fixture' });
  assert.deepEqual(crisp.crispStop(facts({ selectedQuality: 1 })), { stop: true, reason: 'Declared objective satisfied.' });
  assert.equal(crisp.crispStop(facts({ selectedQuality: 1, objective: 'source-size' })).stop, false);
  assert.deepEqual(crisp.crispStop(facts({ disposition: 'no-hypothesis' })), { stop: true, reason: 'No further evidenced change.' });
  assert.deepEqual(crisp.crispStop(facts({ disposition: 'no-opportunity' })), { stop: true, reason: 'No further evidenced change.' });
  assert.deepEqual(crisp.crispStop(facts({ iteration: 2 })), { stop: true, reason: 'Declared experiments completed.' });
  assert.deepEqual(crisp.crispStop(facts()), { stop: false, reason: 'Continue with a distinct hypothesis.' });
  assert.equal(crisp.verifyStop({ stop: true, reason: 'enough' }, facts()).stop, true, 'stopping early is allowed');
  assert.throws(() => crisp.verifyStop({ stop: false, reason: 'more' }, facts({ iteration: 2 })), /maxExperiments \(3\)/);
});

const candidate = (id, value, extra = {}) => ({ id, value, parents: id === 'base' ? [] : ['base'], train: { results: [] }, validation: { quality: 0.5, results: [{ caseId: 'a', quality: 0.5 }] }, ...extra });

test('the engine move is the former schedule, and compose needs two members', () => {
  for (let iteration = 0; iteration < 12; iteration++) for (const members of [1, 2, 5]) {
    assert.equal(crispMove('gepa', iteration, members), iteration % 4 === 3 && members > 1 ? 'compose' : 'edit');
    assert.equal(crispMove('reflection', iteration, members), 'edit');
  }
  assert.equal(verifyMove('edit', 1), 'edit');
  assert.equal(verifyMove('compose', 2), 'compose');
  assert.throws(() => verifyMove('compose', 1), /population has one member/);
  assert.throws(() => verifyMove('merge', 3), /`edit` or `compose`/);
});

test('engine facts and component bounds', () => {
  const baseline = { k1: { text: 'a' }, k2: { text: 'b' } };
  const population = [candidate('base', baseline), candidate('m1', { k1: { text: 'changed' }, k2: { text: 'b' } }), candidate('m2', { k1: { text: 'a' }, k2: { text: 'edited' } })];
  assert.deepEqual(moveFacts(3, population, baseline).members, [{ id: 'base', changedKeys: [] }, { id: 'm1', changedKeys: ['k1'] }, { id: 'm2', changedKeys: ['k2'] }]);
  assert.deepEqual(parentChoices(population).map(item => [item.id, item.timesParent]), [['base', 2], ['m1', 0], ['m2', 0]]);
  assert.equal(verifyParent('m1', population), 'm1');
  assert.throws(() => verifyParent('nope', population), /population member/);
  const parent = candidate('p', baseline, { train: { results: [{ coverage: new Set(['k1']), quality: 0 }, { coverage: new Set(['k1', 'k2']), quality: 1 }] } });
  assert.deepEqual(componentFactsOf(['k1', 'k2'], parent, { k1: { proposals: 2, accepts: 1 } }).eligible,
    [{ key: 'k1', coveredCases: 2, failingCases: 1, proposals: 2, accepts: 1 }, { key: 'k2', coveredCases: 1, failingCases: 0, proposals: 0, accepts: 0 }]);
  const targets = [{ key: 'a', dependsOn: ['b'] }, { key: 'b', dependsOn: ['c'] }, { key: 'c' }, { key: 'd' }];
  assert.deepEqual(closeUnderDependencies(['a'], targets), ['a', 'b', 'c']);
  assert.deepEqual(closeUnderDependencies(['d', 'b'], targets).sort(), ['b', 'c', 'd']);
  assert.deepEqual(verifyComponents(['a'], ['a', 'd']), ['a']);
  assert.throws(() => verifyComponents([], ['a']), /at least one/);
  assert.throws(() => verifyComponents(['z'], ['a', 'd']), /eligible ones \(a, d\); these are not: z/);
  assert.deepEqual(checkPolicySettings(undefined), {});
  assert.throws(() => checkPolicySettings({ chooseNothing: 'nl' }), /search policies are chooseParent, chooseMove, chooseComponents/);
});
