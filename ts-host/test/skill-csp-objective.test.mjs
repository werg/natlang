import test from 'node:test';
import assert from 'node:assert/strict';
const objectiveModule = process.env.CSP_OBJECTIVE_MODULE ?? '../dist/skills/csp-objective.js';
const { cspProgressBound, scoreCspProgress, solveFiniteCsp, checkCspAssignment } = await import(objectiveModule);
import { buildCspPacket } from '../scripts/skills/build-csp-episodes.mjs';

const unique = {
  schema: 'natlang.finite-csp/1', template: 'test-three-slot', narrative: 'Assign distinct ordered slots.',
  variables: [{ id: 'a', domain: [1, 2, 3] }, { id: 'b', domain: [1, 2, 3] }, { id: 'c', domain: [1, 2, 3] }],
  constraints: [
    { kind: 'allDifferent', vars: ['a', 'b', 'c'] },
    { kind: 'sum', vars: ['a', 'b'], op: 'eq', value: 3 },
    { kind: 'order', before: 'a', after: 'b', op: 'lt' },
    { kind: 'equal', variable: 'c', value: 3 },
  ],
};

test('finite CSP validates rules and bounded search distinguishes unique, multiple, unsat, and limit', () => {
  assert.deepEqual(solveFiniteCsp(unique).solutions, [{ a: 1, b: 2, c: 3 }]);
  assert.equal(solveFiniteCsp(unique).status, 'unique');
  assert.equal(solveFiniteCsp({ ...unique, constraints: unique.constraints.slice(0, 1) }).status, 'multiple');
  assert.equal(solveFiniteCsp({ ...unique, constraints: [...unique.constraints, { kind: 'equal', variable: 'a', value: 3 }] }).status, 'unsat');
  assert.equal(solveFiniteCsp(unique, { maxNodes: 1 }).status, 'limit');
  assert.equal(checkCspAssignment(unique, { a: 1, b: 2 }).valid, true);
  assert.equal(checkCspAssignment(unique, { a: 2 }).valid, true); // local consistency can miss cross-constraint dead ends; the scorer searches extensions.
});

test('progress reward is host-computed, requires unique binding and a feasible completion', () => {
  const expected = cspProgressBound(unique);
  assert.equal(scoreCspProgress(unique, { assignment: { a: 1 } }, expected).quality, 1 / 3);
  assert.equal(scoreCspProgress(unique, { assignment: { a: 1, b: 2, c: 3 } }, expected).quality, 1);
  assert.equal(scoreCspProgress(unique, { assignment: { a: 2 } }, expected).quality, 0);
  assert.equal(scoreCspProgress(unique, { assignment: { a: 1, b: 2, c: 3 }, feasible: false, objective: -999 }, expected).quality, 1);
  assert.equal(scoreCspProgress(unique, { assignment: { a: 1 } }, { ...expected, instance_sha256: '0'.repeat(64) }).quality, 0);
});

test('line-run constraints check the displayed runs and malformed/oversized definitions fail closed', () => {
  const grid = { schema: 'natlang.finite-csp/1', template: 'line', narrative: 'Binary line.',
    variables: ['x1', 'x2', 'x3', 'x4'].map(id => ({ id, domain: [0, 1] })),
    constraints: [{ kind: 'lineRuns', vars: ['x1', 'x2', 'x3', 'x4'], runs: [2, 1] }] };
  assert.equal(checkCspAssignment(grid, { x1: 1, x2: 1, x3: 0, x4: 1 }).valid, true);
  assert.equal(checkCspAssignment(grid, { x1: 1, x2: 0, x3: 0, x4: 1 }).valid, false);
  assert.throws(() => solveFiniteCsp({ ...grid, variables: Array.from({ length: 29 }, (_, i) => ({ id: `v${i}`, domain: [0, 1] })) }), /variable count cap/);
});

test('generated episodes keep roles and held-out structures distinct and expose no hidden assignments', () => {
  const packet = buildCspPacket({ support: 2, query: 2, transfer: 2 });
  assert.equal(packet.manifest.episodes, 15);
  assert.deepEqual(new Set(packet.manifest.families), new Set(['csp-order-schedule', 'csp-logic-grid', 'csp-resource-placement', 'csp-latin-grid', 'csp-nonogram-grid']));
  assert.equal(packet.manifest.by_split.train, 12);
  assert.equal(packet.manifest.by_split.validation, 3);
  assert.equal(packet.caseAudit.length, 28);
  for (const episode of packet.all) {
    const support = new Set(episode.support.cases.map(row => row.group));
    const query = new Set(episode.query.cases.map(row => row.group));
    for (const group of support) assert.equal(query.has(group), false);
    if (episode.transfer) {
      for (const group of episode.transfer.cases.map(row => row.group)) {
        assert.equal(support.has(group), false);
        assert.equal(query.has(group), false);
      }
    }
    const skill = episode.library.skills['finite-constraint-method']['SKILL.md'];
    for (const row of episode.query.cases) {
      const visibleInstance = JSON.parse(row.args[0]);
      assert.equal(Object.hasOwn(visibleInstance, 'assignment'), false);
      assert.equal(Object.hasOwn(visibleInstance, 'solution'), false);
      assert.ok(row.expected.assignment);
    }
    assert.equal(episode.target.files['solve.nl'].includes('Return JSON with an assignment'), true);
  }
  const variants = packet.all.filter(row => row.family === 'csp-order-schedule' && row.split === 'train');
  assert.notEqual(variants.find(row => row.provenance.skill_condition === 'generic').library.skills['finite-constraint-method']['SKILL.md'],
    variants.find(row => row.provenance.skill_condition === 'description-tuned').library.skills['finite-constraint-method']['SKILL.md']);
  assert.notEqual(variants.find(row => row.provenance.skill_condition === 'description-tuned').library.skills['finite-constraint-method']['SKILL.md'],
    variants.find(row => row.provenance.skill_condition === 'body-tuned').library.skills['finite-constraint-method']['SKILL.md']);
});
