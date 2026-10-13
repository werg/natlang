import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkOracle } from '../dist/evaluation/oracles.js';

const expected = {
  clusters: [['A', 'B'], ['C']],
  edges: [['A', 'B'], ['C', 'D']],
  route: ['start', 'middle', 'end'],
  note: 'The source attributes this to Ada.',
};
const spec = { level: 'normalized', normalization: 'semantic-structured', comparison_spec: {
  schema_version: 'semantic-structured-comparison/1',
  unordered_paths: ['/clusters', '/clusters/*', '/edges'],
  ordered_paths: ['/route'],
  prose_paths: ['/note'],
  witness_review_required: false,
} };

test('declared unordered paths accept bijective permutations but preserve directed and ordered fields', async () => {
  const equivalent = { ...expected, clusters: [['C'], ['B', 'A']], edges: [['C', 'D'], ['A', 'B']] };
  assert.equal((await checkOracle(equivalent, expected, spec)).accepted, true);
  for (const wrong of [
    { ...equivalent, edges: [['C', 'D'], ['B', 'A']] },
    { ...equivalent, clusters: [['C'], ['B', 'B']] },
    { ...equivalent, clusters: [['C'], ['B', 'A'], ['A']] },
    { ...equivalent, route: [...expected.route].reverse() },
    { ...equivalent, edges: [['C', 'D']] },
  ]) {
    const verdict = await checkOracle(wrong, expected, spec);
    assert.equal(verdict.accepted, false);
    assert.notEqual(verdict.needs_review, true);
  }
});

test('conflicting comparison paths fail closed', async () => {
  const conflicting = { ...spec, comparison_spec: { ...spec.comparison_spec,
    ordered_paths: ['/route', '/edges'] } };
  const verdict = await checkOracle(expected, expected, conflicting);
  assert.equal(verdict.accepted, false);
  assert.equal(verdict.comparison_version, 'invalid-semantic-structured-spec/1');
});

test('prose alternatives and canonical-witness alternatives remain review pending', async () => {
  const paraphrase = { ...expected, note: 'Ada is named as the source.' };
  const prose = await checkOracle(paraphrase, expected, spec);
  assert.equal(prose.accepted, false);
  assert.equal(prose.needs_review, true);
  const witness = await checkOracle({ ...expected, route: ['start', 'other', 'end'] }, expected,
    { ...spec, comparison_spec: { ...spec.comparison_spec, witness_review_required: true } });
  assert.equal(witness.accepted, false);
  assert.equal(witness.needs_review, true);
  const permissiveJudge = async () => ({ accepted: true, verdict: 'looks plausible' });
  const witnessWithJudge = await checkOracle({ ...expected, route: ['start', 'other', 'end'] }, expected,
    { ...spec, comparison_spec: { ...spec.comparison_spec, witness_review_required: true } }, permissiveJudge);
  assert.equal(witnessWithJudge.accepted, false);
  assert.equal(witnessWithJudge.needs_review, true);
});
