import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { scoreGraded, multisetF1 } from '../dist/skills/graded.js';

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'graded-sql-')); mkdirSync(join(root, 'shop'));
  const db = new DatabaseSync(join(root, 'shop', 'shop.sqlite'));
  db.exec("CREATE TABLE item(id INTEGER, name TEXT, price REAL); INSERT INTO item VALUES (1,'a',2.5),(2,'b',4),(3,'c',4);");
  db.close();
  return root;
}

test('SQL answers are scored by result-set F1 against the executed gold query', () => {
  const metric = { schema: 'natlang.skill-graded/1', kind: 'sql-result-f1', database_root: fixture() };
  const gold = { kind: 'sql-gold', db: 'shop/shop.sqlite', sql: 'SELECT name FROM item WHERE price = 4' };
  assert.equal(scoreGraded(metric, 'SELECT name FROM item WHERE price >= 4', gold).quality, 1);
  assert.equal(scoreGraded(metric, '```sql\nSELECT name FROM item\n```', gold).quality, 0.8);
  assert.equal(scoreGraded(metric, 'SELECT nope FROM item', gold).gates.executes, false);
  assert.equal(scoreGraded(metric, 'DELETE FROM item', gold).gates.executes, false);
  assert.equal(scoreGraded(metric, 'SELECT 1; SELECT 2', gold).gates.single_statement, false);
  assert.equal(scoreGraded(metric, 'SELECT 1', { ...gold, db: '../escape.sqlite' }).gates.database_path, false);
  assert.equal(scoreGraded(metric, { sql: 'SELECT name FROM item WHERE price = 4' }, gold).quality, 1);
});

test('multiset F1 counts duplicates and treats integral numbers alike', () => {
  assert.equal(multisetF1([[1], [1]], [[1]]).precision, 0.5);
  assert.equal(multisetF1([[2.0]], [[2]]).f1, 1);
  assert.equal(multisetF1([], []).f1, 1);
});

test('the shared scorer registry keeps objective identities and adds graded SQL scorers', async () => {
  const { episodeScoring, episodeScorings } = await import('../dist/skills/scoring.js');
  const pins = { 'skills/objective.js': 'aaa', 'skills/graded.js': 'bbb' };
  assert.equal(episodeScoring({ schema: 'natlang.skill-objective/1', kind: 'tsp' }, { pins }).identity, 'natlang.skill-objective/1:tsp:aaa');
  const sql = episodeScoring({ schema: 'natlang.skill-graded/1', kind: 'sql-result-f1' }, { pins, databaseRoot: fixture() });
  assert.equal(sql.identity, 'natlang.skill-graded/1:sql-result-f1:bbb');
  assert.equal(sql.score({ args: [], expected: { kind: 'sql-gold', db: 'shop/shop.sqlite', sql: 'SELECT id FROM item' } }, { value: 'SELECT id FROM item' }).quality, 1);
  assert.equal(sql.score({ args: [], expected: {} }, { error: 'boom' }).gates.completed, false);
  assert.throws(() => episodeScoring({ schema: 'natlang.skill-graded/1', kind: 'sql-result-f1' }, { pins }), /database root/);
  assert.throws(() => episodeScoring({ schema: 'other', kind: 'x' }, { pins }), /unsupported/);
  assert.throws(() => episodeScorings({ metric: { schema: 'natlang.skill-objective/1', kind: 'tsp' } }, true, { pins }), /transfer requires/);
  assert.equal(episodeScorings(undefined, false, { pins }).scoring, undefined);
});

test('Python answers are scored by the fraction of reference unit tests passed in a network-less sandbox', async t => {
  const { spawnSync } = await import('node:child_process');
  const { PYTHON_SANDBOX_IMAGE } = await import('../dist/skills/graded.js');
  if (spawnSync('docker', ['image', 'inspect', PYTHON_SANDBOX_IMAGE]).status !== 0) return t.skip('sandbox image unavailable');
  const metric = { schema: 'natlang.skill-graded/1', kind: 'python-tests' };
  const tests = 'from solution import add\ndef test_a():\n    assert add(1, 2) == 3\ndef test_b():\n    assert add(2, 2) == 5\n';
  const reference = { kind: 'python-tests', tests };
  assert.equal(scoreGraded(metric, '```python\ndef add(a, b):\n    return a + b\n```', reference).quality, 0.5);
  assert.equal(scoreGraded(metric, 'import socket\nsocket.create_connection(("1.1.1.1", 80), timeout=2)\n', reference).gates.imports, false);
  assert.equal(scoreGraded(metric, 'while True:\n    pass\n', reference).gates.runs, false);
  assert.equal(scoreGraded(metric, 42, reference).gates.returned_code, false);
});

test('answer token F1, ranking NDCG, assignment accuracy and call F1 give partial credit', async () => {
  const { scoreGraded: score, ndcg, tokenF1 } = await import('../dist/skills/graded.js');
  const m = kind => ({ schema: 'natlang.skill-graded/1', kind });
  assert.equal(tokenF1('The Eiffel Tower', 'eiffel tower'), 1);
  assert.equal(score(m('answer-token-f1'), 'Paris, France', { kind: 'gold-answer', value: 'Paris' }).quality, 2 / 3);
  assert.equal(score(m('answer-token-f1'), 42, { kind: 'gold-answer', value: 'x' }).gates.returned_answer, false);
  assert.equal(ndcg(['a', 'b'], ['a', 'b']), 1);
  const swapped = score(m('ranking-ndcg'), '["c", "a", "b"]', { kind: 'relevant-set', items: ['a', 'b'] });
  assert.ok(swapped.quality > 0 && swapped.quality < 1 && swapped.gates.relevant_first === false);
  const kk = { kind: 'assignment', value: { Ann: 'knight', Bob: 'knave', Cy: 'knight', Di: 'knave' } };
  assert.equal(score(m('assignment-accuracy'), { Ann: 'knight', Bob: 'knight', Cy: 'knight', Di: 'knave' }, kk).quality, 0.75);
  assert.equal(score(m('assignment-accuracy'), 'not json', kk).quality, 0);
  const calls = { kind: 'function-calls', calls: [{ name: 'f', arguments: { a: 1, b: 'x' } }] };
  assert.equal(score(m('call-f1'), '```json\n[{"name":"f","arguments":{"b":"x","a":1}}]\n```', calls).quality, 1);
  const partial = score(m('call-f1'), [{ name: 'f', arguments: { a: 2, b: 'x' } }], calls);
  assert.ok(Math.abs(partial.quality - 2 / 3) < 1e-9);
});

test('choice Brier rewards calibrated probabilities and accepts a bare label as certainty', async () => {
  const { scoreGraded: score } = await import('../dist/skills/graded.js');
  const m = { schema: 'natlang.skill-graded/1', kind: 'choice-brier' };
  const ref = { kind: 'choice', answer: 'B', options: ['A', 'B', 'C', 'D'] };
  assert.equal(score(m, 'B', ref).quality, 1);
  assert.equal(score(m, 'A', ref).quality, 0);
  const hedge = score(m, { probabilities: { A: 0.25, B: 0.5, C: 0.25 } }, ref);
  assert.ok(Math.abs(hedge.quality - (1 - 0.375 / 2)) < 1e-9 && hedge.gates.top_correct);
  assert.equal(score(m, '{"B": 2, "C": 2}', ref).quality, 1 - 0.5 / 2, 'weights are normalised');
  assert.equal(score(m, { E: 1 }, ref).gates.returned_distribution, false);
  assert.equal(score(m, { A: -1, B: 2 }, ref).gates.returned_distribution, false);
});

test('binary Brier takes labels or target frequencies; ordinal RPS takes distributions, levels or fractional scores', async () => {
  const { scoreGraded: score, rankedProbabilityScore } = await import('../dist/skills/graded.js');
  const b = { schema: 'natlang.skill-graded/1', kind: 'binary-brier' };
  assert.equal(score(b, 0.8, { kind: 'binary', answer: true }).quality, 1 - 0.04);
  assert.ok(Math.abs(score(b, '{"probability": 0.3}', { kind: 'binary', answer: 0.3 }).quality - 1) < 1e-12, 'soft label');
  assert.equal(score(b, true, { kind: 'binary', answer: false }).quality, 0);
  assert.equal(score(b, 1.5, { kind: 'binary', answer: true }).gates.returned_probability, false);
  const o = { schema: 'natlang.skill-graded/1', kind: 'ordinal-rps' };
  const ref = { kind: 'ordinal', levels: ['1', '2', '3', '4', '5'], answer: 3 };
  assert.equal(score(o, '4', ref).quality, 1);
  assert.equal(score(o, { score: 3 }, ref).quality, 1);
  const near = score(o, { probabilities: { 3: 0.5, 4: 0.5 } }, ref).quality, far = score(o, '1', ref).quality;
  assert.ok(near > far && near < 1 && far >= 0, `${near} ${far}`);
  // A fractional target (a mean rating) is matched by the same fractional score.
  assert.ok(Math.abs(score(o, 2.4, { ...ref, answer: 2.4 }).quality - 1) < 1e-12);
  assert.equal(rankedProbabilityScore([1, 0, 0], [0, 0, 1]), 1);
});
