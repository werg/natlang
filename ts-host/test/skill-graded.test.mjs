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
