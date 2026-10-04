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
