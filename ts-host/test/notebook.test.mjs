import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, openFolder } from '../dist/index.js';
import { NotebookWorkspace, answerRequest, runNotebook } from '../../applications/dist/notebook/index.js';
import { scriptedModel } from './support/natlang.mjs';

test('natlang orders SQL and JavaScript cells, then explains the checked samples', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-notebook-files-'));
  writeFileSync(join(folder, 'schema.md'), 'facts has category and amount columns.');
  const notebook = new NotebookWorkspace([
    { id: 'totals', engine: 'sqlite', needs: [], description: 'aggregate amounts by category',
      source: 'SELECT category, SUM(amount) AS total FROM facts GROUP BY category ORDER BY category' },
    { id: 'view', engine: 'javascript', needs: ['totals'], description: 'make presentation labels',
      source: 'return deps.totals.map(row => ({ label: row.category.toUpperCase(), total: row.total }));' },
    { id: 'unrelated', engine: 'javascript', needs: [], description: 'unrelated work', source: 'return 999;' },
  ], { facts: [{ category: 'alpha', amount: 2 }, { category: 'alpha', amount: 3 }, { category: 'beta', amount: 4 }] });
  const model = scriptedModel(opening => {
    if (opening.includes('Choose one offered ready cell')) return 'return ready[0].id';
    return 'const note = await files.file("schema.md").readText();\n' +
      'return `${run.results.map(r => r.id + "@" + r.revision + " " + r.sample).join("; ")} (${note.length} note chars)`';
  });
  try {
    const run = await createNatlangRuntime({ model: model.driver }).run(() =>
      runNotebook(notebook, 'view', 'What are the category totals? See schema.md.', openFolder(folder).root()));
    assert.equal(run.status, 'done', run.detail);
    assert.deepEqual(run.order, ['totals', 'view']);
    assert.match(run.answer, /view@0 \[\{"label":"ALPHA","total":5\}/);
    assert.deepEqual(notebook.outputs.get('view').value, [{ label: 'ALPHA', total: 5 }, { label: 'BETA', total: 4 }]);
    const edit = notebook.edit('totals', 'SELECT category, SUM(amount) AS total FROM facts WHERE amount > 2 GROUP BY category ORDER BY category');
    assert.deepEqual(edit.invalidated, ['totals', 'view']);
    assert.equal(notebook.outputs.has('view'), false);
    assert.equal((await notebook.execute('view')).status, 'failed');
    assert.equal((await notebook.execute('totals')).status, 'ok');
    assert.equal((await notebook.execute('view')).status, 'ok');
    assert.deepEqual(notebook.outputs.get('view').value, [{ label: 'ALPHA', total: 3 }, { label: 'BETA', total: 4 }]);
  } finally { notebook.close(); rmSync(folder, { recursive: true, force: true }); }
});

test('a request chooses its goal cell; a cell with no ready path is reported blocked', async () => {
  const notebook = new NotebookWorkspace([
    { id: 'base', engine: 'javascript', needs: [], description: 'a number', source: 'return 2;' },
    { id: 'double', engine: 'javascript', needs: ['base'], description: 'twice the number', source: 'return deps.base * 2;' },
    { id: 'orphan', engine: 'javascript', needs: ['missing'], description: 'needs a missing cell', source: 'return 0;' },
  ]);
  const model = scriptedModel(opening => {
    if (opening.includes('Choose exactly one offered cell ID')) return 'return request.includes("orphan") ? "orphan" : "double"';
    if (opening.includes('Choose one offered ready cell')) return 'return ready[0].id';
    return 'return run.status';
  });
  const runtime = createNatlangRuntime({ model: model.driver });
  try {
    const run = await runtime.run(() => answerRequest(notebook, 'What is twice the number?'));
    assert.deepEqual([run.goal, run.status, run.answer], ['double', 'done', 'done']);
    const blocked = await runtime.run(() => answerRequest(notebook, 'Run the orphan'));
    assert.deepEqual([blocked.status, blocked.blocked], ['blocked', ['base', 'double', 'orphan']]);
  } finally { notebook.close(); }
});

test('SQL cells reject writes and preserve NULL distinctly from empty text', async () => {
  const notebook = new NotebookWorkspace([
    { id: 'read', engine: 'sqlite', needs: [], source: 'SELECT label FROM t ORDER BY rowid' },
    { id: 'write', engine: 'sqlite', needs: [], source: 'WITH x AS (SELECT 1) DELETE FROM t' },
  ], { t: [{ label: null }, { label: '' }] });
  try {
    assert.equal((await notebook.execute('read')).status, 'ok');
    assert.deepEqual(notebook.outputs.get('read').value, [{ label: null }, { label: '' }]);
    assert.equal((await notebook.execute('write')).status, 'failed');
    assert.deepEqual(notebook.query('SELECT label FROM t ORDER BY rowid'), [{ label: null }, { label: '' }]);
  } finally { notebook.close(); }
});

test('notebook can import cells and tables after startup', async () => {
  const notebook = new NotebookWorkspace([], {});
  try {
    const loaded = notebook.importConfig({ tables: { measurements: [{ amount: 2 }, { amount: 5 }] }, cells: [
      { id: 'total', engine: 'sqlite', needs: [], description: 'sum', source: 'SELECT SUM(amount) AS total FROM measurements' }] });
    assert.deepEqual(loaded.tables, ['measurements']);
    assert.equal(notebook.catalog()[0].id, 'total');
    assert.match((await notebook.execute('total')).sample, /7/);
    notebook.importConfig({ tables: { measurements: [{ amount: 11 }] }, cells: [] });
    assert.match((await notebook.execute('total')).sample, /11/);
  } finally { notebook.close(); }
});
