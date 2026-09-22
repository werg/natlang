import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NatlangHost, NodeFileTree, TypeScriptEnvironment } from '../dist/index.js';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { NotebookWorkspace } from '../../applications/notebook.mjs';
import { fileURLToPath } from 'node:url';
import { evalTurn } from './support/eval-turn.mjs';

const path = fileURLToPath(new URL('../../codebases/notebook/run.nl', import.meta.url));

test('natlang orders SQL and TypeScript cells, then explains the checked samples', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-notebook-files-'));
  writeFileSync(join(folder, 'schema.md'), 'facts has category and amount columns.');
  const notebook = new NotebookWorkspace([
    { id: 'totals', engine: 'sqlite', needs: [], description: 'aggregate amounts by category',
      source: 'SELECT category, SUM(amount) AS total FROM facts GROUP BY category ORDER BY category' },
    { id: 'view', engine: 'typescript-host', needs: ['totals'],
      description: 'make presentation labels',
      source: 'return deps.totals.map(row => ({ label: row.category.toUpperCase(), total: row.total }));' },
    { id: 'unrelated', engine: 'typescript-host', needs: [], description: 'unrelated work',
      source: 'return 999;' },
  ], { facts: [
    { category: 'alpha', amount: 2 }, { category: 'alpha', amount: 3 },
    { category: 'beta', amount: 4 },
  ] }, { environment: new TypeScriptEnvironment({ mode: 'fresh' }) });
  const host = new NatlangHost({ host: { notebook,
    drainEvents: () => notebook.drainEvents() }, mode: 'retained' });
  let chosenCount = 0;
  const modelTurn = turn => {
    const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
    if (prompt.includes('function run(')) return evalTurn(turn,
      'const initial = await prepare(goal);\n' +
      'const finished = await step(initial, files);\n' +
      'const answer = await explain(question, finished, files);\n' +
      'await attach(finished, answer)');
    if (prompt.includes('function step(')) return evalTurn(turn,
      'const ready = await ready_cells(state);\n' +
      'if (ready.length === 0) { await stall(state); }\n' +
      'const chosen = await choose(ready, state.goal, files);\n' +
      'await advance(state, chosen)');
    if (prompt.includes('Choose one offered ready cell')) {
      const chosen = chosenCount++ === 0 ? 'totals' : 'view';
      // Each run executes totals before view; this scripted driver only checks the host boundary.
      return evalTurn(turn, JSON.stringify(chosen));
    }
    return evalTurn(turn, '"The checked table has alpha 5 and beta 4."');
  };
  try {
    const result = await host.run({ source: { kind: 'file', path },
      inputs: { goal: 'view', question: 'What are the category totals?', files: new NodeFileTree(folder) }, modelTurn,
    });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'done');
    assert.deepEqual(Array.from(result.value.order), ['totals', 'view']);
    assert.deepEqual(notebook.outputs.get('view').value, [
      { label: 'ALPHA', total: 5 }, { label: 'BETA', total: 4 } ]);
    const edit = notebook.edit('totals', 'SELECT category, SUM(amount) AS total FROM facts WHERE amount > 2 GROUP BY category ORDER BY category');
    assert.deepEqual(edit.invalidated, ['totals', 'view']);
    assert.equal(notebook.outputs.has('view'), false);
    assert.equal((await notebook.execute('view')).status, 'failed');
    assert.equal((await notebook.execute('totals')).status, 'ok');
    assert.equal((await notebook.execute('view')).status, 'ok');
    assert.deepEqual(notebook.outputs.get('view').value, [
      { label: 'ALPHA', total: 3 }, { label: 'BETA', total: 4 } ]);
  } finally { host.close(); notebook.close(); rmSync(folder, { recursive: true, force: true }); }
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
