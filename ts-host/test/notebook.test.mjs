import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, openFolder } from '../dist/index.js';
import { NotebookWorkspace, answerRequest, checkCitations, evidenceOf, finalCells, limitsOf, lowestId, readyCells, remainingCells, runNotebook }
  from '../../applications/dist/notebook/index.js';
import { scriptedModel } from './support/natlang.mjs';

const STAGES = [['goal', 'Choose the one cell whose result answers'], ['next', 'Choose one cell of ready to run next'],
  ['explain', 'Answer question from evidence'], ['note', 'Find the supporting file']];
/** A scripted model: `answers` maps a stage to eval code (or a function of the call count and opening); `calls` records the stages in order. */
function script(answers) {
  const calls = [];
  const model = scriptedModel(opening => {
    const stage = STAGES.find(([, text]) => opening.includes(text))?.[0];
    calls.push(stage);
    const answer = answers[stage];
    const code = typeof answer === 'function' ? answer(calls.filter(name => name === stage).length, opening) : answer;
    return code ?? (stage === 'note' ? 'return ""' : 'return "no scripted answer"');
  });
  return { ...model, calls };
}
const EXPLAIN_IDS = 'return evidence.map(r => r.id + "@" + r.revision + " " + r.sample).join("; ")';

test('cells run in dependency order with the crisp default; the explanation reads host-computed evidence and the note', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-notebook-files-'));
  writeFileSync(join(folder, 'schema.md'), 'facts has category and amount columns.');
  const notebook = new NotebookWorkspace([
    { id: 'totals', engine: 'sqlite', needs: [], description: 'aggregate amounts by category',
      source: 'SELECT category, SUM(amount) AS total FROM facts GROUP BY category ORDER BY category' },
    { id: 'view', engine: 'javascript', needs: ['totals'], description: 'make presentation labels',
      source: 'return deps.totals.map(row => ({ label: row.category.toUpperCase(), total: row.total }));' },
    { id: 'unrelated', engine: 'javascript', needs: [], description: 'unrelated work', source: 'return 999;' },
  ], { facts: [{ category: 'alpha', amount: 2 }, { category: 'alpha', amount: 3 }, { category: 'beta', amount: 4 }] });
  const model = script({ note: 'return await files.file("schema.md").readText()',
    explain: 'return evidence.map(r => r.id + "@" + r.revision + " " + r.sample).join("; ") + " (" + note.length + " note chars)"' });
  try {
    const run = await createNatlangRuntime({ model: model.driver }).run(() =>
      runNotebook(notebook, 'view', 'What are the category totals? See schema.md.', openFolder(folder).root()));
    assert.equal(run.status, 'done', run.detail);
    assert.deepEqual(run.order, ['totals', 'view']);
    assert.match(run.answer, /view@0 \[\{"label":"ALPHA","total":5\}/);
    assert.match(run.answer, /\(38 note chars\)/);
    assert.deepEqual(model.calls, ['note', 'explain'], 'no model call chooses the order');
    assert.doesNotMatch(model.openings.at(-1), /[0-9a-f]{64}/, 'the output hash is not model-visible');
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

test('a request chooses its goal from the final cells; a wrong id returns once with the problem; a cell with no ready path is blocked', async () => {
  const notebook = new NotebookWorkspace([
    { id: 'base', engine: 'javascript', needs: [], description: 'a number', source: 'return 2;' },
    { id: 'double', engine: 'javascript', needs: ['base'], description: 'twice the number', source: 'return deps.base * 2;' },
    { id: 'orphan', engine: 'javascript', needs: ['missing'], description: 'needs a missing cell', source: 'return 0;' },
  ]);
  const seen = [];
  const model = script({ goal: (count, opening) => { seen.push(opening); return count === 1 ? 'return "no_such_cell"' : 'return finals.includes("double") && !finals.includes("base") ? "double" : "wrong"'; },
    explain: 'return evidence.map(r => r.id + "@" + r.revision).join(",")' });
  model.calls.length = 0;
  const runtime = createNatlangRuntime({ model: model.driver });
  try {
    const run = await runtime.run(() => answerRequest(notebook, 'What is twice the number?'));
    assert.deepEqual([run.goal, run.status], ['double', 'done']);
    assert.equal(model.calls.filter(stage => stage === 'goal').length, 2);
    assert.match(seen[1], /not one of the cells: base, double, orphan/);
    const blocked = await runtime.run(() => runNotebook(notebook, 'orphan', 'Run the orphan'));
    assert.deepEqual([blocked.status, blocked.blocked], ['blocked', ['base', 'double', 'orphan']]);
    // A goal that never names a cell ends invalid after the single retry.
    const lost = script({ goal: 'return "still_wrong"' });
    const invalid = await createNatlangRuntime({ model: lost.driver }).run(() => answerRequest(notebook, 'anything'));
    assert.deepEqual([invalid.status, lost.calls.length], ['invalid', 2]);
  } finally { notebook.close(); }
});

test('the next ready cell is a pluggable: crisp default, nl choice, shadow serves crisp and traces agreement', async () => {
  const cells = [
    { id: 'a', engine: 'javascript', needs: [], description: 'load a', source: 'return 1;' },
    { id: 'b', engine: 'javascript', needs: [], description: 'load b', source: 'return 2;' },
    { id: 'goal', engine: 'javascript', needs: ['a', 'b'], description: 'sum', source: 'return deps.a + deps.b;' },
  ];
  const last = 'return ready[ready.length - 1].id';
  const traces = [];
  const orderOf = async (mode, nextAnswers) => {
    const notebook = new NotebookWorkspace(cells);
    const model = script({ next: nextAnswers, explain: EXPLAIN_IDS });
    try {
      const run = await createNatlangRuntime({ model: model.driver, trace: trace => traces.push(trace) })
        .run(() => runNotebook(notebook, 'goal', 'sum?', undefined, mode ? { nextCellMode: mode } : {}));
      return { order: run.order, next: model.calls.filter(stage => stage === 'next').length, status: run.status };
    } finally { notebook.close(); }
  };
  assert.deepEqual(await orderOf(undefined, last), { order: ['a', 'b', 'goal'], next: 0, status: 'done' });
  assert.deepEqual(await orderOf('crisp', last), { order: ['a', 'b', 'goal'], next: 0, status: 'done' });
  assert.deepEqual(await orderOf('nl', last), { order: ['b', 'a', 'goal'], next: 3, status: 'done' });
  traces.length = 0;
  assert.deepEqual(await orderOf('shadow', last), { order: ['a', 'b', 'goal'], next: 3, status: 'done' });
  const shadows = traces.flatMap(trace => trace.events).filter(event => event.kind === 'pluggable_shadow');
  assert.deepEqual(shadows.map(event => event.agree), [false, true, true]);
  assert.equal(shadows[0].name, 'notebook.nextCell');
  // An nl answer that is not ready returns once with the problem, then ends the run invalid.
  const bad = await orderOf('nl', 'return "goal"');
  assert.equal(bad.status, 'invalid');
  assert.equal(bad.next, 2);
  assert.equal((await orderOf('nl', (count, opening) => count === 1 ? 'return "goal"' : 'return ready[0].id')).status, 'done');
});

test('the loop ends by a structural measure: required cells not yet run', async () => {
  const notebook = new NotebookWorkspace([
    { id: 'x', engine: 'javascript', needs: ['y'], description: 'x', source: 'return 1;' },
    { id: 'y', engine: 'javascript', needs: ['x'], description: 'y', source: 'return 1;' },
    { id: 'p', engine: 'javascript', needs: [], description: 'p', source: 'return 1;' },
    { id: 'q', engine: 'javascript', needs: ['p'], description: 'q', source: 'return deps.p;' },
    { id: 'r', engine: 'javascript', needs: [], description: 'unrelated', source: 'return 1;' },
  ]);
  try {
    const run = { goal: 'q', cells: notebook.catalog(), order: [], results: [], blocked: [], answer: '', detail: '', status: 'running' };
    assert.equal(remainingCells(run), 2);
    assert.deepEqual(readyCells(run).map(cell => cell.id), ['p']);
    assert.equal(remainingCells({ ...run, order: ['p'] }), 1);
    assert.equal(remainingCells({ ...run, order: ['p'], status: 'failed' }), 0);
    const model = script({ explain: EXPLAIN_IDS });
    const runtime = createNatlangRuntime({ model: model.driver });
    const cyclic = await runtime.run(() => runNotebook(notebook, 'x', 'x?'));
    assert.deepEqual([cyclic.status, cyclic.blocked], ['blocked', ['p', 'q', 'r', 'x', 'y']]);
    assert.deepEqual(await runtime.run(() => runNotebook(notebook, 'q', 'q?')).then(done => [done.status, done.order]), ['done', ['p', 'q']]);
    assert.equal(lowestId(notebook.catalog().filter(cell => cell.id !== 'p')), 'q');
    assert.deepEqual(finalCells(notebook.catalog()), ['q', 'r']);
  } finally { notebook.close(); }
});

test('evidence carries the facts the host knows: NULLs, emptiness, truncation and limits; the explanation states them', async () => {
  const notebook = new NotebookWorkspace([
    { id: 'nulls', engine: 'sqlite', needs: [], source: 'SELECT label FROM t ORDER BY rowid' },
    { id: 'none', engine: 'sqlite', needs: [], source: 'SELECT label FROM t WHERE label = \'zzz\'' },
    { id: 'big', engine: 'javascript', needs: [], source: 'return Array.from({ length: 400 }, (_, i) => ({ i }));' },
    { id: 'plain', engine: 'javascript', needs: [], source: 'return { n: 1 };' },
    { id: 'broken', engine: 'javascript', needs: [], source: 'throw new Error("boom");' },
    { id: 'after', engine: 'javascript', needs: ['broken'], source: 'return 1;' },
  ], { t: [{ label: null }, { label: '' }] });
  try {
    const results = [];
    for (const id of ['nulls', 'none', 'big', 'plain', 'broken']) results.push(await notebook.execute(id));
    const evidence = evidenceOf(notebook, { results });
    const facts = Object.fromEntries(evidence.map(row => [row.id, [row.has_null, row.empty, row.truncated, row.status]]));
    assert.deepEqual(facts, { nulls: [true, false, false, 'ok'], none: [false, true, false, 'ok'], big: [false, false, true, 'ok'],
      plain: [false, false, false, 'ok'], broken: [false, false, false, 'failed'] });
    assert.ok(evidence.every(row => !('output_sha256' in row)));

    const model = script({ explain: 'return limits.join("|") + " " + evidence.map(r => r.id + "@" + r.revision).join(",")' });
    const run = await createNatlangRuntime({ model: model.driver }).run(() => runNotebook(notebook, 'after', 'What is after?'));
    assert.equal(run.status, 'failed');
    assert.match(run.answer, /^cell broken failed \(Error: boom\)/);
    assert.deepEqual(limitsOf({ ...run, status: 'done', results: [] }), []);
  } finally { notebook.close(); }
});

test('citations are checked against the cells that ran, with one retry', async () => {
  const notebook = new NotebookWorkspace([{ id: 'totals', engine: 'javascript', needs: [], description: 'one', source: 'return 1;' }]);
  try {
    const repaired = script({ explain: (count, opening) => count === 1 ? 'return "The answer is in totals@9."' : 'return "The total is 1 (totals@0)."' });
    const run = await createNatlangRuntime({ model: repaired.driver }).run(() => runNotebook(notebook, 'totals', 'total?'));
    assert.equal(run.answer, 'The total is 1 (totals@0).');
    assert.equal(run.detail, '');
    assert.match(repaired.openings.at(-1), /totals@9 does not name a cell and revision that ran/);
    const stuck = script({ explain: 'return "The total is 1."' });
    const unchecked = await createNatlangRuntime({ model: stuck.driver }).run(() => runNotebook(notebook, 'totals', 'total?'));
    assert.equal(stuck.calls.filter(stage => stage === 'explain').length, 2);
    assert.match(unchecked.detail, /^citation check: the answer cites no cell/);
    assert.equal(checkCitations('x', [{ id: 'a', revision: 0, status: 'failed' }]), null);
    assert.match(checkCitations('a@1', [{ id: 'a', revision: 0, status: 'ok' }]), /a@1 does not/);
  } finally { notebook.close(); }
});

test('a JavaScript cell stops at the workspace timeout setting', async () => {
  const notebook = new NotebookWorkspace([{ id: 'spin', engine: 'javascript', needs: [], source: 'while (true) {}' }], {}, { cellTimeoutMs: 50 });
  try {
    const result = await notebook.execute('spin');
    assert.equal(result.status, 'failed');
    assert.match(result.detail, /timed out/);
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
