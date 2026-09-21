import assert from 'node:assert/strict';
import { test } from 'node:test';
import { examples, exampleCategories } from '../playground/examples.mjs';
import { crispExamples } from '../playground/examples/crisp.mjs';
import { interfaceExamples } from '../playground/examples/interfaces.mjs';

async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

test('example library is diverse, self-contained, and valid for the browser loader', async () => {
  const { newPlaygroundProject, validatePlaygroundProject } = await api();
  assert.ok(examples.length >= 10);
  assert.ok(examples.every(item => item.modelRequired), 'the public gallery should demonstrate natlang');
  assert.equal(new Set(examples.map(item => item.id)).size, examples.length);
  assert.ok(exampleCategories.includes('Composed agents'));
  for (const item of examples) {
    assert.ok(item.description && item.concepts.length && item.level && item.category, item.id);
    assert.equal(item.modelRequired, item.root.endsWith('.nl'), item.id);
    const project = newPlaygroundProject(item.name, item.root, item.files, item.inputs, item.expected);
    assert.deepEqual(validatePlaygroundProject(project), [], item.id);
    if (item.category === 'Composed agents') assert.ok(Object.keys(item.files).length > 1, item.id);
  }
});

test('every crisp example executes to its advertised expected result', async () => {
  const { BrowserNatlangHost, newPlaygroundProject, runPlaygroundProject } = await api();
  const host = new BrowserNatlangHost();
  try {
    for (const item of [...crispExamples, ...interfaceExamples.filter(item => !item.modelRequired)]) {
      const project = newPlaygroundProject(item.name, item.root, item.files, item.inputs, item.expected);
      const run = await runPlaygroundProject(host, project);
      assert.equal(run.outcome.kind, 'done', `${item.id}: ${run.outcome.detail}`);
      assert.equal(run.correct, true, `${item.id}: ${JSON.stringify(run.value)}`);
    }
  } finally { host.close(); }
});
