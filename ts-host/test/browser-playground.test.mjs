import assert from 'node:assert/strict';
import { test } from 'node:test';

async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

test('playground validates linked files while keeping invalid edits available', async () => {
  const { newPlaygroundProject, editPlaygroundProject, validatePlaygroundProject,
    validProjectPath } = await api();
  assert.equal(validProjectPath('../escape.ts'), false);
  const project = newPlaygroundProject('Math', 'math/add.ts', {
    'math/add.ts': 'export default function add(a: number): Result {\n  return a + 2;\n}',
    'math/types.ts': 'type Result = number;',
  }, { a: 5 }, 7);
  assert.deepEqual(validatePlaygroundProject(project), []);
  const invalid = editPlaygroundProject(project, { files: { ...project.files,
    'math/add.ts': project.files['math/add.ts'].replace('return a + 2;', 'return (;') } });
  assert.notEqual(invalid.revision, project.revision);
  assert.equal(project.files['math/add.ts'].includes('return a + 2;'), true);
  assert.match(validatePlaygroundProject(invalid)[0].message, /typescript/i);
});

test('playground pins source, reconstructs trace frames, and admits exact captured outcomes', async () => {
  const { BrowserNatlangHost, newPlaygroundProject, editPlaygroundProject,
    runPlaygroundProject, traceFrame, admitPlaygroundRun } = await api();
  const project = newPlaygroundProject('Add', 'add.ts', {
    'add.ts': 'export default function add(a: number): number {\n  return a + 2;\n}',
  }, { a: 5 }, 7);
  const host = new BrowserNatlangHost();
  let run;
  try { run = await runPlaygroundProject(host, project); }
  finally { host.close(); }
  assert.equal(run.value, 7);
  assert.equal(run.correct, true);
  const edited = editPlaygroundProject(project, { files: { 'add.ts': project.files['add.ts'].replace('+ 2', '+ 9') } });
  assert.notEqual(run.revision, edited.revision);
  assert.equal(run.source.files['add.ts'].includes('+ 2'), true);
  const first = traceFrame(run.trace, 0), final = traceFrame(run.trace, run.trace.length - 1);
  assert.equal(first.event.kind, 'manifest');
  assert.equal(final.state.phase, 'final');
  assert.deepEqual(admitPlaygroundRun(run, { outcome: 'done', value: 7, effects: [] }).admitted, true);
  assert.throws(() => admitPlaygroundRun(run, { outcome: 'done', value: 8 }), /does not match/);
  assert.throws(() => admitPlaygroundRun(run, { outcome: 'done', value: 7,
    requiredActions: [{ name: 'write' }] }), /required ordered action absent/);
});
