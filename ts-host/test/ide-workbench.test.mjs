import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { IdeWorkbench } from '../../applications/ide_workbench.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const editPath = fileURLToPath(new URL('../../codebases/ide/edit.nl', import.meta.url));
const runPath = fileURLToPath(new URL('../../codebases/ide/run.nl', import.meta.url));
const viewPath = fileURLToPath(new URL('../../codebases/ide/view.nl', import.meta.url));

test('natlang edits, runs, inspects and renders a source-pinned child programme', async () => {
  const ide = new IdeWorkbench({ main: { args: { input: 'string' }, returns: 'string',
    code: 'return input.toUpperCase();' } }, 'main');
  const original = ide.snapshot();
  const host = new NatlangHost({ host: { ide, drainEvents: () => ide.drainEvents() } });
  try {
    const edited = await host.run({ source: { kind: 'file', path: editPath },
      inputs: { request: 'Make output lowercase' },
      modelTurn: request => evalTurn(request,
        'const snapshot = await inspect();\n' +
        'const patch = await interpret(request, snapshot);\n' +
        'await apply(patch)') });
    assert.equal(edited.value.status, 'edited');
    assert.notEqual(ide.snapshot().revision, original.revision);
    const run = await host.run({ source: { kind: 'file', path: runPath },
      inputs: { input: 'HeLLo' },
      modelTurn: request => evalTurn(request,
        'const snapshot = await inspect();\n' +
        'const checked = await check(snapshot.revision);\n' +
        'await execute(input, checked.revision)') });
    assert.equal(run.outcome.kind, 'done', JSON.stringify(run.outcome));
    assert.equal(run.value.status, 'done');
    assert.equal(run.value.value_text, '"hello"');
    const view = await host.run({ source: { kind: 'file', path: viewPath },
      inputs: { run_id: run.value.run_id, index: 0 },
      modelTurn: request => evalTurn(request,
        'const snapshot = await inspect();\n' +
        'const checked = await check(snapshot.revision);\n' +
        'const event = await trace(run_id, index);\n' +
        'const page = await describe(snapshot, checked, event);\n' +
        'await render(page)') });
    assert.equal(view.outcome.kind, 'done');
    assert.match(view.value, /IDE &lt;view&gt;/);
    assert.match(view.value, /&lt;recorded event&gt;/);
    assert.ok(ide.inspect(run.value.run_id, 0).event_json.length > 0);
    ide.addScenario({ id: 'lowercase', inputs: { input: 'HeLLo' }, expected: 'hello' });
    assert.equal((await ide.evaluate(['lowercase'])).passed, true);
  } finally { host.close(); }
});

test('stale edits and invalid syntax remain visible without replacing a checked revision', async () => {
  const ide = new IdeWorkbench({ main: { args: { input: 'string' }, returns: 'string',
    code: 'return input;' } }, 'main');
  const base = ide.snapshot().revision;
  const bad = ide.edit({ name: 'main', start: 0, end: 0, text: 'return (',
    expected_revision: base });
  assert.equal(bad.status, 'edited');
  assert.equal(ide.check().status, 'invalid');
  assert.equal((await ide.run({ input: 'x' })).status, 'invalid-source');
  assert.equal(ide.edit({ name: 'main', start: 0, end: 0, text: 'x',
    expected_revision: base }).status, 'stale');
  assert.equal(ide.check(base).status, 'checked');
});
