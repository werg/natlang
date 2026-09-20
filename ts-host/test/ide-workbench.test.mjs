import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { IdeWorkbench } from '../../applications/ide_workbench.mjs';

const editPath = fileURLToPath(new URL('../../codebases/ide/edit.nl', import.meta.url));
const runPath = fileURLToPath(new URL('../../codebases/ide/run.nl', import.meta.url));
const viewPath = fileURLToPath(new URL('../../codebases/ide/view.nl', import.meta.url));

test('natlang edits, runs, inspects and renders a source-pinned child programme', async () => {
  const ide = new IdeWorkbench({ main: { args: { input: 'Text' }, returns: 'Text',
    code: 'return args.input.toUpperCase();' } }, 'main');
  const original = ide.snapshot();
  const host = new NatlangHost({ host: { ide, drainEvents: () => ide.drainEvents() } });
  const complete = turn => turn.messages.filter(m => m.role === 'assistant').length > 1;
  try {
    const edited = await host.run({ source: { kind: 'file', path: editPath },
      inputs: { request: 'Make output lowercase' },
      options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (complete(turn)) return { calls: [], text: 'done', completion_tokens: 1 };
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function edit(')) return { calls: [
          ['call', { function: 'inspect', to: 'let/snapshot' }],
          ['call', { function: 'interpret', to: 'let/patch', inputs: {
            request: 'args/request', snapshot: 'let/snapshot' } }],
          ['call', { function: 'apply', to: 'return', inputs: { patch: 'let/patch' } }],
        ], completion_tokens: 1 };
        return { calls: [['write', { path: 'return', value: {
          name: 'main', start: 18, end: 29, text: 'toLowerCase',
          expected_revision: original.revision } }]], completion_tokens: 1 };
      } });
    assert.equal(edited.value.status, 'edited');
    assert.notEqual(ide.snapshot().revision, original.revision);
    const run = await host.run({ source: { kind: 'file', path: runPath },
      inputs: { input: 'HeLLo' }, options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (complete(turn)) return { calls: [], text: 'done', completion_tokens: 1 };
        return { calls: [
          ['call', { function: 'inspect', to: 'let/snapshot' }],
          ['call', { function: 'check', to: 'let/checked', inputs: {
            revision: 'let/snapshot/revision' } }],
          ['call', { function: 'execute', to: 'return', inputs: {
            input: 'args/input', revision: 'let/snapshot/revision' } }],
        ], completion_tokens: 1 };
      } });
    assert.equal(run.outcome.kind, 'done', JSON.stringify(run.outcome));
    assert.equal(run.value.status, 'done');
    assert.equal(run.value.value_text, '"hello"');
    const view = await host.run({ source: { kind: 'file', path: viewPath },
      inputs: { run_id: run.value.run_id, index: 0 },
      options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (complete(turn)) return { calls: [], text: 'done', completion_tokens: 1 };
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function view(')) return { calls: [
          ['call', { function: 'inspect', to: 'let/snapshot' }],
          ['call', { function: 'check', to: 'let/checked', inputs: {
            revision: 'let/snapshot/revision' } }],
          ['call', { function: 'trace', to: 'let/event', inputs: {
            run_id: 'args/run_id', index: 'args/index' } }],
          ['call', { function: 'describe', to: 'let/page', inputs: {
            snapshot: 'let/snapshot', checked: 'let/checked', event: 'let/event' } }],
          ['call', { function: 'render', to: 'return', inputs: { page: 'let/page' } }],
        ], completion_tokens: 1 };
        return { calls: [['write', { path: 'return', value: { title: 'IDE <view>',
          panels: [{ heading: 'Source', body: 'return args.input.toLowerCase();' },
            { heading: 'Trace', body: '<recorded event>' }] } }]], completion_tokens: 1 };
      } });
    assert.equal(view.outcome.kind, 'done');
    assert.match(view.value, /IDE &lt;view&gt;/);
    assert.match(view.value, /&lt;recorded event&gt;/);
    assert.ok(ide.inspect(run.value.run_id, 0).event_json.length > 0);
    ide.addScenario({ id: 'lowercase', inputs: { input: 'HeLLo' }, expected: 'hello' });
    assert.equal((await ide.evaluate(['lowercase'])).passed, true);
  } finally { host.close(); }
});

test('stale edits and invalid syntax remain visible without replacing a checked revision', async () => {
  const ide = new IdeWorkbench({ main: { args: { input: 'Text' }, returns: 'Text',
    code: 'return args.input;' } }, 'main');
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
