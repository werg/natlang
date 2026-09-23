import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { IdeWorkbench, requestEdit, viewTrace } from '../../applications/dist/ide/index.js';
import { scriptedModel } from './support/natlang.mjs';

const MAIN = 'export function main(input: { text: string }): string { return input.text.toUpperCase(); }\n';

test('natlang edits, runs, inspects and renders a source-pinned project', async () => {
  const model = scriptedModel(opening => {
    if (opening.includes('Translate the request into one exact text edit')) return 'const file = snapshot.files[0];\n' +
      'const start = file.source.indexOf("toUpperCase");\n' +
      'return { name: file.name, start, end: start + "toUpperCase".length, text: "toLowerCase", expected_revision: snapshot.revision }';
    if (opening.includes('Describe a useful editor view')) return 'return { title: "IDE <view>", panels: [' +
      '{ heading: "Source", body: snapshot.files[0].source }, { heading: "Diagnostics", body: checked.status }, ' +
      '{ heading: "Trace", body: "<recorded event>" }] }';
    if (opening.includes('Lower-case')) return 'return text.toLowerCase()';
    return null;
  });
  const runtime = createNatlangRuntime({ model: model.driver });
  const ide = new IdeWorkbench({ 'main.ts': MAIN }, 'main.ts', runtime);
  const original = ide.snapshot();
  const edited = await runtime.run(() => requestEdit(ide, 'Make the output lowercase'));
  assert.equal(edited.status, 'edited', edited.detail);
  assert.notEqual(ide.snapshot().revision, original.revision);
  assert.match(ide.snapshot().files[0].source, /toLowerCase/);
  const run = await ide.run({ text: 'HeLLo' });
  assert.equal(run.status, 'done', run.detail);
  assert.equal(run.value_text, '"hello"');
  const natural = new IdeWorkbench({ 'lower.nl': '---\nargs:\n  text: string\nreturns: string\n---\nLower-case text.\n' }, 'lower.nl', runtime);
  const traced = await natural.run({ text: 'ABC' });
  assert.equal(traced.value_text, '"abc"');
  assert.ok(traced.trace_events > 0);
  const html = await runtime.run(() => viewTrace(natural, traced.run_id, 0));
  assert.match(html, /IDE &lt;view&gt;/);
  assert.match(html, /&lt;recorded event&gt;/);
  assert.match(natural.inspect(traced.run_id, 0).event_json, /manifest/);
  ide.addScenario({ id: 'lowercase', inputs: { text: 'HeLLo' }, expected: 'hello' });
  assert.equal((await ide.evaluate(['lowercase'])).passed, true);
});

test('stale edits and invalid syntax remain visible without replacing a checked revision', async () => {
  const ide = new IdeWorkbench({ 'main.ts': MAIN }, 'main.ts', createNatlangRuntime());
  const base = ide.snapshot().revision;
  assert.equal(ide.edit({ name: 'main.ts', start: 0, end: 0, text: 'return (', expected_revision: base }).status, 'edited');
  assert.equal(ide.check().status, 'invalid');
  assert.equal((await ide.run({ text: 'x' })).status, 'invalid-source');
  assert.equal(ide.edit({ name: 'main.ts', start: 0, end: 0, text: 'x', expected_revision: base }).status, 'stale');
  assert.equal(ide.check(base).status, 'checked');
});
