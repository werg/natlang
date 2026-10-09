import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { IdeWorkbench, requestEdit, viewTrace, locate, assembleView } from '../../applications/dist/ide/index.js';
import { scriptedModel } from './support/natlang.mjs';

const MAIN = 'export function main(input: { text: string }): string { return input.text.toUpperCase(); }\n';

test('natlang edits, runs, inspects and renders a source-pinned project', async () => {
  const model = scriptedModel(opening => {
    if (opening.includes('Propose one text edit')) return 'return { anchor: "toUpperCase", placement: "replace", text: "toLowerCase" }';
    if (opening.includes('Write a reading of the check result')) return 'return "no errors"';
    if (opening.includes('Explain one recorded trace event')) return 'return "<recorded event>"';
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
  assert.match(html, /Editor view of revision/);
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

test('locate computes offsets and the expected revision from a quoted anchor', () => {
  const snapshot = { revision: 'r1', root: 'a.ts', files: [{ name: 'a.ts', source: 'one two one' }, { name: 'b.ts', source: 'x' }] };
  assert.deepEqual(locate(snapshot, 'a.ts', { anchor: 'two', placement: 'replace', text: '2' }).patch,
    { name: 'a.ts', start: 4, end: 7, text: '2', expected_revision: 'r1' });
  assert.equal(locate(snapshot, 'a.ts', { anchor: 'two', placement: 'insert-before', text: 'x' }).patch.start, 4);
  const after = locate(snapshot, 'a.ts', { anchor: 'two', placement: 'insert-after', text: 'x' }).patch;
  assert.deepEqual([after.start, after.end], [7, 7]);
  assert.equal(locate(snapshot, 'a.ts', { anchor: 'one', placement: 'replace', text: '' }).occurrences, 2);
  assert.equal(locate(snapshot, 'a.ts', { anchor: 'missing', placement: 'replace', text: '' }).occurrences, 0);
  assert.equal(locate(snapshot, 'a.ts', { anchor: '', placement: 'replace', text: '' }).ok, false);
});

test('an ambiguous anchor returns to the model once with its count, then the edit is rejected', async () => {
  const prompts = [];
  const model = scriptedModel(opening => {
    if (!opening.includes('Propose one text edit')) return null;
    prompts.push(opening);
    return prompts.length === 1 ? 'return { anchor: "a", placement: "replace", text: "b" }' :
      'return { anchor: "a = 1", placement: "replace", text: "a = 2" }';
  });
  const runtime = createNatlangRuntime({ model: model.driver });
  const ide = new IdeWorkbench({ 'main.ts': 'const a = 1;\nconst aa = 3;\n' }, 'main.ts', runtime);
  const report = await runtime.run(() => requestEdit(ide, 'set a to 2'));
  assert.equal(report.status, 'edited', report.detail);
  assert.equal(prompts.length, 2);
  assert.equal(ide.snapshot().files[0].source, 'const a = 2;\nconst aa = 3;\n');
  const stubborn = scriptedModel(opening => opening.includes('Propose one text edit') ? 'return { anchor: "nowhere", placement: "replace", text: "" }' : null);
  const second = createNatlangRuntime({ model: stubborn.driver });
  const other = new IdeWorkbench({ 'main.ts': 'x' }, 'main.ts', second);
  const rejected = await second.run(() => requestEdit(other, 'edit'));
  assert.equal(rejected.status, 'rejected');
  assert.match(rejected.detail, /does not occur/);
});

test('with several files the model chooses the file', async () => {
  const model = scriptedModel(opening => {
    if (opening.includes('Choose the file a request concerns')) return 'return "b.ts"';
    if (opening.includes('Propose one text edit')) return 'return { anchor: "two", placement: "replace", text: "2" }';
    return null;
  });
  const runtime = createNatlangRuntime({ model: model.driver });
  const ide = new IdeWorkbench({ 'a.ts': 'one', 'b.ts': 'two' }, 'a.ts', runtime);
  assert.equal((await runtime.run(() => requestEdit(ide, 'change b'))).status, 'edited');
  assert.equal(ide.snapshot().files.find(file => file.name === 'b.ts').source, '2');
});

test('scenario comparison ignores key order; the view panels are built by crisp code', async () => {
  const runtime = createNatlangRuntime();
  const ide = new IdeWorkbench({ 'main.ts': 'export function main(input: { n: number }): { b: number, a: number } { return { b: input.n, a: 1 }; }\n' }, 'main.ts', runtime);
  ide.addScenario({ id: 'order', inputs: { n: 2 }, expected: { a: 1, b: 2 } });
  assert.equal((await ide.evaluate(['order'])).passed, true);
  const view = assembleView(ide.snapshot(), { status: 'invalid', revision: 'r', detail: 'main.ts: bad' }, 'one problem', 'explained', { index: 0, total: 3 });
  assert.deepEqual(view.panels.map(panel => panel.heading), ['Source', 'Diagnostics', 'Trace event 1 of 3']);
  assert.match(view.panels[0].body, /main\.ts/);
  assert.match(view.panels[1].body, /one problem[\s\S]*main\.ts: bad/);
});
