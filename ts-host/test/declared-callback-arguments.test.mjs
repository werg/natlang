import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, defineNatlang, CallStore } from '../dist/index.js';
import { definitionKey } from '../dist/calls/recorder.js';
import { inlineCallable } from '../dist/runtime/callable.js';
import { planDefinition } from '../dist/runtime/lowered.js';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { analyzeInlineLambdas } from '../dist/compiler/inline.js';
import { definitionFor } from '../dist/calls/offline.js';
import { scriptedModel } from './support/natlang.mjs';

async function mapped(declaration, application = 'items.map(judge)') {
  const root = defineNatlang('---\nargs: { items: "string[]" }\nreturns: "string[]"\n---\nMap the declared callback.\n');
  const source = `${declaration}; return await Promise.all(${application});`;
  const model = scriptedModel(opening => opening.includes('Map the declared callback.') ? source : 'return "ok";');
  const store = CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-callback-')));
  try {
    const output = await createNatlangRuntime({ model: model.driver, calls: store }).run(() => root(['first', 'second']));
    assert.deepEqual(output, ['ok', 'ok']);
    const records = store.calls().map(summary => store.call(summary.call_id)).filter(record => record.parent_call_id);
    assert.equal(records.length, 2);
    return records.map(record => ({ names: Object.keys(record.inputs), values: Object.fromEntries(Object.entries(record.inputs).map(([name, ref]) => [name, store.value(ref)])), identity: record.definition, rebuilt: definitionFor(store, record).definition })).sort((a,b) => String(Object.values(a.values)[0]).localeCompare(String(Object.values(b.values)[0])));
  } finally { const path = store.root; store.close(); rmSync(path, { recursive: true, force: true }); }
}

for (const [parameters, names] of [
  ['item: string', ['item']],
  ['item: string, index: number', ['item', 'index']],
  ['item: string, index: number, array: string[]', ['item', 'index', 'array']],
]) test(`Array.map preserves exactly declared callback parameters: ${names.join(', ')}`, async () => {
  const records = await mapped(`const judge: (${parameters}) => Promise<string> = nl<string>\`Describe the item.\``);
  for (const [index, record] of records.entries()) {
    assert.deepEqual(record.names, names);
    assert.equal(record.values.item, ['first', 'second'][index]);
    if (names.includes('index')) assert.equal(record.values.index, index);
    if (names.includes('array')) assert.deepEqual(record.values.array, ['first', 'second']);
    assert.equal(record.identity.declaredParameters, true);
    assert.equal(record.rebuilt.declaredParameters, true);
    assert.equal(definitionKey(record.rebuilt), record.identity.key);
  }
});

test('explicit function type closes callback boundary', async () => {
  const records = await mapped('const judge = nl<(item: string) => Promise<string>>`Describe the item.`');
  assert.deepEqual(records.map(record => record.names), [['item'], ['item']]);
});

test('open and legacy unmarked lowered callables preserve extra positional values', async () => {
  for (const openParameters of [true, false]) {
    const target = { natlang: 'string', text: 'string', aliases: {} };
    const plan = { definitionId: 'legacy', sourceSpan: { file: 'legacy.ts', line: 1 }, instructions: 'Describe all supplied values.', parameters: [{ name: 'input', type: target }], returns: target, captures: [], inheritedCodebaseRevision: '', ...(openParameters ? { openParameters: true } : {}) };
    const definition = planDefinition(JSON.parse(JSON.stringify(plan)));
    const callable = inlineCallable(definition, definition.body, {});
    const store = CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-open-callback-')));
    try {
      const model = scriptedModel(() => 'return "ok";');
      assert.equal(await createNatlangRuntime({ model: model.driver, calls: store }).run(() => callable('first', 0, ['first'])), 'ok');
      const record = store.call(store.calls()[0].call_id);
      assert.deepEqual(Object.values(record.inputs).map(ref => store.value(ref)), ['first', 0, ['first']]);
      assert.equal(record.definition.declaredParameters, undefined);
    } finally { const path = store.root; store.close(); rmSync(path, { recursive: true, force: true }); }
  }
});

test('ordinary rest forwarding preserves declared index without array leakage', async () => {
  const records = await mapped('const judge: (item: string, index: number) => Promise<string> = nl<string>`Describe the item.`', 'items.map((...args: [string, number, string[]]) => judge(args[0], args[1]))');
  assert.deepEqual(records.map(record => record.names), [['item', 'index'], ['item', 'index']]);
});

test('declared boundary participates in call compilation identity without changing legacy keys', () => {
  const definition = { id: 'inline', body: 'Describe.', params: [{ name: 'item', type: 'string' }], returns: 'string', types: {}, subtype: 'function' };
  const legacy = definitionKey(definition);
  assert.equal(definitionKey({ ...definition, declaredParameters: undefined }), legacy);
  assert.notEqual(definitionKey({ ...definition, declaredParameters: true }), legacy);
});

test('contextual map contracts survive lowered-plan JSON roundtrip; unsupported rest stays rejected', () => {
  const source = 'async function run(items: string[]) { return await Promise.all(items.map(nl<string>`Describe item.`)); }';
  const program = createVirtualProgram({ '/callback.ts': source });
  const result = analyzeInlineLambdas(program, [program.getSourceFile('/callback.ts')]);
  assert.deepEqual(result.diagnostics, []);
  assert.equal(result.plans[0].declaredParameters, true);
  const definition = planDefinition(JSON.parse(JSON.stringify(result.plans[0])));
  assert.equal(definition.declaredParameters, true);
  assert.equal(definition.params.length, 3);
  const rest = 'const judge = nl<(...items: string[]) => Promise<string>>`Describe items.`;';
  const restProgram = createVirtualProgram({ '/rest.ts': rest });
  const checked = analyzeInlineLambdas(restProgram, [restProgram.getSourceFile('/rest.ts')]);
  assert.ok(checked.diagnostics.some(diagnostic => diagnostic.code === 'nl-spread'));
});

test('inline capture rebinding retains declared callback boundary', async () => {
  const records = await mapped('const judge: (item: string) => Promise<string> = nl.with<string>({})`Describe the item.`', 'items.map(judge.with({}))');
  for (const record of records) {
    assert.deepEqual(record.names, ['item']);
    assert.equal(record.identity.declaredParameters, true);
  }
});
