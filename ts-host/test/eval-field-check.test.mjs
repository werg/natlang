/**
 * Reads of fields a value's declared type does not have are rejected before the eval runs (undeclared-field), with the
 * type's fields named; reads the language allows (optional and union fields, callable intrinsics) are not.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, loadNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const ENTRIES = `export type Item = { type: string; id: string };
export type EntryRecord = { id: number; kind: string; model?: { role: string; content: Item[] }[] };
/** The entry with this id. */
export function entry(id: number): Promise<EntryRecord | null>;`;

/** Run one eval of `code` in a call of `f(id: number, item: Item): string`; returns the value or the eval's error text. */
async function run(code) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-fields-'));
  writeFileSync(join(root, 'types.ts'), 'export type Item = { type: string; id: string };\n');
  writeFileSync(join(root, 'f.nl'), '---\nargs:\n  id: number\n  item: Item\nreturns: string\n---\nLook up entry id with svc.entry.\n');
  const errors = [];
  const model = scriptedModel(() => code);
  const driver = async request => {
    const last = request.messages.at(-1);
    if (last.role === 'tool' && /^(?:rejected|error)|\nerror|undeclared-field|has no field/.test(String(last.content))) errors.push(String(last.content));
    return model.driver(request);
  };
  const runtime = createNatlangRuntime({ model: driver });
  const entries = { entry: async id => ({ id, kind: 'pi.assistant', extra: 'kept', model: [{ role: 'assistant', content: [{ type: 'toolCall', id: 'c1' }] }] }) };
  try {
    return { value: await runtime.run(() => loadNatlang(join(root, 'f.nl'))(3, { type: 'toolCall', id: 'c9' }),
      { services: { svc: entries }, serviceDeclarations: { svc: ENTRIES } }) };
  } catch { return { error: errors.join('\n') }; }
}

test('a read of a field the declared type lacks is rejected before the eval runs, naming the fields', async () => {
  const { error } = await run('const message = await svc.entry(id); const calls = message?.content ?? []; return String(calls.length);');
  assert.match(error, /undeclared-field: message has no field content: EntryRecord has id, kind, model\. Read the field that holds what you need\./);
  assert.match((await run('return item.name;')).error, /item has no field name: Item has type, id/);
});

test('declared, optional and present fields read as before, and writes are not checked', async () => {
  assert.deepEqual(await run(`const message = await svc.entry(id);
    const calls = message?.model?.[0]?.content ?? [];
    const out: any = {}; out.note = 'written';
    return calls.map(call => call.id).join(",") + ":" + out.note + ":" + item.id;`), { value: 'c1:written:c9' });
});

test('a field one member of a union declares is how code tells the members apart', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-fields-'));
  writeFileSync(join(root, 'types.ts'), 'export type Applied = { base: string; content: string } | { error: string };\n');
  writeFileSync(join(root, 'g.nl'), '---\nargs:\n  applied: Applied\nreturns: string\n---\nSay what applied holds.\n');
  const model = scriptedModel(() => `return applied.error ? "error " + applied.error : String((applied as any).content);`);
  const runtime = createNatlangRuntime({ model: model.driver });
  const g = loadNatlang(join(root, 'g.nl'));
  assert.equal(await runtime.run(() => g({ base: 'b', content: 'c' })), 'c');
  assert.equal(await runtime.run(() => g({ error: 'no match' })), 'error no match');
});
