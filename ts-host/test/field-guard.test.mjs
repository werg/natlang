/**
 * Reads of fields a value's declared type does not have: rejected before the eval runs when the checker can see them
 * (undeclared-field), and refused at run time through values whose static type was lost (field guards).
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

test('through a value whose static type was lost, the field guard refuses the same read at run time', async () => {
  const { error } = await run(`const message: any = await svc.entry(id);
    const calls = message.content ?? [];
    return String(calls.length);`);
  assert.match(error, /svc\.entry\(…\) has no field content: its type EntryRecord has id, kind, model/);
  const nested = await run(`const message = await svc.entry(id);
    return message!.model![0]!.content.filter((part: any) => part.name === 'x').length + "";`);
  assert.match(nested.error, /svc\.entry\(…\)\.model\[0\]\.content\[0\] has no field name: its type Item has type, id/);
  const input = await run(`const anything: any = item; return anything.name ?? 'none';`);
  assert.match(input.error, /item has no field name: its type Item has type, id/);
});

test('guarded values read like the data they hold: present undeclared fields, JSON, spreading and copies', async () => {
  assert.deepEqual(await run(`const message: any = await svc.entry(id);
    const copy = { ...message, model: undefined };
    return [message.extra, JSON.stringify(message.model[0].content), Object.keys(copy).join("|"), Array.isArray(message.model)].join(" ");`),
  { value: 'kept [{"type":"toolCall","id":"c1"}] id|kind|extra|model true' });
});

test('a field one member of a union declares is how code tells the members apart, statically and at run time', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-fields-'));
  writeFileSync(join(root, 'types.ts'), 'export type Applied = { base: string; content: string } | { error: string };\n');
  writeFileSync(join(root, 'g.nl'), '---\nargs:\n  applied: Applied\nreturns: string\n---\nSay what applied holds.\n');
  const model = scriptedModel(() => `const loose: any = applied; return applied.error ? "error " + applied.error : loose.content;`);
  const runtime = createNatlangRuntime({ model: model.driver });
  const g = loadNatlang(join(root, 'g.nl'));
  assert.equal(await runtime.run(() => g({ base: 'b', content: 'c' })), 'c');
  assert.equal(await runtime.run(() => g({ error: 'no match' })), 'error no match');
});

test('cyclic data behind a guard stays finite for everything that walks it', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-fields-'));
  writeFileSync(join(root, 'h.nl'), '---\nargs: {}\nreturns: string\n---\nWalk the ring.\n');
  const model = scriptedModel(() => `const ring = await graph.ring(); const kept = ring; return String(kept.next.next.id) + ":" + JSON.stringify(Object.keys(ring));`);
  const runtime = createNatlangRuntime({ model: model.driver });
  const a = { id: 1, next: null }, b = { id: 2, next: a }; a.next = b;
  const value = await runtime.run(() => loadNatlang(join(root, 'h.nl'))(), {
    services: { graph: { ring: async () => a } },
    serviceDeclarations: { graph: 'export type Node = { id: number; next: Node | null };\nexport function ring(): Promise<Node>;' } });
  assert.equal(value, '1:["id","next"]');
});
