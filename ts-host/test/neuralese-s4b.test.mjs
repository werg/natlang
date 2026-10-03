import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { buildProject, createNatlangRuntime, loadNatlang, save, softFunction } from '../dist/index.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { neuraleseSentinel, sourceWithLiteralCalls } from '../dist/native/neuralese.js';

const runtimeModule = { url: new URL('../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

const standIn = (width = 8) => {
  const store = new MemoryNeuraleseStore();
  return { store, port: new StandInNeuralesePort(store, hashingEmbedder(width), width, 'nd:natlang@1') };
};
const neuraleseDriver = fn => Object.assign(fn, { neuralese: true });

function folder(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-s4b-'));
  for (const [path, content] of Object.entries(files)) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), content);
  }
  return root;
}

const TONE_SKILL = `---
name: tone
description: Matching a customer's tone.
natlang:
  scope:
    greeting: { type: string, value: Hello there }
---
Mirror the customer's register.
`;

test('a companion folder binds its .nz files and skills by default; imported blocks join the runtime store', async () => {
  const source = standIn();
  const memo = await source.port.write('the customer prefers email');
  const bytes = await save({ memo: { type: 'Neuralese<string>', value: neuraleseRef('Neuralese<string>', memo.id) } },
    { store: source.store, dialect: 'nd:natlang@1' });
  const root = folder({ 'reply.nl': '---\nargs:\n  ticket: string\nreturns: string\n---\nReply to the ticket using notes.memo.\n',
    'reply/notes.nz': bytes, 'reply/skills/tone/SKILL.md': TONE_SKILL });
  const reply = loadNatlang(join(root, 'reply.nl'), root);
  const target = standIn();
  assert.equal(await target.store.has(memo.id), false);
  const seen = [];
  const driver = neuraleseDriver(({ messages }) => { seen.push(messages);
    return { calls: [['return_result', { status: 'success', value: 'ok' }]] }; });
  const runtime = createNatlangRuntime({ model: driver, neuralese: target });
  assert.equal(await runtime.run(() => reply('late refund')), 'ok');
  const opening = JSON.stringify(seen[0]);
  const opened = seen[0].find(message => message.tool_calls?.[0]?.id === 'scope_0').tool_calls[0].function.arguments;
  const reading = Array.isArray(opened) ? opened.map(part => part.text ?? '').join('') : typeof opened === 'string' ? JSON.parse(opened).code : opened.code;
  assert.match(reading, /const greeting: string = \\"Hello there\\";\s*\/\/ from skill tone/);
  assert.match(opening, /notes/);
  assert.match(opening, /- tone: Matching a customer's tone\./);
  assert.ok(Array.isArray(opened) && opened.some(part => part.type === 'neuralese' && part.id === memo.id),
    'the bound .nz block reaches the call as a literal');
  assert.equal(await target.store.has(memo.id), true, 'the imported block was adopted by the runtime store');
});

test('a .nz file name must be an identifier and must not shadow an item', () => {
  const root = folder({ 'f.nl': '---\nreturns: string\n---\nSay hi.\n', 'f/g.nl': '---\nreturns: string\n---\nSay hi.\n', 'f/g.nz': new Uint8Array([1]) });
  assert.throws(() => loadNatlang(join(root, 'f.nl'), root), /both an item and a \.nz file/);
});

test('a soft body grants no authority (fn-soft-no-authority)', async () => {
  const { store, port } = standIn();
  const body = await port.write('send the customer an email about their refund');
  const notify = softFunction({ type: '(t: string) => string', body: body.id });
  const results = [];
  let turn = 0;
  const driver = neuraleseDriver(({ messages }) => {
    if (turn++) { results.push(JSON.stringify(messages.at(-1).content)); return { calls: [['return_result', { status: 'success', value: 'no' }]] }; }
    return { calls: [['eval', { code: 'await mail.send("customer", "refund"); return "sent";' }]] };
  });
  const sent = [];
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port },
    services: { mail: { send: (...args) => { sent.push(args); return 'ok'; } } },
    serviceDeclarations: { mail: 'export function send(to: string, text: string): string;' }, serviceScopes: { mail: ['mailer.nl'] } });
  assert.equal(await runtime.run(() => notify('refund')), 'no');
  assert.match(results.join('\n'), /mail is not defined/);
  assert.equal(sent.length, 0);
});

const SCOPE = { types: {}, inputs: [], locals: [{ name: 'rubric', type: 'string', mutable: false }], captures: [], imports: [], returns: 'number' };

test('soft-bodied nl takes only nl.with captures; text nl.with lists them (soft-literal-requires-explicit-captures)', () => {
  const analyze = code => analyzeEvalSnippet(code, SCOPE);
  const errors = result => result.diagnostics.filter(item => item.severity === 'error').map(item => item.code);
  const id = `nz1_${'a'.repeat(52)}`;
  // A soft body without nl.with captures nothing, though rubric is in scope (and may be what the body is about).
  const soft = analyze(sourceWithLiteralCalls('const f: Neuralese<(t: string) => Promise<string>> = nl`' + neuraleseSentinel(id) + '`; return 1;'));
  assert.deepEqual(errors(soft), []);
  assert.equal(soft.plans[0].softBody, id);
  assert.deepEqual(soft.plans[0].captures, []);
  const untyped = analyze(sourceWithLiteralCalls('const f = nl`' + neuraleseSentinel(id) + '`; return 1;'));
  assert.ok(errors(untyped).includes('neuralese-untyped-literal'));
  const listed = analyze('let count = 0; const f = nl.with({ rubric, count: live(count) })`Add items to count using rubric.`; await f([1]); return count;');
  assert.deepEqual(errors(listed), []);
  assert.deepEqual(listed.plans[0].captures.map(capture => [capture.name, capture.mode]), [['rubric', 'snapshot'], ['count', 'live']]);
  assert.ok(errors(analyze('const k = 1; const f = nl.with({ k: live(k) })`x`; return 1;')).includes('nl-const-capture-write'));
  assert.ok(errors(analyze('const f = nl.with<() => Promise<string>>({ rubric })`Use \\`other\\`.`; return 1;')).includes('nl-unknown-name'));
  // Text nl keeps implicit captures by mention.
  assert.deepEqual(analyze('const f = nl<() => Promise<string>>`Use rubric.`; return 1;').plans[0].captures.map(capture => capture.name), ['rubric']);
});

test('a project lowers nl.with: snapshot captures are read once, live captures are written back', async () => {
  const root = folder({
    'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext', strict: true,
      skipLibCheck: true }, include: ['*.ts'] }),
    'package.json': JSON.stringify({ type: 'module' }),
    'app.ts': "import { nl, live } from '@natlang/node';\n" + 'export async function run(ticket: string): Promise<string> {\n  let count = 0;\n  let rubric = "billing first";\n' +
      '  const label = nl.with({ rubric, count: live(count) })<(t: string) => string>`Label t using rubric; add one to count.`;\n' +
      '  rubric = "changed";\n  const result = await label(ticket);\n  return `${result}:${count}`;\n}\n',
  });
  const outDir = mkdtempSync(join(tmpdir(), 'natlang-s4b-out-'));
  const built = buildProject({ project: root, outDir, runtimeModule });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  const app = await import(pathToFileURL(join(outDir, 'app.js')).href);
  const seen = [];
  const runtime = createNatlangRuntime({ agent: async session => {
    seen.push(Object.fromEntries(Object.entries(session.lam.captures).map(([name, cell]) => [name, cell.get()])));
    const event = await session.applyAsync('eval', { code: 'count = count + 2; return rubric;' });
    assert.equal(event.kind, 'ok', event.text);
  } });
  assert.equal(await runtime.run(() => app.run('charged twice')), 'billing first:2');
  assert.deepEqual(seen, [{ rubric: 'billing first', count: 0 }]);
});
