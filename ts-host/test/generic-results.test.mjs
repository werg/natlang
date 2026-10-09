import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as runtimeNamespace from '../dist/index.js';
import { CallStore, compileVirtualProject, createNatlangRuntime, defineNatlang, invokeAt, loadVirtualNatlang } from '../dist/index.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { isNeuraleseRef } from '../dist/native/neuralese.js';
import { modelTurnsSoFar } from '../dist/native/agent.js';
import { parseNatlang, PATH_ONLY } from '../dist/runtime/loader.js';

// Representation chosen by use (plans/neuralese/DECISIONS.md 2026-10-09): one function body, a generic result, and
// each call site runs the instance its expected type picks.

const GIST = '---\nargs:\n  text: string\ngeneric:\n  R: string | Neuralese<string>\nreturns: R\n---\nGive the gist of text.\n';
const PLAIN = '---\nargs:\n  text: string\nreturns: string\n---\nGive the gist of text.\n';
const CONSUME = '---\nargs:\n  notes: Neuralese<string>\nreturns: string\n---\nAnswer from the notes.\n';

const standIn = () => {
  const store = new MemoryNeuraleseStore();
  return { store, port: new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1') };
};

/** A Neuralese-capable scripted model: every call returns `value`; `requests` keeps what each call was sent. */
const model = (value = 'the gist') => {
  const requests = [];
  const driver = Object.assign(async request => {
    requests.push(request);
    return { calls: [['return_result', { status: 'success', value }]] };
  }, { neuralese: true });
  return { driver, requests };
};

test('a generic result is declared in the frontmatter; the record keeps its crisp instance as returns', () => {
  const record = parseNatlang('gist.nl', GIST, {}, PATH_ONLY);
  assert.equal(record.returns, 'string');
  assert.deepEqual(record.generic, { name: 'R', constraint: 'string | Neuralese<string>', crisp: 'string',
    neuralese: 'Neuralese<string>', dialect: 'DefaultDialect' });
  assert.equal(parseNatlang('plain.nl', PLAIN, {}, PATH_ONLY).generic, undefined);
  for (const [text, message] of [
    [GIST.replace('returns: R', 'returns: string'), /generic declares R for the result, so the result is returns: R/],
    [GIST.replace('R: string | Neuralese<string>', 'R: string | Neuralese<number>'), /R is a crisp type and its Neuralese form/],
    [GIST.replace('R: string | Neuralese<string>', 'R: string'), /R is a crisp type and its Neuralese form/],
    [GIST.replace('---\nGive', 'readout: decision\n---\nGive'), /readout: decision scores a finite crisp result/],
    [GIST.replace('  text: string', '  text: R'), /R is the generic result/],
  ]) assert.throws(() => defineNatlang(text, { name: 'gist' }), message);
});

test('one function runs as text or as a template write, as the call asks', async () => {
  const { store, port } = standIn();
  const { driver, requests } = model();
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  const gist = defineNatlang(GIST, { name: 'gist' });
  // A plain call has no expected type: the crisp instance, as ordinary execution.
  assert.equal(await runtime.run(() => gist('a long text')), 'the gist');
  assert.equal(requests.at(-1).template, undefined);
  assert.equal(await runtime.run(() => invokeAt(gist, ['a long text'], { kind: 'crisp' })), 'the gist');
  // The Neuralese instance: the same body, its first reply forced to return_result and the value written.
  const block = await runtime.run(() => invokeAt(gist, ['a long text'], { kind: 'neuralese' }));
  assert.ok(isNeuraleseRef(block), JSON.stringify(block));
  assert.equal(block.$neuralese.type, 'Neuralese<string>');
  assert.equal(requests.at(-1).template.value, 'write');
  assert.equal(requests.at(-1).template.call, 'return_result');
  assert.equal((await store.meta(block.$neuralese.id)).dialect, 'nd:natlang@1');
  // The reader's dialect, named explicitly, is the same instance.
  assert.ok(isNeuraleseRef(await runtime.run(() => invokeAt(gist, ['x'], { kind: 'neuralese', dialect: 'nd:natlang@1' }))));
  await assert.rejects(runtime.run(() => invokeAt(gist, ['x'], { kind: 'neuralese', dialect: 'nd:other@1' })),
    /neuralese-dialect-mismatch: gist is asked for Neuralese in dialect "nd:other@1", but its generic result writes "nd:natlang@1"/);
  assert.throws(() => invokeAt(gist, ['x'], { kind: 'soft' }), /a representation is \{ kind: "crisp" \}/);
});

test('call records keep the definition and the representation it ran at', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-generic-'));
  const calls = CallStore.open(root);
  try {
    const { store, port } = standIn();
    const runtime = createNatlangRuntime({ model: model().driver, neuralese: { store, port }, calls });
    const gist = defineNatlang(GIST, { name: 'gist' });
    await runtime.run(() => gist('first'));
    await runtime.run(() => invokeAt(gist, ['second'], { kind: 'neuralese' }));
    const records = calls.calls({ limit: 10 }).map(row => calls.call(row.call_id));
    const byInput = Object.fromEntries(records.map(record => [calls.value(record.inputs.text), record.definition]));
    assert.deepEqual(byInput.first.representation, { kind: 'crisp' });
    assert.deepEqual(byInput.second.representation, { kind: 'neuralese' });
    assert.deepEqual(byInput.first.generic, { name: 'R', constraint: 'string | Neuralese<string>' });
    assert.equal(byInput.first.id, byInput.second.id, 'one definition');
    assert.equal(byInput.first.returns, 'string');
    assert.equal(byInput.second.returns, 'Neuralese<string>');
    assert.equal(byInput.second.readout, 'template');
    assert.notEqual(byInput.first.key, byInput.second.key, 'each instance is compiled and specialized on its own');
  } finally { calls.close(); rmSync(root, { recursive: true, force: true }); }
});

test('without a reader dialect the Neuralese instance fails, naming the call site; a crisp function teaches the generic form', async () => {
  const runtime = createNatlangRuntime({ model: model().driver });
  const gist = defineNatlang(GIST, { name: 'gist' });
  await assert.rejects(runtime.run(() => invokeAt(gist, ['x'], { kind: 'neuralese' }, 'app.ts:7')),
    /neuralese-unsupported-backend: gist is called for its Neuralese<string> result at app\.ts:7, but this runtime has no Neuralese reader/);
  assert.equal(await runtime.run(() => gist('x')), 'the gist', 'the crisp instance needs no Neuralese');
  const plain = defineNatlang(PLAIN, { name: 'plain' });
  await assert.rejects(runtime.run(() => invokeAt(plain, ['x'], { kind: 'neuralese' })),
    /plain returns string, so it cannot run for Neuralese<string>\. .*declare `generic: \{ R: string \| Neuralese<string> \}` with `returns: R`/s);
  assert.equal(await runtime.run(() => invokeAt(plain, ['x'], { kind: 'crisp' })), 'the gist');
});

const PROJECT = {
  'gist.nl': GIST,
  'plain.nl': PLAIN,
  'consume.nl': CONSUME,
  'main.ts': `import gist from './gist.nl';
import consume from './consume.nl';
export async function soft(text: string): Promise<Neuralese<string>> {
  const block: Neuralese<string> = await gist(text);
  return block;
}
export async function crisp(text: string): Promise<string> {
  const words = await gist(text);
  return words.toUpperCase();
}
export async function passed(text: string): Promise<string> {
  return await consume(await gist(text));
}
`,
};

test('the compiler instantiates each call from its expected type: a typed const, a typed parameter, or nothing', async () => {
  const compiled = compileVirtualProject({ files: PROJECT }, runtimeNamespace, { target: 'node' });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  assert.match(Object.entries(compiled.declarations).find(([path]) => path.endsWith('gist.d.nl.ts'))[1],
    /NatlangGenericFunction<\[text: string\], string \| Neuralese<string>, string>/);
  const main = Object.entries(compiled.outputs).find(([path]) => path.endsWith('main.js'))[1];
  const sites = [...main.matchAll(/instantiate\([^,]+, (\{[^}]*\}), "([^"]+)"\)/g)].map(match => [JSON.parse(match[1]).kind, match[2]]);
  assert.deepEqual(sites, [['neuralese', 'main.ts:4'], ['crisp', 'main.ts:8'], ['neuralese', 'main.ts:12']]);

  const { store, port } = standIn();
  const { driver, requests } = model();
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  const app = compiled.require('main.ts');
  const block = await runtime.run(() => app.soft('a long text'));
  assert.ok(isNeuraleseRef(block));
  assert.equal(requests.at(-1).template.value, 'write');
  assert.equal(await runtime.run(() => app.crisp('a long text')), 'THE GIST');
  assert.equal(requests.at(-1).template, undefined);
  const before = requests.length;
  assert.equal(await runtime.run(() => app.passed('a long text')), 'the gist');
  assert.equal(requests[before].template.value, 'write', 'the producer wrote the block its consumer reads');

  // The call site is named when the runtime cannot read Neuralese.
  await assert.rejects(createNatlangRuntime({ model: driver }).run(() => app.soft('x')),
    /neuralese-unsupported-backend: gist is called for its Neuralese<string> result at main\.ts:4/);
});

test('a crisp result used where Neuralese is expected is reported with the generic form', () => {
  const compiled = compileVirtualProject({ files: { 'plain.nl': PLAIN,
    'main.ts': "import plain from './plain.nl';\nexport async function soft(text: string) {\n  const n: Neuralese<string> = await plain(text);\n  return n;\n}\n" } },
    runtimeNamespace, { target: 'node' });
  assert.equal(compiled.ok, false);
  const hint = compiled.diagnostics.find(item => item.code === 'neuralese-crisp-result');
  assert.ok(hint, JSON.stringify(compiled.diagnostics));
  assert.equal(hint.line, 3);
  assert.match(hint.message, /plain returns string, and this call's result is used as Neuralese<string>\. Declare plain's result representation-generic \(`generic: \{ R: string \| Neuralese<string> \}` with `returns: R`\)/);
  assert.ok(compiled.diagnostics.some(item => item.code === 'typescript'), 'the type error itself stays');
});

test('eval code instantiates generic results from its own expected types', async () => {
  const files = {
    'outer.nl': '---\nargs:\n  text: string\nreturns: Neuralese<string>\n---\nUse gist on text for a block.\n',
    'outer/gist.nl': GIST,
    'words.nl': '---\nargs:\n  text: string\nreturns: string\n---\nUse gist on text for words.\n',
    'words/gist.nl': GIST,
  };
  const { store, port } = standIn();
  const requests = [];
  const driver = Object.assign(async request => {
    requests.push(request);
    if (requests.length > 40) throw new Error('runaway');
    const opening = JSON.stringify(request.messages);
    if (String(request.messages[1].content).includes('Give the gist of text'))
      return { calls: [['return_result', { status: 'success', value: 'the gist' }]] };
    if (modelTurnsSoFar(request.messages) === 0)
      return { calls: [['eval', { code: opening.includes('for a block') ?
        'const g: Neuralese<string> = await gist(text);\nreturn g;' : 'const g = await gist(text);\nreturn g;' }]] };
    return { calls: [['return_result', { status: 'success' }]] };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: driver, neuralese: { store, port } });
  const outer = loadVirtualNatlang(files, 'outer.nl');
  const soft = await runtime.run(() => outer('a long text'));
  assert.ok(isNeuraleseRef(soft), JSON.stringify(soft));
  const gistRequests = requests.filter(request => JSON.stringify(request.messages).includes('Give the gist of text'));
  assert.equal(gistRequests.at(-1).template?.value, 'write');
  const words = loadVirtualNatlang(files, 'words.nl');
  assert.equal(await runtime.run(() => words('a long text')), 'the gist');
  const after = requests.filter(request => JSON.stringify(request.messages).includes('Give the gist of text'));
  assert.equal(after.at(-1).template, undefined, 'an unannotated local runs the crisp instance');
});
