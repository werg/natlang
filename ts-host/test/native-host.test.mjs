import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DesktopBindings, NatlangHost, NativeNatlangHost, NativeRuntime, NativeSourceWorkspace, TypeScriptEnvironment } from '../dist/index.js';

test('public NatlangHost defaults to the Python-free native interpreter', () => {
  assert.equal(NatlangHost, NativeNatlangHost);
});

test('public native host runs a program without starting Python', async () => {
  const host = new NativeNatlangHost();
  const result = await host.run({ source: { kind: 'program', program: { $lambda: {
    type: 'Lambda<{ item: Num }, Num>', code: 'return args.item * 4;', engine: 'typescript-host' } } },
    inputs: { item: 3 } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 12);
  host.close();
});

test('native file loader links companion functions and lexical types', async () => {
  const file = fileURLToPath(new URL('../../examples/triage/main.nl', import.meta.url));
  const host = new NativeNatlangHost();
  let looked = false;
  const result = await host.run({ source: { kind: 'file', path: file },
    inputs: { tickets: [], rubric: 'none' }, modelTurn: request => {
      looked = true;
      assert.ok(request.tools.some(tool => tool.function.name === 'run_function'));
      return { calls: [['report_blocker', { missing: 'No tickets were supplied for triage.' }]], completion_tokens: 8 };
    }, options: { model: { max_turns: 2 } } });
  assert.equal(looked, true);
  assert.equal(result.outcome.kind, 'quiesced');
  host.close();
});

test('retained native eval sees the same application object across runs', async () => {
  const state = { count: 0 };
  const environment = new TypeScriptEnvironment({ mode: 'retained', host: state });
  const host = new NativeNatlangHost({ environment });
  const source = { kind: 'program', program: { $lambda: { type: 'Lambda<{}, Num>',
    code: 'host.count++; return host.count;', engine: 'typescript-host' } } };
  assert.equal((await host.run({ source })).value, 1);
  assert.equal((await host.run({ source })).value, 2);
  assert.equal(state.count, 2);
  host.close(); environment.close();
});

test('host observations survive an eval failure after a native mutation', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-host-trace-'));
  const native = { count: 0, events: [], change() {
    this.count++; this.events.push({ operation: 'sample.changed', count: this.count });
    throw new Error('failed after mutation');
  }, drainEvents() { return this.events.splice(0); } };
  const host = new NativeNatlangHost({ host: native });
  try {
    const result = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', engine: 'typescript-host', code: 'return host.change();',
    } } }, tracePath: join(folder, 'run.jsonl') });
    assert.equal(result.outcome.kind, 'quiesced');
    assert.equal(native.count, 1);
    assert.ok(result.trace.some(event => event.kind === 'host' && event.event?.operation === 'sample.changed'));
  } finally { host.close(); rmSync(folder, { recursive: true, force: true }); }
});

test('native cancellation prevents model actions after an interrupted turn', async () => {
  const host = new NativeNatlangHost();
  const abort = new AbortController();
  const source = { kind: 'program', program: { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return 1.' } } };
  await assert.rejects(host.run({ source, signal: abort.signal, modelTurn: () => {
    abort.abort();
    return { calls: [['write', { path: 'return', type: 'Num', value: 1 }]], completion_tokens: 1 };
  } }), /aborted/);
  host.close();
});

test('native timeout returns while a model callback is still pending', async () => {
  const host = new NativeNatlangHost();
  const source = { kind: 'program', program: { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return 1.' } } };
  try {
    await assert.rejects(host.run({ source, modelTurn: () => new Promise(() => {}), timeoutMs: 15 }), /timed out/);
    const next = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', engine: 'typescript-host', code: 'return 2;' } } } });
    assert.equal(next.value, 2);
  } finally { host.close(); }
});

test('native host loads JSON program files and accepts a write with inferred slot type', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-native-file-'));
  const path = join(folder, 'program.json');
  writeFileSync(path, JSON.stringify({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return 12.' } }));
  const host = new NativeNatlangHost();
  let turn = 0;
  try {
    const result = await host.run({ source: { kind: 'file', path }, modelTurn: () => ++turn === 1 ?
      { calls: [['write', { path: 'return', value: 12 }]], completion_tokens: 1 } :
      { calls: [], text: 'done', completion_tokens: 1 } });
    assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 12);
  } finally { host.close(); rmSync(folder, { recursive: true, force: true }); }
});

test('native host consumes an async Fold stream in order', async () => {
  async function* source() { yield 2; yield 3; }
  const host = new NativeNatlangHost();
  try {
    const result = await host.run({ source: { kind: 'program', program: { $fold: { type: 'Fold<Num, Num>',
      init: 1, step: { $lambda: { type: 'Lambda<{ acc: Num, item: Num }, Num>',
        engine: 'typescript-host', code: 'return args.acc + args.item;' } } } } }, streams: { over: source() } });
    assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 6);
  } finally { host.close(); }
});

test('native crisp TypeScript can await an application method in the shared environment', async () => {
  const state = { count: 0, async next() { await Promise.resolve(); return ++this.count; } };
  const host = new NativeNatlangHost({ host: state, mode: 'retained' });
  const source = { kind: 'program', program: { $lambda: {
    type: 'Lambda<{}, Num>', engine: 'typescript-host', code: 'return await host.next();' } } };
  try {
    assert.equal((await host.run({ source })).value, 1);
    assert.equal((await host.run({ source })).value, 2);
    assert.equal(state.count, 2);
  } finally { host.close(); }
});

test('a native workspace versions source and can invoke a checked child from crisp eval', async () => {
  const workspace = new NativeSourceWorkspace({ double: { args: { item: 'Num' }, returns: 'Num',
    code: 'return args.item * 2;', engine: 'typescript-host' } }, 'double');
  const edited = workspace.edited('double', { args: { item: 'Num' }, returns: 'Num',
    code: 'return args.item * 3;', engine: 'typescript-host' });
  assert.notEqual(workspace.revision, edited.revision);
  assert.equal(workspace.describe().signature, 'double(item: Num) -> Num');
  assert.equal(workspace.typeCheck('Num', 'Num').fits, true);
  const host = new NativeNatlangHost({ host: { workspace } });
  try {
    const result = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{ item: Num }, Num>', engine: 'typescript-host',
      code: 'const child = await host.workspace.invoke("double", { item: args.item }); return child.value;',
    } } }, inputs: { item: 4 } });
    assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 8);
    const child = await workspace.invoke('double', { item: 4 }, { parentCallId: 'root@1' });
    assert.equal(child.trace[0].parent_call_id, 'root@1');
    assert.equal(child.source_revision, workspace.revision);
  } finally { host.close(); }
});

test('native declared effects await asynchronous application callbacks before completion', async () => {
  const host = new NativeNatlangHost();
  let settled = false;
  try {
    const result = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', effects: ['counter.add'], engine: 'typescript-host',
      code: 'return await fx.counter.add(3);',
    } } }, capabilities: { 'counter.add': async ([n]) => { await Promise.resolve(); settled = true; return n + 2; } } });
    assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 5);
    assert.equal(settled, true);
  } finally { host.close(); }
});

test('nested source invocations share the parent episode budget', async () => {
  const workspace = new NativeSourceWorkspace({ leaf: { returns: 'Num', instructions: 'Return one.' } }, 'leaf');
  const parent = new NativeRuntime({ maxEpisodes: 1 });
  const modelTurn = request => request.messages.some(message => message.role === 'tool' &&
    String(message.content).includes('ok')) ? { calls: [], text: 'done', completion_tokens: 1 } :
    { calls: [['write', { path: 'return', value: 1 }]], completion_tokens: 1 };
  const first = await workspace.invoke('leaf', {}, { parentRuntime: parent, modelTurn });
  assert.equal(first.outcome, 'done'); assert.equal(first.value, 1);
  assert.equal(parent.episodesStarted, 1);
  const second = await workspace.invoke('leaf', {}, { parentRuntime: parent, modelTurn });
  assert.equal(second.outcome, 'quiesced'); assert.equal(parent.episodesStarted, 1);
  parent.close();
});

test('native host runs TypeScript syntax, model eval and desktop bindings', async () => {
  const mapped = new NativeNatlangHost();
  try {
    const result = await mapped.run({ source: { kind: 'program', program: { $map: {
      type: 'Map<Num, Num>', over: [1, 2, 3], fn: { $lambda: {
        type: 'Lambda<{ item: Num }, Num>', engine: 'typescript-host',
        code: 'enum Scale { Double = 2 }; return args.item * Scale.Double;',
      } },
    } } } });
    assert.deepEqual(result.value, [2, 4, 6]);
  } finally { mapped.close(); }

  const native = { count: 0, increase(n) { this.count += n; return this.count; } };
  const modelHost = new NativeNatlangHost({ host: native, mode: 'retained' });
  let turn = 0;
  try {
    const result = await modelHost.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', instructions: 'Increment the counter and return it.',
    } } }, modelTurn: () => ++turn === 1 ?
      { calls: [['run_code', { engine: 'typescript-host', code: 'host.increase(4)' }]], completion_tokens: 1 } :
      turn === 2 ? { calls: [['write', { path: 'return', value: native.count }]], completion_tokens: 1 } :
        { calls: [], text: 'done', completion_tokens: 1 } });
    assert.equal(result.value, 4); assert.equal(native.count, 4);
  } finally { modelHost.close(); }

  const folder = mkdtempSync(join(tmpdir(), 'natlang-native-desktop-'));
  const path = join(folder, 'input.txt'); writeFileSync(path, 'sample');
  const desktop = new DesktopBindings();
  const desktopHost = new NativeNatlangHost({ host: desktop, mode: 'retained' });
  try {
    const result = await desktopHost.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Text>', engine: 'typescript-host',
      code: `const content = host.readText(${JSON.stringify(path)}); return host.run(["node", "-e", "process.stdout.write(process.argv[1].toUpperCase())", content]).stdout;`,
    } } } });
    assert.equal(result.value, 'SAMPLE');
  } finally { desktopHost.close(); desktop.close(); rmSync(folder, { recursive: true, force: true }); }
});
