import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DesktopBindings, NatlangHost, TypeScriptEnvironment, portable } from '../dist/index.js';

const lambda = (type, code, engine = 'typescript-host') => ({ $lambda: { type, engine, code } });
test('host and isolated engines use the same crisp standard library source', () => {
  const canonical = fileURLToPath(new URL('../../natlang/prelude.js', import.meta.url));
  const copied = fileURLToPath(new URL('../prelude.js', import.meta.url));
  assert.equal(readFileSync(copied, 'utf8'), readFileSync(canonical, 'utf8'));
});
const run = async (program, environment = new TypeScriptEnvironment()) => {
  const host = new NatlangHost({ environment });
  try { return await host.run({ source: { kind: 'program', program } }); }
  finally { host.close(); }
};

test('full interpreter runs TypeScript generated syntax and finite Map', async () => {
  const root = { $map: { type: 'Map<Num, Num>', over: [1, 2, 3],
    fn: lambda('Lambda<{ item: Num }, Num>', 'enum Scale { Double = 2 }; return args.item * Scale.Double;') } };
  const result = await run(root);
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(result.value, [2, 4, 6]);
});

test('model turns use the native tool surface and shared eval sees host identity', async () => {
  const native = { count: 0, increase(n) { this.count += n; return this.count; } };
  const environment = new TypeScriptEnvironment({ mode: 'retained', host: native });
  const host = new NatlangHost({ environment });
  let turn = 0;
  try {
    const result = await host.run({ source: { kind: 'program', program: {
      $lambda: { type: 'Lambda<{}, Num>', instructions: 'Increment the host counter and return its value.' },
    } }, modelTurn: () => {
      turn++;
      if (turn === 1) return { calls: [['run_code', { engine: 'typescript-host',
        code: 'host.increase(4)' }]], completion_tokens: 4 };
      if (turn === 2) return { calls: [['write', { path: 'return', value: native.count }]], completion_tokens: 4 };
      return { calls: [], text: 'done', completion_tokens: 1 };
    } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 4);
    assert.equal(native.count, 4);
    assert.equal(turn, 3);
  } finally { host.close(); }
});

test('checked definitions, input binding and trace work from TypeScript', async () => {
  const host = new NatlangHost();
  const dir = mkdtempSync(join(tmpdir(), 'natlang-ts-trace-'));
  try {
    const tracePath = join(dir, 'trace.jsonl');
    const result = await host.run({ source: { kind: 'definitions', root: 'main', entries: {
      main: { args: { price: 'Num' }, returns: 'Num', engine: 'typescript-host',
        code: 'return args.price * 1.25;' },
    } }, inputs: { price: 8 }, tracePath });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 10);
    assert.equal(result.trace[0].kind, 'manifest');
    assert.deepEqual(result.trace[0].engine_contracts['typescript-host'], {
      environment_mode: 'fresh', authority: 'shared-node-host', native_state_replayable: false,
    });
    assert.match(readFileSync(tracePath, 'utf8'), /typescript-host/);
  } finally { host.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('file source uses the existing Python loader', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-ts-source-'));
  const path = join(dir, 'program.json');
  writeFileSync(path, JSON.stringify(lambda('Lambda<{}, Num>', 'return 12;')));
  const host = new NatlangHost();
  try {
    const result = await host.run({ source: { kind: 'file', path } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 12);
  } finally { host.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('fresh and retained eval contexts, immutable snapshots and boundary failures', () => {
  const request = code => ({ code, scope: { args: { n: 3 } }, body: false, path: 'eval', effectful: false });
  const fresh = new TypeScriptEnvironment({ mode: 'fresh' });
  assert.equal(fresh.execute(request('globalThis.marker = 7; 7')).result, 7);
  assert.equal(fresh.execute(request('globalThis.marker ?? 0')).result, 0);
  const retained = new TypeScriptEnvironment({ mode: 'retained' });
  assert.equal(retained.execute(request('globalThis.marker = 9; 9')).result, 9);
  assert.equal(retained.execute(request('globalThis.marker')).result, 9);
  assert.throws(() => retained.execute(request('args.n = 10')), /read only|read-only|Cannot assign/);
  assert.throws(() => retained.execute(request('new Date()')), /native object/);
  assert.throws(() => retained.execute(request('2n ** 60n')), /unsupported or inexact/);
  assert.throws(() => portable({ x: undefined }), /unsupported or inexact/);
  retained.close();
  assert.throws(() => retained.execute(request('1')), /disposed/);
});

test('desktop bindings retain bytes and jobs while reporting effects', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-ts-desktop-'));
  const path = join(dir, 'data.txt');
  writeFileSync(path, 'abc');
  const native = new DesktopBindings();
  const observed = [];
  const environment = new TypeScriptEnvironment({ mode: 'retained', host: native,
    observe: event => observed.push(event) });
  try {
    const id = environment.execute({ code: `host.readBytes(${JSON.stringify(path)})`, scope: {}, body: false,
      path: 'eval', effectful: true }).result;
    assert.equal(environment.execute({ code: `host.buffer(${JSON.stringify(id)}).length`, scope: {}, body: false,
      path: 'eval', effectful: true }).result, 3);
    assert.ok(observed.some(event => event.operation === 'file.readBytes'));
    const job = native.start(['node', '-e', 'process.stdout.write("done")']);
    let state;
    for (let i = 0; i < 40; i++) {
      state = native.poll(job);
      if (state.status !== 'running') break;
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    assert.equal(state.status, 'finished');
    assert.equal(state.stdout, 'done');
    native.release(id); native.release(job);
    assert.throws(() => native.buffer(id), /disposed/);
    assert.ok(native.drainEvents().some(event => event.operation === 'process.completed'));
  } finally { environment.close(); native.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('an eval may mutate shared state before its result fails validation', async () => {
  const native = { count: 0 };
  const environment = new TypeScriptEnvironment({ mode: 'retained', host: native });
  const host = new NatlangHost({ environment });
  try {
    const result = await host.run({ source: { kind: 'program', program:
      lambda('Lambda<{}, Num>', 'host.count++; return { wrong: true };') } });
    assert.equal(result.outcome.kind, 'quiesced');
    assert.equal(native.count, 1);
  } finally { host.close(); }
});

test('declared fx capabilities remain available through isolated QuickJS', async () => {
  const result = await run({ $lambda: { type: 'Lambda<{}, Num>', engine: 'quickjs-isolated',
    effects: ['out.emit'], code: 'fx.out.emit({ kind: "seen" }); return 5;' } });
  assert.equal(result.outcome.kind, 'done');
  assert.equal(result.value, 5);
  assert.deepEqual(result.emitted, [{ kind: 'seen' }]);
});

test('application capabilities bridge back into the declared effect journal', async () => {
  const seen = [];
  const host = new NatlangHost();
  try {
    const result = await host.run({ source: { kind: 'program', program: {
      $lambda: { type: 'Lambda<{}, Num>', engine: 'quickjs-isolated',
        effects: ['counter.add'], code: 'return fx.counter.add(3);' },
    } }, capabilities: { 'counter.add': ([n]) => { seen.push(n); return n + 2; } } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 5);
    assert.deepEqual(seen, [3]);
  } finally { host.close(); }
});

test('eventful Fold consumes an async TypeScript stream in order', async () => {
  async function* source() {
    yield 2;
    await new Promise(resolve => setTimeout(resolve, 5));
    yield 3;
  }
  const host = new NatlangHost();
  try {
    const result = await host.run({ source: { kind: 'program', program: { $fold: {
      type: 'Fold<Num, Num>', init: 1, over: [],
      step: lambda('Lambda<{ acc: Num, item: Num }, Num>', 'return args.acc + args.item;'),
    } } }, streams: { over: source() } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 6);
  } finally { host.close(); }
});

test('retained shared host object is reused by distinct crisp Map calls', async () => {
  const native = { seen: [] };
  const result = await run({ $map: { type: 'Map<Num, Num>', over: [3, 5, 7],
    fn: lambda('Lambda<{ item: Num }, Num>', 'host.seen.push(args.item); return host.seen.length;'),
  } }, new TypeScriptEnvironment({ mode: 'retained', host: native }));
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(result.value, [1, 2, 3]);
  assert.deepEqual(native.seen, [3, 5, 7]);
});

test('desktop bindings work through authored natlang code', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-ts-app-'));
  const path = join(dir, 'input.txt');
  writeFileSync(path, 'sample');
  const native = new DesktopBindings();
  const env = new TypeScriptEnvironment({ mode: 'retained', host: native });
  try {
    const result = await run(lambda('Lambda<{}, Text>',
      `const content: string = host.readText(${JSON.stringify(path)});\n` +
      'const converted = host.run(["node", "-e", "process.stdout.write(process.argv[1].toUpperCase())", content]);\n' +
      'return converted.stdout;'), env);
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 'SAMPLE');
  } finally { native.close(); rmSync(dir, { recursive: true, force: true }); }
});
