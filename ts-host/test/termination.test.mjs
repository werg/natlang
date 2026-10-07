/**
 * Restricted code cannot loop or recurse without bound by itself (spec: Iteration and termination). Each case runs model-
 * written eval code in a child process with a timeout, so a regression that hangs fails here instead of stalling the suite.
 */
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

const runner = fileURLToPath(new URL('./support/eval-case.mjs', import.meta.url));

function evalCase(spec, timeout = 20000) {
  return new Promise((resolve, reject) => {
    execFile(process.execPath, ['--max-old-space-size=256', runner, JSON.stringify(spec)], { timeout },
      (error, stdout, stderr) => {
        if (error) return reject(new Error(`eval case did not finish (${error.signal ?? error.code}): ${stderr.slice(-500)}`));
        resolve(JSON.parse(stdout.trim().split('\n').at(-1)));
      });
  });
}

test('a counted loop needs a finite bound', async () => {
  const result = await evalCase({ code: 'let n = 0; for (let i = 0; i < Infinity; i++) { n++; if (n > 5) break; } return `ran ${n}`;' });
  assert.match(result.error, /finite number as its bound, but got Infinity/);
});

test('a counted loop reads its bound once, when it starts', async () => {
  const getter = await evalCase({ code:
    'let k = 0; { const o = { get n() { return 3 + counter.tick() * 0; } }; for (let i = 0; i < o.n; i++) k++; } return `k ${k}`;' });
  assert.equal(getter.ok, 'k 3'); assert.equal(getter.ticksAtReturn, 1);
  const alias = await evalCase({ code: 'const a = [0]; const b = a; for (let i = 0; i < a.length; i++) b.push(i); return `length ${a.length}`;' });
  assert.equal(alias.ok, 'length 2');
});

test('recursion through methods, properties and promise callbacks is refused', async () => {
  const cases = {
    objectMethod: 'const o = { w: async (f: any, n: number): Promise<number> => { await new Promise(r => setTimeout(r, 1)); counter.tick(); ' +
      'return n >= 40 ? n : f(f, n + 1); } }; return `reached ${await o.w(o.w, 0)}`;',
    classMethod: 'class A { async m(): Promise<number> { await null; counter.tick(); return this.m(); } } await new A().m(); return "ran";',
    thenChain: 'const holder: any = {}; holder.go = () => Promise.resolve().then(() => { counter.tick(); return holder.go(); }); ' +
      'await holder.go(); return "ran";',
    flatMapSelf: 'const f = (n: number): number[] => { counter.tick(); return [n].flatMap(f); }; f(3); return "ran";',
  };
  for (const [name, code] of Object.entries(cases)) {
    const result = await evalCase({ code });
    assert.match(result.error ?? '', /was called while it was still running/, name);
    assert.ok(result.ticksAtReturn <= 1, `${name}: ${result.ticksAtReturn} ticks`);
  }
});

test('unawaited recursion cannot starve the event loop or outlive the call', async () => {
  const cases = {
    microtasks: 'const o = { w: async (f: any): Promise<number> => { await null; counter.tick(); return f(f); } }; o.w(o.w); return "started";',
    timers: 'const o = { w: async (f: any): Promise<number> => { await new Promise(r => setTimeout(r, 1)); counter.tick(); return f(f); } }; ' +
      'o.w(o.w); return "started";',
    rescheduling: 'const tick = () => { counter.tick(); setTimeout(tick, 1); }; tick(); return "started";',
  };
  for (const [name, code] of Object.entries(cases)) {
    const result = await evalCase({ code, settleMs: 200 });
    assert.equal(result.ok, 'started', name);
    assert.ok(result.ticksAfter <= 1, `${name}: ${result.ticksAfter} ticks`);
  }
});

test('setInterval is not available; timers end with their call', async () => {
  const interval = await evalCase({ code: 'setInterval(() => counter.tick(), 5); return "scheduled";' });
  assert.match(interval.error, /setInterval` is not available here; repeat with `iterateOn/);
  const pending = await evalCase({ code: 'setTimeout(() => counter.tick(), 100); return "scheduled";', settleMs: 300 });
  assert.equal(pending.ok, 'scheduled'); assert.equal(pending.ticksAfter, 0);
  const awaited = await evalCase({ code: 'await new Promise(r => setTimeout(r, 20)); counter.tick(); return "waited";' });
  assert.equal(awaited.ok, 'waited'); assert.equal(awaited.ticksAtReturn, 1);
});

test('restricted code cannot define iterators', async () => {
  const cases = {
    symbol: 'const inf = { [Symbol.iterator]() { return { next: () => ({ done: false, value: 1 }) }; } }; return `size ${new Set(inf).size}`;',
    from: 'const it = Iterator.from({ next: () => ({ done: false, value: 1 }) }); return `size ${new Set(it).size}`;',
    subclass: 'class Forever extends Iterator { next() { return { done: false, value: 1 }; } } return "defined";',
  };
  for (const [name, code] of Object.entries(cases)) {
    const result = await evalCase({ code });
    assert.match(result.error ?? '', /Defining iterators is not available here/, name);
  }
  const builtIn = await evalCase({ code: 'const m = new Map([["a", 1]]); return [...m.keys()].join(",");' });
  assert.equal(builtIn.ok, 'a');
});

test('prelude helpers refuse arguments that would never finish', async () => {
  for (const [code, pattern] of [['return String(chunk([1, 2, 3], 0).length);', /chunk size must be a whole number of at least 1/],
    ['return String(windows([1, 2, 3], 2, 0).length);', /window step must be a whole number of at least 1/],
    ['return String(range(1, Infinity).length);', /range needs finite bounds/]]) {
    const result = await evalCase({ code });
    assert.match(result.error ?? '', pattern, code);
  }
  const fine = await evalCase({ code: 'return JSON.stringify([chunk([1, 2, 3], 2), windows([1, 2, 3], 2), range(1, 3)]);' });
  assert.equal(fine.ok, '[[[1,2],[3]],[[1,2],[2,3]],[1,2,3]]');
});

test('callable-folder code cannot use setInterval or define iterators', async () => {
  for (const [source, pattern] of [
    ['export default function start(): string { setInterval(() => {}, 5); return "started"; }\n', /setInterval` is not available here/],
    ['export default function start(): string { const it = { [Symbol.iterator]: () => ({ next: () => ({ done: false, value: 1 }) }) }; ' +
      'return String([...(it as any)].length); }\n', /Defining iterators is not available here/],
  ]) {
    const result = await evalCase({ code: 'return String(await start());', files: { 'probe/start.ts': source } });
    assert.match(result.error ?? '', pattern);
  }
});

// Cancellation within a call (spec: Eval). The slow child waits two seconds, then ticks; a stopped child never ticks.
const SLOW = 'await new Promise(r => setTimeout(r, 2000)); counter.tick(); return "slow";';
const children = { 'Answer slowly.': SLOW, 'Answer quickly.': 'return "fast";',
  'Answer after a moment.': 'await new Promise(r => setTimeout(r, 200)); return "moment";', 'Refuse.': 'throw new Error("refused");' };

test('the loser of a race is stopped when its eval finishes, without failing the eval', async () => {
  const race = await evalCase({ children, settleMs: 2500, code:
    'return await Promise.race([nl<string>`Answer slowly.`(), nl<string>`Answer quickly.`()]);' });
  assert.equal(race.ok, 'fast'); assert.ok(race.elapsedMs < 1800, `${race.elapsedMs} ms`); assert.equal(race.ticksAfter, 0);
  const timeout = await evalCase({ children, settleMs: 2500, code:
    'return await Promise.race([nl<string>`Answer slowly.`(), new Promise<string>(r => setTimeout(() => r("timed out"), 100))]);' });
  assert.equal(timeout.ok, 'timed out'); assert.ok(timeout.elapsedMs < 1800, `${timeout.elapsedMs} ms`); assert.equal(timeout.ticksAfter, 0);
  const later = await evalCase({ children, code: 'const moment = nl<string>`Answer after a moment.`(); ' +
    'const first = await Promise.race([moment, nl<string>`Answer quickly.`()]); return first + " " + await moment;' });
  assert.equal(later.ok, 'fast moment');
});

test('a failed or timed-out eval stops the calls it started', async () => {
  const thrown = await evalCase({ children, settleMs: 2500, code: 'const pending = nl<string>`Answer slowly.`(); throw new Error("boom");' });
  assert.match(thrown.error, /boom/); assert.ok(thrown.elapsedMs < 1800, `${thrown.elapsedMs} ms`); assert.equal(thrown.ticksAfter, 0);
  const timedOut = await evalCase({ children, timeoutMs: 200, settleMs: 2500, code: 'return await nl<string>`Answer slowly.`();' });
  assert.match(timedOut.error, /timed out after 200 ms/); assert.ok(timedOut.elapsedMs < 1800, `${timedOut.elapsedMs} ms`);
  assert.equal(timedOut.ticksAfter, 0);
});

test('an eval that ends with calls still running stops them and says so', async () => {
  const dropped = await evalCase({ children, settleMs: 2500, code: 'nl<string>`Answer slowly.`(); return "done";' });
  assert.match(dropped.error, /1 natural-language call it started was still running; it was stopped/);
  assert.ok(dropped.elapsedMs < 1800, `${dropped.elapsedMs} ms`); assert.equal(dropped.ticksAfter, 0);
  const iteration = await evalCase({ settleMs: 500, code:
    'iterateOn(async (s: number) => { await new Promise(r => setTimeout(r, 50)); counter.tick(); return s + 1; }, 0)' +
    '.withLimit({ maxSteps: 100 }).until(s => s >= 100); return "started";' });
  assert.match(iteration.error, /still running; it was stopped/); assert.ok(iteration.ticksAfter <= 2, `${iteration.ticksAfter} ticks`);
});

test('Promise.all keeps JavaScript semantics: siblings of a rejected call keep running', async () => {
  // In a block: a top-level local holding promises is awaited when the eval ends.
  const result = await evalCase({ children, code: 'let second = ""; { const calls = [nl<string>`Refuse.`(), nl<string>`Answer after a moment.`()]; ' +
    'try { await Promise.all(calls); } catch { /* one refused */ } second = await calls[1]; } return second;' });
  assert.equal(result.ok, 'moment');
});
