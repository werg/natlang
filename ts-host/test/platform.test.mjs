import assert from 'node:assert/strict';
import { AsyncLocalStorage } from 'node:async_hooks';
import test from 'node:test';
import { NatlangPlatformUnavailableError, NatlangRuntime, createNatlangRuntime, currentFrame, loadVirtualNatlang, nodePlatform,
  realmPlatform } from '../dist/index.js';
import { createNatlangRuntime as createKernelRuntime } from '../dist/runtime/runtime.js';

const files = { 'double.nl': '---\nargs:\n  n: number\nreturns: number\n---\nDouble n.\n' };

/** An agent that answers by evaluating `return n * 2` through the runtime's evaluator. */
const doubler = async session => {
  const result = await session.applyAsync('eval', { code: 'return n * 2' });
  if (result.kind !== 'ok' && result.kind !== 'value') throw new Error(`eval -> ${result.kind}: ${result.text}`);
  if (!session.finish()) throw new Error('did not finish');
};

/** A realm over the Function constructor that records the bodies it runs, so a test can tell whose evaluator ran. */
function countingRealm(label, seen) {
  return { call(body, parameters, args) {
    seen.push(label);
    return new Function(...parameters, `"use strict";\n${body}`)(...args);
  } };
}

test('a kernel runtime without a platform is refused, naming the platforms to pass', () => {
  assert.throws(() => createKernelRuntime({}), NatlangPlatformUnavailableError);
  assert.throws(() => new NatlangRuntime({}), /nodePlatform\(\)/);
});

test('the Node entry defaults to the Node platform, which is one shared value', () => {
  assert.equal(nodePlatform(), nodePlatform());
  assert.equal(createNatlangRuntime().platform, nodePlatform());
});

test('two runtimes with different platforms coexist, each evaluating in its own realm', async () => {
  const seen = [];
  const left = realmPlatform({ name: 'left', contextStore: nodePlatform().contextStore, realm: countingRealm('left', seen) });
  const right = realmPlatform({ name: 'right', contextStore: nodePlatform().contextStore, realm: countingRealm('right', seen) });
  const a = createNatlangRuntime({ platform: left, agent: doubler, calls: false });
  const b = createNatlangRuntime({ platform: right, agent: doubler, calls: false });
  const double = loadVirtualNatlang(files, 'double.nl');
  // Interleaved: both tasks are in flight at once.
  const [x, y] = await Promise.all([a.run(() => double(4)), b.run(() => double(5))]);
  assert.equal(x, 8);
  assert.equal(y, 10);
  assert.deepEqual([...seen].sort(), ['left', 'right']);
  // The default Node platform is untouched by both: its evaluator is the vm one.
  seen.length = 0;
  assert.equal(await createNatlangRuntime({ agent: doubler, calls: false }).run(() => double(6)), 12);
  assert.deepEqual(seen, []);
});

test('runtimes on platforms with their own context stores find their own task, the innermost first', async () => {
  const own = new AsyncLocalStorage();
  const separate = realmPlatform({ name: 'separate', realm: countingRealm('separate', []),
    contextStore: { current: () => own.getStore(), run: (frame, fn) => own.run(frame, fn) } });
  const outer = createNatlangRuntime({ calls: false });
  const inner = createNatlangRuntime({ platform: separate, calls: false });
  let outerTask, innerTask, afterInner;
  await outer.run(async () => {
    outerTask = currentFrame().task;
    await inner.run(async () => { innerTask = currentFrame().task; });
    afterInner = currentFrame().task;
  });
  assert.equal(outerTask.runtime, outer);
  assert.equal(innerTask.runtime, inner, 'a runtime started inside another runtime\'s task is the innermost');
  assert.equal(afterInner, outerTask, 'leaving it returns to the outer task');
});

test('module instances belong to the platform value', async () => {
  const modules = { 'main.nl': '---\nargs: {}\nreturns: number\n---\nCount.\n',
    'main/counter.ts': 'let n = 0;\nexport function next(): number { return ++n; }\n' };
  const one = nodePlatform();
  const two = { ...nodePlatform() };
  const counter = (platform) => loadVirtualNatlang(modules, 'main.nl', '/project', platform).counter;
  const ones = counter(one), twos = counter(two);
  assert.deepEqual([ones.next(), ones.next(), twos.next()], [1, 2, 1], 'a variant of a platform has its own module instances');
  assert.equal(ones.next(), 3, 'the same platform value keeps its instance');
});
