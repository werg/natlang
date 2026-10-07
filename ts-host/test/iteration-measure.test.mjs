import test from 'node:test';
import assert from 'node:assert/strict';
import { createNatlangRuntime, iterateOn } from '../dist/runtime/node.js';
import { IterationDivergedError, IterationLimitError, IterationStepError } from '../dist/runtime/iterate.js';
import { Folder, APPLY_TO_FOLDER } from '../dist/native/scoped-fs.js';

const run = fn => createNatlangRuntime().run(fn);

test('a decreasing measure and semantic review cooperate without replenishing remaining work', async () => {
  let reviews = 0;
  await assert.rejects(() => run(() => iterateOn(n => n - 1, 12)
    .withMeasure(n => n)
    .checkProgress(async trajectory => {
      reviews++; assert.equal(trajectory.state(0), 12);
      return { verdict: 'continue', reason: 'still useful' };
    }).until(() => false)), error => {
    assert.ok(error instanceof IterationLimitError); assert.equal(error.lastState, 0);
    assert.equal(error.trajectory.length, 13); return true;
  });
  assert.equal(reviews, 1);
});

test('semantic divergence stops a mechanically decreasing iteration early', async () => {
  await assert.rejects(() => run(() => iterateOn(n => n - 1, 100).withMeasure(n => n)
    .checkProgress(async () => ({ verdict: 'divergent', reason: 'unproductive edits' }))
    .until(() => false)), error => {
    assert.ok(error instanceof IterationDivergedError); assert.equal(error.lastState, 90); return true;
  });
});

test('measures cannot stall, increase, become fractional or be replenished', async () => {
  for (const next of [3, 4]) {
    await assert.rejects(() => run(() => iterateOn(() => next, 3).withMeasure(n => n)
      .checkProgress('off').until(() => false)), error => {
      assert.ok(error instanceof IterationDivergedError); assert.equal(error.lastState, 3);
      assert.equal(error.trajectory.length, 1); return true;
    });
  }
  for (const initial of [-1, 0.5, Infinity, NaN, Number.MAX_SAFE_INTEGER + 1])
    await assert.rejects(() => run(() => iterateOn(n => n - 1, initial).withMeasure(n => n)
      .until(() => false)), /nonnegative safe integer/);
  assert.equal(await run(() => iterateOn(n => n - 1, 3).withMeasure(n => n)
    .checkProgress('off').until(n => n === 0)), 0);
  assert.equal(await run(() => iterateOn(() => {throw Error('no work');}, 0)
    .withMeasure(n => n).until(n => n === 0)), 0);
});

test('folder measure violations retain the last joint source/state checkpoint', async () => {
  const folder = Folder.fromFiles({'entry.ts':'before'});
  const reducer = { async [APPLY_TO_FOLDER](draft) { await draft.file('entry.ts').writeText('after'); return {remaining:2}; } };
  await assert.rejects(() => run(() => folder.iterateOn(reducer, {remaining:2})
    .withMeasure(state => state.remaining).checkProgress('off').until(() => false)), asyncError => {
    assert.ok(asyncError instanceof IterationDivergedError);
    assert.equal(asyncError.lastState.folder.digest, folder.snapshot().digest);
    assert.equal(asyncError.lastState.state.remaining, 2); return true;
  });
  assert.equal(await folder.readText('entry.ts'), 'before');
});

test('nested finite iterations use their own work measures, without instruction gas', async () => {
  assert.equal(await run(() => iterateOn(async n => {
    assert.equal(await iterateOn(x => x - 1, 3).withMeasure(x => x).checkProgress('off').until(x => x === 0), 0);
    return n - 1;
  }, 2).withMeasure(n => n).checkProgress('off').until(n => n === 0)), 0);
});

test('Promise.all preserves concurrent iterateOn results when each run is tracked', async () => {
  assert.deepEqual(await run(() => Promise.all([
    iterateOn(n => n + 1, 0).withMeasure(n => Math.max(0, 2 - n)).checkProgress('off').until(n => n === 2),
    iterateOn(n => n + 1, 4).withMeasure(n => Math.max(0, 7 - n)).checkProgress('off').until(n => n === 7),
  ])), [2, 7]);
});

test('an invalid next measure exposes the previous checked state', async () => {
  await assert.rejects(() => run(() => iterateOn(() => NaN, 2).withMeasure(n => n).until(() => false)), error => {
    assert.ok(error instanceof IterationStepError); assert.equal(error.lastState, 2);
    assert.equal(error.trajectory.length, 1); return true;
  });
});

test('iteration refuses work with no mechanical stopping bound, independently of semantic review',async()=>{
 let calls=0;await assert.rejects(()=>run(()=>iterateOn(n=>{calls++;return n+1;},0).checkProgress('off').until(()=>false)),/requires withLimit/);
 assert.equal(calls,0);
});

test('iteration arguments stay fixed while ordered progress advances in state', async () => {
  const seen = [];
  const edits = ['first', 'second', 'third'];
  const result = await run(() => iterateOn((state, fixed) => {
    seen.push(fixed);
    return { draft: [...state.draft, edits[state.nextPass]], nextPass: state.nextPass + 1 };
  }, { draft: [], nextPass: 0 }, 0).withLimit({ maxSteps: edits.length })
    .checkProgress('off').until(state => state.nextPass === edits.length));
  assert.deepEqual(seen, [0, 0, 0]);
  assert.deepEqual(result, { draft: edits, nextPass: 3 });
});

test('a satisfied initial stopping check returns without applying an edit', async () => {
  let calls = 0;
  const original = { nextPass: 3 };
  const result = await run(() => iterateOn(state => { calls++; return state; }, original)
    .withLimit({ maxSteps: 3 }).until(state => state.nextPass === 3));
  assert.equal(result, original);
  assert.equal(calls, 0);
});
