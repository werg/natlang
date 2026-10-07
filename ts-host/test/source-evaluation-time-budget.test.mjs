import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Folder } from '../dist/native/scoped-fs.js';
import { SourceEvaluator } from '../dist/improvement/host.js';
import { UsageGateway } from '../dist/evaluation/usage.js';
import { scriptedModel } from './support/natlang.mjs';

const folder = () => Folder.fromFiles({ 'solve.nl': '---\nreturns: number\n---\nReturn seven.\n' }).snapshot();
const cases = [{ id: 'one', group: 'one', split: 'train', args: [], expected: 7 }];
function delayed(delay) {
  const model = scriptedModel(() => 'return 7');
  return async (request, signal) => {
    await new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, delay);
      const abort = () => { clearTimeout(timer); reject(new Error('cancelled')); };
      signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) abort();
    });
    const turn = await model.driver(request, signal);
    for (const [name,args] of turn.calls ?? []) if (name === 'eval') args.finish = true;
    return turn;
  };
}
const evaluator = (driver, options = {}) => new SourceEvaluator({ entry: 'solve.nl', exportName: 'default', programId: 'budget-test' },
  cases, driver, new UsageGateway({ maxModelCalls: 10, maxRollouts: 5, maxProposals: 0 }), { executorId: 'delayed-test', timeoutMs: 3000, ...options });

test('explicit local execution budget excludes pending model time', async () => {
  // The local work (cold compile and run) must fit the budget on a loaded machine; the model wait alone exceeds it.
  const report = await evaluator(delayed(3500), { excludeModelWaitFromTimeout: true }).evaluate(folder(), { split: 'train' });
  assert.equal(report.quality, 1, JSON.stringify(report.outcomes));
  assert.equal(report.modelCalls, 1);
  const old = await evaluator(delayed(3500)).evaluate(folder(), { split: 'train' });
  assert.equal(old.outcomes[0].failureKind, 'timeout');
  assert.equal(old.quality, 0);
});

test('local computation still times out with model-wait exclusion', async () => {
  const source = Folder.fromFiles({ 'solve.ts': 'export function solve(): number { let total = 0; for (let i = 0; i < 1000000000000; i++) total += i; return total; }' }).snapshot();
  const host = new SourceEvaluator({ entry: 'solve.ts', exportName: 'solve', programId: 'budget-loop' }, cases,
    () => { throw Error('no model expected'); }, new UsageGateway({ maxModelCalls: 1, maxRollouts: 1, maxProposals: 0 }),
    { executorId: 'exact-loop', timeoutMs: 500, excludeModelWaitFromTimeout: true });
  const report = await host.evaluate(source, { split: 'train' });
  assert.equal(report.outcomes[0].failureKind, 'timeout');
  assert.equal(report.quality, 0);
});

test('host cancellation interrupts a pending model wait', async () => {
  const control = new AbortController();
  const timer = setTimeout(() => control.abort(), 800);
  try { await assert.rejects(() => evaluator(delayed(5000), { excludeModelWaitFromTimeout: true, signal: control.signal }).evaluate(folder(), { split: 'train' }), /cancelled/); }
  finally { clearTimeout(timer); }
});
