// Tests for WS2 R4 (`escalate` as a pluggable crisp/nl/shadow policy with `shouldEscalate.nl` and `fallback` alias)
// and WS8 N3 (`{ carry: 'neuralese' }` on std combinators, viewed by the runtime's builtin view operator).
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, std } from '../dist/index.js';
import {
  escalate,
  fixedPoint,
  semanticMap,
  semanticReduce,
  shouldEscalate,
} from '../dist/std/index.js';

const textOf = req =>
  req.messages.flatMap(m => [
    typeof m.content === 'string' ? m.content : JSON.stringify(m.content),
    ...(m.tool_calls ?? []).map(c => c.function?.arguments ?? ''),
  ]).join('\n');

test('WS2 R4: escalate supports fallback alias and pluggable crisp / nl / shadow policy modes', async () => {
  const fn = loadVirtualNatlang({
    'verdict.nl': [
      '---',
      'args:',
      '  x: string',
      'returns: "allow" | "block"',
      'readout: decision',
      '---',
      'Choose allow or block for x.',
    ].join('\n'),
  }, 'verdict.nl');

  const traces = [];
  const driver = async () => ({ text: '"allow"', calls: [] });
  driver.decide = async req => {
    // High confidence 0.90 on 'allow', low risk 0.10 on 'block' -> crisp says do NOT escalate
    return {
      log_probs: req.options.map(o => Math.log(JSON.parse(o) === 'allow' ? 0.90 : 0.10)),
    };
  };
  const frontier = async () => ({ text: '"block"', calls: [] });
  frontier.decide = async req => ({
    log_probs: req.options.map(o => Math.log(JSON.parse(o) === 'block' ? 0.95 : 0.05)),
  });

  const runtime = createNatlangRuntime({
    model: driver,
    models: { frontier },
    trace: t => traces.push(t),
  });

  // 1. policy: 'nl' with custom nlPolicy returning true escalates even when crisp confidence >= floor
  let nlCalls = 0;
  const nlWrapped = escalate(fn, {
    fallback: 'frontier',
    floor: 0.8,
    policy: 'nl',
    history: { totalCalls: 50, escalationRate: 0.12, disagreementRate: 0.35 },
    cost: { local: 1, fallback: 20, ratio: 20 },
    nlPolicy: async (_decision, floor, _risk, history, cost) => {
      nlCalls++;
      assert.equal(floor, 0.8);
      assert.equal(history?.disagreementRate, 0.35);
      assert.equal(cost?.ratio, 20);
      return true;
    },
  });
  const nlRes = await runtime.run(() => nlWrapped('case-1'));
  assert.equal(nlCalls, 1);
  assert.equal(nlRes.escalated, true);
  assert.equal(nlRes.reason, 'policy');
  assert.equal(nlRes.value, 'block');

  // 2. policy: 'shadow' runs both crisp (false) and nl (true), serves crisp (false), and emits pluggable_shadow
  const shadowWrapped = escalate(fn, {
    fallback: 'frontier',
    floor: 0.8,
    policy: 'shadow',
    nlPolicy: async () => true,
  });
  const shadowRes = await runtime.run(() => shadowWrapped('case-2'));
  assert.equal(shadowRes.escalated, false, 'shadow serves crisp decision');
  assert.equal(shadowRes.value, 'allow');
  const shadowEvents = traces.flatMap(t => t.events ?? []).filter(e => e.kind === 'pluggable_shadow' && e.name === 'std.escalate');
  assert.equal(shadowEvents.length, 1);
  assert.equal(shadowEvents[0].served, 'crisp');
  assert.equal(shadowEvents[0].agree, false);
  assert.equal(typeof shouldEscalate, 'function');
});

test('WS8 N3: carry: "neuralese" compacts intermediate states across semanticReduce, semanticMap, and fixedPoint', async () => {
  // The std ships no second compaction operator: views come from the runtime's builtin view (or the view option).
  assert.equal(std.compactValue, undefined);

  const driver = async () => ({ text: '"ok"', calls: [] });
  const runtime = createNatlangRuntime({ model: driver });

  // 1. semanticReduce with carry: 'neuralese' views initial & intermediate states and reads once at the end.
  const viewed = [];
  let reads = 0;
  const reduced = await runtime.run(() =>
    semanticReduce(
      ['a', 'b', 'c'],
      'init',
      async (state, item) => `${state}+${item}`,
      {
        mode: 'linear',
        carry: 'neuralese',
        viewInstructions: 'keep sequence',
        view: (val, instr) => {
          viewed.push([val, instr]);
          return `V(${val})`;
        },
        read: val => {
          reads++;
          return `READ(${val})`;
        },
      },
    ),
  );
  assert.equal(reads, 1, 'read called exactly once at the end of semanticReduce');
  assert.equal(viewed.length, 3, 'view called on initial and intermediate steps');
  assert.equal(reduced, 'READ(V(V(V(init)+a)+b)+c)');

  // 2. semanticMap with carry: 'neuralese' views each item before passing to fn.
  const mappedViews = [];
  const mapped = await runtime.run(() =>
    semanticMap(
      ['doc1', 'doc2'],
      async item => `SUMMARY:${item}`,
      {
        carry: 'neuralese',
        view: val => {
          mappedViews.push(val);
          return `SOFT(${val})`;
        },
      },
    ),
  );
  assert.deepEqual(mappedViews, ['doc1', 'doc2']);
  assert.deepEqual(mapped, ['SUMMARY:SOFT(doc1)', 'SUMMARY:SOFT(doc2)']);

  // 3. fixedPoint with carry: 'neuralese' views intermediate states and reads once at settlement.
  let fpReads = 0;
  let fpViews = 0;
  const converged = await runtime.run(() =>
    fixedPoint(async s => `${s}!`, 'v0', {
      maxSteps: 3,
      carry: 'neuralese',
      settled: s => String(s).length >= 6,
      view: s => { fpViews++; return `[${s}]`; },
      read: s => { fpReads++; return `FINAL:${s}`; },
    }),
  );
  assert.equal(fpReads, 1, 'fixedPoint reads once at the end');
  assert.ok(fpViews >= 1);
  assert.match(String(converged.value), /^FINAL:/);
});

test('WS8 N3: carry: "neuralese" without a Neuralese view refuses with what to do instead', async () => {
  const runtime = createNatlangRuntime({ model: async () => ({ text: '"ok"', calls: [] }) });
  await assert.rejects(runtime.run(() => semanticMap(['a'], async item => item, { carry: 'neuralese' })),
    /neuralese-unsupported-backend: carry: "neuralese" needs a runtime with a Neuralese server that serves view; leave carry out/);
});

test('WS2 R4: escalate to a model the runtime does not have keeps the local answer, marked unavailable, instead of rerunning it', async () => {
  const fn = loadVirtualNatlang({ 'verdict.nl': '---\nargs:\n  x: string\nreturns: "allow" | "block"\nreadout: decision\n---\nChoose.\n' }, 'verdict.nl');
  const driver = async () => { return { text: '"allow"', calls: [] }; };
  driver.decide = async req => { return { log_probs: req.options.map(o => Math.log(JSON.parse(o) === 'allow' ? 0.6 : 0.4)) }; };
  const runtime = createNatlangRuntime({ model: driver });
  const result = await runtime.run(() => escalate(fn, { to: 'teacher' })('x'));
  assert.equal(result.escalated, false);
  assert.equal(result.value, 'allow');
  assert.match(result.reason, /^unavailable: this runtime has no model teacher \(confidence\)/);
});

test('WS2 R4: the nl escalate policy reads the function\'s escalation history and tier cost from the call store itself', async () => {
  const { CallStore } = await import('../dist/index.js');
  const { mkdtempSync, rmSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const root = mkdtempSync(join(tmpdir(), 'natlang-escalate-history-'));
  const store = CallStore.open(root);
  store.writeSettings({ minFreeBytes: 0 });
  try {
    const fn = loadVirtualNatlang({ 'verdict.nl': '---\nargs:\n  x: string\nreturns: "allow" | "block"\nreadout: decision\n---\nChoose.\n' }, 'verdict.nl');
    const other = loadVirtualNatlang({ 'other.nl': '---\nargs:\n  x: string\nreturns: "allow" | "block"\nreadout: decision\n---\nOther.\n' }, 'other.nl');
    const driver = async () => ({ text: '"allow"', calls: [] });
    driver.decide = async req => ({ log_probs: req.options.map(o => Math.log(JSON.parse(o) === 'allow' ? 0.9 : 0.1)) });
    const frontier = async () => ({ text: '"block"', calls: [] });
    frontier.decide = async req => ({ log_probs: req.options.map(o => Math.log(JSON.parse(o) === 'block' ? 0.9 : 0.1)) });
    const runtime = createNatlangRuntime({ model: driver, models: { frontier }, calls: store });

    const seen = [];
    const policy = escalateEvery => async (_decision, _floor, _risk, history, cost) => { seen.push({ history, cost }); return escalateEvery; };
    // First run: nothing recorded yet; this wrapper's own runs join its history as they happen.
    const first = escalate(fn, { fallback: 'frontier', policy: 'nl', nlPolicy: policy(true) });
    await runtime.run(async () => { await first('a'); await first('b'); });
    assert.deepEqual(seen[0].history, { totalCalls: 0 });
    assert.equal(seen[1].history.totalCalls, 1);
    assert.equal(seen[1].history.escalationRate, 1);
    assert.equal(seen[1].history.disagreementRate, 1, 'the fallback answered block where the local model said allow');
    // An unrelated function's escalations are not this function's history.
    await runtime.run(() => escalate(other, { fallback: 'frontier', policy: 'nl', nlPolicy: policy(false) })('c'));
    assert.deepEqual(seen[2].history, { totalCalls: 0 });

    // A new wrapper (another process, a later run) reads the recorded runs from the store.
    seen.length = 0;
    await runtime.run(() => escalate(fn, { fallback: 'frontier', policy: 'nl', nlPolicy: policy(false) })('d'));
    assert.equal(seen[0].history.totalCalls, 2);
    assert.equal(seen[0].history.escalationRate, 1);
    assert.equal(seen[0].history.disagreementRate, 1);
    assert.equal(typeof seen[0].cost.local, 'number');
    assert.equal(typeof seen[0].cost.fallback, 'number');

    // Given statistics win; false opts out.
    seen.length = 0;
    await runtime.run(() => escalate(fn, { fallback: 'frontier', policy: 'nl', nlPolicy: policy(false), history: { totalCalls: 9 }, cost: false })('e'));
    await runtime.run(() => escalate(fn, { fallback: 'frontier', policy: 'nl', nlPolicy: policy(false), history: false })('f'));
    assert.deepEqual(seen[0], { history: { totalCalls: 9 }, cost: undefined });
    assert.equal(seen[1].history, undefined);
  } finally {
    store.close();
    rmSync(root, { recursive: true, force: true });
  }
});
