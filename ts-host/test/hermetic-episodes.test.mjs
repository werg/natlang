// Collector episodes are hermetic (machine-store.ts runInEpisodeCallStore, teacher/collector.ts hermeticEpisode): each
// runs with a fresh call store of its own, nothing another episode or run stored changes what it does, compiled cases
// never serve its calls, and its own records stay readable within it.
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { executeProgram } from '../dist/teacher/collector.js';
import { episodeCallStoreRoot, openMachineStore } from '../dist/machine-store.js';
import { currentFrame } from '../dist/runtime/context.js';
import { caseHashes, createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const machine = mkdtempSync(join(tmpdir(), 'natlang-hermetic-machine-'));
process.env.NATLANG_CALL_STORE = machine;
process.on('exit', () => rmSync(machine, { recursive: true, force: true }));

const PRICE = '---\nargs: { id: string }\nreturns: number\n---\nLook up the price of id.\n';
const record = id => ({ version: 'natlang.program/2', id, kind: 'lambda_source',
  semantics: { root: 'solve.nl', files: { 'solve.nl': '---\nargs: {}\nreturns: number\n---\nReturn the price of abc.\n', 'solve/price.nl': PRICE },
    inputs: {}, expected: 30 } });
const options = runId => ({ rootSeed: 1, runId, systemPrompt: TOOLS_PROMPT, contextTokens: 8192, maxTurns: 4 });
const isPrice = request => request.messages.some(message => String(message.content).includes('Look up the price of id.'));

/** A policy that has the root call price and price answer 30; `inspect` runs inside every policy turn. */
function policy(inspect = () => {}) {
  const seen = { price: 0, root: 0 };
  const driver = async request => {
    inspect(request);
    if (isPrice(request)) { seen.price++; return { calls: [['return_result', { status: 'success', value: 30 }]] }; }
    return ++seen.root === 1 ? { calls: [['eval', { code: 'return await price("abc")', finish: true }]] } :
      { calls: [['return_result', { status: 'success' }]] };
  };
  return { driver, seen };
}

test('two concurrent episodes in one process each get a fresh store and do not see each other', async () => {
  const roots = {};
  const run = name => {
    const { driver } = policy(() => {
      const root = episodeCallStoreRoot();
      assert.ok(root && root !== machine, 'the episode records to a store of its own');
      (roots[name] ??= new Set()).add(root);
      const store = openMachineStore(root);
      const statistics = store.iterationStatistics();
      assert.equal(statistics.read(`planted-by-${name === 'a' ? 'b' : 'a'}`), undefined, 'the other episode\'s state is invisible');
      statistics.write(`planted-by-${name}`, { successes: 1, meanSteps: 1, m2Steps: 0, meanMs: 1, m2Ms: 0, failures: 0, wallMs: 1 });
      assert.equal(currentFrame()?.task.specializationMode(), 'off', 'no compiled case may serve a collector call');
    });
    return executeProgram(record(`hermetic-${name}`), driver, options(`hermetic-${name}`));
  };
  const [a, b] = await Promise.all([run('a'), run('b')]);
  assert.equal(a.outcome.accepted, true, JSON.stringify(a.outcome));
  assert.equal(b.outcome.accepted, true);
  assert.equal(roots.a.size, 1); assert.equal(roots.b.size, 1);
  const [rootA] = roots.a, [rootB] = roots.b;
  assert.notEqual(rootA, rootB);
  assert.ok(!existsSync(rootA) && !existsSync(rootB), 'episode stores are removed with their episode');
  assert.equal(openMachineStore(machine).calls({ limit: 10 }).length, 0, 'nothing reached the machine store');
});

test('the episode keeps its own records readable while it runs (the item-result binding reads them)', async () => {
  let checked = false, rootTurns = 0;
  const driver = async request => {
    if (isPrice(request)) return { calls: [['return_result', { status: 'success', value: 30 }]] };
    if (++rootTurns === 1) return { calls: [['eval', { code: 'return await price("abc")' }]] };
    const store = currentFrame()?.task.runtime.callStore();
    assert.equal(store?.root, episodeCallStoreRoot());
    const children = store.calls({ limit: 10 }).filter(row => row.definition_name === 'price' && row.outcome === 'done');
    assert.equal(children.length, 1, 'the child call is in the episode store');
    assert.equal(store.value(store.call(children[0].call_id).output), 30, 'with its output value');
    checked = true;
    return { calls: [['return_result', { status: 'success' }]] };
  };
  const run = await executeProgram(record('hermetic-own-records'), driver, options('hermetic-own-records'));
  assert.equal(run.outcome.accepted, true);
  assert.ok(checked, 'the root\'s second turn saw its child\'s record');
});

test('an active compiled case in the machine store never serves a collector call, even with the shared-store opt-out', async () => {
  // Record price once in the machine store, compile an always-matching case for it and make it active.
  process.env.NATLANG_EPISODE_CALL_STORE = 'machine';
  try {
    await executeProgram(record('hermetic-seed'), policy().driver, options('hermetic-seed'));
    const store = openMachineStore(machine);
    const priceCall = store.calls({ limit: 50 }).find(row => row.definition_name === 'price');
    assert.ok(priceCall, 'the shared-store opt-out records to the machine store');
    const recorded = store.call(priceCall.call_id);
    const text = 'export const cases = [\n  { when: (args: { id: string }) => true, run: async (args: { id: string }) => 999 },\n];\n';
    const hashes = caseHashes(text);
    store.saveCompilation({ definitionKey: recorded.definition.key, definitionId: recorded.definition.id, definitionName: 'price',
      definitionSource: recorded.definition.source, interfaceHash: recorded.definition.interface, programRoot: null,
      files: { 'cases.ts': text }, caseHashes: hashes });
    store.setTier(hashes[0], 'active');
    store.writeSettings({ specialization: 'on' });
    // Control: outside a collector, the same compiled case does serve the call (so the assertions below can fail).
    const product = createNatlangRuntime({ model: async () => { throw new Error('a served call asks no model'); }, calls: store });
    const price = loadVirtualNatlang(record('control').semantics.files, 'solve/price.nl');
    assert.equal(await product.run(() => price('abc')), 999);
    assert.equal(store.calls({ executor: 'crisp', limit: 10 }).length, 1);

    const shared = policy();
    const viaMachine = await executeProgram(record('hermetic-shared'), shared.driver, options('hermetic-shared'));
    assert.equal(viaMachine.outcome.accepted, true, 'the policy answered 30, not the compiled 999');
    assert.equal(shared.seen.price, 1, 'the policy answered price');
    assert.equal(store.calls({ executor: 'crisp', limit: 10 }).length, 1, 'no collector call was served crisp');
  } finally { delete process.env.NATLANG_EPISODE_CALL_STORE; }

  const hermetic = policy();
  const run = await executeProgram(record('hermetic-default'), hermetic.driver, options('hermetic-default'));
  assert.equal(run.outcome.accepted, true);
  assert.equal(hermetic.seen.price, 1);
});
