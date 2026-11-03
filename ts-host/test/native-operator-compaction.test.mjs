import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { OPERATOR_FRAMES } from '../dist/native/operator-frames.js';
import { createNatlangRuntime } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

const ID = 'nzs1_' + 'a'.repeat(52);

async function run(compaction) {
  const sites = [], calls = [];
  const compactor = async site => { sites.push(site); return neuraleseRef('Neuralese<string>', ID); };
  const driver = Object.assign(async request => {
    calls.push(request);
    return { calls: [['return_result', { status: 'success', value: 7 }]], completion_tokens: 1 };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ calls: false, model: driver,
    neuralese: { store: new MemoryNeuraleseStore(), dialect: 'native-state/test', compactor, ...(compaction ? { compaction } : {}) } });
  const fn = loadVirtualNatlang({ 'run.nl': '---\nargs: {value: string}\nreturns: number\n---\nCount the lines of value.\n' }, 'run.nl');
  let error;
  try { assert.equal(await runtime.run(() => fn('line of data\n'.repeat(20000))), 7); } catch (caught) { error = caught; }
  return { sites, error, request: calls.length ? JSON.stringify(calls[0].messages) : '' };
}

test('the learned compact-params operator writes the listing state of a cut-off argument when selected', async () => {
  const { sites, error } = await run('nl');
  assert.equal(sites.length, 1);
  assert.equal(sites[0].operator, 'compact-params');
  assert.match(sites[0].instructions, /receives this argument as value \(string\) answers: Count the lines of value\./);
  assert.equal(sites[0].value.length, JSON.stringify('line of data\n'.repeat(20000)).length);
  // The fake state has no archived artifact, so the model turn refuses it for provenance: it was placed in the request.
  assert.match(String(error?.message), new RegExp(`${ID} is unavailable for provenance`));
});

test('the crisp default keeps the text cut and never calls the operator', async () => {
  const { sites, request, error } = await run(undefined);
  assert.equal(error, undefined);
  assert.equal(sites.length, 0);
  assert.ok(!request.includes(ID));
  assert.match(request, /cut off/);
});

test('consumer frames equal the Python operator registry', async () => {
  assert.equal(OPERATOR_FRAMES['compact-history'].close, '\n\nContinue from the most recent steps.');
});
