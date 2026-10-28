import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readLearnedSettings } from '../dist/tune/learned.js';
import '../dist/runtime/node.js';
import { runInFrame } from '../dist/runtime/context.js';
import { decodeNativeState, MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { nativeStateFixture } from './fixtures/native-state.mjs';
import { nativeBridge } from '../../applications/learned-optimizer/native-bridge.mjs';

test('learned execution binds a controller release and permits crisp only as an explicit baseline', () => {
  const root = mkdtempSync(join(tmpdir(), 'lopt-controller-'));
  const path = join(root, 'spec.json');
  const base = { bank: 'bank.json', state: 'state.json', store: 'store', scopes: { lambda: 2 } };
  const read = value => { writeFileSync(path, JSON.stringify(value)); return readLearnedSettings(path); };
  assert.throws(() => read(base), /earned controller/);
  assert.equal(read({ ...base, policy: 'crisp-baseline' }).policy, 'crisp-baseline');
  const bound = read({ ...base, controller: { root: 'controller', release: 'earned' } });
  assert.deepEqual(bound.controller, { root: join(root, 'controller'), release: 'earned' });
  assert.throws(() => read({ ...base, controller: { root: 'controller', release: '' } }), /earned controller/);
});

test('controller evidence crosses into the optimizer with its verified native identity', async () => {
  const store = new MemoryNeuraleseStore(), state = await store.putNativeState(nativeStateFixture());
  const requests = [], python = { request: async request => { requests.push(request); } };
  const repository = new URL('../..', import.meta.url).pathname;
  const bridge = nativeBridge(repository, python, '/plan', '/out');
  await runInFrame({ task: { runtime: { options: { neuralese: { store } } } }, chain: [] },
    () => bridge([{ binding: 'b', digest: { $neuralese: { id: state.id } } }]));
  assert.equal(requests[0].op, 'import-artifacts');
  assert.equal(decodeNativeState(Buffer.from(requests[0].artifacts[0].bytes, 'base64')).id, state.id);
  await assert.rejects(() => bridge([{ binding: 'b', digest: { $neuralese: { id: state.id } } }]), /executing native state store/);
});
