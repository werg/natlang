import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { readLearnedSettings } from '../dist/tune/learned.js';

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
