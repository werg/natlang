// Native model profiles: where a tuned program's base checkpoint is found, and how a managed server is launched.
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { findNativeCheckpoint, nativePort, nativeServerCommand, validateNativeSettings } from '../dist/model/native-managed.js';

const sha = 'ab'.repeat(32), other = 'cd'.repeat(32);

test('a tuned base is found as a pin, then in the program\'s tune plans, then behind a native source pointer', () => {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-native-')), models = join(directory, 'models'), program = join(directory, 'program');
  const environment = { ...process.env, NATLANG_NATIVE_MODELS: models };
  try {
    mkdirSync(models);
    const checkpoint = join(directory, 'student.pt'), pointer = join(directory, 'current.json');
    writeFileSync(checkpoint, 'weights');
    assert.equal(findNativeCheckpoint(sha, { programRoot: program }, environment), undefined);
    writeFileSync(pointer, JSON.stringify({ global_step: 7, local_full_state: { path: checkpoint, sha256: sha, bytes: 7 } }));
    assert.deepEqual(findNativeCheckpoint(sha, { sources: { latest: pointer } }, environment),
      { pointer: { path: checkpoint, sha256: sha, bytes: 7, step: 7 }, step: 7 });
    mkdirSync(join(program, '.natlang', 'tune', 'jobs', 'k1'), { recursive: true });
    writeFileSync(join(program, '.natlang', 'tune', 'jobs', 'k1', 'plan.json'), JSON.stringify({ base: { checkpoint_sha256: sha }, student: { checkpoint } }));
    assert.deepEqual(findNativeCheckpoint(sha, { programRoot: program }, environment), { pointer: { path: checkpoint, sha256: sha } });
    assert.equal(findNativeCheckpoint(other, { programRoot: program }, environment), undefined, 'a plan for another base names nothing');
    writeFileSync(join(models, `${sha}.pt`), 'weights');
    writeFileSync(join(models, 'pins.json'), JSON.stringify({ [sha]: { step: 9 } }));
    assert.deepEqual(findNativeCheckpoint(sha, { programRoot: program }, environment), { pinned: join(models, `${sha}.pt`), step: 9 });
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('a native profile launches its pinned checkpoint on a port derived from it, with prefix reuse unless turned off', () => {
  const native = { checkpoint: '/pins/x.pt', sha256: sha, python: '/venv/python', pythonPath: ['/repo/training'] };
  validateNativeSettings(native);
  assert.equal(nativePort(native), nativePort({ ...native }));
  assert.notEqual(nativePort(native), nativePort({ ...native, sha256: other }));
  const { command, env } = nativeServerCommand(native, {});
  assert.deepEqual(command.slice(0, 5), ['/venv/python', '-u', '-m', 'natlang_neuralese.serve', '--native-checkpoint']);
  assert.ok(command.includes('--prefix-cache-gb') && env.PYTHONPATH === '/repo/training');
  assert.ok(!nativeServerCommand({ ...native, prefixCacheGb: 0 }, {}).command.includes('--prefix-cache-gb'));
  const launched = nativeServerCommand({ ...native, launcher: ['ledger', 'run', '--unit', '{unit}', '--'], stop: ['stop', '{unit}'] }, {}).command;
  assert.deepEqual(launched.slice(0, 6), ['ledger', 'run', '--unit', `natlang-native-${sha.slice(0, 12)}-${nativePort(native)}`, '--', 'env']);
  assert.throws(() => validateNativeSettings({ checkpoint: '/x', sha256: 'short' }), /sha256/);
  assert.throws(() => validateNativeSettings({ ...native, stop: ['x'] }), /launcher/);
});
