// Native model profiles: where a tuned program's base checkpoint is found, and how a managed server is launched.
import assert from 'node:assert/strict';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { findNativeCheckpoint, nativePort, nativeServerCommand, pinNativeCheckpoint, removeNativePin, validateNativeSettings } from '../dist/model/native-managed.js';

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

test('a pin carries its compact actor view; the server loads it, and a pin made before views gains one', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-native-')), models = join(directory, 'models');
  const environment = { ...process.env, NATLANG_NATIVE_MODELS: models };
  // Stands in for the training interpreter: the pin script copies, native_bundle actor-view prints a receipt.
  const python = join(directory, 'python'), calls = join(directory, 'calls');
  writeFileSync(python, `#!/bin/sh
echo "$1 $3" >> ${calls}
if [ "$1" = -c ]; then cp "$3" "$4"; echo '{"sha256":"${sha}","bytes":7,"source_sha256":"${sha}"}'; exit 0; fi
if [ "$FAIL_VIEW" = 1 ]; then echo 'actor view source lacks its actual optimizer/RNG/source validation' >&2; exit 1; fi
mkdir -p "$7/${sha}"; echo receipt > "$7/${sha}/r.json"; echo '{"path":"'"$7/${sha}/r.json"'","sha256":"${other}","bytes":8}'
`);
  chmodSync(python, 0o755);
  try {
    const checkpoint = join(directory, 'student.pt');
    writeFileSync(checkpoint, 'weights');
    const failing = await pinNativeCheckpoint({ path: checkpoint, sha256: sha, step: 3 }, { python }, { ...environment, FAIL_VIEW: '1' }, { write() {} });
    assert.equal(failing.actorView, undefined, 'a failed view keeps the complete checkpoint path');
    const pin = await pinNativeCheckpoint({ path: join(models, `${sha}.pt`), sha256: sha }, { python }, environment, { write() {} });
    assert.equal(pin.reused, true);
    assert.deepEqual(pin.actorView, { path: join(models, 'actor-views', sha, 'r.json'), sha256: other, bytes: 8 });
    assert.deepEqual(JSON.parse(readFileSync(join(models, 'pins.json'), 'utf8'))[sha].actorView, pin.actorView);
    const again = await pinNativeCheckpoint({ path: checkpoint, sha256: sha }, { python }, environment, { write() {} });
    assert.deepEqual(again.actorView, pin.actorView);
    assert.equal(readFileSync(calls, 'utf8').trim().split('\n').length, 3, 'one copy, one failed and one prepared view; then reuse');
    const native = { checkpoint: pin.checkpoint, sha256: sha, python, actorView: pin.actorView };
    validateNativeSettings(native);
    const { command } = nativeServerCommand(native, {});
    assert.deepEqual(command.slice(command.indexOf('--native-actor-view'), command.indexOf('--native-actor-view') + 4),
      ['--native-actor-view', pin.actorView.path, '--native-actor-view-sha256', other]);
    assert.throws(() => validateNativeSettings({ ...native, actorView: { path: 'x', sha256: 'short' } }), /actorView/);
    assert.ok(removeNativePin(pin.checkpoint, environment));
    assert.ok(!existsSync(join(models, 'actor-views', sha)));
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
