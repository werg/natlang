import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';

const helperUrl = pathToFileURL(new URL('../dist/teacher/collection-liveness.js', import.meta.url).pathname).href;
function child(source) {
  return spawnSync(process.execPath, ['--input-type=module', '-e', source], { encoding: 'utf8', timeout: 5000 });
}

test('idle unresolved collection emits bounded diagnosis and exits nonzero', () => {
  const result = child(`import {observeCollectionPromise} from ${JSON.stringify(helperUrl)};
    await observeCollectionPromise(new Promise(() => {}), () => ({stage:'case_run', active_cases:[{index:27,program_id:'fixture:case'}]}));`);
  assert.equal(result.status, 1);
  const lines = result.stderr.trim().split('\n');
  assert.equal(lines.length, 1);
  assert.deepEqual(JSON.parse(lines[0]), {
    event: 'incomplete_collection', version: 'natlang.incomplete_collection/1', stage: 'case_run',
    active_cases: [{ index: 27, program_id: 'fixture:case' }], active_cases_truncated: false,
    pending_calls: { available: false }, provider_state: { available: false },
  });
});

test('settled collection removes beforeExit listener and exits normally', () => {
  const result = child(`import {observeCollectionPromise} from ${JSON.stringify(helperUrl)};
    const value = await observeCollectionPromise(Promise.resolve('done'), () => ({stage:'case_run'}));
    process.stdout.write(value);`);
  assert.equal(result.status, 0);
  assert.equal(result.stdout, 'done');
  assert.equal(result.stderr, '');
});

test('explicit collection errors retain their original failure behavior', () => {
  const result = child(`import {observeCollectionPromise} from ${JSON.stringify(helperUrl)};
    try { await observeCollectionPromise(Promise.reject(new Error('fixture failure')), () => ({stage:'case_run'})); }
    catch (error) { process.stderr.write(error.message); process.exitCode = 7; }`);
  assert.equal(result.status, 7);
  assert.equal(result.stderr, 'fixture failure');
});

test('repeated beforeExit emits once and preserves a prior nonzero exit code', () => {
  const result = child(`import {observeCollectionPromise} from ${JSON.stringify(helperUrl)};
    process.exitCode = 9;
    void observeCollectionPromise(new Promise(() => {}), () => ({stage:'case_run'}));
    process.emit('beforeExit', 9);
    process.emit('beforeExit', 9);`);
  assert.equal(result.status, 9);
  assert.equal(result.stderr.trim().split('\n').length, 1);
  assert.equal(JSON.parse(result.stderr).event, 'incomplete_collection');
});
