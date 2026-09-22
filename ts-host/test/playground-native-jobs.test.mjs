import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const read = path => readFile(new URL(`../../${path}`, import.meta.url), 'utf8');

test('live playground conversion, SFT export, and turn evaluation launch Node scripts', async () => {
  const jobs = await read('ts-host/scripts/playground-jobs.mjs');
  assert.match(jobs, /case 'cases_ir':[\s\S]*?cmd: 'node',[\s\S]*?import-playground-cases\.mjs/);
  assert.match(jobs, /case 'teacher':[\s\S]*?cmd: 'node',[\s\S]*?playground-teacher\.mjs/);
  assert.match(jobs, /case 'export':[\s\S]*?cmd: 'node',[\s\S]*?export-native-sft\.mjs/);
  assert.match(jobs, /case 'evaluate':[\s\S]*?cmd: 'node',[\s\S]*?eval-playground-turns\.mjs/);
  assert.doesNotMatch(jobs, /cmd:\s*['"]python['"]/);
});

test('unsupported generic materialization fails closed and is absent from the Jobs selector', async () => {
  const [jobs, page] = await Promise.all([
    read('ts-host/scripts/playground-jobs.mjs'), read('ts-host/playground/index.html'),
  ]);
  assert.match(jobs, /case 'materialize':[\s\S]*?fail\('generic program IR materialization is not available in the Node workbench/);
  assert.doesNotMatch(page, /option value="materialize"/);
});
