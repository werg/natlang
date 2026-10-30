/** Real-browser authoring and run workflow, backed by an isolated deterministic model fixture. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';
import { startDashboard } from '../dist/dashboard/server.js';

const root = mkdtempSync(join(tmpdir(), 'natlang-dashboard-smoke-'));
const model = createServer((request, response) => {
  request.resume(); request.on('end', () => {
    response.setHeader('content-type', 'application/json');
    response.end(JSON.stringify(request.url.endsWith('/models') ? { data: [] } : {
      choices: [{ message: { content: '"billing"', tool_calls: [] }, finish_reason: 'stop' }], usage: { completion_tokens: 1 } }));
  });
});
await new Promise(resolve => model.listen(0, '127.0.0.1', resolve));
const environment = { ...process.env, NATLANG_SERVER: `http://127.0.0.1:${model.address().port}`, NATLANG_MODEL: 'browser-fixture',
  NATLANG_CONFIG_HOME: join(root, '.config'), NATLANG_STATE_HOME: join(root, '.state'), NATLANG_CACHE_HOME: join(root, '.cache'),
  NATLANG_CALL_STORE: 'off', NATLANG_HASH_CACHE: join(root, 'hashes.jsonl') };
let browser, dashboard;
try {
  dashboard = await startDashboard({ root, environment, version: 'smoke', port: 0, log: { write() {} } });
  browser = await chromium.launch({ executablePath: chromiumPath(), headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(dashboard.url);
  await page.getByRole('link', { name: 'Programs', exact: true }).click();
  await page.getByRole('button', { name: 'Create program', exact: true }).click();
  await page.getByRole('dialog').getByRole('textbox').fill('hello');
  await page.getByRole('dialog').getByRole('button', { name: 'Create program' }).click();
  await page.getByRole('heading', { name: 'hello 0.1.0', exact: true }).waitFor();
  const manifestPath = join(root, 'hello', 'natlang.json'), manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  manifest.targets.main.arguments.push({ name: 'count', type: 'number' }, { name: 'enabled', type: 'boolean' }, { name: 'settings', type: 'json' });
  writeFileSync(manifestPath, JSON.stringify(manifest)); await page.reload();
  const input = page.getByLabel('input', { exact: false }).first();
  await input.fill('My browser test input');
  await page.getByLabel('count', { exact: true }).fill('2');
  await page.getByLabel('enabled', { exact: true }).selectOption('false');
  await page.getByLabel('settings', { exact: true }).fill('invalid JSON');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByText('settings must be valid JSON.', { exact: true }).waitFor();
  await page.getByLabel('settings', { exact: true }).fill('{"mode":"test"}');
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.getByText('done', { exact: true }).first().waitFor({ timeout: 30000 });
  assert.equal(await page.getByRole('button', { name: 'Output', exact: true }).getAttribute('class'), 'active');
  assert.match(await page.locator('pre.console').innerText(), /billing/);
  await page.getByRole('link', { name: 'Programs', exact: true }).click();
  await page.getByText('Inputs restored from the last successful run.').waitFor();
  assert.equal(await page.getByLabel('input', { exact: false }).first().inputValue(), 'My browser test input');
  assert.equal(await page.getByLabel('enabled', { exact: true }).inputValue(), 'false');
  assert.equal(await page.getByLabel('settings', { exact: true }).inputValue(), '{"mode":"test"}');
  await page.locator('summary').filter({ hasText: 'answer' }).click();
  await page.getByRole('button', { name: 'Edit source', exact: true }).click();
  const source = page.getByRole('textbox', { name: 'Source of answer' });
  await source.fill((await source.inputValue()) + '\nBe concise.\n');
  await page.getByRole('button', { name: 'Save source', exact: true }).click();
  await page.getByText('Saved. Call the function or run the program to try the change.').waitFor();
  assert.match(readFileSync(join(root, 'hello', 'answer.nl'), 'utf8'), /Be concise/);
  await page.getByText('Save an expected result as a test', { exact: true }).click();
  await page.getByLabel('Expected result (JSON)', { exact: true }).fill('"other"');
  await page.getByRole('button', { name: 'Save test', exact: true }).click();
  await page.getByText('Test saved to examples.json. Use Run tests to verify it.').waitFor();
  await page.getByRole('button', { name: 'Run tests', exact: true }).click();
  await page.getByText('The run failed.', { exact: true }).waitFor({ timeout: 30000 });
  assert.match(await page.locator('pre.console').innerText(), /FAIL.*duplicate charge/s);
  await page.getByRole('link', { name: 'Playground', exact: true }).click();
  await page.getByRole('button', { name: 'Save as program', exact: true }).click();
  await page.getByRole('dialog').getByRole('textbox').fill('playground-copy');
  await page.getByRole('dialog').getByRole('button', { name: 'Create program' }).click();
  await page.getByRole('heading', { name: 'playground-copy 0.1.0', exact: true }).waitFor();
  assert.match(readFileSync(join(root, 'playground-copy', 'answer.nl'), 'utf8'), /Summarize the input/);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'mobile view must not overflow');
  assert.deepEqual(errors, []);
  process.stdout.write('Dashboard browser smoke passed: create, run, restore inputs, edit, test failure, Playground export, mobile layout.\n');
} finally {
  await browser?.close(); await dashboard?.close(); await new Promise(resolve => model.close(resolve));
  rmSync(root, { recursive: true, force: true });
}
