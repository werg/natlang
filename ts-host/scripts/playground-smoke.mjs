#!/usr/bin/env node
/** Real-browser smoke for the self-contained playground; no model artifact required. */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { chromium } from 'playwright-core';

const server = spawn(process.execPath, [new URL('./serve-browser-local.mjs', import.meta.url).pathname,
  '--playground', '--port=0'], { stdio: ['ignore', 'pipe', 'pipe'] });
const base = await new Promise((resolve, reject) => {
  let output = '';
  const timer = setTimeout(() => reject(new Error('playground server did not start')), 10000);
  server.stdout.on('data', chunk => {
    output += chunk;
    const match = /(http:\/\/127\.0\.0\.1:\d+\/ts-host\/playground\/)/.exec(output);
    if (match) { clearTimeout(timer); resolve(match[1]); }
  });
  server.once('error', reject);
  server.once('exit', code => reject(new Error(`playground server exited ${code}: ${output}`)));
});
let browser;
try {
  browser = await chromium.launch({ headless: true,
    executablePath: process.env.NATLANG_CHROMIUM || chromium.executablePath(),
    args: ['--no-sandbox', '--enable-unsafe-swiftshader'] });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#projectSelect')?.options.length > 0);
  assert.equal(await page.locator('#diagnosticCount').textContent(), 'No diagnostics');
  await page.locator('#runButton').click();
  await page.getByText('Expected value matched').waitFor();
  assert.match(await page.locator('#resultValue').textContent(), /"sum": 8/);
  await page.locator('[data-panel=trace]').click();
  assert.match(await page.locator('#tracePosition').textContent(), /^Event \d+ \/ \d+$/);
  await page.locator('#tracePrev').click();
  assert.notEqual(await page.locator('#traceSlider').inputValue(), await page.locator('#traceSlider').getAttribute('max'));
  await page.locator('[data-panel=result]').click();
  await page.locator('#captureCase').click();
  await page.locator('#caseDialog button[value=confirm]').click();
  await page.locator('[data-panel=cases]').click();
  await page.locator('.case-card').first().getByText('Accept').click();
  await page.getByText('train · accepted').waitFor();
  await page.locator('#editor').fill('/*---\nreturns: Num\nengine: typescript-host\n---*/\nreturn (;');
  await page.getByText('1 diagnostic').waitFor();
  assert.equal(await page.locator('#runButton').isDisabled(), true);
  await page.locator('#editor').fill('/*---\nreturns: Num\nengine: typescript-host\n---*/\nreturn 9;');
  await page.getByText('No diagnostics').waitFor();
  await page.getByText('Saved locally').waitFor();
  await page.reload();
  await page.waitForFunction(() => document.querySelector('#projectSelect')?.options.length > 0);
  assert.match(await page.locator('#editor').inputValue(), /return 9;/);
  await page.locator('[data-panel=jobs]').click();
  await page.getByText('Local pipeline service ready').waitFor();
  const rejected = await page.evaluate(async () => (await fetch('/api/playground/jobs', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
  })).status);
  assert.equal(rejected, 403);
  assert.deepEqual(errors, []);
  console.log('PASS browser playground: edit, validate, run, inspect, admit, persist, and local job API');
} finally {
  await browser?.close();
  server.kill('SIGTERM');
}
