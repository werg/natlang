/** Exercise live UI updates over the real dashboard API. The fixture writes the runtime's JSONL event format. */
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, appendFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';
import { startDashboard } from '../dist/dashboard/server.js';

const root = mkdtempSync(join(tmpdir(), 'natlang-live-ui-')), state = join(root, 'state'), id = 'execution-fixture-01';
const directory = join(state, 'dashboard', 'jobs', id);
const environment = { ...process.env, NATLANG_STATE_HOME: state, NATLANG_CONFIG_HOME: join(root, 'config'), NATLANG_CALL_STORE: 'off' };
let browser, dashboard;
try {
  dashboard = await startDashboard({ root, environment, version: 'live-smoke', port: 0, log: { write() {} } });
  mkdirSync(directory, { recursive: true });
  const job = { id, kind: 'run', title: 'run streaming-example', argv: ['run', 'streaming-example'], cwd: root,
    status: 'running', exitCode: null, pid: process.pid, startedAt: new Date().toISOString(), endedAt: null, interactive: true };
  const saveJob = () => writeFileSync(join(directory, 'job.json'), JSON.stringify(job));
  saveJob();
  for (const name of ['stdout.log', 'stderr.log', 'live.jsonl']) writeFileSync(join(directory, name), '');
  const emit = (callId, data, parentCallId = null) => appendFileSync(join(directory, 'live.jsonl'), JSON.stringify({
    at: Date.now(), taskId: 'fixture', callId, parentCallId, definitionId: `fixture:${callId}`, name: callId === 'child' ? 'nl@main.ts:12' : 'main', ...data }) + '\n');
  const trace = (callId, event, parent = null) => emit(callId, { type: 'trace', event }, parent);
  browser = await chromium.launch({ executablePath: chromiumPath(), headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }), errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(dashboard.url);
  await page.goto(new URL(`/#runs-${id}`, dashboard.url).href);
  await page.getByRole('heading', { name: 'Live execution', exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Follow output', exact: true }).count(), 0, 'following output needs no extra control until the reader scrolls back');
  assert.match(await page.locator('.execution-headline').innerText(), /Preparing the program/);
  emit('parent', { type: 'call_start' }); trace('parent', { kind: 'model_request', phase: 'start', turn: 1 });
  emit('parent', { type: 'model_delta', turn: 1, delta: { type: 'reasoning', text: 'Inspecting the task.' } });
  const parent = page.locator('.lambda[data-call-id="parent"]');
  await parent.getByText('Working through the instructions', { exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Show all 1 top-level calls', exact: true }).count(), 0, 'small executions do not show an unnecessary show-all button');
  assert.equal(await parent.getAttribute('open'), null, 'lambda details start collapsed');
  await parent.locator(':scope > summary').click();
  trace('parent', { kind: 'proposal', phase: 'generated', turn: 1, calls: [['eval', { code: 'return await helper(input)' }]] });
  emit('child', { type: 'call_start' }, 'parent'); trace('child', { kind: 'model_request', phase: 'start', turn: 1 }, 'parent');
  emit('child', { type: 'model_delta', turn: 1, delta: { type: 'text', text: 'First streamed fragment' } }, 'parent');
  const child = page.locator('.lambda[data-call-id="child"]');
  await child.getByText('Inline λ · main.ts:12', { exact: true }).waitFor();
  await parent.getByText('Waiting for 1 child call', { exact: true }).waitFor();
  await child.locator(':scope > summary').click();
  await child.locator('.lambda-step > summary').click();
  const draft = child.getByLabel('Model output for step 1', { exact: true });
  assert.equal(await draft.innerText(), 'First streamed fragment');
  await draft.evaluate(node => { node.dataset.stableNode = 'yes'; });
  const input = page.getByRole('textbox', { name: 'Program input', exact: true });
  await input.fill('Input I am still typing');
  emit('child', { type: 'model_delta', turn: 1, delta: { type: 'text', text: ' — more text' } }, 'parent');
  await page.waitForFunction(() => document.querySelector('[data-stable-node]')?.textContent.includes('more text'));
  assert.equal(await input.inputValue(), 'Input I am still typing');
  assert.equal(await input.evaluate(node => document.activeElement === node), true, 'stream updates preserve input focus');
  assert.equal(await child.getAttribute('open'), '', 'updates preserve expanded lambdas');
  assert.equal(await child.locator('.lambda-step').getAttribute('open'), '', 'updates preserve expanded steps');

  await page.getByRole('button', { name: 'Pause updates', exact: true }).click();
  emit('child', { type: 'model_delta', turn: 1, delta: { type: 'text', text: ' — caught up after pause' } }, 'parent');
  await page.waitForResponse(response => response.url().includes(`/api/jobs/${id}?`) && response.ok());
  assert.ok(!(await draft.innerText()).includes('caught up'), 'pause freezes the display while events keep arriving');
  await page.getByRole('button', { name: 'Resume updates', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('[data-stable-node]')?.textContent.includes('caught up'));

  let offline = true;
  await page.route(`**/api/jobs/${id}?*`, route => offline ? route.abort() : route.continue());
  await page.getByText('Reconnecting · your program may still be running', { exact: true }).waitFor();
  offline = false;
  await page.getByText('Live · updates as your program runs', { exact: true }).waitFor();
  assert.equal(await draft.getAttribute('data-stable-node'), 'yes', 'reconnecting does not rebuild the call view');

  appendFileSync(join(directory, 'stdout.log'), Array.from({ length: 150 }, (_, index) => `Program output line ${index}`).join('\n') + '\n');
  const output = page.getByLabel('Program output', { exact: true });
  await page.waitForFunction(() => document.querySelector('.program-output')?.textContent.includes('line 149'));
  await output.evaluate(node => { node.scrollTop = 0; node.dispatchEvent(new Event('scroll')); });
  await page.getByRole('button', { name: 'Follow output', exact: true }).waitFor();
  appendFileSync(join(directory, 'stdout.log'), 'Another output fragment\n');
  await page.waitForFunction(() => document.querySelector('.program-output')?.textContent.includes('Another output fragment'));
  assert.equal(await output.evaluate(node => node.scrollTop), 0, 'new output does not pull readers away from earlier output');
  await page.getByRole('button', { name: 'Follow output', exact: true }).click();
  assert.ok(await output.evaluate(node => node.scrollTop > 0));
  await page.getByRole('button', { name: 'Follow output', exact: true }).waitFor({ state: 'hidden' });

  appendFileSync(join(directory, 'stdout.log'), 'Report {"summary":"A clean streamed record","details":{"nested":{"status":{"ok":true}}}}\nReport {"summary":');
  await output.getByText('A clean streamed record', { exact: true }).waitFor();
  await output.getByText('Receiving details…', { exact: true }).waitFor();
  assert.ok(!(await output.innerText()).includes('"summary":'), 'partial structured records are never displayed as JSON');
  const expandedValue = output.locator('details[data-value-path]').last();
  await expandedValue.locator(':scope > summary').click();
  await output.locator('.value-boolean.yes').waitFor();
  appendFileSync(join(directory, 'stdout.log'), '"The next clean record"}\n');
  await output.getByText('The next clean record', { exact: true }).waitFor();
  assert.equal(await output.locator('details[data-value-path][open]').count(), 1, 'structured record disclosures survive new output');

  emit('child', { type: 'model_delta', turn: 1, delta: { type: 'tool_call', index: 0, name: 'eval', arguments: '{"code":"return ' } }, 'parent');
  await child.getByText('Execute code · in progress', { exact: true }).waitFor();
  await child.locator('.lambda-action > summary').click();
  emit('child', { type: 'model_delta', turn: 1, delta: { type: 'tool_call', index: 0, arguments: '7","finish":true}' } }, 'parent');
  await child.locator('.lambda-action pre').first().getByText('return 7', { exact: true }).waitFor();
  await page.screenshot({ path: '/tmp/natlang-streaming-desktop.png', fullPage: true });
  trace('child', { kind: 'action', name: 'eval', arguments: { code: 'return 7', finish: true }, outcome: 'applied', result_text: '7' }, 'parent');
  emit('child', { type: 'call_end', outcome: 'done', detail: '' }, 'parent');
  emit('parent', { type: 'call_end', outcome: 'failed', detail: 'The final report was missing a required field.' });
  job.status = 'failed'; job.exitCode = 1; job.endedAt = new Date().toISOString(); saveJob();
  await page.getByText('The run failed.', { exact: true }).waitFor();
  appendFileSync(join(directory, 'stderr.log'), 'natlang: the final report was missing a required field.\n');
  await page.locator('.run-failure-message').getByText('natlang: the final report was missing a required field.', { exact: true }).waitFor();
  await page.getByText('Recorded execution', { exact: true }).waitFor();
  assert.equal(await page.getByRole('textbox', { name: 'Program input', exact: true }).count(), 0, 'finished programs no longer offer input');
  assert.match(await parent.locator(':scope > summary').innerText(), /Needs attention/);
  assert.match(await child.locator(':scope > summary').innerText(), /Finished/);
  assert.match(await page.locator('.execution-panel').innerText(), /0 running · 1 finished · 1 needs attention/);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), 'live view fits a narrow screen');
  assert.deepEqual(errors, []);
  process.stdout.write('Live execution browser smoke passed: incremental text/code/output, nested lambdas, preserved focus/disclosure/scroll, pause, reconnect, completion and mobile layout.\n');
} finally {
  await browser?.close(); await dashboard?.close(); rmSync(root, { recursive: true, force: true });
}
