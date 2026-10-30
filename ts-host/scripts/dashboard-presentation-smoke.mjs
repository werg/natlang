/** End-to-end reading and editing of structured values, with no serialized-data display or editor. */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';
import { startDashboard } from '../dist/dashboard/server.js';
import { createProgram } from '../dist/package/authoring.js';

const root = mkdtempSync(join(tmpdir(), 'natlang-designed-values-'));
const report = { summary: 'Two customers need help with checkout. Billing issues are ready for review.', urgent: 2, by_label: { billing: 3, technical: 2 },
  actions: ['Review the duplicate charge', 'Investigate checkout failures'], tickets: [{ subject: 'Charged twice', urgent: true }, { subject: 'Checkout unavailable', urgent: false }] };
const model = createServer((request, response) => {
  request.resume(); request.on('end', () => { response.setHeader('content-type', 'application/json'); response.end(JSON.stringify(request.url.endsWith('/models') ? { data: [] } : {
    choices: [{ message: { content: JSON.stringify(report), tool_calls: [] }, finish_reason: 'stop' }], usage: { prompt_tokens: 30, completion_tokens: 20 } })); });
});
await new Promise(resolve => model.listen(0, '127.0.0.1', resolve));
const endpoint = `http://127.0.0.1:${model.address().port}`;
const environment = { ...process.env, NATLANG_SERVER: endpoint, NATLANG_MODEL: 'reading-fixture', NATLANG_CONFIG_HOME: join(root, '.config'),
  NATLANG_STATE_HOME: join(root, '.state'), NATLANG_CACHE_HOME: join(root, '.cache'), NATLANG_HASH_CACHE: join(root, 'hashes.jsonl'), NATLANG_CALL_STORE: join(root, '.calls') };
createProgram(join(root, 'report'), { instruction: 'Summarize input as a support report.', input: 'Two checkout failures and a duplicate charge.',
  returns: '{ summary: string; urgent: number; by_label: Record<string, number>; actions: string[]; tickets: { subject: string; urgent: boolean }[] }' });
writeFileSync(join(root, 'report', 'review.nl'), '---\ndescription: Review a typed ticket.\nargs:\n  ticket: "{ title: string; priority: 1 | 2 }"\nreturns: boolean\n---\nReview ticket.\n');
let browser, dashboard;
try {
  dashboard = await startDashboard({ root, environment, version: 'presentation-test', port: 0, log: { write() {} } });
  browser = await chromium.launch({ executablePath: chromiumPath(), headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }), errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const noDump = async () => {
    assert.equal(await page.locator('.json-input, .json-editor, .json-toggle').count(), 0);
    const visible = await page.locator('body').innerText();
    assert.ok(!visible.includes('"summary":') && !visible.includes('"by_label":') && !visible.includes('[object Object]'), 'data must be read as fields and prose');
  };
  await page.goto(dashboard.url);
  await page.getByRole('link', { name: 'Programs', exact: true }).click();
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await page.locator('.pill[data-status=done]').first().waitFor({ timeout: 30000 });
  const result = page.getByLabel('Program output', { exact: true });
  await result.getByText(report.summary, { exact: true }).waitFor();
  assert.equal(await result.locator('.value-metrics').count(), 1);
  assert.equal(await result.getByRole('table').count(), 1);
  assert.match(await result.innerText(), /Charged twice/); assert.match(await result.innerText(), /Yes/);
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: 'Download', exact: true }).click()]);
  assert.deepEqual(JSON.parse(readFileSync(await download.path(), 'utf8')), report, 'downloads retain the exact underlying result');
  await noDump(); await page.screenshot({ path: '/tmp/natlang-designed-result.png', fullPage: true });

  await page.getByRole('link', { name: 'Programs', exact: true }).click();
  const answer = page.locator('.function-card').filter({ has: page.getByText('answer', { exact: true }) });
  await answer.locator(':scope > summary').click();
  await answer.getByText('Save an expected result as a test', { exact: true }).click();
  const expected = answer.locator('.save-test');
  await expected.getByRole('textbox', { name: 'Expected result/summary', exact: true }).fill('A corrected summary.');
  await expected.getByRole('textbox', { name: 'Expected result/urgent', exact: true }).fill('4');
  await expected.locator('input[type=file]').first().setInputFiles({ name: 'expected-result.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ ...report, summary: 'A corrected summary.', urgent: 4 })) });
  await page.getByText('Imported expected-result.json', { exact: true }).waitFor();
  await expected.getByRole('button', { name: 'Save test', exact: true }).click();
  await page.getByText('Test saved. Use Run tests to verify the expected result.').waitFor();
  const saved = JSON.parse(readFileSync(join(root, 'report', 'examples.json'), 'utf8'))[0];
  assert.equal(saved.expected.summary, 'A corrected summary.'); assert.equal(saved.expected.urgent, 4); assert.deepEqual(saved.expected.tickets, report.tickets);
  await noDump();

  const url = new URL(dashboard.url), token = url.searchParams.get('token');
  const request = async path => (await fetch(url.origin + path, { headers: { 'x-natlang-token': token, cookie: `natlang_dashboard=${token}` } })).json();
  const calls = await request('/api/traces'), callId = calls[0].call_id;
  assert.equal(calls.length, 1, 'the test uses its own isolated call store');
  const trace = await request(`/api/traces/${encodeURIComponent(callId)}`);
  assert.ok(trace.record, trace.error || 'Trace record absent');
  trace.record.effects = [{ service: 'billing', method: 'review', args: { reference: 'R-42', policy: { maximum: 10, automatic: false } }, result: { accepted: true, reference: 'R-42' } }];
  trace.annotations = [{ kind: 'fix', value: { summary: 'A previous correction', urgent: 1 }, source: 'Reviewer', created_at: new Date().toISOString() }];
  await page.route(`**/api/traces/${encodeURIComponent(callId)}`, route => route.fulfill({ contentType: 'application/json', body: JSON.stringify(trace) }));
  await page.goto(new URL(`/#traces-call-${encodeURIComponent(callId)}`, dashboard.url).href);
  await page.getByRole('heading', { name: 'Result', exact: true }).waitFor();
  await page.getByText('billing · review', { exact: true }).click();
  await page.getByRole('button', { name: 'Explore event history', exact: true }).click();
  await page.locator('.event-item').first().waitFor();
  await page.locator('.event-details > summary').first().click();
  await page.getByRole('button', { name: 'Save correction', exact: true }).click();
  await page.getByText('Correction saved.', { exact: true }).waitFor();
  const corrections = readFileSync(join(root, 'report', 'tune.examples.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.deepEqual(corrections[0].expected, report, 'typed trace corrections retain numbers, booleans, records and lists');
  await noDump(); await page.screenshot({ path: '/tmp/natlang-designed-trace.png', fullPage: true });
  // Corrections are native values; a text correction must never be parsed as JSON source or dropped when empty.
  for (const expectedText of ['A plain text correction', '']) {
    const response = await fetch(`${url.origin}/api/traces/${encodeURIComponent(callId)}/feedback`, { method: 'POST',
      headers: { 'x-natlang-token': token, cookie: `natlang_dashboard=${token}`, 'content-type': 'application/json' },
      body: JSON.stringify({ verdict: 'fix', expected: expectedText }) });
    assert.equal(response.status, 200, await response.text());
    assert.equal(JSON.parse(readFileSync(join(root, 'report', 'tune.examples.jsonl'), 'utf8').trim()).expected, expectedText);
  }
  const config = await request('/api/config');
  config.profiles['trained-model'] = { kind: 'native', model: 'Trained model', profile: { native: { source: 'Local checkpoint', checkpoint: '/models/checkpoint', contextTokens: 32768 } } };
  await page.route('**/api/config', route => route.fulfill({ contentType: 'application/json', body: JSON.stringify(config) }));
  await page.getByRole('link', { name: 'Models', exact: true }).click();
  await page.getByRole('button', { name: 'Edit', exact: true }).click();
  await page.getByText('Local checkpoint', { exact: true }).waitFor(); await noDump();
  await page.getByRole('link', { name: 'System', exact: true }).click(); await noDump();
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.goto(new URL('/#programs-report', dashboard.url).href);
  await page.getByRole('heading', { name: 'report 0.1.0', exact: true }).waitFor();
  await page.setViewportSize({ width: 390, height: 844 });
  const review = page.locator('.function-card').filter({ has: page.getByText('review', { exact: true }) });
  await review.locator(':scope > summary').click();
  await review.getByRole('textbox', { name: 'ticket/title', exact: true }).fill('A ticket entered in the form');
  await review.getByRole('radio', { name: '2', exact: true }).click();
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'structured editors fit narrow screens');
  await page.screenshot({ path: '/tmp/natlang-designed-mobile.png', fullPage: true });
  await noDump(); assert.deepEqual(errors, []);
  process.stdout.write('Presentation browser smoke passed: structured results, metrics, tables, typed expected values, trace inputs/results/services/events, model/system views and mobile layout.\n');
} finally { await browser?.close(); await dashboard?.close(); await new Promise(resolve => model.close(resolve)); rmSync(root, { recursive: true, force: true }); }
