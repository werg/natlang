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
  const openRuntimeFixture = async id => {
    const name = await page.evaluate(async fixtureId => {
      const [{ crispExamples }, { interfaceExamples }, { storage }, { newPlaygroundProject }] = await Promise.all([
        import('./examples/crisp.mjs'), import('./examples/interfaces.mjs'),
        import('./storage.mjs'), import('../dist/browser/natlang.js'),
      ]);
      const template = [...crispExamples, ...interfaceExamples].find(item => item.id === fixtureId);
      if (!template) throw new Error(`Unknown fixture: ${fixtureId}`);
      const project = newPlaygroundProject(template.name, template.root, template.files, template.inputs, template.expected);
      await storage.put('projects', project);
      localStorage.setItem('natlang-project', project.id);
      return project.name;
    }, id);
    await page.reload();
    await page.waitForFunction(expected => document.querySelector('#projectName')?.textContent === expected, name);
  };
  await page.goto(base);
  await page.waitForFunction(() => document.querySelector('#projectSelect')?.options.length > 0);
  await page.emulateMedia({ colorScheme: 'dark' });
  assert.deepEqual(await page.evaluate(() => ({
    scheme: getComputedStyle(document.documentElement).colorScheme,
    body: getComputedStyle(document.body).backgroundColor,
    toolbar: getComputedStyle(document.querySelector('.workbench-toolbar')).backgroundColor,
    inspector: getComputedStyle(document.querySelector('.inspector')).backgroundColor,
    input: getComputedStyle(document.querySelector('#inputs')).backgroundColor,
  })), {
    scheme: 'dark', body: 'rgb(21, 27, 30)', toolbar: 'rgb(25, 33, 37)',
    inspector: 'rgb(28, 37, 41)', input: 'rgb(25, 34, 38)',
  });
  await page.locator('#modelButton').click();
  assert.equal(await page.locator('#modelDialog').evaluate(element => getComputedStyle(element).backgroundColor), 'rgb(25, 34, 38)');
  await page.locator('#modelDialog button[value=cancel]').first().click();
  await page.locator('#libraryNav').click();
  assert.equal(await page.locator('.example-card').first().evaluate(element => getComputedStyle(element).backgroundColor), 'rgb(25, 34, 38)');
  assert.deepEqual(await page.locator('#exampleCategory option').nth(1).evaluate(element => ({
    background: getComputedStyle(element).backgroundColor,
    color: getComputedStyle(element).color,
    scheme: getComputedStyle(element).colorScheme,
  })), { background: 'rgb(25, 34, 38)', color: 'rgb(228, 235, 237)', scheme: 'dark' });
  await page.locator('#closeExamples').click();
  await page.emulateMedia({ colorScheme: 'light' });
  assert.equal(await page.locator('.inspector').evaluate(element => getComputedStyle(element).backgroundColor), 'rgb(255, 255, 255)');
  assert.equal(await page.locator('#projectSelect option').first().evaluate(element => getComputedStyle(element).backgroundColor), 'rgb(255, 255, 255)');
  assert.equal(await page.locator('#diagnosticCount').textContent(), 'No diagnostics');
  assert.equal(await page.locator('#workbenchView').isVisible(), true);
  assert.equal(await page.locator('#discoverView').count(), 0);
  const catalogResponse = await page.request.get(new URL('/models/browser-catalog.json', base).href);
  assert.equal(catalogResponse.status(), 200);
  const catalog = await catalogResponse.json();
  assert.equal(catalog.schema, 'natlang.browser-model-catalog/1');
  if (catalog.models.length) {
    assert.equal(catalog.defaultId, catalog.models[0].id);
    assert.equal((await page.request.head(new URL(catalog.models[0].templateUrl, base).href)).status(), 200);
    assert.equal((await page.request.head(new URL(catalog.models[0].url, base).href)).status(), 200);
  } else assert.equal(catalog.defaultId, '');
  assert.equal(await page.locator('#projectSelect').inputValue(), await page.evaluate(() => window.natlangPlayground.project.id));
  assert.match(await page.locator('#editor').inputValue(), /find the person's full name and email address/);
  assert.equal(await page.locator('#fileList').isVisible(), true);
  assert.equal(await page.locator('#inputFields input[aria-label=message]').inputValue(), 'Please contact Maya Chen at maya@example.com.');
  await page.locator('#inputFields input[aria-label=message]').fill('Contact Ada at ada@example.com.');
  await page.locator('#inputFields input[aria-label=message]').press('Tab');
  await page.waitForFunction(() => window.natlangPlayground.project.inputs.message === 'Contact Ada at ada@example.com.');
  await openRuntimeFixture('structured-calculation');
  assert.equal(await page.locator('#inputFields input[aria-label=a]').inputValue(), '3');
  assert.equal(await page.locator('#inputFields input[aria-label=b]').inputValue(), '5');
  await page.locator('#inputFields input[aria-label=a]').fill('');
  await page.locator('#runButton').click();
  assert.match(await page.locator('#inputError').textContent(), /a must be a number/);
  assert.equal(await page.evaluate(() => window.natlangPlayground.runs.length), 0);
  await page.locator('#inputFields input[aria-label=a]').fill('3');
  await page.locator('#inputFields input[aria-label=a]').press('Tab');
  await page.locator('#rawInputs summary').click();
  await page.locator('#inputs').fill('{"a":"invalid","b":5}');
  await page.locator('#inputs').press('Tab');
  assert.match(await page.locator('#inputError').textContent(), /a must match number/);
  await page.locator('#inputs').fill('{"a":3,"b":5}');
  await page.locator('#inputs').press('Tab');
  assert.equal(await page.locator('#inputFields input[aria-label=a]').inputValue(), '3');
  await page.locator('#rawInputs summary').click();
  await page.locator('#runButton').click();
  await page.getByText('Expected value matched').waitFor();
  assert.match(await page.locator('#resultValue').textContent(), /"sum": 8/);
  const firstRevision = await page.locator('#revisionInfo').textContent();
  await page.locator('#runButton').click();
  await page.waitForFunction(() => window.natlangPlayground.runs.length === 2);
  assert.equal(await page.locator('#revisionInfo').textContent(), firstRevision);
  await page.locator('.result-details summary').click();
  await page.locator('#compareRun').selectOption({ index: 1 });
  assert.match(await page.locator('#compareSummary').textContent(), /"sameRevision": true/);
  assert.match(await page.locator('#compareSummary').textContent(), /"sameOutput": true/);
  await page.locator('[data-panel=trace]').click();
  assert.match(await page.locator('#tracePosition').textContent(), /^Event \d+ \/ \d+$/);
  await page.locator('#tracePrev').click();
  assert.notEqual(await page.locator('#traceSlider').inputValue(), await page.locator('#traceSlider').getAttribute('max'));
  await page.locator('[data-panel=result]').click();
  await page.locator('#captureCase').click();
  await page.locator('#caseDialog button[value=confirm]').click();
  if (!(await page.locator('[data-panel=cases]').isVisible())) await page.locator('#advancedButton').click();
  await page.locator('[data-panel=cases]').click();
  await page.locator('.case-card').first().getByText('Accept').click();
  await page.getByText('train · accepted').waitFor();
  await page.locator('#newCase').click();
  await page.locator('#caseDialog button[value=confirm]').click();
  await page.locator('.case-card').first().getByText('Run case').click();
  await page.getByText('Case run recorded: done').waitFor();
  await page.locator('[data-panel=cases]').click();
  await page.locator('.case-card').first().getByText('Accept').click();
  await page.waitForFunction(() => [...document.querySelectorAll('.case-split')]
    .filter(element => element.textContent === 'train · accepted').length === 2);
  assert.equal(await page.getByText('train · accepted').count(), 2);
  await page.locator('#editor').fill('import type { Summary } from "./types.js";\n\nexport default function summarize(a: number, b: number): Summary { return (; }');
  await page.getByText('1 diagnostic').waitFor();
  assert.equal(await page.locator('#runButton').isDisabled(), true);
  await page.locator('#editor').fill('import type { Summary } from "./types.js";\n\nexport default function summarize(a: number, b: number): Summary { return { sum: 9, larger: 9 }; }');
  await page.getByText('No diagnostics').waitFor();
  await page.getByText('Saved locally').waitFor();
  await page.reload();
  await page.waitForFunction(() => document.querySelector('#projectSelect')?.options.length > 0);
  assert.match(await page.locator('#editor').inputValue(), /return 9;/);
  await page.locator('#advancedButton').click();
  await page.locator('[data-panel=jobs]').click();
  await page.getByText('Local pipeline service ready').waitFor();
  const rejected = await page.evaluate(async () => (await fetch('/api/playground/jobs', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}',
  })).status);
  assert.equal(rejected, 403);
  await page.locator('#libraryNav').click();
  assert.equal(await page.locator('.example-card').count(), 14);
  await page.locator('#exampleSearch').fill('contact');
  assert.match(await page.locator('#exampleCount').textContent(), /^1 of \d+ examples$/);
  await page.locator('.example-card').click();
  await page.waitForFunction(() => document.querySelector('#projectName').textContent === 'Extract contact');
  assert.match(await page.locator('#runButton').textContent(), /Load model to run/);
  // Open a live program and follow its event/state/view loop without a model.
  await openRuntimeFixture('living-counter');
  assert.equal(await page.locator('#inputFields input[aria-label="state.count"]').inputValue(), '3');
  assert.equal(await page.locator('#inputFields input[aria-label="state.step"]').inputValue(), '1');
  await page.locator('#libraryNav').click();
  await page.locator('#exampleCategory').selectOption('Live interfaces');
  await page.locator('#exampleSearch').fill('counter');
  assert.equal(await page.locator('.example-card').count(), 1);
  await page.locator('#closeExamples').click();
  await page.locator('#runButton').click();
  await page.locator('#liveCanvas output').waitFor();
  await page.emulateMedia({ colorScheme: 'dark' });
  assert.equal(await page.locator('#liveCanvas').evaluate(element => getComputedStyle(element).backgroundColor), 'rgb(28, 37, 41)');
  assert.equal(await page.locator('#liveCanvas button').first().evaluate(element => getComputedStyle(element).backgroundColor), 'rgb(40, 51, 56)');
  await page.emulateMedia({ colorScheme: 'light' });
  assert.equal(await page.locator('#liveCanvas output').textContent(), '3');
  await page.getByRole('button', { name: '+ Increase', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#liveCanvas output')?.textContent === '4');
  await page.locator('#liveTimeline').fill('0');
  assert.equal(await page.locator('#liveCanvas output').textContent(), '3');
  assert.equal(await page.locator('#liveCanvas').evaluate(node => node.inert), true);
  await page.locator('#liveLatest').click();
  assert.equal(await page.locator('#liveCanvas output').textContent(), '4');
  // Applying edited source keeps live state; recorded history stays immutable.
  await page.locator('#autoPreview').check();
  const counterSource = await page.locator('#editor').inputValue();
  await page.locator('#editor').fill(counterSource.replace("text: 'Counter'", "text: 'Updated counter'"));
  await page.getByRole('heading', { name: 'Updated counter' }).waitFor();
  assert.equal(await page.locator('#liveCanvas output').textContent(), '4');
  await page.locator('#liveTimeline').fill('0');
  assert.equal(await page.locator('#liveCanvas h2').textContent(), 'Counter');
  await page.locator('#liveLatest').click();
  await page.locator('#restartPreview').click();
  await page.waitForFunction(() => document.querySelector('#liveCanvas output')?.textContent === '3');
  await page.locator('#libraryNav').click();
  await page.locator('#exampleCategory').selectOption('Live interfaces');
  await page.locator('#exampleSearch').fill('counter');
  assert.equal(await page.locator('.example-card').count(), 1);
  await page.locator('.example-card').click();
  await page.waitForFunction(() => window.natlangPlayground.project.name === 'Generated counter UI');
  assert.equal(await page.locator('#autoPreview').isDisabled(), true);
  assert.match(await page.locator('#runButton').textContent(), /Load model to run/);
  await page.locator('#runButton').click();
  assert.equal(await page.locator('#modelDialog').isVisible(), true);
  await page.locator('#modelDialog button[value=cancel]').last().click();
  // Typed input controls and derived values in the second live example.
  await openRuntimeFixture('reactive-pricing');
  await page.locator('#runButton').click();
  await page.waitForFunction(() => document.querySelector('#liveCanvas output')?.textContent === '€50 / month');
  await page.getByRole('textbox', { name: 'Team seats', exact: true }).fill('10');
  await page.getByRole('textbox', { name: 'Team seats', exact: true }).press('Tab');
  await page.waitForFunction(() => document.querySelector('#liveCanvas output')?.textContent === '€100 / month');
  await page.getByRole('button', { name: 'Monthly billing · save 20% annually', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#liveCanvas output')?.textContent === '€80 / month');
  await page.locator('[data-panel=result]').click();
  assert.match(await page.locator('#resultValue').textContent(), /€80/);
  await page.locator('[data-panel=trace]').click();
  assert.match(await page.locator('#tracePosition').textContent(), /^Event \d+ \/ \d+$/);
  // Small-screen access to navigation, examples, files, and advanced tools.
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.locator('#toggleExplorer').click();
  assert.equal(await page.locator('#fileList').isVisible(), false);
  await page.locator('#reopenExplorer').click();
  assert.equal(await page.locator('#fileList').isVisible(), true);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.deepEqual(errors, []);
  console.log('PASS browser playground: edit, run, inspect, admit, persist, live UI, time travel, model gate, mobile, and local job API');
} finally {
  await browser?.close();
  server.kill('SIGTERM');
}
