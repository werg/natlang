/** Save/load a live-selected artifact through installed Node and Chromium consumers. */
import { createServer } from 'node:http';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { resolve, join, extname, dirname } from 'node:path';
import { createRequire } from 'node:module';
import * as host from '@natlang/node';
import { loadAdaptation, bindAdaptation } from '@natlang/node/adaptation';
import { loadEvaluationSuite, UsageGateway } from '@natlang/node/evaluation';
import { loadModelConfiguration, createResolvedModelSession, executorIdentityForChoice } from '@natlang/node/model';
const root = resolve(import.meta.dirname);
const require = createRequire(import.meta.url);
const staged = (() => {
  try { return dirname(require.resolve('@natlang/browser')); }
  catch { return resolve(root, '../../npm-packages/browser/dist'); }
})();
const { chromium } = (() => {
  try { return require('playwright-core'); }
  catch (error) {
    // Workspace scripts keep browser pilot dependencies in ts-host.
    if (!existsSync(resolve(root, '../../ts-host/package.json'))) throw error;
    return createRequire(resolve(root, '../../ts-host/package.json'))('playwright-core');
  }
})();
const prepared = await loadEvaluationSuite(join(root, 'stateful/suite.ts'));
const artifact = loadAdaptation(join(root, 'evidence/stateful-lambda-only-gepa.json'));
const choice = loadModelConfiguration('default').choice, identity = executorIdentityForChoice(choice);
const binding = bindAdaptation(artifact, prepared.program, identity);
const files = Object.fromEntries(['main.ts', 'record.nl'].map(path => [path, readFileSync(join(root, 'stateful', path), 'utf8')]));
const policy = artifact.policy;
const configuration = { program: prepared.program, executorIdentity: identity, codeEdits: policy.codeEdits,
  ...(policy.limits ? { limits: policy.limits } : {}) };
const payload = { files, artifact, program: prepared.program, identity, policy };
const report = { schema: 'natlang.live-artifact-consumers/v1', started: new Date().toISOString(),
  artifact: artifact.digest, programHash: prepared.program.buildHash, executor: identity,
  limitations: ['The selected stateful artifact retains baseline instructions; this verifies deployment, not improvement.',
    'Chromium uses a Node bridge to the same local deployment model; optimization is not bundled into browser code.'], consumers: [] };
const output = join(root, 'evidence/consumers-live-2026-09-29.json');
const save = () => writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
const session = createResolvedModelSession(choice, process.env, process.stderr);
const server = createServer((request, response) => {
  const path = new URL(request.url, 'http://localhost').pathname;
  if (path === '/') { response.setHeader('Content-Type', 'text/html'); response.end('<!doctype html><title>Artifact consumer</title>'); return; }
  const file = resolve(staged, '.' + path);
  if (!file.startsWith(staged + '/') || !existsSync(file)) { response.writeHead(404); response.end(); return; }
  response.setHeader('Content-Type', extname(file) === '.js' ? 'text/javascript' : extname(file) === '.wasm' ? 'application/wasm' : 'application/octet-stream');
  response.end(readFileSync(file));
});
let browser;
const summarize = (kind, result, error, log, untouched, traces, ledger) => ({ kind, outcome: error ? { kind: 'failed', detail: error } : { kind: 'returned', value: result ?? null },
  observation: { log, untouched }, quality: log === 'start\ntwo\n' && result === 'ok' ? 1 : 0,
  gates: { completed: !error, preservesOtherFiles: untouched === 'keep' },
  provenance: traces.map(trace => trace.adaptation), ledger });
try {
  const app = host.compileVirtualProject({ files }, host, { target: 'node', programId: prepared.program.id });
  if (!app.ok) throw new Error(host.formatDiagnostics(app.diagnostics));
  const gateway = new UsageGateway({ maxRollouts: 1, maxProposals: 0, maxModelCalls: 20, maxElapsedMs: 120000 });
  const traces = [], folder = host.Folder.fromFiles({ 'log.txt': 'start\n', 'untouched.txt': 'keep' });
  const runtime = host.createNatlangRuntime({ ...configuration, adaptation: binding, trace: trace => traces.push(trace),
    model: { ...policy.settings, driver: (request, signal) => gateway.request((req, cancellation) => session.turn(req, cancellation), request, signal) } });
  let result, error;
  try { result = await runtime.run(() => folder.apply(app.require('main.ts').append, 'two')); }
  catch (failure) { error = String(failure); }
  report.consumers.push(summarize('installed-node', result, error, await folder.readText('log.txt'), await folder.readText('untouched.txt'), traces, gateway.snapshot())); save();
  await new Promise(done => server.listen(0, '127.0.0.1', done));
  browser = await chromium.launch({ headless: true, executablePath: process.env.NATLANG_CHROMIUM || chromium.executablePath(), args: ['--no-sandbox', '--disable-gpu'] });
  const page = await browser.newPage();
  const browserGateway = new UsageGateway({ maxRollouts: 1, maxProposals: 0, maxModelCalls: 20, maxElapsedMs: 120000 });
  await page.exposeFunction('deploymentTurn', request => {
    delete request.signal;
    return browserGateway.request((req, signal) => session.turn(req, signal), request, AbortSignal.timeout(120000));
  });
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  const observed = await page.evaluate(async payload => {
    const host = await import('/natlang.js');
    const app = host.compileVirtualProject({ files: payload.files }, host, { target: 'browser', programId: payload.program.id });
    if (!app.ok) throw new Error(host.formatDiagnostics(app.diagnostics));
    const binding = host.bindAdaptation(host.parseAdaptation(JSON.stringify(payload.artifact)), payload.program, payload.identity);
    const traces = [], folder = host.Folder.fromFiles({ 'log.txt': 'start\n', 'untouched.txt': 'keep' });
    const runtime = host.createNatlangRuntime({ program: payload.program, adaptation: binding, executorIdentity: payload.identity,
      codeEdits: payload.policy.codeEdits, ...(payload.policy.limits ? { limits: payload.policy.limits } : {}),
      trace: trace => traces.push({ adaptation: trace.adaptation }),
      model: { ...payload.policy.settings, driver: request => window.deploymentTurn(request) } });
    let result, error;
    try { result = await runtime.run(async () => { await Promise.resolve(); return await folder.apply(app.require('main.ts').append, 'two'); }); }
    catch (failure) { error = String(failure); }
    return { result: result ?? null, error: error ?? null, log: await folder.readText('log.txt'), untouched: await folder.readText('untouched.txt'), traces };
  }, payload);
  report.consumers.push(summarize('installed-browser-chromium', observed.result, observed.error, observed.log, observed.untouched, observed.traces, browserGateway.snapshot())); save();
  for (const consumer of report.consumers) if (!consumer.provenance.length || consumer.provenance.some(item => item?.artifact !== artifact.digest)) throw new Error('consumer did not trace the selected artifact');
  report.status = 'completed';
} catch (error) { report.status = 'infrastructure-failed'; report.error = String(error); throw error; }
finally { report.finished = new Date().toISOString(); save(); await browser?.close(); if (server.listening) await new Promise(done => server.close(done)); await session.close(); }
console.log(output);
