#!/usr/bin/env node
/**
 * Real-browser smoke of the local playground on natlang's GGUF (the page default, ts-host/src/model-default.ts): the
 * playground loads the model with its OPFS native-state store, a turn produces a native state (a grammar-forced
 * producer span) that is archived in OPFS as the engine stored it, a natural-language example program runs to a
 * recorded result, and after a page reload (a fresh engine) a turn naming the state gets it from OPFS into the engine
 * and consumes it exactly.
 *
 *   node scripts/playground-native-state-smoke.mjs --model MODEL.gguf
 *
 * MODEL.gguf is the default model's file (its sha256 must match); it is linked into the repository's models/ for the
 * duration, as `serve-browser-local.mjs` serves the builtin catalog from there.
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, existsSync, symlinkSync, unlinkSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';
import { chromiumPath } from './chromium-path.mjs';
import { exitUnlessFreshBrowserBuild } from './browser-build-freshness.mjs';
import { BROWSER_MODEL_RELEASE as release } from '../dist/model-default.js';

exitUnlessFreshBrowserBuild('scripts/playground-native-state-smoke.mjs');
const source = process.argv[process.argv.indexOf('--model') + 1];
if (!process.argv.includes('--model') || !existsSync(source)) throw new Error('usage: playground-native-state-smoke.mjs --model MODEL.gguf');
const hash = createHash('sha256');
for await (const chunk of createReadStream(source)) hash.update(chunk);
assert.equal(hash.digest('hex'), release.sha256, `${source} is not the default model ${release.file}`);
const linked = resolve(import.meta.dirname, '../../models', release.file);
const link = !existsSync(linked);
if (link) symlinkSync(resolve(source), linked);

const server = spawn(process.execPath, [new URL('./serve-browser-local.mjs', import.meta.url).pathname, '--playground', '--port=0'],
  { stdio: ['ignore', 'pipe', 'pipe'] });
const log = (...items) => console.log(new Date().toISOString().slice(11, 19), ...items);
let browser;
try {
  const base = await new Promise((done, fail) => {
    let output = '';
    const timer = setTimeout(() => fail(new Error('playground server did not start')), 10000);
    server.stdout.on('data', chunk => {
      output += chunk;
      const match = /(http:\/\/127\.0\.0\.1:\d+\/ts-host\/playground\/)/.exec(output);
      if (match) { clearTimeout(timer); done(match[1]); }
    });
  });
  browser = await chromium.launch({ headless: true, executablePath: chromiumPath(), args: ['--no-sandbox'] });
  const context = await browser.newContext();
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const loadModel = async () => {
    await page.goto(base);
    await page.waitForFunction(() => document.querySelector('#projectSelect')?.options.length > 0);
    await page.locator('#modelButton').click();
    assert.match(await page.locator('#modelSelect').textContent(), new RegExp(release.label));
    await page.locator('#loadModel').click();
    await page.waitForFunction(() => window.natlangPlayground.model?.loaded || /Load failed/.test(document.querySelector('#modelProgress').textContent),
      null, { timeout: 15 * 60_000 });
    assert.equal(await page.evaluate(() => !!window.natlangPlayground.model?.loaded), true, await page.locator('#modelProgress').textContent());
    if (await page.locator('#modelDialog').isVisible()) await page.locator('#modelDialog button[value=cancel]').last().click();
    assert.equal(await page.evaluate(() => window.natlangPlayground.model.diagnostics.contextTokens), release.contextTokens);
    const consumer = await page.evaluate(() => window.natlangPlayground.model.neuraleseRuntime()?.nativeStateConsumer);
    log('model loaded', JSON.stringify(consumer));
    assert.equal(consumer?.checkpoint_sha256, release.trainingRun.split(' ').at(-1), 'the runtime gets the model\'s native identity');
  };

  await loadModel();
  // A produced state: the turn's engine request carries a grammar that forces a producer span.
  const produced = await page.evaluate(async () => {
    const model = window.natlangPlayground.model, engine = model.engine;
    const metadata = engine.getModelMetadata().meta;
    const [open, close] = [metadata['natlang.produce_open'], metadata['natlang.produce_close']];
    const plain = engine.createChatCompletion.bind(engine);
    engine.createChatCompletion = request => plain({ ...request,
      grammar: `root ::= "Billing: " <[${open}]> " refund request for order 4411" <[${close}]> " done"` });
    try {
      const turn = await model.turn({ messages: [{ role: 'user', content: 'Summarize the ticket.' }], tools: [], temperature: 0,
        seed: null, max_tokens: 40 });
      const text = JSON.stringify(turn.raw_response);
      const id = /nzs1_[a-z2-7]{52}/.exec(text)?.[0];
      const { openBrowserNeuraleseStore } = await import('../dist/browser/natlang.js');
      const store = await openBrowserNeuraleseStore();
      const wire = id ? await store.getNativeStateWire(id) : undefined;
      return { id, archived: wire?.length ?? 0, store: store.constructor.name, text: id ? (await store.getNativeState(id))?.metadata.text : undefined, raw: id ? undefined : text.slice(0, 1500), turnText: turn.text };
    } finally { engine.createChatCompletion = plain; }
  });
  log('produced', JSON.stringify(produced));
  assert.ok(produced.id, 'the turn produced a native state');
  assert.match(produced.store, /OpfsNeuraleseStore/);
  assert.ok(produced.archived > 0, 'the produced state is archived in OPFS');
  assert.equal(produced.text, ' refund request for order 4411');

  // A natural-language program runs in the playground on this model, its runtime bound to the page's native store.
  await page.locator('#libraryNav').click();
  await page.locator('#exampleSearch').fill('contact');
  await page.locator('.example-card').first().click();
  await page.waitForFunction(() => document.querySelector('#projectName').textContent === 'Extract contact');
  const before = await page.evaluate(() => window.natlangPlayground.runs.length);
  await page.locator('#runButton').click();
  await page.waitForFunction(count => window.natlangPlayground.runs.length > count, before, { timeout: 30 * 60_000 });
  const run = await page.evaluate(() => { const run = window.natlangPlayground.runs[0];
    return { outcome: run.outcome, value: run.value, turns: run.trace?.length ?? 0 }; });
  log('program run', JSON.stringify(run).slice(0, 600));
  assert.ok(!/native|neuralese/i.test(JSON.stringify(run.outcome)), 'the program run fails on no native-state error');
  assert.ok(!/context size|kv_cache_full|model context is full/i.test(JSON.stringify(run.outcome)), 'the page loads the model with its catalog context');

  // After a reload the engine is new: a turn naming the state gets it from OPFS and consumes it exactly.
  await loadModel();
  const imported = await page.evaluate(async id => {
    const { nativeStateSentinel } = await import('../dist/native/neuralese.js');
    const model = window.natlangPlayground.model, engine = model.engine, puts = [];
    const put = engine.putNativeState.bind(engine);
    engine.putNativeState = async (state, expected) => { const result = await put(state, expected); puts.push({ expected, ...result }); return result; };
    try {
      const turn = await model.turn({ messages: [{ role: 'user', content: `Which order is this about? ${nativeStateSentinel(id)}` }],
        tools: [], temperature: 0, seed: null, max_tokens: 16 });
      return { puts, reply: turn.text };
    } finally { engine.putNativeState = put; }
  }, produced.id);
  log('imported', JSON.stringify(imported));
  assert.equal(imported.puts.length, 1, 'the state is put into the new engine once');
  assert.equal(imported.puts[0].meta.id, produced.id);
  assert.equal(imported.puts[0].mode, 'exact-native-state');
  assert.deepEqual(errors, []);
  console.log('PASS playground native states: produced into OPFS, a program run, imported after reload');
} finally {
  await browser?.close();
  server.kill();
  if (link) unlinkSync(linked);
}
