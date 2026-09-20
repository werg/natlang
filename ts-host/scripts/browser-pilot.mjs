#!/usr/bin/env node
/** Actual Chromium smoke, with an optional real-model pilot. */
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat, writeFile } from 'node:fs/promises';
import { extname, resolve, sep } from 'node:path';
import { chromium } from 'playwright-core';

const root = resolve(import.meta.dirname, '../..');
const liveModel = process.argv.includes('--model');
const cpu = process.argv.includes('--cpu');
const gpu = process.argv.includes('--gpu');
const broad = process.argv.includes('--broad');
const probe = process.argv.find(arg => arg.startsWith('--probe='))?.slice('--probe='.length);
const probeTokens = Number(process.argv.find(arg => arg.startsWith('--probe-tokens='))?.slice('--probe-tokens='.length) ?? 8);
const suite = process.argv.includes('--suite');
const taskId = process.argv.find(arg => arg.startsWith('--task='))?.slice('--task='.length);
const contextTokens = Number(process.argv.find(arg => arg.startsWith('--context='))?.slice('--context='.length)
  ?? (suite || taskId ? 4096 : 2048));
const output = process.argv.find(arg => arg.startsWith('--output='))?.slice('--output='.length);
const types = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm', '.json': 'application/json' };
const server = createServer(async (request, response) => {
  try {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (!pathname.startsWith('/ts-host/') && !pathname.startsWith('/models/'))
      throw new Error('path is not a pilot asset');
    const file = resolve(root, '.' + decodeURIComponent(pathname), pathname.endsWith('/') ? 'index.html' : '');
    if (!file.startsWith(root + sep)) throw new Error('path outside root');
    const size = (await stat(file)).size;
    const range = /^bytes=(\d+)-(\d*)$/.exec(request.headers.range ?? '');
    const start = range ? Number(range[1]) : 0;
    const end = range && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    if (start > end || end >= size) { response.writeHead(416, { 'Content-Range': `bytes */${size}` }); response.end(); return; }
    response.writeHead(range ? 206 : 200, {
      'Content-Type': types[extname(file)] ?? 'application/octet-stream',
      'Content-Length': end - start + 1, 'Accept-Ranges': 'bytes',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      ...(range ? { 'Content-Range': `bytes ${start}-${end}/${size}` } : {}),
    });
    if (request.method === 'HEAD') response.end();
    else createReadStream(file, { start, end }).pipe(response);
  } catch (error) { response.writeHead(404); response.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const url = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch({ headless: true,
  executablePath: process.env.NATLANG_CHROMIUM || chromium.executablePath(),
  args: gpu ? ['--no-sandbox', '--enable-unsafe-webgpu',
    '--enable-features=Vulkan,WebGPU', '--use-angle=vulkan'] :
    ['--no-sandbox', '--enable-unsafe-swiftshader'] });
try {
  const page = await browser.newPage();
  const errors = [];
  const backendLogs = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', message => {
    const line = message.text();
    if (/webgpu|offload|gpu.layers|backend|shader.f16/i.test(line) && backendLogs.length < 200)
      backendLogs.push(line.slice(0, 500));
  });
  await page.goto(`${url}/ts-host/test/browser-smoke.html`);
  await page.getByText('PASS browser interpreter').waitFor({ timeout: 30000 });
  console.log('PASS actual Chromium browser interpreter smoke');
  if (errors.length) throw new Error(errors.join('\n'));
  if (liveModel) {
    await page.goto(`${url}/ts-host/examples/browser-local/`);
    if (broad) await page.locator('#schema').selectOption('broad');
    if (cpu) await page.locator('#gpuLayers').fill('0');
    await page.locator('#context').fill(String(contextTokens));
    await page.locator('#load').click();
    await page.waitForFunction(() => /Model ready|Load failed/.test(document.querySelector('#status').textContent),
      null, { timeout: 600000 });
    const loadStatus = await page.locator('#status').textContent();
    console.log(loadStatus);
    if (!loadStatus.startsWith('Model ready')) throw new Error(loadStatus);
    if (probe) {
      const result = await page.evaluate(async ({ kind, probeTokens }) => {
        const model = window.natlangPilot.model;
        const started = performance.now();
        let request;
        if (kind.startsWith('natlang')) {
          await window.natlangPilot.host.run({
            source: { kind: 'program', program: { $lambda: {
              type: 'Lambda<{}, Num>', instructions: 'Write the number 7 to return.' } } },
            modelTurn: async turn => { request = turn; return { calls: [], text: 'probe', completion_tokens: 1 }; },
            options: { model: { max_turns: 1 } },
          });
        }
        const tools = kind === 'tool' ? [{ type: 'function', function: { name: 'answer',
          description: 'Answer with a number', parameters: { type: 'object',
            properties: { value: { type: 'number' } }, required: ['value'] } } }] : [];
        if (kind === 'natlang-prompt') request = { ...request, tools: [] };
        if (kind === 'natlang-tools') request = { ...request,
          messages: [{ role: 'user', content: 'Write seven.' }] };
        try {
          if (kind === 'natlang-raw') {
            const { compileBrowserTools } = await import('/ts-host/dist/browser/natlang.js');
            const compiled = compileBrowserTools(request.tools, model.schemaMode);
            const response = await model.engine.createChatCompletion({
              messages: request.messages, tools: compiled.tools, tool_choice: 'auto',
              temperature: 0, seed: 1, max_tokens: 32 });
            return { elapsed_ms: Math.round(performance.now() - started),
              request_bytes: JSON.stringify(request).length, tool_count: compiled.tools.length,
              response };
          }
          const turn = await model.turn(request ? { ...request, max_tokens: probeTokens } : { messages: [{ role: 'user',
            content: kind === 'tool' ? 'Call answer with value 7.' : 'Say hello.' }],
            tools, temperature: 0, seed: 1, max_tokens: probeTokens });
          return { elapsed_ms: Math.round(performance.now() - started),
            request_bytes: request ? JSON.stringify(request).length : null, turn,
            metrics: model.lastTurn };
        } catch (error) {
          return { elapsed_ms: Math.round(performance.now() - started),
            request_bytes: request ? JSON.stringify(request).length : null, error: String(error) };
        }
      }, { kind: probe, probeTokens });
      result.backendLogs = backendLogs;
      if (output) await writeFile(output, JSON.stringify(result, null, 2) + '\n');
      console.log(JSON.stringify(result, null, 2));
      if (result.error) process.exitCode = 1;
    } else {
    await page.evaluate(() => {
      const model = window.natlangPilot.model;
      window.natlangTurnEvents = [];
      const turn = model.turn;
      model.turn = async (...args) => {
        const event = { phase: 'start', at: performance.now(),
          messages: args[0].messages.length, tools: args[0].tools.length,
          max_tokens: args[0].max_tokens };
        window.natlangTurnEvents.push(event);
        try {
          const result = await turn(...args);
          window.natlangTurnEvents.push({ phase: 'done', at: performance.now(),
            calls: result.calls.length, tokens: result.completion_tokens });
          return result;
        } catch (error) {
          window.natlangTurnEvents.push({ phase: 'error', at: performance.now(), error: String(error) });
          throw error;
        }
      };
    });
    if (taskId) {
      await page.evaluate(id => {
        const task = window.natlangPilot.tasks.find(item => item.id === id);
        if (!task) throw new Error(`unknown pilot task: ${id}`);
        void window.natlangPilot.run([task]);
      }, taskId);
    } else await page.locator(suite ? '#suite' : '#run').click();
    await page.waitForFunction(() => /correct|Run failed|Load failed/.test(document.querySelector('#status').textContent),
      null, { timeout: suite ? 900000 : 360000 });
    const result = { status: await page.locator('#status').textContent(),
      diagnostics: JSON.parse(await page.locator('#diagnostics').textContent()),
      task: JSON.parse(await page.locator('#result').textContent()),
      report: await page.evaluate(() => window.natlangPilot.reports.at(-1)),
      turnEvents: await page.evaluate(() => window.natlangTurnEvents),
      backendLogs };
    if (output) await writeFile(output, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify({ status: result.status, task: result.task }, null, 2));
    if (!result.task.every(task => task.correct)) process.exitCode = 1;
    }
  }
} finally { await browser.close(); server.close(); }
