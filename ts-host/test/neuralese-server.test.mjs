/**
 * End to end against the Python reference server (training/neuralese, `python -m natlang_neuralese.serve`) with the
 * base LFM2.5-350M weights and untrained port heads: the content of written blocks is meaningless; these tests check
 * the plumbing. Skipped when the Neuralese venv or the model is not available (set NATLANG_NEURALESE_PYTHON to
 * point at a Python with torch and transformers).
 */
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { after, before, test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { isNeuraleseRef, neuraleseRef } from '../dist/native/neuralese.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { decodeBlockBody, encodeBlockBody, neuraleseServerModelTurn } from '../dist/model/neuralese-server.js';
import { session as open } from './support/natlang.mjs';

const repo = fileURLToPath(new URL('../..', import.meta.url));
const python = process.env.NATLANG_NEURALESE_PYTHON ?? join(repo, '.venv-neuralese', 'bin', 'python');
const fallbackPython = join(homedir(), 'natlang', '.venv-neuralese', 'bin', 'python');
const executable = existsSync(python) ? python : existsSync(fallbackPython) ? fallbackPython : null;
const model = join(homedir(), '.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M');
const skip = !executable || !existsSync(model) ? 'Neuralese venv or LFM2.5-350M not available' : false;

let server, endpoint;
before(async () => {
  if (skip) return;
  server = spawn(executable, ['-m', 'natlang_neuralese.serve', '--port', '0', '--max-block', '4', '--threads', '8'], {
    cwd: join(repo, 'training', 'neuralese'), env: { ...process.env, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' },
    stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  server.stderr.on('data', chunk => { stderr += chunk; });
  endpoint = await new Promise((resolve, reject) => {
    let out = '';
    server.stdout.on('data', chunk => {
      out += chunk;
      const line = out.split('\n').find(item => item.startsWith('{'));
      if (line) resolve(JSON.parse(line).listening);
    });
    server.on('exit', code => reject(new Error(`server exited (${code}): ${stderr.slice(-2000)}`)));
  });
});
after(() => server?.kill());

const EVAL_WITH_LITERAL = ["<|tool_call_start|>[eval(code='const note: Neuralese<string> = ", { neuralese: 'write' },
  ";\\nreturn note;')]<|tool_call_end|>"];
const RETURN = ["<|tool_call_start|>[return_result(status='success')]<|tool_call_end|>"];

/** One scripted driver per turn (the forced plans are a server test hook), sharing a store and endpoint. */
function scripted(store, plans, options = {}) {
  const exchanges = [];
  const drivers = plans.map(plan => neuraleseServerModelTurn({ endpoint, model: 'natlang-neuralese', store,
    request: { x_natlang_forced: plan }, onExchange: exchange => exchanges.push(exchange), ...options }));
  let turn = 0;
  const driver = Object.assign((request, signal) => drivers[Math.min(turn++, drivers.length - 1)](request, signal),
    { neuralese: true });
  return { driver, exchanges, blocks: drivers[0].blocks };
}

test('block bodies round-trip through the safetensors encoding', () => {
  const data = new Uint8Array(new Float32Array([1, 2, 3, 4, 5, 6]).buffer);
  const meta = { id: 'nz1_x', dialect: 'nd:natlang@1', length: 2, width: 3, dtype: 'f32', type: 'Neuralese<string>' };
  const back = decodeBlockBody(encodeBlockBody(meta, data));
  assert.deepEqual(back.meta, meta);
  assert.deepEqual([...back.data], [...data]);
});

test('a model-written literal in eval code is written by the server, stored, typed, read back and traced', { skip }, async () => {
  const store = new MemoryNeuraleseStore();
  const { driver, exchanges, blocks } = scripted(store, [EVAL_WITH_LITERAL, RETURN], { neuraleseTemperature: 0.5 });
  const { session } = open({ type: '() => Neuralese<string>', instructions: 'Write a note for later.' });
  await new NativeToolAgent(driver, { maxTurns: 4, neuralese: { store } }).run(session);
  // Typed: the call returns a reference of the declared type.
  assert.ok(isNeuraleseRef(session.lam.return), JSON.stringify(session.lam.return));
  assert.equal(session.lam.return.$neuralese.type, 'Neuralese<string>');
  const id = session.lam.return.$neuralese.id;
  // Stored on both sides under the same content ID, with the write record.
  const local = await store.meta(id);
  assert.equal(local.dialect, 'nd:natlang@1');
  assert.ok(local.length >= 1 && local.length <= 4);
  assert.equal(local.producer.kind, 'write');
  assert.equal(local.producer.temperature, 0.5);
  assert.equal(typeof local.producer.seed, 'number');
  assert.match(local.producer.mean, /^nz1_/);
  assert.ok(await blocks.has(id));
  // Written inside the eval code string: the first reply's code argument carried the block as a part.
  const first = exchanges[0].wireResponse.choices[0].message.tool_calls[0];
  assert.equal(first.function.name, 'eval');
  assert.equal(exchanges[0].wireResponse.neuralese.blocks[0].id, id);
  // Re-rendered and read back: the next request carries the block as a part in the eval call it repeats.
  const parts = JSON.stringify(exchanges[1].wireRequest.messages);
  assert.ok(parts.includes(`{"type":"neuralese","id":"${id}"}`), 'second request sends the block as a part');
  // Traced: the call's trace refers to the block.
  assert.ok(JSON.stringify(session.runtime.trace.events).includes(id), 'trace mentions the block');
});

test('a soft input from the runtime store is uploaded to the server and read there', { skip }, async () => {
  const store = new MemoryNeuraleseStore();
  const info = await (await fetch(`${endpoint}/v1/neuralese/info`)).json();
  const values = new Float32Array(3 * info.width).map((_, index) => Math.sin(index) * 0.05);
  const meta = await store.put({ dialect: info.dialects[0], length: 3, width: info.width, dtype: 'f32',
    data: new Uint8Array(values.buffer), type: 'Neuralese<string>' });
  const { driver, exchanges, blocks } = scripted(store, [["<|tool_call_start|>[return_result(status='success', value='ok')]<|tool_call_end|>"]]);
  assert.equal(await blocks.has(meta.id), false);
  const { session } = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Answer from memo.',
    args: { memo: neuraleseRef('Neuralese<string>', meta.id) } });
  await new NativeToolAgent(driver, { maxTurns: 2, neuralese: { store } }).run(session);
  assert.equal(session.lam.return, 'ok');
  assert.ok(await blocks.has(meta.id), 'uploaded under the same content ID');
  assert.ok(exchanges[0].wireResponse.usage.prompt_tokens > 0);
  // A block the server cannot read in its dialect is refused rather than sent as text.
  const foreign = await store.put({ dialect: 'nd:other@1', length: 1, width: info.width, dtype: 'f32',
    data: new Uint8Array(new Float32Array(info.width).buffer) });
  const second = scripted(store, [["<|tool_call_start|>[return_result(status='success', value='x')]<|tool_call_end|>"]]);
  const other = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Answer from memo.',
    args: { memo: neuraleseRef('Neuralese<string>', foreign.id) } }).session;
  await assert.rejects(() => new NativeToolAgent(second.driver, { maxTurns: 2, neuralese: { store } }).run(other),
    /neuralese-dialect-mismatch/);
});
