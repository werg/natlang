#!/usr/bin/env node
/** Run frozen Studio cases through an HTTP teacher with atomic, resumable jobs. */
import { createHash } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { apps } from '../studio/apps/index.mjs';
import { loadProgram } from '../studio/shared/program.mjs';
import { applyOperation } from '../studio/shared/host.mjs';
import { simulateInventory } from '../studio/apps/worlds.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const digest = value => createHash('sha256').update(typeof value === 'string' || Buffer.isBuffer(value) ? value : JSON.stringify(value)).digest('hex');
const sleep = ms => new Promise(resolvePromise => setTimeout(resolvePromise, ms));

async function api() {
  const processValue = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = processValue; }
}

function services() {
  const values = new Map();
  return { studio: { apply: (state, event, decision) => applyOperation(
    apps.find(spec => spec.id === JSON.parse(event.value).application) ??
      apps.find(spec => spec.decisionType.includes(`action: ${JSON.stringify(decision.action)}`)),
    state, decision, {}) },
    trial: config => simulateInventory(config),
    readValue: async id => values.get(id),
    cell: async (cell, deps) => { const value = cell.id === 'numbers' ? [2,3,5,7,11] :
      Object.values(deps).flat().reduce((a, b) => a + Number(b), 0);
      const id = `value-${values.size}`; values.set(id, value); return { id, preview: JSON.stringify(value) }; },
    service: async operation => {
      if (operation === 'media.sample') return { asset: 'asset-fixture.mp4' };
      if (operation === 'media.transform') return { asset: 'asset-transformed.mp4' };
      if (operation === 'build.run') return { status: 'ok', output: 'FIXTURE' };
      if (operation === 'terminal.run') return { status: 'succeeded', output: 'fixture stdout' };
      if (operation === 'packages.catalog') return { packages: [{ name: 'greetings', versions: ['1.0.0', '1.1.0'] }] };
      if (operation === 'packages.resolve') return { locks: [{ id: 'fixture-lock' }] };
      if (operation === 'packages.install') return { status: 'installed', id: 'fixture-install' };
      if (operation === 'repository.check') return { status: 'reviewable', checks: [] };
      throw new Error(`Unexpected fixture operation ${operation}`);
    } };
}

function decodeArguments(raw) {
  try { return JSON.parse(raw || '{}'); }
  catch { return { __unparsed__: raw }; }
}

function teacherDriver({ server, exchanges }) {
  return async request => {
    const tools = structuredClone(request.tools);
    for (const tool of tools) {
      if (tool.function.name === 'call') tool.function.name = 'call_function';
      if (tool.function.name === 'write') {
        const properties = tool.function.parameters?.properties;
        if (properties?.value) properties.value = { type: 'string', description:
          'For Text use plain text. For every other type use JSON text of the value itself.' };
      }
      if (tool.function.parameters)
        for (const key of Object.keys(tool.function.parameters))
          if (key.startsWith('x-')) delete tool.function.parameters[key];
    }
    const messages = structuredClone(request.messages);
    for (const message of messages) for (const call of message.tool_calls ?? [])
      if (call.function?.name === 'call') call.function.name = 'call_function';
    const payload = { messages, tools, tool_choice: 'auto', parallel_tool_calls: true,
      temperature: request.temperature, seed: request.seed, thinking_budget_tokens: 256,
      top_p: 0.95, top_k: 20, chat_template_kwargs: { reasoning_effort: 'low' } };
    if (request.max_tokens !== null) payload.max_tokens = request.max_tokens;
    const started = performance.now();
    const response = await fetch(server + '/v1/chat/completions', { method: 'POST',
      headers: { 'content-type': 'application/json' }, body: JSON.stringify(payload) });
    const body = await response.json();
    if (!response.ok) throw new Error(`teacher HTTP ${response.status}: ${JSON.stringify(body).slice(0, 2000)}`);
    const message = body.choices?.[0]?.message ?? {};
    const calls = (message.tool_calls ?? []).map(call => {
      const name = call.function?.name === 'call_function' ? 'call' : call.function?.name;
      const args = decodeArguments(call.function?.arguments);
      if (name === 'write' && typeof args.value === 'string' && args.type !== 'Text') {
        try { args.value = JSON.parse(args.value); } catch { /* runtime may coerce a simple scalar string */ }
      }
      return [name, args];
    });
    exchanges.push({ request, response: body, duration_ms: Math.round(performance.now() - started),
      assistant: { content: String(message.content ?? '').trim(),
        reasoning: message.reasoning_content ?? message.reasoning ?? message.thinking ?? null,
        calls: calls.map(([tool, arguments_]) => ({ tool, arguments: arguments_ })) } });
    return { calls, text: String(message.content ?? '').trim(), raw_calls: message.tool_calls ?? [],
      completion_tokens: body.usage?.completion_tokens, prompt_tokens: body.usage?.prompt_tokens,
      raw_response: body };
  };
}

async function sourceFor(spec) {
  return loadProgram(spec, path => readFile(resolve(here, '../studio', path.replace('./', '')), 'utf8'));
}

async function runCase(frozen, options, bindings) {
  const spec = apps.find(item => `studio:${item.id}` === frozen.target);
  if (!spec) throw new Error(`Unknown target ${frozen.target}`);
  const exchanges = [];
  const event = { ...frozen.event,
    value: JSON.stringify({ ...JSON.parse(frozen.event.value), application: spec.id }) };
  const host = services();
  host.studio.apply = (state, rawEvent, decision) => applyOperation(spec, state, decision, host);
  const client = new bindings.BrowserNatlangClient({ host });
  let app;
  app = new bindings.BrowserNatlangApplication({ client, source: await sourceFor(spec),
    initialState: frozen.initial_state, seedRoot: options.seed,
    modelTurn: teacherDriver({ server: options.server, exchanges }) });
  try {
    await app.start();
    const transition = await app.dispatch(event);
    const actual = { state: transition.state, ok: transition.reducerRun.outcome.kind === 'done',
      detail: transition.state.notice };
    const accepted = isDeepStrictEqual(actual.state, frozen.expected.state) && actual.ok === frozen.expected.ok;
    return { schema: 'natlang.studio_teacher_trajectory/1', id: `teacher:${frozen.id}:${options.seed}`,
      case: frozen, provenance: { model: options.model, seed: options.seed,
        source_revision: frozen.source_revision, transport: 'openai-chat/tools-v1' },
      outcome: { accepted, expected: frozen.expected, actual },
      runs: { reducer: transition.reducerRun, view: transition.viewRun }, exchanges };
  } finally { await app.close(); await client.close(); }
}

async function writeAtomic(path, value) {
  const temporary = `${path}.tmp-${process.pid}-${Math.random().toString(16).slice(2)}`;
  await writeFile(temporary, JSON.stringify(value) + '\n');
  await rename(temporary, path);
}

async function validResult(path, frozen, options) {
  try { const row = JSON.parse(await readFile(path, 'utf8'));
    return row.case.id === frozen.id && row.case.source_revision === frozen.source_revision &&
      row.provenance.model === options.model && row.provenance.seed === options.seed;
  } catch { return false; }
}

async function main() {
  const args = process.argv.slice(2), take = flag => { const at = args.indexOf(flag); return at < 0 ? null : args[at + 1]; };
  const positional = args.filter((value, index) => !value.startsWith('--') && !args[index - 1]?.startsWith('--'));
  if (positional.length < 2) throw new Error('usage: collect-studio-teacher.mjs CASES.jsonl JOB_DIR --model ID --seed N [--server URL] [--workers N]');
  const casesPath = resolve(positional[0]), jobs = resolve(positional[1]);
  const options = { model: take('--model'), seed: Number(take('--seed')), server: take('--server') ?? 'http://127.0.0.1:8081', workers: Number(take('--workers') ?? 1) };
  if (!options.model || !Number.isSafeInteger(options.seed) || !Number.isInteger(options.workers) || options.workers < 1)
    throw new Error('--model, integer --seed, and positive --workers are required');
  await mkdir(jobs, { recursive: true });
  const cases = (await readFile(casesPath, 'utf8')).trim().split('\n').filter(Boolean).map(JSON.parse);
  const bindings = await api();
  let cursor = 0, completed = 0;
  async function worker() {
    while (cursor < cases.length) {
      const index = cursor++, frozen = cases[index], path = resolve(jobs, `${String(index).padStart(6,'0')}-${digest(frozen).slice(0,16)}.result.json`);
      if (await validResult(path, frozen, options)) { completed++; continue; }
      try { const result = await runCase(frozen, options, bindings); await writeAtomic(path, result); completed++;
        console.log(`${index} ${frozen.id}: accepted=${result.outcome.accepted} (${completed}/${cases.length})`); }
      catch (error) { await writeAtomic(resolve(jobs, `${String(index).padStart(6,'0')}.error.json`),
        { case: frozen.id, error: String(error), stack: error?.stack });
        console.error(`${index} ${frozen.id}: ${error}`); }
    }
  }
  await Promise.all(Array.from({ length: options.workers }, worker));
  const missing = [];
  for (const [index, frozen] of cases.entries()) {
    const path = resolve(jobs, `${String(index).padStart(6,'0')}-${digest(frozen).slice(0,16)}.result.json`);
    if (!await validResult(path, frozen, options)) missing.push(index);
  }
  await writeAtomic(resolve(jobs, 'manifest.json'), { schema: 'natlang.studio_teacher_batch/1',
    cases_sha256: digest(await readFile(casesPath)), model: options.model, seed: options.seed,
    completed: cases.length - missing.length, missing });
  if (missing.length) process.exitCode = 2;
}

await main();
