#!/usr/bin/env node
// Run the live probe cases against an OpenAI-compatible server and write one readable transcript per
// case (prompt, reasoning, calls, results, server timings) plus summary.json.
// node scripts/live-probe/probe.mjs --server http://127.0.0.1:8081 --out /tmp/probe/base [--cases a,b] [--repeat N]
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { Folder, createNatlangRuntime, loadVirtualNatlang, openAICompatibleModelTurn } from '../../dist/index.js';
import { CASES } from './cases.mjs';

const { values } = parseArgs({ options: { server: { type: 'string', default: 'http://127.0.0.1:8081' },
  model: { type: 'string', default: 'probe' }, out: { type: 'string' }, cases: { type: 'string' },
  repeat: { type: 'string', default: '1' } } });
if (!values.out) throw new Error('--out DIRECTORY is required');
const out = resolve(values.out);
mkdirSync(out, { recursive: true });
// Calls run in an empty workspace, as a small application's calls would, not in this checkout.
process.chdir(mkdtempSync(join(tmpdir(), 'natlang-probe-')));
// One case's stray error must not end the round: report it and keep going.
process.on('uncaughtException', error => process.stdout.write(`UNCAUGHT ${error?.stack ?? error}\n`));
process.on('unhandledRejection', error => process.stdout.write(`UNHANDLED ${error?.stack ?? error}\n`));
const selected = values.cases ? CASES.filter(item => values.cases.split(',').includes(item.id)) : CASES;

const block = (text, fence = '') => '```' + fence + '\n' + String(text ?? '').trimEnd() + '\n```\n';
const messageText = message => {
  if (message.role === 'tool') return block(message.content);
  if (message.role === 'assistant') return (message.content ? block(message.content) : '') +
    (message.tool_calls ?? []).map(call => `→ ${call.function.name} ` + block(prettyArgs(call.function.arguments), 'json')).join('');
  return block(message.content);
};
const prettyArgs = raw => {
  try { const args = typeof raw === 'string' ? JSON.parse(raw) : raw;
    return typeof args.code === 'string' && Object.keys(args).length === 1 ? args.code : JSON.stringify(args, null, 1); }
  catch { return String(raw); }
};

async function runCase(item, attempt) {
  const exchanges = [];
  const turn = openAICompatibleModelTurn({ endpoint: values.server, model: values.model,
    onExchange: exchange => { exchanges.push({ ...exchange, at: Date.now() }); } });
  // The server samples with its own settings and seeds, so repeated attempts are independent draws.
  const runtime = createNatlangRuntime({ model: turn, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang(item.files, item.root);
  const started = Date.now();
  const folder = item.folder ? Folder.fromFiles(item.folder) : undefined;
  let value, error;
  try { value = await runtime.run(() => folder ? folder.apply(fn, ...item.args) : fn(...item.args)); }
  catch (failure) { error = failure instanceof Error ? failure.message : String(failure); }
  finally { runtime.close?.(); }
  const verdict = await item.check({ value, error, folder: folder?.root() });
  const seconds = Math.round((Date.now() - started) / 1000);
  const lines = [`# ${item.id} (attempt ${attempt})`, '',
    `**${verdict === true ? 'PASS' : 'FAIL'}** in ${seconds}s, ${exchanges.length} model turns` +
      (verdict === true ? '' : ` — ${verdict}`), '',
    error ? `Error: ${error}\n` : `Value: \`${JSON.stringify(value)}\`\n`];
  // One section per natural-language call (child calls interleave with their parent), keyed by the call's opening.
  const calls = new Map();
  for (const exchange of exchanges) {
    const key = JSON.stringify(exchange.request.messages[1] ?? null);
    if (!calls.has(key)) calls.set(key, []);
    calls.get(key).push(exchange);
  }
  for (const [callIndex, callExchanges] of [...calls.values()].entries()) {
    const seconds = Math.round((callExchanges.at(-1).at - callExchanges[0].at) / 1000);
    lines.push(`## Call ${callIndex + 1}: ${callExchanges.length} turns, about ${seconds}s`, '');
    let shownCount = 0;
    for (const [index, exchange] of callExchanges.entries()) {
      const messages = exchange.request.messages;
      lines.push(`### Turn ${index + 1}`, '');
      if (index === 0) lines.push(`tools: ${exchange.request.tools.map(tool => tool.function?.name).join(', ') || '(none)'}`, '');
      for (const message of messages.slice(messages.length >= shownCount ? shownCount : 0)) lines.push(`#### ${message.role}`, '', messageText(message));
      shownCount = messages.length + 1;
      const choice = exchange.wireResponse.choices?.[0]?.message ?? {};
      const timings = exchange.wireResponse.timings;
      lines.push(`#### model` + (timings ? ` (prompt ${timings.prompt_n} tok ${Math.round(timings.prompt_ms)}ms, cached ${timings.cache_n ?? '?'}; ` +
      `output ${timings.predicted_n} tok ${Math.round(timings.predicted_ms)}ms)` : ''), '');
      if (choice.reasoning_content) lines.push('<details><summary>reasoning</summary>', '', block(choice.reasoning_content), '</details>', '');
      if (choice.content) lines.push(block(choice.content));
      for (const call of choice.tool_calls ?? []) lines.push(`→ ${call.function.name} ` + block(prettyArgs(call.function.arguments), 'json'));
    }
  }
  writeFileSync(join(out, `${item.id}${attempt > 1 ? `-${attempt}` : ''}.md`), lines.join('\n') + '\n');
  return { id: item.id, attempt, pass: verdict === true, reason: verdict === true ? null : verdict,
    seconds, turns: exchanges.length, value: value ?? null, error: error ?? null };
}

const results = [];
for (let attempt = 1; attempt <= Number(values.repeat); attempt++)
  for (const item of selected) {
    const result = await runCase(item, attempt);
    results.push(result);
    process.stdout.write(`${result.pass ? 'PASS' : 'FAIL'} ${item.id} ${result.seconds}s ${result.turns} turns` +
      `${result.pass ? '' : ` — ${result.reason}`}\n`);
    writeFileSync(join(out, 'summary.json'), JSON.stringify(results, null, 2) + '\n');
  }
const passed = results.filter(result => result.pass).length;
process.stdout.write(`${passed}/${results.length} passed; transcripts in ${out}\n`);
