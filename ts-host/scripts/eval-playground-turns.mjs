#!/usr/bin/env node
/** Paired teacher-forced turn evaluation for the localhost browser workbench. */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const sha = value => createHash('sha256').update(value).digest('hex');
function stable(value) {
  if (Array.isArray(value)) return `[${value.map(stable).join(', ')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort()
    .map(key => `${JSON.stringify(key)}: ${stable(value[key])}`).join(', ')}}`;
  return JSON.stringify(value);
}
const digest = value => sha(stable(value));
const group = family => family.startsWith('cb_') ? 'codebase' : family === 'composed' ? 'composed' :
  ['judge', 'classify', 'extract', 'crisp_scalar'].includes(family) ? 'leaf' : 'shape';
const cell = row => [group(row.family), row.skill];
const cellKey = row => JSON.stringify(cell(row));
const fail = message => { throw new Error(message); };

export function selectSamples(rows, manifestPath, perCell = 12, seed = 0) {
  if (!Number.isInteger(perCell) || perCell < 1) fail('--per-cell must be positive');
  const byId = new Map();
  for (const row of rows) {
    if (byId.has(row.id)) fail(`Duplicate evaluation ID: ${row.id}`);
    byId.set(row.id, row);
  }
  if (manifestPath) {
    return readFile(manifestPath, 'utf8').then(text => {
      const manifest = JSON.parse(text), selected = [];
      for (const sample of manifest.samples) {
        const row = byId.get(sample.id);
        if (!row) fail(`Manifest sample missing: ${sample.id}`);
        if (digest(row) !== sample.sha256) fail(`Evaluation sample changed: ${sample.id}; use a new manifest for a new corpus`);
        selected.push(row);
      }
      return { samples: selected, manifest };
    }, error => {
      if (error.code !== 'ENOENT') throw error;
      return createManifest(rows, manifestPath, perCell, seed);
    });
  }
  return createManifest(rows, null, perCell, seed);
}

async function createManifest(rows, manifestPath, perCell, seed) {
  const pools = new Map();
  for (const row of rows) {
    const key = cellKey(row), pool = pools.get(key) ?? [];
    const rank = BigInt(`0x${sha(JSON.stringify([seed, row.id]))}`);
    pool.push({ rank, row });
    pool.sort((a, b) => a.rank < b.rank ? -1 : a.rank > b.rank ? 1 : 0);
    if (pool.length > perCell) pool.shift();
    pools.set(key, pool);
  }
  const samples = [...pools.entries()].sort(([a], [b]) => a.localeCompare(b))
    .flatMap(([, pool]) => pool.map(item => item.row));
  if (!samples.length) fail('Evaluation corpus is empty');
  const manifest = { version: 'turn-manifest/1', seed, per_cell: perCell, samples: samples.map(row =>
    ({ id: row.id, sha256: digest(row) })) };
  if (manifestPath) {
    await mkdir(dirname(manifestPath), { recursive: true });
    await writeFile(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`, { flag: 'wx' });
  }
  return { samples, manifest };
}

export function scoreTurn(sample, calls, text = '') {
  const expected = (sample.target?.tool_calls ?? []).map(call => [call.function.name,
    typeof call.function.arguments === 'string' ? JSON.parse(call.function.arguments) : call.function.arguments]);
  const actual = calls.map(call => [call.function.name, typeof call.function.arguments === 'string' ?
    JSON.parse(call.function.arguments) : call.function.arguments]);
  const same = (a, b) => stable(a) === stable(b);
  const functions = list => list.filter(([name]) => name === 'call').map(([, args]) => args.function);
  return { id: sample.id, cell: cell(sample), reply: expected.length === 0, exact: same(actual, expected),
    right_tool: same(actual.map(([name]) => name), expected.map(([name]) => name)),
    right_function: functions(expected).length ? same(functions(actual), functions(expected)) : null,
    expected, actual, text };
}

export async function evaluate(samples, server) {
  const results = [];
  for (const sample of samples) {
    const response = await fetch(`${server.replace(/\/$/, '')}/v1/chat/completions`, { method: 'POST',
      headers: { 'content-type': 'application/json' }, body: JSON.stringify({ model: 'local-model',
        messages: sample.messages, tools: sample.tools, tool_choice: 'auto', temperature: 0, seed: 1,
        max_tokens: 500 }) });
    const body = await response.json();
    if (!response.ok) fail(`model HTTP ${response.status}: ${JSON.stringify(body).slice(0, 2000)}`);
    const message = body.choices?.[0]?.message ?? {};
    const calls = (message.tool_calls ?? []).map(call => ({ function: { name: call.function?.name,
      arguments: call.function?.arguments } }));
    results.push(scoreTurn(sample, calls, String(message.content ?? '')));
  }
  return results;
}

const summary = rows => ({ n: rows.length, exact: rows.filter(row => row.exact).length,
  right_tool: rows.filter(row => row.right_tool).length });

async function main() {
  const args = process.argv.slice(2), data = resolve(args[0] ?? ''), take = (flag, fallback) => {
    const at = args.indexOf(flag); return at < 0 ? fallback : args[at + 1];
  };
  if (!args[0]) fail('usage: eval-playground-turns.mjs DATA.jsonl --out FILE --model-label LABEL [--server URL] [--per-cell N] [--seed N] [--manifest FILE]');
  const server = take('--server', 'http://127.0.0.1:8080'), modelLabel = take('--model-label', 'unspecified');
  const perCell = Number(take('--per-cell', '12')), seed = Number(take('--seed', '0'));
  const out = resolve(take('--out', 'runs/eval-playground.json'));
  const manifestPath = resolve(take('--manifest', data.replace(/\.jsonl(?:\.gz)?$/, '.eval-manifest.json')));
  const rows = (await readFile(data, 'utf8')).split(/\r?\n/).filter(Boolean).map(JSON.parse);
  const { samples, manifest } = await selectSamples(rows, manifestPath, perCell, seed);
  const results = await evaluate(samples, server), cells = new Map();
  for (const result of results) { const key = JSON.stringify(result.cell); const list = cells.get(key) ?? []; list.push(result); cells.set(key, list); }
  for (const [key, list] of [...cells.entries()].sort()) {
    const value = summary(list), [category, skill] = JSON.parse(key);
    console.log(`${category.padEnd(9)} ${skill.padEnd(22)} n=${String(value.n).padEnd(3)} exact=${Math.round(value.exact / value.n * 100)}% right tool=${Math.round(value.right_tool / value.n * 100)}%`);
  }
  const totals = { all: summary(results), actions: summary(results.filter(row => !row.reply)),
    replies: summary(results.filter(row => row.reply)) };
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify({ model: modelLabel, server, manifest: manifestPath,
    manifest_sha256: digest(manifest), summary: totals, rows: results }, null, 2)}\n`);
  console.log(`manifest: ${manifestPath}; results: ${out}`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error instanceof Error ? error.stack : String(error)); process.exitCode = 1; });
