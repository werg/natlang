#!/usr/bin/env node
// Neuralese language migration of the natlang corpus (plans/neuralese/S1_DATA.md §7,
// training/api-migrations/neuralese-language.json): eager typing and explicit captures.
//
// Compiler-checked candidate migration. Execution replay is still required before admission.
// Original rows, lineage and split identities are preserved; output never replaces a run.
//
// Usage:
//   node scripts/migrate-neuralese-language.mjs --input prepared/teacher.jsonl --out DIR [--inventory] [--limit N]
import { createWriteStream, mkdirSync, writeFileSync, renameSync, readdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { finished } from 'node:stream/promises';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { jsonlRows, fileDigest } from './jsonl-stream.mjs';

const MIGRATION = 'neuralese-language';
const CHANGES = ['eager-typing', 'explicit-captures'];

function args(argv) {
  const out = { inventory: false, limit: Infinity };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--input') out.input = argv[++i];
    else if (a === '--out') out.out = argv[++i];
    else if (a === '--compiler-module') out.compilerModule = resolve(argv[++i]);
    else if (a === '--inventory') out.inventory = true;
    else if (a === '--limit') out.limit = Number(argv[++i]);
    else throw new Error(`unknown argument ${a}`);
  }
  if (!out.input || !out.out) throw new Error('--input and --out are required');
  if (!(out.limit === Infinity || Number.isSafeInteger(out.limit) && out.limit > 0)) throw new Error('--limit must be a positive integer');
  if (!out.inventory && !out.compilerModule) throw new Error('Rewrite requires --compiler-module pointing to a separately built compiler');
  return out;
}

// Model-written eval code: every eval tool call in the history and in the target turn.
export function evalSites(row) {
  const sites = [];
  const turns = [...(row.messages ?? []).map((m, i) => [`messages.${i}`, m]), ['target', row.target]];
  for (const [where, message] of turns) {
    if (message?.role !== 'assistant') continue;
    for (const [j, call] of (message.tool_calls ?? []).entries()) {
      if (call.function?.name !== 'eval' || call.id === 'scope_0') continue;
      let code;
      try { code = (typeof call.function.arguments === 'string' ? JSON.parse(call.function.arguments) : call.function.arguments)?.code; } catch { code = undefined; }
      if (typeof code === 'string') sites.push({ where: `${where}.tool_calls.${j}`, code });
    }
  }
  return sites;
}

// Program sources carried with the row; inline nl sites live here and in eval code.
export function programSources(row) {
  const files = row.task?.program_ir?.semantics?.files ?? {};
  return Object.entries(files).filter(([, text]) => typeof text === 'string').map(([path, text]) => ({ path, text }));
}

const INLINE_NL = /\bnl\s*`/g;
const DECLARATION = /\b(?:const|let)\s+[\w$[{]/g;
const count = (re, text) => (text.match(re) ?? []).length;

// Removing just the rewritten eval code must leave the complete row unchanged.
function invariantProjection(row) {
  const value = structuredClone(row);
  for (const message of [...(value.messages ?? []), value.target]) {
    if (message?.role !== 'assistant') continue;
    for (const call of message.tool_calls ?? []) {
      if (call.function?.name !== 'eval' || call.id === 'scope_0') continue;
      const raw = call.function.arguments;
      let args;
      try { args = typeof raw === 'string' ? JSON.parse(raw) : structuredClone(raw); } catch { continue; }
      if (args && typeof args.code === 'string') {
        args.code = '<migrated-eval-code>';
        call.function.arguments = args;
      }
    }
  }
  return JSON.stringify(value);
}
const rowDigest = row => createHash('sha256').update(JSON.stringify(row)).digest('hex');
async function compilerTreeDigest(root) {
  const hash = createHash('sha256');
  async function walk(directory) {
    for (const item of readdirSync(directory, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      const path = resolve(directory, item.name);
      if (item.isDirectory()) await walk(path);
      else if (item.isFile()) hash.update(path.slice(root.length)).update(await fileDigest(path));
      else throw new Error(`Compiler tree contains unsupported link: ${path}`);
    }
  }
  await walk(root);
  return hash.digest('hex');
}

async function main() {
  const opts = args(process.argv.slice(2));
  const out = resolve(opts.out);
  mkdirSync(resolve(out, '..'), { recursive: true });
  mkdirSync(out); // Exclusive reservation preserves interrupted and completed runs.
  const inputSha256 = await fileDigest(opts.input);
  const compilerSha256 = opts.compilerModule ? await fileDigest(opts.compilerModule) : null;
  const compilerRoot = opts.compilerModule ? resolve(opts.compilerModule, '../..') : null;
  const compilerTreeSha256 = compilerRoot ? await compilerTreeDigest(compilerRoot) : null;
  const passes = opts.inventory ? null : await import(pathToFileURL(opts.compilerModule).href);
  const report = {
    version: 'natlang.api-migration-run/1', migration: MIGRATION, changes: CHANGES,
    input: resolve(opts.input), mode: opts.inventory ? 'inventory' : 'rewrite',
    rows: 0, rows_with_eval: 0, eval_sites: 0, declarations: 0, inline_nl_sites: 0,
    program_inline_nl_sites: 0, by_family: {}, rewritten: 0,
    input_sha256: inputSha256, compiler_module: opts.compilerModule ?? null, compiler_sha256: compilerSha256,
    compiler_tree_root: compilerRoot, compiler_tree_sha256: compilerTreeSha256,
    status: opts.inventory ? 'inventory_only' : 'candidate_pending_execution_replay',
    execution_replay: 'not_performed', publication_allowed: false,
    compiler_failed_rows: 0, annotations: 0, capture_sites: 0, skipped: {},
    program_sources: 'inventoried_only; not rewritten without program-specific compilation context',
  };
  const sink = opts.inventory ? null : createWriteStream(resolve(out, 'teacher.candidate.jsonl.pending'), { flags: 'wx' });
  let streamError;
  sink?.on('error', error => { streamError = error; });
  for await (const row of jsonlRows(opts.input)) {
    if (report.rows >= opts.limit) break;
    report.rows++;
    const sites = evalSites(row);
    const family = row.task_family ?? row.family ?? 'unknown';
    const f = (report.by_family[family] ??= { rows: 0, eval_sites: 0, declarations: 0, inline_nl_sites: 0 });
    f.rows++;
    if (sites.length) report.rows_with_eval++;
    for (const { code } of sites) {
      const d = count(DECLARATION, code), n = count(INLINE_NL, code);
      report.eval_sites++; report.declarations += d; report.inline_nl_sites += n;
      f.eval_sites++; f.declarations += d; f.inline_nl_sites += n;
    }
    for (const { text } of programSources(row)) report.program_inline_nl_sites += count(INLINE_NL, text);
    if (sink) {
      if (!Array.isArray(row.messages)) throw new Error(`Row ${report.rows} has no messages: refusing silent schema loss`);
      const { record, stats } = passes.rewriteTrajectory(row);
      if (invariantProjection(row) !== invariantProjection(record)) throw new Error(`Row ${report.rows}: non-code invariant changed`);
      const changed = rowDigest(row) !== rowDigest(record);
      report.rewritten += Number(changed);
      report.compiler_failed_rows += Number(stats.failed.length > 0);
      report.annotations += stats.annotated;
      report.capture_sites += stats.sites;
      for (const [reason, count] of Object.entries(stats.skipped)) report.skipped[reason] = (report.skipped[reason] ?? 0) + count;
      const migrated = { ...record, data_rewrite: {
        migration: MIGRATION, version: passes.DATA_REWRITE_VERSION, changes: CHANGES,
        parent_row_sha256: rowDigest(row), rewritten_row_sha256: rowDigest(record),
        compiler_sha256: compilerSha256, changed, stats,
        execution_replay: 'pending', admission: 'held_pending_execution_replay',
        previous_data_rewrite: row.data_rewrite ?? null,
      } };
      if (streamError) throw streamError;
      if (!sink.write(JSON.stringify(migrated) + '\n')) await once(sink, 'drain');
    }
    if (report.rows % 500 === 0) writeFileSync(resolve(out, 'progress.json'), JSON.stringify(report, null, 2) + '\n');
  }
  if (sink) { sink.end(); await finished(sink); }
  if (await fileDigest(opts.input) !== inputSha256) throw new Error('Input changed during migration');
  if (opts.compilerModule && await fileDigest(opts.compilerModule) !== compilerSha256) throw new Error('Compiler changed during migration');
  if (compilerRoot && await compilerTreeDigest(compilerRoot) !== compilerTreeSha256) throw new Error('Compiler dependency tree changed during migration');
  if (sink) {
    report.output_sha256 = await fileDigest(resolve(out, 'teacher.candidate.jsonl.pending'));
    renameSync(resolve(out, 'teacher.candidate.jsonl.pending'), resolve(out, 'teacher.candidate.jsonl'));
  }
  writeFileSync(resolve(out, opts.inventory ? 'inventory.json' : 'manifest.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ rows: report.rows, eval_sites: report.eval_sites, declarations: report.declarations,
    inline_nl_sites: report.inline_nl_sites, program_inline_nl_sites: report.program_inline_nl_sites }));
}

main().catch(error => { console.error(error.message); process.exit(1); });
