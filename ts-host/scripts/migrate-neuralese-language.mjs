#!/usr/bin/env node
// Neuralese language migration of the natlang corpus (plans/neuralese/S1_DATA.md §7,
// training/api-migrations/neuralese-language.json): eager typing and explicit captures.
//
// Skeleton. The row walk, site inventory, snapshot layout and manifest are here; the two
// rewrites are compiler passes owned by S4 and are not implemented yet. Until they land,
// only --inventory runs: it counts the sites each pass will touch without writing rows.
//
// Usage:
//   node scripts/migrate-neuralese-language.mjs --input prepared/teacher.jsonl --out DIR [--inventory] [--limit N]
import { createWriteStream, mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { jsonlRows } from './jsonl-stream.mjs';

const MIGRATION = 'neuralese-language';
const CHANGES = ['eager-typing', 'explicit-captures'];

function args(argv) {
  const out = { inventory: false, limit: Infinity };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--input') out.input = argv[++i];
    else if (a === '--out') out.out = argv[++i];
    else if (a === '--inventory') out.inventory = true;
    else if (a === '--limit') out.limit = Number(argv[++i]);
    else throw new Error(`unknown argument ${a}`);
  }
  if (!out.input || !out.out) throw new Error('--input and --out are required');
  return out;
}

// Model-written eval code: every eval tool call in the history and in the target turn.
export function evalSites(row) {
  const sites = [];
  const turns = [...(row.messages ?? []).map((m, i) => [`messages.${i}`, m]), ['target', row.target]];
  for (const [where, message] of turns) {
    if (message?.role !== 'assistant') continue;
    for (const [j, call] of (message.tool_calls ?? []).entries()) {
      if (call.function?.name !== 'eval') continue;
      let code;
      try { code = JSON.parse(call.function.arguments).code; } catch { code = undefined; }
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

// --- Compiler passes (S4). Each returns { code, rewritten, skipped } for one site. ---------------
// eager-typing: insert the checker's inferred natlang type on each unannotated const/let in eval
// code, given the call's scope types; skip any/unknown/inexpressible types and count them.
function eagerTyping(_code, _scope) {
  throw new Error('eager-typing pass is not implemented (S4 compiler pass)');
}
// explicit-captures: rewrite nl`…` to nl.with({ … })`…` with exactly the analysed capture set;
// captured lets written back become live(x); function-typed captures stay plain snapshots.
function explicitCaptures(_code, _scope) {
  throw new Error('explicit-captures pass is not implemented (S4 compiler pass)');
}
// Validation: the rewritten eval recompiles in the same scope with the same diagnostics, and the
// recorded execution replays unchanged (checked results, effects, files).
function validate(_row, _migrated) {
  throw new Error('migration validation is not implemented (S4 compiler and replay)');
}
// ------------------------------------------------------------------------------------------------

function migrateRow(row) {
  const scope = row.decision ?? null;
  const migrated = structuredClone(row);
  const stats = { declarations: { rewritten: 0, skipped: 0 }, captures: { rewritten: 0 } };
  for (const site of evalSites(row)) {
    const typed = eagerTyping(site.code, scope);
    const captured = explicitCaptures(typed.code, scope);
    stats.declarations.rewritten += typed.rewritten;
    stats.declarations.skipped += typed.skipped;
    stats.captures.rewritten += captured.rewritten;
    // TODO(S4): write captured.code back to the call at site.where.
  }
  validate(row, migrated);
  migrated.migrations = [...(row.migrations ?? []), { id: MIGRATION, changes: CHANGES }];
  return { migrated, stats };
}

async function main() {
  const opts = args(process.argv.slice(2));
  const out = resolve(opts.out);
  mkdirSync(out, { recursive: true });
  const report = {
    version: 'natlang.api-migration-run/1', migration: MIGRATION, changes: CHANGES,
    input: resolve(opts.input), mode: opts.inventory ? 'inventory' : 'rewrite',
    rows: 0, rows_with_eval: 0, eval_sites: 0, declarations: 0, inline_nl_sites: 0,
    program_inline_nl_sites: 0, by_family: {}, rewritten: null,
  };
  const sink = opts.inventory ? null : createWriteStream(resolve(out, 'teacher.jsonl'));
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
    if (sink) sink.write(JSON.stringify(migrateRow(row).migrated) + '\n');
  }
  if (sink) await new Promise(done => sink.end(done));
  writeFileSync(resolve(out, opts.inventory ? 'inventory.json' : 'manifest.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ rows: report.rows, eval_sites: report.eval_sites, declarations: report.declarations,
    inline_nl_sites: report.inline_nl_sites, program_inline_nl_sites: report.program_inline_nl_sites }));
}

main().catch(error => { console.error(error.message); process.exit(1); });
