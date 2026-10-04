#!/usr/bin/env node
/**
 * Per-turn syntax failure rates of model replies in natlang reduction traces (*.trace.jsonl), to decide how much
 * guided generation can buy (envelope grammar, TypeScript check-and-backtrack). Each generated turn is one of:
 *
 *   ok             parsed tool calls; every eval call's code parses as TypeScript
 *   ts-syntax      parsed tool calls, but an eval call's code does not parse (syntactic diagnostics only)
 *   unclosed       call markup that never closes (cut off by the token limit: runaway or repetition)
 *   malformed      closed call markup the runtime could not parse
 *   text           a reply without call markup
 *
 * Repetition is reported separately: a turn whose text repeats one line at least 4 times.
 *
 *   node scripts/measure-turn-syntax.mjs [--by-dir] DIR_OR_FILE...
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import ts from 'typescript';

const args = process.argv.slice(2);
const byDir = args.includes('--by-dir');
const files = [];
const walk = path => {
  if (statSync(path).isDirectory()) for (const name of readdirSync(path)) walk(join(path, name));
  else if (path.endsWith('.trace.jsonl')) files.push(path);
};
for (const arg of args.filter(a => !a.startsWith('--'))) walk(arg);

function tsErrors(code) {
  const source = ts.createSourceFile('eval.ts', `async function __eval() {\n${code}\n}`, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
  return source.parseDiagnostics.map(d => ts.flattenDiagnosticMessageText(d.messageText, ' '));
}
const repeats = text => {
  const counts = new Map();
  for (const line of text.split(/\\n|\n/).map(l => l.trim()).filter(l => l.length > 8)) counts.set(line, (counts.get(line) ?? 0) + 1);
  return Math.max(0, ...counts.values()) >= 4;
};

const groups = new Map();
const examples = { 'ts-syntax': [], malformed: [] };
for (const file of files) {
  const key = byDir ? dirname(file) : 'all';
  const g = groups.get(key) ?? { turns: 0, ok: 0, 'ts-syntax': 0, unclosed: 0, malformed: 0, text: 0, repetitive: 0, evalCalls: 0 };
  groups.set(key, g);
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.includes('"proposal"')) continue;
    let event;
    try { event = JSON.parse(line); } catch { continue; }
    if (event.kind !== 'proposal' || event.phase !== 'generated') continue;
    g.turns++;
    const text = String(event.text ?? '');
    if (repeats(text)) g.repetitive++;
    const calls = event.calls ?? [];
    if (calls.length) {
      const errors = calls.filter(([name]) => name === 'eval').flatMap(([, a]) => { g.evalCalls++; return tsErrors(String(a?.code ?? '')); });
      if (errors.length) { g['ts-syntax']++; if (examples['ts-syntax'].length < 8) examples['ts-syntax'].push(errors[0]); }
      else g.ok++;
    } else if (/^\s*(<\|tool_call_start\|>)?\s*\[\w+\(/.test(text)) {
      if (/\)\]\s*(<\|tool_call_end\|>)?\s*$/.test(text)) { g.malformed++; if (examples.malformed.length < 5) examples.malformed.push(text.slice(0, 160)); }
      else g.unclosed++;
    } else g.text++;
  }
}
const rate = (n, d) => d ? +(n / d).toFixed(3) : 0;
for (const [key, g] of groups) {
  if (!g.turns) continue;
  console.log(JSON.stringify({ group: key, turns: g.turns, eval_calls: g.evalCalls,
    rates: Object.fromEntries(['ok', 'ts-syntax', 'unclosed', 'malformed', 'text', 'repetitive'].map(k => [k, rate(g[k], g.turns)])) }));
}
console.log(JSON.stringify({ files: files.length, examples }, null, 1));
