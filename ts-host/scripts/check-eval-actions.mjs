#!/usr/bin/env node
/** Compile every eval action of recorded per-call rows with the current runtime's eval compiler.
 *
 * For candidate SFT rows whose task cannot be replayed (scripts/neuralese_data/sft_sources.py, October sources): each
 * row's target tool calls named `eval` are compiled exactly as the realm compiles them (native/eval-compile.ts, an async
 * body with package imports lowered), and a `var` declaration (which eval rejects) counts as a failure. Writes one line
 * per row: {id, evals, failed: [{index, error}]}. Rows without eval calls are reported with evals 0.
 *
 *   node scripts/check-eval-actions.mjs ROWS.jsonl[.zst] OUT.jsonl
 */
import { createReadStream, createWriteStream } from 'node:fs';
import { createInterface } from 'node:readline';
import { createZstdDecompress } from 'node:zlib';
import ts from 'typescript';
import { compileEval } from '../dist/native/eval-compile.js';

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error('usage: node scripts/check-eval-actions.mjs ROWS.jsonl[.zst] OUT.jsonl');
  process.exit(2);
}

/** Whether `code` declares a variable with `var` anywhere. */
export function declaresVar(code) {
  const file = ts.createSourceFile('eval.ts', code, ts.ScriptTarget.Latest, false);
  let found = false;
  const visit = node => {
    if (found) return;
    if (ts.isVariableDeclarationList(node) && !(node.flags & (ts.NodeFlags.Let | ts.NodeFlags.Const | ts.NodeFlags.Using)))
      found = true;
    else ts.forEachChild(node, visit);
  };
  visit(file);
  return found;
}

export function checkCode(code) {
  if (typeof code !== 'string') return 'eval code is not a string';
  try {
    compileEval(code, { body: true, asyncBody: true, imports: () => {} });
  } catch (error) {
    return String(error?.message ?? error).slice(0, 300);
  }
  return declaresVar(code) ? 'var declaration (eval rejects var)' : null;
}

const source = createReadStream(input);
const lines = createInterface({ input: input.endsWith('.zst') ? source.pipe(createZstdDecompress()) : source, crlfDelay: Infinity });
const out = createWriteStream(output);
let rows = 0, evals = 0, failedRows = 0;
for await (const line of lines) {
  if (!line.trim()) continue;
  const row = JSON.parse(line);
  const failed = [];
  let count = 0;
  (row.target?.tool_calls ?? []).forEach((call, index) => {
    if (call?.function?.name !== 'eval') return;
    count++;
    let args = call.function.arguments;
    try { if (typeof args === 'string') args = JSON.parse(args); } catch { failed.push({ index, error: 'arguments are not JSON' }); return; }
    const error = checkCode(args?.code);
    if (error) failed.push({ index, error });
  });
  rows++;
  evals += count;
  if (failed.length) failedRows++;
  out.write(JSON.stringify({ id: row.id, evals: count, failed }) + '\n');
}
out.end();
console.log(JSON.stringify({ input, rows, evals, failed_rows: failedRows }));
