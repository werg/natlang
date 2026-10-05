#!/usr/bin/env node
// Static demonstrations from verified curriculum cases: replay each case's reference solution (root eval programs that
// map nl lambdas over items, named helper calls, crisp aggregation; children answering from the case's hidden labels)
// through the collector's execution path, and write ordinary collector rows plus their native training decisions.
// No model calls. The reasoning of a replayed turn is a stock action note (provenance.synthetic_reasoning
// 'action-notes/1'), kept as context but never trained as a target. Rows are admitted like teacher rows.
//   node scripts/inline-curriculum/replay-demonstrations.mjs CASES.ir.jsonl OUT.results.jsonl OUT.turns.jsonl
import { appendFile, readFile, writeFile } from 'node:fs/promises';
import * as collector from '../../dist/teacher/collector.js';
import * as curriculum from '../../dist/teacher/curriculum.js';
import * as materializer from '../../dist/teacher/native-materializer.js';
import { loadRationales, rationaleKey } from './rationales.mjs';

const [input, rowsOut, turnsOut, flag, rationalesPath] = process.argv.slice(2);
if (!input || !rowsOut || !turnsOut) throw new Error('usage: replay-demonstrations.mjs CASES ROWS TURNS [--rationales FILE]');
// Synthesized rationales (rationales.mjs) replace the stock action notes, and are trained, when every turn has one.
const rationales = flag === '--rationales' ? loadRationales(rationalesPath) : null;
const cases = (await readFile(input, 'utf8')).trim().split('\n').map(line => JSON.parse(line));
const options = { modelId: 'reference-demonstration', rootSeed: 7301, systemPrompt: collector.defaultSystemPrompt,
  toolSurfaceSha256: await collector.defaultToolSurfaceHash(), contextTokens: 65536, maxTurns: 60 };
const skipped = {};
let admitted = 0, decisions = 0, unlinked = 0;
await writeFile(rowsOut, ''); await writeFile(turnsOut, '');
// Rows are large (whole trajectories with their contexts): replay, admit and materialize in chunks, appending.
for (let start = 0; start < cases.length; start += 20) {
  const rows = [];
  for (const [offset, ir] of cases.slice(start, start + 20).entries()) {
    let missing = 0;
    const hook = rationales ? (context, calls) => {
      const found = rationales.get(rationaleKey(ir.id, context, calls));
      if (!found) missing++;
      return found;
    } : undefined;
    const replay = await curriculum.replayReference(ir, collector.defaultSystemPrompt, hook);
    const provenance = { ...collector.expectedProvenance(ir, options), collection_role: 'reference-demonstration',
      synthetic_reasoning: rationales && !missing ? 'rationalized-actions/1' : 'action-notes/1' };
    const row = collector.programRow(ir, options.modelId, collector.programRunId(start + offset, provenance), provenance, replay.run, replay.trajectory);
    const admission = curriculum.admitRow(row);
    if (!admission.admitted) { for (const reason of admission.reasons) skipped[reason] = (skipped[reason] ?? 0) + 1; continue; }
    rows.push(row);
  }
  const native = materializer.materializeNativeRows(rows);
  admitted += rows.length; decisions += native.turns.length; unlinked += native.unlinked.length;
  // One line at a time: a chunk's decisions together can exceed the longest string V8 allows.
  for (const row of rows) await appendFile(rowsOut, JSON.stringify(row) + '\n');
  for (const turn of native.turns) await appendFile(turnsOut, JSON.stringify(turn) + '\n');
}
console.log(JSON.stringify({ cases: cases.length, admitted, skipped, decisions, unlinked }));
