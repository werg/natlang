#!/usr/bin/env node
// Static demonstrations from verified curriculum cases: replay each case's reference solution (root eval programs that
// map nl lambdas over items, named helper calls, crisp aggregation; children answering from the case's hidden labels)
// through the collector's execution path, and write ordinary collector rows plus their native training decisions.
// No model calls. The reasoning of a replayed turn is a stock action note (provenance.synthetic_reasoning
// 'action-notes/1'), kept as context but never trained as a target. Rows are admitted like teacher rows.
//   node scripts/inline-curriculum/replay-demonstrations.mjs CASES.ir.jsonl OUT.results.jsonl OUT.turns.jsonl
import { readFile, writeFile } from 'node:fs/promises';
import * as collector from '../../dist/teacher/collector.js';
import * as curriculum from '../../dist/teacher/curriculum.js';
import * as materializer from '../../dist/teacher/native-materializer.js';

const [input, rowsOut, turnsOut] = process.argv.slice(2);
if (!input || !rowsOut || !turnsOut) throw new Error('usage: replay-demonstrations.mjs CASES ROWS TURNS');
const cases = (await readFile(input, 'utf8')).trim().split('\n').map(line => JSON.parse(line));
const options = { modelId: 'reference-demonstration', rootSeed: 7301, systemPrompt: collector.defaultSystemPrompt,
  toolSurfaceSha256: await collector.defaultToolSurfaceHash(), contextTokens: 65536, maxTurns: 60 };
const rows = [], skipped = {};
for (const [index, ir] of cases.entries()) {
  const replay = await curriculum.replayReference(ir, collector.defaultSystemPrompt);
  const provenance = { ...collector.expectedProvenance(ir, options), collection_role: 'reference-demonstration',
    synthetic_reasoning: 'action-notes/1' };
  const row = collector.programRow(ir, options.modelId, collector.programRunId(index, provenance), provenance, replay.run, replay.trajectory);
  const admission = curriculum.admitRow(row);
  if (!admission.admitted) { for (const reason of admission.reasons) skipped[reason] = (skipped[reason] ?? 0) + 1; continue; }
  rows.push(row);
}
const native = materializer.materializeNativeRows(rows);
await writeFile(rowsOut, rows.map(row => JSON.stringify(row)).join('\n') + '\n');
await writeFile(turnsOut, native.turns.map(turn => JSON.stringify(turn)).join('\n') + '\n');
const children = native.turns.filter(turn => turn.role === 'child' || turn.call_depth > 0 || turn.child).length;
console.log(JSON.stringify({ cases: cases.length, admitted: rows.length, skipped, decisions: native.turns.length,
  child_decisions: children, unlinked: native.unlinked.length }));
