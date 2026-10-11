#!/usr/bin/env node
// Replay port-record call cases (build-port-record-calls.mjs) into native training turns: the static corpus path
// (references.mjs referenceRow, following the runtime's cut-off pages after a read or eval, so the reference child
// reads the whole evidence), then admitRow and materializeNativeRows with the child answers supervised as direct
// answers (children act without reasoning, owner 2026-10-06). No model calls. Writes OUT.turns.jsonl, OUT.rows.jsonl
// and OUT.summary.json; a case whose replay is not admitted is an omission with its reasons, never written.
//   node replay-port-record-calls.mjs CASES.ir.jsonl OUT_PREFIX
import { open, readFile, rename, writeFile } from 'node:fs/promises';
import { admitRow } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash } from '../../dist/teacher/collector.js';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { referenceRow } from './references.mjs';

const [input, prefix] = process.argv.slice(2);
if (!input || !prefix) throw new Error('usage: replay-port-record-calls.mjs CASES.ir.jsonl OUT_PREFIX');
const cases = (await readFile(input, 'utf8')).split('\n').filter(Boolean).map(line => JSON.parse(line));
const options = { modelId: 'port-record-reference', rootSeed: 7411, systemPrompt: TOOLS_PROMPT, contextTokens: 65536, maxTurns: 200,
  toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference', followCutoffPages: true, followEvalCutoffPages: true };
const turnsOut = await open(`${prefix}.turns.jsonl.tmp`, 'w'), rowsOut = await open(`${prefix}.rows.jsonl.tmp`, 'w');
const stats = { cases: cases.length, admitted: 0, decisions: 0, approved: 0, held: {}, omitted: {}, omissions: [] };
for (const [index, record] of cases.entries()) {
  let row;
  try { row = await referenceRow(record, index, options); } catch (error) {
    stats.omitted.replay_error = (stats.omitted.replay_error ?? 0) + 1;
    stats.omissions.push({ id: record.id, reason: `replay_error: ${String(error).slice(0, 300)}` }); continue;
  }
  const admission = admitRow(row);
  if (!admission.admitted) {
    for (const reason of admission.reasons) stats.omitted[reason] = (stats.omitted[reason] ?? 0) + 1;
    stats.omissions.push({ id: record.id, reason: admission.reasons.join(',') }); continue;
  }
  const native = materializeNativeRows([row], { directAnswers: true });
  if (native.acceptedRows !== 1 || native.unlinked.length) {
    stats.omitted.unlinked = (stats.omitted.unlinked ?? 0) + 1; stats.omissions.push({ id: record.id, reason: 'unlinked' }); continue;
  }
  stats.admitted++;
  await rowsOut.write(JSON.stringify(row) + '\n');
  for (const turn of native.turns) {
    stats.decisions++;
    if (turn.training_admission.approved) stats.approved++;
    else stats.held[turn.training_admission.reason] = (stats.held[turn.training_admission.reason] ?? 0) + 1;
    await turnsOut.write(JSON.stringify(turn) + '\n');
  }
}
await turnsOut.close(); await rowsOut.close();
await rename(`${prefix}.turns.jsonl.tmp`, `${prefix}.turns.jsonl`);
await rename(`${prefix}.rows.jsonl.tmp`, `${prefix}.rows.jsonl`);
await writeFile(`${prefix}.summary.json`, JSON.stringify(stats, null, 1) + '\n');
const { omissions, ...counts } = stats;
console.log(JSON.stringify(counts));
