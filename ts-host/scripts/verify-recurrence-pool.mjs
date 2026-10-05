#!/usr/bin/env node
// Verify a pool against the exact runtime that will collect it; no model calls.
import { readFile, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { createHash } from 'node:crypto';
const [input, runtime, output] = process.argv.slice(2);
if (!input || !runtime || !output) throw new Error('usage: verify-recurrence-pool.mjs CASES RUNTIME REPORT');
const rt = resolve(runtime);
const collector = await import(join(rt, 'dist/teacher/collector.js'));
const curriculum = await import(join(rt, 'dist/teacher/curriculum.js'));
const materializer = await import(join(rt, 'dist/teacher/native-materializer.js'));
const policy = await import(join(rt, 'dist/teacher/curriculum-policy.js'));
const sourceReview = await import(join(rt, 'dist/teacher/source-review.js'));
const bytes = await readFile(input), cases = bytes.toString().trim().split('\n').map(JSON.parse);
const details = [];
for (const [index, ir] of cases.entries()) {
  const holds = [policy.generationHoldReason(ir), policy.quarantineReason(ir), sourceReview.sourceReviewReason(ir)].filter(Boolean);
  if (holds.length) throw new Error(`${ir.id}: ${holds.join(', ')}`);
  const replay = await curriculum.replayReference(ir, collector.defaultSystemPrompt);
  if (!replay.run.outcome.accepted) throw new Error(`${ir.id}: reference rejected`);
  const options = { modelId: 'recurrence-reference', rootSeed: 6105, systemPrompt: collector.defaultSystemPrompt,
    toolSurfaceSha256: await collector.defaultToolSurfaceHash(rt), contextTokens: 65536, maxTurns: 60 };
  const provenance = collector.expectedProvenance(ir, options);
  const row = collector.programRow(ir, options.modelId, collector.programRunId(index, provenance), provenance, replay.run, replay.trajectory);
  const admission = curriculum.admitRow(row);
  const native = materializer.materializeNativeRows([row]);
  if (!admission.admitted || native.acceptedRows !== 1 || native.unlinked.length) throw new Error(`${ir.id}: native admission/link failed`);
  details.push({ id: ir.id, native_decisions: native.turns.length, unlinked: native.unlinked.length,
    depth: ir.generation.recurrence.depth, width: ir.generation.recurrence.width });
}
await writeFile(output, JSON.stringify({ schema: 'natlang.recurrence-pool-proof/1',
  cases_sha256: createHash('sha256').update(bytes).digest('hex'), runtime: rt, verified: details.length,
  details, admission: 'reference-proof-only; generated trajectories require independent admission' }, null, 2) + '\n');
console.log(`${details.length} verified against ${rt}`);
