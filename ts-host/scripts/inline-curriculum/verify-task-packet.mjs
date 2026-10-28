#!/usr/bin/env node
// Re-verify a task packet's cases.jsonl: every case's reference solution is replayed through the collector's
// execution path (verifyCases, as build.mjs does at generation). Prints a JSON summary; exit 1 on any failure.
// node scripts/inline-curriculum/verify-task-packet.mjs CASES.jsonl
// A packet may contain a family the current source-review registry holds (edit_stream); set NATLANG_SOURCE_REVIEWS to a
// registry copy without that hold to verify the reference/oracle only. Record that use in the packet's provenance.
import { readFileSync } from 'node:fs';
import { verifyCases } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';

const path = process.argv[2];
if (!path) throw new Error('usage: verify-task-packet.mjs CASES.jsonl');
const records = readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
const results = await verifyCases(records, TOOLS_PROMPT);
const failures = results.filter(item => !item.ok);
const byFamily = {};
for (const [i, record] of records.entries()) {
  const family = record.curriculum.family;
  byFamily[family] ??= { cases: 0, verified: 0 };
  byFamily[family].cases++;
  if (results[i]?.ok) byFamily[family].verified++;
}
console.log(JSON.stringify({ cases: records.length, verified: records.length - failures.length, by_family: byFamily,
  source_review_registry: process.env.NATLANG_SOURCE_REVIEWS ?? 'training/source-reviews (default)',
  failures: failures.map(item => ({ id: item.id, problems: item.problems })) }, null, 1));
process.exit(failures.length ? 1 : 0);
