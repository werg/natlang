#!/usr/bin/env node
// Run exports of natlang codebase TypeScript modules through the collector's own authoring harness
// (teacher/authoring.ts), so build-time expected values come from exactly the execution path the oracle uses.
// usage: node scripts/rl_tasks/run_exports.mjs RUNTIME_TS_HOST JOBS.jsonl OUT.jsonl
// JOBS: {id, files: {path: text}, module, export, args}; OUT: {id, ok, value} or {id, ok: false, error}.
import fs from 'node:fs';
const [runtime, input, output] = process.argv.slice(2);
const { checkAuthoring } = await import(runtime + '/dist/teacher/authoring.js');
const jobs = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const out = fs.openSync(output, 'w');
const MISS = '\u0000natlang-run-exports-no-expected\u0000';
for (const job of jobs) {
  let row;
  try {
    const result = await checkAuthoring(job.files, { module: job.module, export: job.export, runs: [{ args: job.args, expected: MISS }] });
    const problem = result.problems[0] ?? '';
    row = /^run 0: got /.test(problem) ? { id: job.id, ok: true, value: result.results[0] } :
      { id: job.id, ok: false, error: problem.slice(0, 500) };
  } catch (error) { row = { id: job.id, ok: false, error: String(error).slice(0, 500) }; }
  fs.writeSync(out, JSON.stringify(row) + '\n');
}
fs.closeSync(out);
console.log(JSON.stringify({ jobs: jobs.length }));
