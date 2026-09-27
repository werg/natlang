#!/usr/bin/env node
// Snapshot atomic completed job files; ignore traces, journals and in-flight temporary files.
import { readdir, readFile, rename, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
export async function snapshotJobs(jobs, output) {
  const names = (await readdir(jobs)).filter(name => name.endsWith('.result.json')).sort();
  const temporary = `${output}.tmp-${process.pid}-${randomUUID()}`;
  // Validate every row before replacing a usable snapshot. Never silently drop corrupt results.
  const rows = [];
  for (const name of names) {
    const row = JSON.parse(await readFile(join(jobs, name), 'utf8'));
    if (!row.task?.program_ir?.id || !row.outcome) throw new Error(`Invalid teacher result: ${join(jobs, name)}`);
    rows.push(JSON.stringify(row));
  }
  await writeFile(temporary, rows.length ? rows.join('\n') + '\n' : '');
  await rename(temporary, output);
  return rows.length;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [jobs, output] = process.argv.slice(2);
  if (!jobs || !output) throw new Error('usage: snapshot_teacher_jobs.mjs JOBS_DIR OUTPUT.jsonl');
  console.log(`${await snapshotJobs(jobs, output)} completed jobs -> ${output}`);
}
