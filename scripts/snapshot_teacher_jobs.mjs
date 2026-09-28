#!/usr/bin/env node
// Snapshot atomic completed job files; ignore traces, journals and in-flight temporary files.
import { readdir, readFile, rename, open, unlink } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { randomUUID } from 'node:crypto';
export async function snapshotJobs(jobs, output) {
  const names = (await readdir(jobs)).filter(name => name.endsWith('.result.json')).sort();
  const temporary = `${output}.tmp-${process.pid}-${randomUUID()}`;
  // Stream one validated row at a time: folder trajectories exceed V8's single-string limit in aggregate.
  const handle = await open(temporary, 'wx');
  try {
    try {
      for (const name of names) {
        const row = JSON.parse(await readFile(join(jobs, name), 'utf8'));
        if (!row.task?.program_ir?.id || !row.outcome) throw new Error(`Invalid teacher result: ${join(jobs, name)}`);
        await handle.writeFile(JSON.stringify(row) + '\n');
      }
      await handle.sync();
    } finally { await handle.close(); }
    await rename(temporary, output);
  } catch (error) { await unlink(temporary).catch(() => {}); throw error; }
  return names.length;
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const [jobs, output] = process.argv.slice(2);
  if (!jobs || !output) throw new Error('usage: snapshot_teacher_jobs.mjs JOBS_DIR OUTPUT.jsonl');
  console.log(`${await snapshotJobs(jobs, output)} completed jobs -> ${output}`);
}
