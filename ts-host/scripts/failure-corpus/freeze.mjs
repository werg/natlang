#!/usr/bin/env node
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { sha256, writeAtomic } from '../../dist/teacher/collector.js';
import { failureProgramRecords } from './cases.mjs';

export async function freezeFailureCorpus(output) {
  const records = failureProgramRecords();
  const data = records.map(record => JSON.stringify(record)).join('\n') + '\n';
  const manifest = JSON.stringify({ version: 'natlang.failure_repair_corpus/1', count: records.length,
    source_sha256: sha256(await readFile(new URL('./cases.mjs', import.meta.url))),
    output_sha256: sha256(data) }) + '\n';
  for (const [path, content] of [[output, data], [`${output}.manifest.json`, manifest]]) {
    let existing;
    try { existing = await readFile(path, 'utf8'); }
    catch (error) {
      if (error.code !== 'ENOENT') throw error;
    }
    if (existing !== undefined && existing !== content)
      throw new Error(`refusing to overwrite changed failure corpus artifact: ${path}`);
    if (existing === undefined) await writeAtomic(path, content);
  }
  return { output, count: records.length, sha256: sha256(data) };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const path = process.argv[2];
  if (!path) throw new Error('usage: node scripts/failure-corpus/freeze.mjs OUTPUT.jsonl');
  console.log(JSON.stringify(await freezeFailureCorpus(resolve(path))));
}
