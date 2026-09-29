#!/usr/bin/env node
/** Resolve a ready static bundle for build_lora_sft.sh, failing closed on stale/tampered data. */
import { mkdir, readFile, writeFile, open, rename, unlink } from 'node:fs/promises';
import { createHash, randomUUID } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { digest, safePath } from './directory-sources.mjs';
import { admitRow } from '../../dist/teacher/curriculum.js';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { jsonlRows, fileDigest } from '../jsonl-stream.mjs';

export async function staticBundleInput(manifestPath, { optional = false } = {}) {
  let raw;
  try { raw = await readFile(manifestPath, 'utf8'); }
  catch (error) { if (optional && error.code === 'ENOENT') return null; throw error; }
  const manifest = JSON.parse(raw);
  if (manifest.version !== 'natlang.source_static_bundle/1' || !manifest.cases || manifest.results?.rows !== manifest.cases)
    throw new Error('invalid_static_bundle_manifest');
  const root = dirname(resolve(manifestPath));
  const input = resolve(root, safePath(manifest.results.path));
  const irPath = resolve(root, safePath(manifest.ir.path));
  const [resultsHash, irHash] = await Promise.all([fileDigest(input), fileDigest(irPath)]);
  if (resultsHash !== manifest.results.sha256 || irHash !== manifest.ir.sha256)
    throw new Error('static_bundle_checksum_mismatch');
  const records = jsonlRows(irPath);
  let count = 0;
  try { for await (const row of jsonlRows(input)) {
    const record = await records.next();
    if (record.done) throw new Error('static_bundle_row_count_mismatch');
    if (digest(record.value) !== digest(row.task?.program_ir)) throw new Error('static_bundle_ir_result_mismatch');
    if (!row.provenance?.source_conversion || !admitRow(row).admitted) throw new Error('static_bundle_admission_failed');
    count++;
  }
  if (!(await records.next()).done || count !== manifest.cases || manifest.ir.rows !== count)
    throw new Error('static_bundle_row_count_mismatch');
  } finally { await records.return(); }
  return input;
}

export async function materializeStaticBundle(manifestPath, output) {
  const input = await staticBundleInput(manifestPath);
  await mkdir(dirname(resolve(output)), { recursive: true });
  const staged = `${output}.building-${randomUUID()}`, hash = createHash('sha256');
  const handle = await open(staged, 'wx');
  let turns = 0;
  try { for await (const row of jsonlRows(input)) {
    const result = materializeNativeRows([row]);
    if (result.rejectedRows || result.unlinked.length) throw new Error('static_bundle_materialization_failed');
    for (const turn of result.turns) {
      const raw = JSON.stringify(turn) + '\n';
      await handle.writeFile(raw); hash.update(raw); turns++;
    }
  } } catch (error) { await handle.close(); await unlink(staged); throw error; }
  await handle.close();
  await rename(staged, output);
  await writeFile(`${output}.manifest.json`, JSON.stringify({ version: 'natlang.static_bundle_turns/1',
    source_manifest: resolve(manifestPath), source_manifest_sha256: digest(await readFile(manifestPath, 'utf8')),
    sha256: hash.digest('hex'), turns, model_calls: 0 }, null, 2) + '\n');
  return output;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const [manifestPath, ...args] = process.argv.slice(2);
  if (!manifestPath) throw new Error('usage: static-bundle-input.mjs MANIFEST [--optional]');
  if (args.includes('--turns-out')) {
    const output = args[args.indexOf('--turns-out') + 1];
    if (!output) throw new Error('--turns-out requires a path');
    console.log(await materializeStaticBundle(manifestPath, output));
  } else {
  const path = await staticBundleInput(manifestPath, { optional: args.includes('--optional') });
  if (path) console.log(path);
  }
}
