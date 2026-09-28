#!/usr/bin/env node
/** Resolve a ready static bundle for build_lora_sft.sh, failing closed on stale/tampered data. */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { digest, safePath } from './directory-sources.mjs';
import { admitRow } from '../../dist/teacher/curriculum.js';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';

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
  const [results, ir] = await Promise.all([readFile(input, 'utf8'), readFile(irPath, 'utf8')]);
  if (digest(results) !== manifest.results.sha256 || digest(ir) !== manifest.ir.sha256)
    throw new Error('static_bundle_checksum_mismatch');
  const records = ir.split('\n').filter(Boolean).map(JSON.parse);
  const rows = results.split('\n').filter(Boolean).map(JSON.parse);
  if (rows.length !== manifest.cases || records.length !== manifest.cases) throw new Error('static_bundle_row_count_mismatch');
  for (const [index, row] of rows.entries()) {
    if (digest(records[index]) !== digest(row.task?.program_ir)) throw new Error('static_bundle_ir_result_mismatch');
    if (!row.provenance?.source_conversion || !admitRow(row).admitted) throw new Error('static_bundle_admission_failed');
  }
  return input;
}

export async function materializeStaticBundle(manifestPath, output) {
  const input = await staticBundleInput(manifestPath);
  const rows = (await readFile(input, 'utf8')).split('\n').filter(Boolean).map(JSON.parse);
  const result = materializeNativeRows(rows);
  if (result.rejectedRows || result.unlinked.length) throw new Error('static_bundle_materialization_failed');
  await mkdir(dirname(resolve(output)), { recursive: true });
  const raw = result.turns.map(turn => JSON.stringify(turn)).join('\n') + '\n';
  await writeFile(output, raw);
  await writeFile(`${output}.manifest.json`, JSON.stringify({ version: 'natlang.static_bundle_turns/1',
    source_manifest: resolve(manifestPath), source_manifest_sha256: digest(await readFile(manifestPath, 'utf8')),
    sha256: digest(raw), turns: result.turns.length, model_calls: 0 }, null, 2) + '\n');
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
