#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NATIVE_SYNTHETIC_FAMILIES, NATIVE_SYNTHETIC_GENERATOR_VERSION, syntheticRecords } from '../dist/teacher/synthetic-generator.js';

const hash = value => createHash('sha256').update(value).digest('hex');
function parse(argv) {
  const options = new Map();
  for (let index = 0; index < argv.length; index++) {
    const argument = argv[index];
    if (argument === '--help' || argument === '-h') { options.set('--help', 'true'); continue; }
    if (!argument?.startsWith('--')) throw new Error(`unexpected argument: ${argument}`);
    const equals = argument.indexOf('=');
    if (equals >= 0) options.set(argument.slice(0, equals), argument.slice(equals + 1));
    else if (argv[index + 1] && !argv[index + 1].startsWith('--')) options.set(argument, argv[++index]);
    else options.set(argument, 'true');
  }
  return options;
}
const integer = (options, name, fallback) => {
  const value = Number(options.get(name) ?? fallback);
  if (!Number.isSafeInteger(value)) throw new Error(`${name} must be a safe integer`);
  return value;
};
async function main() {
  const options = parse(process.argv.slice(2));
  if (options.has('--help')) {
    process.stdout.write('usage: node ts-host/scripts/generate-synthetic-ir.mjs --out FILE [--seed N] [--start-index N] [--n N] [--families FAMILY... ]\n\n' +
      `Native families: ${NATIVE_SYNTHETIC_FAMILIES.join(', ')}\n` +
      'The generator emits natlang.program/2 JSONL and an adjacent provenance manifest. Run npm --prefix ts-host run build:node first.\n');
    return;
  }
  const outputOption = options.get('--out');
  if (!outputOption) throw new Error('--out is required');
  const output = resolve(outputOption), seed = integer(options, '--seed', 0);
  const start = integer(options, '--start-index', 0), count = integer(options, '--n', 100);
  if (start < 0 || count < 0) throw new Error('--start-index and --n must be nonnegative');
  const families = options.has('--families') ? options.get('--families').split(',') : [...NATIVE_SYNTHETIC_FAMILIES];
  for (const family of families) if (!NATIVE_SYNTHETIC_FAMILIES.includes(family)) throw new Error(`unsupported family ${family}; choose ${NATIVE_SYNTHETIC_FAMILIES.join(', ')}`);
  const records = syntheticRecords(seed, start, count, families);
  const sourcePath = resolve(dirname(fileURLToPath(import.meta.url)), '../src/teacher/synthetic-generator.ts');
  const generatorSha256 = hash(await readFile(sourcePath));
  const lines = records.map(record => JSON.stringify(record)).join('\n') + (records.length ? '\n' : '');
  const bytes = Buffer.from(lines);
  const staging = `${output}.building`;
  await writeFile(staging, bytes);
  await rename(staging, output);
  const manifest = { version: 'natlang.synthetic_dataset.native/1', generator: NATIVE_SYNTHETIC_GENERATOR_VERSION,
    generator_sha256: generatorSha256, seed, start_index: start, programs: count, families,
    output_sha256: hash(bytes), ir_version: 'natlang.program/2' };
  await writeFile(`${output}.manifest.json.building`, `${JSON.stringify(manifest, null, 2)}\n`);
  await rename(`${output}.manifest.json.building`, `${output}.manifest.json`);
  process.stdout.write(`${count} programs -> ${output}\n`);
}
main().catch(error => { process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`); process.exitCode = 1; });
