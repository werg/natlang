#!/usr/bin/env node
import { randomUUID } from 'node:crypto';
import { link, mkdir, readFile, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';

const args = process.argv.slice(2), positional = args.filter(value => !value.startsWith('--'));
const [inputPath, outputPath] = positional;
if (!inputPath || !outputPath || positional.length !== 2 || args.some(value => value.startsWith('--') && value !== '--replace')) {
  console.error('usage: node scripts/materialize-native-teacher.mjs INPUT.jsonl OUTPUT.jsonl [--replace]');
  process.exit(2);
}
const input = resolve(inputPath), output = resolve(outputPath);
const rows = (await readFile(input, 'utf8')).split(/\r?\n/).filter(line => line.trim()).map(line => JSON.parse(line));
const result = materializeNativeRows(rows);
const data = result.turns.map(turn => JSON.stringify(turn)).join('\n') + (result.turns.length ? '\n' : '');
await mkdir(dirname(output), { recursive: true });
const staged = `${output}.building-${process.pid}-${randomUUID()}`;
await writeFile(staged, data, { flag: 'wx' });
if (args.includes('--replace')) await rename(staged, output);
else {
  try { await link(staged, output); }
  catch (error) {
    await unlink(staged);
    if (error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST')
      throw new Error(`refusing to overwrite ${output}`);
    throw error;
  }
  await unlink(staged);
}
console.log(JSON.stringify({ output, accepted_rows: result.acceptedRows,
  rejected_rows: result.rejectedRows, training_decisions: result.turns.length }));
