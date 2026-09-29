#!/usr/bin/env node
import { randomUUID } from 'node:crypto';
import { link, mkdir, open, rename, unlink } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { jsonlRows } from './jsonl-stream.mjs';

const args = process.argv.slice(2), positional = args.filter(value => !value.startsWith('--'));
const [inputPath, outputPath] = positional;
if (!inputPath || !outputPath || positional.length !== 2 || args.some(value => value.startsWith('--') && !['--replace', '--direct-answers'].includes(value))) {
  console.error('usage: node scripts/materialize-native-teacher.mjs INPUT.jsonl OUTPUT.jsonl [--replace] [--direct-answers]');
  process.exit(2);
}
const input = resolve(inputPath), output = resolve(outputPath);
// --direct-answers: train answers given without reasoning towards them, for a student that answers directly.
await mkdir(dirname(output), { recursive: true });
const staged = `${output}.building-${process.pid}-${randomUUID()}`;
// Line by line: every turn carries its whole context, and thousands of them do not fit in one string.
const handle = await open(staged, 'wx');
let accepted = 0, rejected = 0, turns = 0;
try { for await (const row of jsonlRows(input)) {
  const result = materializeNativeRows([row], { directAnswers: args.includes('--direct-answers') });
  accepted += result.acceptedRows; rejected += result.rejectedRows;
  for (const missed of result.unlinked)
    console.error(`  ${missed.id}: not used, ${missed.outcomes} action outcomes could not be linked to their decisions`);
  for (const turn of result.turns) { await handle.writeFile(JSON.stringify(turn) + '\n'); turns++; }
} } catch (error) { await handle.close(); await unlink(staged); throw error; }
await handle.close();
console.error(`${accepted} rows -> ${turns} turns (${rejected} rows not used)`);
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
console.log(JSON.stringify({ output, accepted_rows: accepted,
  rejected_rows: rejected, training_decisions: turns }));
