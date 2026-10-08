#!/usr/bin/env node
import { randomUUID } from 'node:crypto';
import { link, mkdir, open, rename, unlink } from 'node:fs/promises';
import { readFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { dirname, resolve } from 'node:path';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { jsonlRows } from './jsonl-stream.mjs';

const {values, positionals: positional} = parseArgs({allowPositionals:true, options:{replace:{type:'boolean'}, 'direct-answers':{type:'boolean'}, 'decision-review':{type:'string'}}});
const [inputPath, outputPath] = positional;
if (!inputPath || !outputPath || positional.length !== 2) {
  console.error('usage: node scripts/materialize-native-teacher.mjs INPUT.jsonl OUTPUT.jsonl [--replace] [--direct-answers] [--decision-review FILE]');
  process.exit(2);
}
const input = resolve(inputPath), output = resolve(outputPath);
const review = values['decision-review'] ? JSON.parse(await readFile(values['decision-review'], 'utf8')) : null;
if (review && (review.schema !== 'natlang.native-decision-review/1' || !Array.isArray(review.holds)))
  throw new Error('invalid native decision review schema');
const seen = new Set();
// --direct-answers: train answers given without reasoning towards them, for a student that answers directly.
await mkdir(dirname(output), { recursive: true });
const staged = `${output}.building-${process.pid}-${randomUUID()}`;
const authoredOutput = `${output}.authored-actions.jsonl`;
const stagedAuthored = `${authoredOutput}.building-${process.pid}-${randomUUID()}`;
const handle = await open(staged, 'wx');
let authoredHandle, accepted = 0, rejected = 0, turns = 0, authoredActions = 0;
try { for await (const row of jsonlRows(input)) {
  seen.add(row.id);
  const result = materializeNativeRows([row], { directAnswers: values['direct-answers'], decisionHolds: review?.holds });
  accepted += result.acceptedRows; rejected += result.rejectedRows;
  for (const missed of result.unlinked)
    console.error(`  ${missed.id}: not used, ${missed.outcomes} action outcomes could not be linked to their decisions`);
  for (const turn of result.turns) { await handle.writeFile(JSON.stringify(turn) + '\n'); turns++; }
  if (result.authored_actions.length) {
    authoredHandle ??= await open(stagedAuthored, 'wx');
    for (const action of result.authored_actions) {
      await authoredHandle.writeFile(JSON.stringify(action) + '\n'); authoredActions++;
    }
  }
} } catch (error) {
  await handle.close(); await authoredHandle?.close(); await unlink(staged);
  if (authoredHandle) await unlink(stagedAuthored);
  throw error;
}
if (review?.holds.some(hold => !seen.has(hold.trajectory_id))) {
  await handle.close(); await authoredHandle?.close(); await unlink(staged);
  if (authoredHandle) await unlink(stagedAuthored);
  throw new Error('semantic review references absent input trajectory');
}
await handle.close(); await authoredHandle?.close();
console.error(`${accepted} rows -> ${turns} turns (${rejected} rows not used); ${authoredActions} held authored actions`);
if (values.replace) {
  await rename(staged, output);
  if (authoredHandle) await rename(stagedAuthored, authoredOutput);
  else await unlink(authoredOutput).catch(error => { if (error?.code !== 'ENOENT') throw error; });
} else {
  let linkedOutput = false;
  try {
    await link(staged, output); linkedOutput = true;
    if (authoredHandle) await link(stagedAuthored, authoredOutput);
  } catch (error) {
    await unlink(staged);
    if (linkedOutput) await unlink(output);
    if (authoredHandle) await unlink(stagedAuthored);
    if (error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST')
      throw new Error(`refusing to overwrite ${output}`);
    throw error;
  }
  await unlink(staged);
  if (authoredHandle) await unlink(stagedAuthored);
}
console.log(JSON.stringify({ output, ...(authoredActions ? { authored_actions_output: authoredOutput } : {}),
  accepted_rows: accepted, rejected_rows: rejected, training_decisions: turns, authored_actions_held: authoredActions }));
