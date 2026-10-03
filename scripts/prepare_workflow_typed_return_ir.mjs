#!/usr/bin/env node
/** Create an immutable, source-preserving typed-return variant of legacy workflow reducer IR. */
import { createReadStream } from 'node:fs';
import { StringDecoder } from 'node:string_decoder';
import { createHash } from 'node:crypto';
import { open, rename, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { applyReviewedWorkflowTypedReturns, WORKFLOW_TYPED_RETURN_VERSION } from '../ts-host/scripts/inline-curriculum/workflow-typed-return-reviewed.mjs';

const [inputArg, outputArg] = process.argv.slice(2);
if (!inputArg || !outputArg) throw Error('usage: prepare_workflow_typed_return_ir.mjs INPUT.ir.jsonl NEW_OUTPUT.ir.jsonl');
const input = resolve(inputArg), output = resolve(outputArg);
if (input === output) throw Error('preserve the original IR and write a new versioned output');
const receiptPath = `${output}.receipt.json`, temp = `${output}.preparing`;
const outputHandle = await open(output, 'wx'); await outputHandle.close();
let receiptHandle;
try { receiptHandle = await open(receiptPath, 'wx'); }
catch (error) { await unlink(output).catch(() => {}); throw error; }
let handle, tempOwned = false;
const inputHash = createHash('sha256'), outputHash = createHash('sha256');
const changed = [], counts = { cases: 0, transformed: 0, boolean_fields: 0, choice_fields: 0, score_fields: 0 };
async function* exactLines(path) {
  const decoder = new StringDecoder('utf8'); let pending = '';
  for await (const chunk of createReadStream(path)) {
    inputHash.update(chunk); pending += decoder.write(chunk);
    let newline;
    while ((newline = pending.indexOf('\n')) >= 0) {
      yield pending.slice(0, newline).replace(/\r$/, ''); pending = pending.slice(newline + 1);
    }
  }
  pending += decoder.end();
  if (pending.length) yield pending.replace(/\r$/, '');
}
try {
  handle = await open(temp, 'wx'); tempOwned = true;
  for await (const line of exactLines(input)) {
    if (!line.trim()) continue;
    const base = JSON.parse(line), next = applyReviewedWorkflowTypedReturns(base);
    counts.cases++;
    if (next !== base) {
      const ir = next?.task?.kind === 'whole_program' ? next.task.program_ir : next;
      const info = ir.generation.workflow_typed_return_review;
      counts.transformed++;
      for (const field of info.field_types_from_visible_question_schemas) {
        if (field.questionType === 'noul') counts.boolean_fields++;
        else if (field.questionType === 'choice') counts.choice_fields++;
        else if (field.questionType === 'score') counts.score_fields++;
      }
      changed.push({ base_id: base.id, variant_id: next.id, base_ir_sha256: info.base_ir_sha256,
        variant_ir_sha256: createHash('sha256').update(JSON.stringify(next)).digest('hex'),
        field_types: info.field_types_from_visible_question_schemas });
    }
    const raw = `${JSON.stringify(next)}\n`;
    outputHash.update(raw); await handle.writeFile(raw);
  }
  await handle.sync(); await handle.close(); handle = undefined; await rename(temp, output);
  const receipt = { version: 'natlang.workflow_typed_return_ir_preparation/1', adapter_version: WORKFLOW_TYPED_RETURN_VERSION,
    created_at: new Date().toISOString(), input, input_bytes_sha256: inputHash.digest('hex'), output, output_sha256: outputHash.digest('hex'),
    ...counts, changes: changed, golds_inputs_visible_files_sources_groups_licenses_references_preserved: true,
    model_calls: 0, ready_for_generation: false, next_step: 'Native reference replay, current admission, source conversion and materialization review required.' };
  await receiptHandle.writeFile(`${JSON.stringify(receipt, null, 2)}\n`); await receiptHandle.sync(); await receiptHandle.close(); receiptHandle = undefined;
  process.stdout.write(`${JSON.stringify(receipt)}\n`);
} catch (error) {
  if (handle) await handle.close(); if (receiptHandle) await receiptHandle.close();
  if (tempOwned) await unlink(temp).catch(() => {}); await unlink(output).catch(() => {}); await unlink(receiptPath).catch(() => {}); throw error;
}
