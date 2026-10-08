#!/usr/bin/env node
/** Materialize exact reviewed provider actions as explicitly held native preview rows. */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { materializeNativeRows } from '../ts-host/dist/teacher/native-materializer.js';

const [selectionPath, receiptsPath, outPath] = process.argv.slice(2);
if (!selectionPath || !receiptsPath || !outPath) {
  throw new Error('usage: node scripts/materialize_held_neuralese_action_preview.mjs SELECTED_ACTIONS.jsonl RAW_RECEIPTS.jsonl OUT.jsonl');
}
const sha = value => createHash('sha256').update(value).digest('hex');
const readLines = path => readFileSync(path, 'utf8').split(/\r?\n/).filter(Boolean);
const selectedLines = readLines(selectionPath), receiptLines = readLines(receiptsPath);
if (selectedLines.length !== receiptLines.length) throw new Error('selected action/raw receipt count mismatch');
const selections = selectedLines.map(JSON.parse), receipts = receiptLines.map(JSON.parse);
const rawByPath = new Map(), materializedByPath = new Map(), output = [], evidence = [];
for (let index = 0; index < selections.length; index++) {
  const selection = selections[index], receipt = receipts[index], review = selection.source_review_record;
  if (selection.proposal_disposition !== 'held_for_root_review') throw new Error(`row ${index}: not held`);
  if (receipt.candidate_key !== selection.candidate_key || review.invocation_id !== receipt.invocation_id)
    throw new Error(`row ${index}: candidate identity mismatch`);
  let raw = rawByPath.get(receipt.raw_result_path);
  if (!raw) {
    const bytes = readFileSync(receipt.raw_result_path);
    if (sha(bytes) !== receipt.raw_result_file_sha256) throw new Error(`raw file hash mismatch: ${receipt.raw_result_path}`);
    const lines = bytes.toString('utf8').split(/\r?\n/).filter(Boolean);
    const matching = lines.filter(line => sha(Buffer.from(line)) === receipt.raw_result_row_sha256);
    if (matching.length !== 1) throw new Error(`raw row identity mismatch: ${receipt.raw_result_path}`);
    raw = JSON.parse(matching[0]);
    rawByPath.set(receipt.raw_result_path, raw);
    materializedByPath.set(receipt.raw_result_path, materializeNativeRows([raw], { failedRuns: true }).turns);
  }
  const roles = receipt.provider_action_roles || {};
  const generation = roles.target_generation_turn || roles.selected_output_action;
  if (!generation || !Number.isSafeInteger(generation.trajectory_index)
      || typeof generation.request_sha256 !== 'string' || typeof generation.raw_response_sha256 !== 'string')
    throw new Error(`row ${index}: missing exact provider target-generation identity`);
  const sourceTurn = raw.trajectory[generation.trajectory_index];
  if (!sourceTurn || sourceTurn.invocation_id !== review.invocation_id
      || sourceTurn.request_sha256 !== generation.request_sha256
      || sourceTurn.raw_response_sha256 !== generation.raw_response_sha256)
    throw new Error(`row ${index}: target-generation identity differs from raw trajectory`);
  const matches = materializedByPath.get(receipt.raw_result_path).filter(turn =>
    turn.source_ref?.invocation_id === review.invocation_id
    && turn.decision?.source_raw_response_sha256 === generation.raw_response_sha256);
  if (matches.length !== 1) throw new Error(`row ${index}: expected one native action, found ${matches.length}`);
  const row = structuredClone(matches[0]);
  row.id = `held-neuralese-action:${selection.candidate_key}`;
  row.split = review.split;
  row.source_groups = [review.source_group];
  row.source_ids = [review.source_id];
  row.review_disposition = 'held_for_root_review';
  row.training_admission = { approved: false, status: 'held-review-only', reason: 'source/action candidate; root admission pending' };
  row.decision = { ...row.decision, training_approved: false, failed_action: row.decision?.failed_action ?? false };
  row.preview_source_selection = {
    candidate_key: selection.candidate_key,
    source_review_sha256: sha(Buffer.from(JSON.stringify(review))),
    source_result_path: receipt.raw_result_path,
    source_result_file_sha256: receipt.raw_result_file_sha256,
    source_result_row_sha256: receipt.raw_result_row_sha256,
    target_generation_turn: generation,
    selected_output_action: roles.selected_output_action || null,
    role_class: roles.role_class || null,
    v3_receipt_sha256: sha(Buffer.from(JSON.stringify(receipt))),
  };
  output.push(row);
  evidence.push({ id: row.id, candidate_key: selection.candidate_key, split: row.split,
    source_group: review.source_group, invocation_id: review.invocation_id,
    generation_turn: generation, native_decision_index: row.decision.index,
    native_target_sha256: sha(Buffer.from(JSON.stringify(row.target))),
    review_disposition: row.review_disposition, training_approved: row.training_admission.approved });
}
writeFileSync(outPath, output.map(row => JSON.stringify(row)).join('\n') + '\n', { flag: 'wx' });
console.log(JSON.stringify({ rows: output.length, source_files: rawByPath.size,
  train: output.filter(row => row.split === 'train').length,
  test: output.filter(row => row.split === 'test').length,
  held_only: output.every(row => row.training_admission.approved === false), evidence }, null, 2));
