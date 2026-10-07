import assert from 'node:assert/strict';
import { gunzipSync } from 'node:zlib';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import test from 'node:test';
import { validateSoftStateConversionEvidence } from '../scripts/inline-curriculum/soft-state-conversion-evidence.mjs';
import { nativeRowDigest } from '../dist/teacher/native-materializer.js';

const fixture = JSON.parse(readFileSync(new URL('./fixtures/neuralese-v83-soft-edges-minimal.json', import.meta.url), 'utf8'));
const actualResultBytes = gunzipSync(readFileSync(new URL('./fixtures/neuralese-v83-actual-result.jsonl.gz', import.meta.url)));
const actualResult = JSON.parse(actualResultBytes.toString('utf8'));
const actualActions = fixture.actions.map(item => item.record);
const sha = value => createHash('sha256').update(value).digest('hex');

function withEvidence(t, mutate = () => {}) {
  const dir = mkdtempSync(path.join(tmpdir(), 'natlang-soft-edge-evidence-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const result = structuredClone(actualResult);
  const review = structuredClone(fixture.evidence_review);
  mutate(result, review);
  const resultBytes = Buffer.from(JSON.stringify(result) + '\n');
  review.result.sha256 = sha(resultBytes);
  const resultPath = path.join(dir, 'result.jsonl');
  const reviewPath = path.join(dir, 'review.json');
  writeFileSync(resultPath, resultBytes);
  writeFileSync(reviewPath, JSON.stringify(review));
  const actionRows = structuredClone(actualActions);
  for (const row of actionRows) row.source_ref.source_row_sha256 = nativeRowDigest(result);
  return { resultPath, reviewPath, actionRows };
}

test('conversion evidence reruns the shared graph validator and binds all five materialized actions', t => {
  const evidence = validateSoftStateConversionEvidence(withEvidence(t));
  assert.equal(evidence.edges.length, 5);
  assert.equal(evidence.validation.revalidated_edges, 5);
  assert.equal(evidence.edges[4].writer_node, 'task-1-gbrjm9/35#46');
  assert.equal(evidence.source.learned_vectors, false);
  assert.equal(evidence.source.qualification_certificate, false);
  assert.equal(evidence.source.training_admission, false);
});

test('conversion evidence rejects a changed review edge even when its result hash is fresh', t => {
  const args = withEvidence(t, (_result, review) => { review.actual_graph.edges[0].reader_node = 'forged-node'; });
  assert.throws(() => validateSoftStateConversionEvidence(args), /shared graph validation does not reproduce reviewed edge/);
});

test('conversion evidence rejects missing provider-visible body expansion', t => {
  const args = withEvidence(t, result => {
    const edge = fixture.evidence_review.actual_graph.edges[0];
    for (const turn of result.trajectory.filter(item => item.invocation_id === edge.reader_call))
      turn.model_response.transport_provenance.expanded_input_blocks = [];
  });
  assert.throws(() => validateSoftStateConversionEvidence(args), /no provider-visible expansion/);
});

test('conversion evidence rejects a stale result hash in the review', t => {
  const args = withEvidence(t);
  const review = JSON.parse(readFileSync(args.reviewPath, 'utf8'));
  review.result.sha256 = '0'.repeat(64);
  writeFileSync(args.reviewPath, JSON.stringify(review));
  assert.throws(() => validateSoftStateConversionEvidence(args), /does not bind this accepted completed result/);
});

test('conversion evidence rejects an action row whose source program partition was projected from a different source', t => {
  const args = withEvidence(t);
  args.actionRows[0].task.program_ir.source_groups = ['forged-source-group'];
  assert.throws(() => validateSoftStateConversionEvidence(args), /source partition/);
});

test('conversion evidence rejects a changed producer action target even when trajectory hash is valid', t => {
  const args = withEvidence(t);
  const writerCall = fixture.evidence_review.actual_graph.edges[0].writer_call;
  const row = args.actionRows.find(item => item.source_ref.invocation_id === writerCall && item.target.tool_calls?.some(call => call.function.name === 'return_result'));
  assert.ok(row);
  row.target.tool_calls[0].function.arguments = JSON.stringify({status:'success', value:'<|neuralese|>forged<|/neuralese|>'});
  assert.throws(() => validateSoftStateConversionEvidence(args), /target differs from exact materializer output/);
});
