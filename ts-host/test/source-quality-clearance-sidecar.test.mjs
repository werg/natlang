import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { canonical } from '../dist/native/canonical-data.js';
import { hexDigest } from '../dist/native/hash.js';
import { sourceQualityClearanceProblems, sourceQualityClearanceRefForProgram,
  sourceQualityClearanceRefProblems, verifySourceQualityClearanceSidecar } from '../dist/teacher/source-quality-clearance.js';
import { sourceConversionProblems } from '../dist/teacher/source-conversion.js';
import { openPacketSourceQualityClearanceResolver, resolvePacketSourceQualityClearance } from '../../scripts/lib/source-quality-clearance-sidecar.mjs';

const sha = value => hexDigest(typeof value === 'string' ? value : canonical(value));
const sourceRow = { id: 'source-1', row_index: 0, row_sha256: '1'.repeat(64), group: 'group-1', split: 'train' };
const quality = { version: 'natlang.source_quality/1', status: 'held', checks: ['source_annotation_reference_only'] };
const program = { version: 'natlang.program/2', id: 'program-1', split: 'train', source_ids: ['source-1'],
  source_groups: ['group-1'], external_source: { original_split: 'train', source_path: 'sources.jsonl',
    snapshot_sha256: '2'.repeat(64), source_manifest_sha256: '3'.repeat(64), source_rows: [sourceRow], quality } };

function verifiedClearance() {
  const evidenceFiles = new Map([
    [sha('quality review fixture'), new TextEncoder().encode('quality review fixture')],
    [sha('closure fixture'), new TextEncoder().encode('closure fixture')],
  ]);
  const review = { schema: 'natlang.source-quality-clearance/1', id: 'clearance-1', scope: 'source-conversion-only',
    decision: 'clear-for-conversion', training_admission: false, trace_admission: false,
    evidence: [{ role: 'source_quality_review', path: 'source-quality-review.json', sha256: sha('quality review fixture') },
      { role: 'source_closure', path: 'source-closure.json', sha256: sha('closure fixture') }],
    source_policy: { path: 'decision_source_quality_holds.json', sha256: '6'.repeat(64), held_source_ids: [], held_source_groups: [] },
    programs: [{ program_id: program.id, program_ir_sha256: sha(program), source_quality_sha256: sha(quality),
      source_path: 'sources.jsonl', source_snapshot_sha256: program.external_source.snapshot_sha256,
      source_manifest_path: 'source-manifest.json', source_manifest_sha256: program.external_source.source_manifest_sha256,
      source_ids: ['source-1'], source_groups: ['group-1'], source_rows: [sourceRow] }], root_approval: {
      schema: 'natlang.source-quality-clearance-approval/1', decision: 'approved', authority: 'root',
      scope: 'source-conversion-only', training_admission: false, trace_admission: false, payload_sha256: '' } };
  const { root_approval: _approval, ...payload } = review;
  review.root_approval.payload_sha256 = sha(payload);
  const bytes = JSON.stringify(review);
  return { bytes: new TextEncoder().encode(bytes), evidenceFiles, clearance: { schema: 'natlang.verified-source-quality-clearance/1',
    review_sha256: sha(bytes), review_canonical_sha256: sha(review), evidence_files_verified: true, review } };
}

test('compact clearance ref resolves only against its exact verified sidecar and program', () => {
  const { bytes, evidenceFiles, clearance } = verifiedClearance();
  assert.deepEqual(sourceQualityClearanceProblems(clearance), []);
  const resolved = verifySourceQualityClearanceSidecar(bytes, clearance.review_sha256, evidenceFiles);
  assert.equal(resolved.review_canonical_sha256, clearance.review_canonical_sha256);
  assert.throws(() => verifySourceQualityClearanceSidecar(bytes, '0'.repeat(64), evidenceFiles),
    /sidecar_file_hash_mismatch/);
  assert.throws(() => verifySourceQualityClearanceSidecar(bytes, clearance.review_sha256, new Map()),
    /evidence_hash_mismatch/);
  const ref = sourceQualityClearanceRefForProgram(clearance, program, 'source-quality-clearance.json', clearance.review_sha256);
  assert.ok(ref);
  assert.equal(JSON.stringify(ref).length < 1000, true);
  assert.deepEqual(sourceQualityClearanceRefProblems(ref, clearance, program), []);
  const row = { task: { program_ir: program }, provenance: { collection_role: 'teacher' },
    source_ref: { source_quality_clearance: ref } };
  assert.deepEqual(sourceConversionProblems(row), [
    'source_quality_held', 'source_quality_clearance_sidecar_unresolved',
  ]);
  assert.deepEqual(sourceConversionProblems(row, clearance), []);
  assert.deepEqual(sourceQualityClearanceRefProblems(ref, undefined, program), ['source_quality_clearance_sidecar_unresolved']);
  assert.deepEqual(sourceQualityClearanceRefProblems({ ...ref, file_sha256: '0'.repeat(64) }, clearance, program),
    ['source_quality_clearance_reference_mismatch']);
  assert.deepEqual(sourceQualityClearanceRefProblems(ref, clearance, { ...program, id: 'other-program' }),
    ['source_quality_clearance_reference_mismatch']);
  assert.deepEqual(sourceQualityClearanceRefProblems({ ...ref, sidecar_path: '../clearance.json' }, clearance, program),
    ['source_quality_clearance_reference_invalid']);
});

test('packet resolver verifies relative sidecar and evidence paths before using a row reference', async () => {
  const { bytes, evidenceFiles, clearance } = verifiedClearance();
  const ref = sourceQualityClearanceRefForProgram(clearance, program, 'source-quality-clearance.json', clearance.review_sha256);
  const root = await mkdtemp(join(tmpdir(), 'natlang-clearance-sidecar-'));
  try {
    await mkdir(root, { recursive: true });
    await writeFile(join(root, 'source-quality-clearance.json'), bytes);
    const evidence = [];
    const names = ['source-quality-review.json', 'source-closure.json'];
    for (const [index, [digest, evidenceBytes]] of [...evidenceFiles.entries()].entries()) {
      await writeFile(join(root, names[index]), evidenceBytes);
      evidence.push({ role: index === 0 ? 'source_quality_review' : 'source_closure', path: names[index], sha256: digest });
    }
    const manifestBytes = JSON.stringify({ source_quality_clearance: {
      path: ref.sidecar_path, file_sha256: ref.file_sha256, review_canonical_sha256: ref.review_canonical_sha256,
      evidence } });
    await writeFile(join(root, 'materialized-manifest.json'), manifestBytes);
    const packet = await openPacketSourceQualityClearanceResolver(root, sha(manifestBytes));
    assert.equal(packet.resolve(ref, program).review.id, clearance.review.id);
    assert.throws(() => packet.resolve({ ...ref, sidecar_path: '../outside.json' }, program), /manifest-mismatch/);
    await assert.rejects(openPacketSourceQualityClearanceResolver(root, '0'.repeat(64)), /manifest-file-hash-mismatch/);
    const resolved = await resolvePacketSourceQualityClearance(root, ref, program);
    assert.equal(resolved.review.id, clearance.review.id);
    await assert.rejects(resolvePacketSourceQualityClearance(root, { ...ref, sidecar_path: '../outside.json' }, program),
      /manifest-mismatch/);
    await writeFile(join(root, 'source-quality-clearance.json'), 'tampered');
    await assert.rejects(resolvePacketSourceQualityClearance(root, ref, program), /sidecar_file_hash_mismatch/);
  } finally { await rm(root, { recursive: true, force: true }); }
});
