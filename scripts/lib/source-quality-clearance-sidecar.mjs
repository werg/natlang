import { readFile } from 'node:fs/promises';
import { isAbsolute, resolve, relative } from 'node:path';
import { nativeDecisionTargetDigest } from '../../ts-host/dist/native/decision-review.js';
import { sourceQualityClearanceRefProblems, verifySourceQualityClearanceSidecar } from '../../ts-host/dist/teacher/source-quality-clearance.js';
import { hexDigest } from '../../ts-host/dist/native/hash.js';

function packetPath(root, value) {
  if (typeof value !== 'string' || !value || isAbsolute(value) || value.includes('\\'))
    throw new Error('source-quality-clearance-sidecar-path-invalid');
  const path = resolve(root, value), rel = relative(root, path);
  if (!rel || rel === '..' || rel.startsWith(`..${process.platform === 'win32' ? '\\' : '/'}`))
    throw new Error('source-quality-clearance-sidecar-path-escapes-packet');
  return path;
}

/** Open a packet once, verifying the manifest and signed clearance/evidence bytes before row-level binding checks. */
export async function openPacketSourceQualityClearanceResolver(packetRootValue, expectedManifestSha256) {
  const root = resolve(packetRootValue);
  const manifestPath = packetPath(root, 'materialized-manifest.json');
  const manifestBytes = await readFile(manifestPath);
  const manifestSha256 = hexDigest(manifestBytes);
  if (expectedManifestSha256 && manifestSha256 !== expectedManifestSha256)
    throw new Error('source-quality-clearance-packet-manifest-file-hash-mismatch');
  const manifest = JSON.parse(manifestBytes.toString('utf8'));
  const sidecar = manifest.source_quality_clearance;
  if (!sidecar || !Array.isArray(sidecar.evidence))
    throw new Error('source-quality-clearance-packet-manifest-mismatch');
  const bytes = await readFile(packetPath(root, sidecar.path));
  const evidence = new Map();
  for (const entry of sidecar.evidence) {
    if (!entry || typeof entry.sha256 !== 'string' || evidence.has(entry.sha256))
      throw new Error('source-quality-clearance-packet-evidence-invalid');
    evidence.set(entry.sha256, await readFile(packetPath(root, entry.path)));
  }
  const clearance = verifySourceQualityClearanceSidecar(bytes, sidecar.file_sha256, evidence);
  if (clearance.review_canonical_sha256 !== sidecar.review_canonical_sha256)
    throw new Error('source-quality-clearance-packet-canonical-hash-mismatch');
  return {
    root, manifest, manifest_sha256: manifestSha256, clearance,
    resolve(reference, program) {
      if (sidecar.path !== reference?.sidecar_path || sidecar.file_sha256 !== reference?.file_sha256 ||
          sidecar.review_canonical_sha256 !== reference?.review_canonical_sha256)
        throw new Error('source-quality-clearance-packet-manifest-mismatch');
      const problems = sourceQualityClearanceRefProblems(reference, clearance, program);
      if (problems.length) throw new Error(problems.join(','));
      return clearance;
    },
  };
}

/** Resolve one row against a packet; multirow consumers should open the packet resolver once. */
export async function resolvePacketSourceQualityClearance(packetRootValue, reference, program, expectedManifestSha256) {
  const packet = await openPacketSourceQualityClearanceResolver(packetRootValue, expectedManifestSha256);
  return packet.resolve(reference, program);
}

async function packetFile(root, entry, label) {
  if (!entry || typeof entry.path !== 'string' || typeof entry.sha256 !== 'string')
    throw new Error(`source-quality-clearance-packet-${label}-reference-invalid`);
  const bytes = await readFile(packetPath(root, entry.path));
  if (hexDigest(bytes) !== entry.sha256)
    throw new Error(`source-quality-clearance-packet-${label}-hash-mismatch`);
  return bytes;
}

/** Validate a complete candidate packet once; used by renderers before owner-approved promotion. */
export async function verifyCandidateSelectionPacket(packetRootValue, expectedManifestSha256) {
  const packet = await openPacketSourceQualityClearanceResolver(packetRootValue, expectedManifestSha256);
  const { root, manifest } = packet;
  if (manifest.schema !== 'natlang.provider-child-action-packet/2' || manifest.candidate_selection !== true ||
      manifest.training_admission !== false || manifest.trace_admission !== false)
    throw new Error('candidate-selection-packet-stage-or-admission-invalid');
  const candidateReviewRef = manifest.candidate_selection_review;
  const semanticReviewRef = manifest.semantic_review;
  const candidateReviewBytes = await packetFile(root, candidateReviewRef, 'candidate-review');
  const semanticReviewBytes = await packetFile(root, semanticReviewRef, 'semantic-review');
  const candidateReview = JSON.parse(candidateReviewBytes.toString('utf8'));
  const semanticReview = JSON.parse(semanticReviewBytes.toString('utf8'));
  const historicalAuditRef = manifest.semantic_review_input_audit;
  const historicalAuditBytes = await packetFile(root, historicalAuditRef, 'historical-semantic-review-audit');
  const historicalAudit = JSON.parse(historicalAuditBytes.toString('utf8'));
  const correctionRef = manifest.selection_review_pin_correction;
  const correctionBytes = await packetFile(root, correctionRef, 'selection-review-pin-correction');
  const correction = JSON.parse(correctionBytes.toString('utf8'));
  if (candidateReview.schema !== 'natlang.provider-child-action-review/1' ||
      candidateReview.training_admission !== false || candidateReview.trace_admission !== false ||
      semanticReview.schema !== 'natlang.provider-child-semantic-review/1' ||
      semanticReview.training_admission !== false || semanticReview.trace_admission !== false ||
      semanticReview.source_cases_sha256 !== manifest.source_cases?.sha256 ||
      historicalAudit.schema !== 'natlang.provider-child-action-candidates/1' ||
      historicalAudit.training_admission !== false ||
      candidateReview.candidate_manifest_sha256 !== historicalAuditRef.sha256 ||
      semanticReview.candidate_audit_sha256 !== historicalAuditRef.sha256 ||
      manifest.semantic_review?.candidate_audit_sha256 !== historicalAuditRef.sha256 ||
      historicalAudit.source_cases?.sha256 !== manifest.source_cases?.sha256 ||
      manifest.review_sha256 !== candidateReviewRef.sha256 ||
      candidateReview.supersedes_review_sha256 !== correction.prior_review_sha256 ||
      correction.schema !== 'natlang.provider-child-review-input-pin-correction/1' ||
      correction.corrected_review_sha256 !== candidateReviewRef.sha256 ||
      correction.corrected_candidate_manifest_sha256 !== historicalAuditRef.sha256 ||
      correction.selected_approval_count !== candidateReview.approvals?.length ||
      correction.selected_action_count !== candidateReview.selected_child_count ||
      candidateReview.approvals?.length !== candidateReview.selected_child_count ||
      !candidateReview.correction_reason?.trim() || correction.reason !== candidateReview.correction_reason ||
      correction.selected_approvals_unchanged !== true)
    throw new Error('candidate-selection-packet-review-schema-or-source-mismatch');
  const candidates = await packetFile(root, { path: manifest.candidate_manifest,
    sha256: manifest.candidate_manifest_sha256 }, 'candidate-manifest');
  const sourceCases = await packetFile(root, manifest.source_cases, 'source-cases');
  const sourceManifest = await packetFile(root, manifest.source_manifest, 'source-manifest');
  const reservations = await packetFile(root, manifest.source_split_reservations, 'split-reservations');
  const reservationSource = await packetFile(root, { path: manifest.source_split_reservations.test_source_path,
    sha256: manifest.source_split_reservations.test_source_sha256 }, 'reservation-source');
  void candidates; void sourceCases; void sourceManifest; void reservations; void reservationSource;
  const sourceRows = sourceCases.toString('utf8')
    .split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
  const sourceById = new Map(sourceRows.map(row => [row.id, row]));
  const approvals = new Map(candidateReview.approvals.filter(row => row.approved === true)
    .map(row => [row.item_id, row]));
  const semanticRows = new Map(semanticReview.rows.map(row => [row.item_id, row]));
  const historicalAuditRows = new Map(historicalAudit.candidates.map(row => [row.item_id, row]));
  const turns = (await readFile(packetPath(root, 'turns.jsonl'), 'utf8'))
    .split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
  if (!Array.isArray(manifest.turns) || manifest.turns.length !== turns.length || approvals.size !== turns.length)
    throw new Error('candidate-selection-packet-row-count-mismatch');
  const manifestRows = new Map(manifest.turns.map(row => [row.id, row]));
  const seenIds = new Set();
  const selected = [];
  for (const row of turns) {
    if (!row || typeof row.id !== 'string' || seenIds.has(row.id))
      throw new Error('candidate-selection-packet-row-id-invalid');
    seenIds.add(row.id);
    const meta = manifestRows.get(row.id);
    if (!meta || row.candidate_selection?.approved !== true || row.training_admission?.approved !== false ||
        row.trace_admission?.admitted !== false || meta.target_sha256 !== nativeDecisionTargetDigest(row.target) ||
        meta.program_id !== row.program_id || meta.trajectory_id !== row.teacher_trajectory_id ||
        meta.trajectory_sha256 !== row.teacher_trajectory_digest || meta.source_group === undefined ||
        !row.source_groups?.includes(meta.source_group) || row.split !== 'train')
      throw new Error(`candidate-selection-packet-row-binding-mismatch:${row.id}`);
    const approval = approvals.get(meta.item_id);
    const semantic = semanticRows.get(meta.item_id);
    const historical = historicalAuditRows.get(meta.item_id);
    const source = sourceById.get(meta.item_id);
    const rowApproval = meta.approval;
    if (!approval || !semantic || !historical || semantic.semantic_review !== 'clear' || !source ||
        approval.program_id !== meta.program_id || approval.item_sha256 !== meta.item_sha256 ||
        approval.group !== meta.source_group || approval.target_sha256 !== meta.target_sha256 ||
        !rowApproval || rowApproval.review_sha256 !== correction.prior_review_sha256 ||
        rowApproval.trajectory_id !== meta.trajectory_id || rowApproval.decision_index !== meta.decision_index ||
        rowApproval.target_sha256 !== meta.target_sha256 ||
        historical.program_id !== meta.program_id || historical.trajectory_id !== meta.trajectory_id ||
        historical.trajectory_sha256 !== meta.trajectory_sha256 || historical.item_sha256 !== meta.item_sha256 ||
        historical.source_group !== meta.source_group || historical.invocation_id !== meta.invocation_id ||
        historical.decision_index !== meta.decision_index || historical.target_sha256 !== meta.target_sha256 ||
        semantic.source_group !== meta.source_group || semantic.source_row_sha256 !== meta.item_sha256 ||
        semantic.target_sha256 !== meta.target_sha256 || semantic.generated_label !== source.answer ||
        source.group !== meta.source_group || source.role !== 'train')
      throw new Error(`candidate-selection-packet-review-binding-mismatch:${row.id}`);
    packet.resolve(row.source_ref?.source_quality_clearance, row.task?.program_ir);
    selected.push({ id: row.id, item_id: meta.item_id, item_sha256: meta.item_sha256,
      target_sha256: meta.target_sha256, split: row.split, source_groups: row.source_groups });
  }
  if (seenIds.size !== manifestRows.size || [...manifestRows.keys()].some(id => !seenIds.has(id)))
    throw new Error('candidate-selection-packet-manifest-turn-closure-mismatch');
  return { schema: 'natlang.verified-candidate-selection-packet/1',
    manifest_sha256: packet.manifest_sha256, candidate_review_sha256: candidateReviewRef.sha256,
    semantic_review_sha256: semanticReviewRef.sha256, historical_audit_sha256: historicalAuditRef.sha256,
    selection_review_pin_correction_sha256: correctionRef.sha256,
    rows: selected };
}
