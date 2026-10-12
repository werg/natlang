#!/usr/bin/env node
/**
 * Selectively materialize provider-sampled child decisions from a partial
 * static decision program. The input remains immutable; an explicit review
 * file binds each admitted child action to its source item and exact target.
 */
import { createHash } from 'node:crypto';
import { mkdir, open, readFile, readdir, stat } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { materializeNativeRows, nativeRowDigest } from '../ts-host/dist/teacher/native-materializer.js';
import { nativeDecisionTargetDigest } from '../ts-host/dist/native/decision-review.js';
import { ordinalDistanceRecordReview } from '../ts-host/dist/evaluation/oracles.js';
import { sourceQualityClearanceForProgram, sourceQualityClearanceProblems } from '../ts-host/dist/teacher/source-quality-clearance.js';
import { canonical as sharedCanonical } from '../ts-host/dist/adaptation/identity.js';
import { hexDigest } from '../ts-host/dist/native/hash.js';
import { providerRenderedTurn, visibleValue } from './lib/decision-child-visibility.mjs';
import { parentEvalChain, projectDeclaredChildResult } from './lib/decision-child-projection.mjs';
import { reviewDecisionChildOutput } from './lib/decision-child-source-review.mjs';

const { values } = parseArgs({ options: {
  'results-dir': { type: 'string' }, 'source-cases': { type: 'string' },
  'source-manifest': { type: 'string' }, 'candidate-out': { type: 'string' },
  review: { type: 'string' }, out: { type: 'string' }, 'source-quality-review': { type: 'string' },
  'semantic-review': { type: 'string' },
  'semantic-source-audit': { type: 'string' },
  'selection-pin-correction': { type: 'string' },
  'source-label-corrections': { type: 'string' },
} });
if (!values['results-dir'] || !values['source-cases'] || !values['source-manifest'] || !values['candidate-out'])
  throw new Error('usage: node scripts/materialize-reviewed-decision-children.mjs --results-dir DIR --source-cases FILE --source-manifest FILE --candidate-out FILE [--review FILE --out DIR]');
if (!!values.review !== !!values.out) throw new Error('--review and --out must be supplied together');

const sha = value => createHash('sha256').update(value).digest('hex');
const resultsRoot = resolve(values['results-dir']);
const packetResultsPath = path => relative(resultsRoot, path).split('\\').join('/');
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(',')}}` : JSON.stringify(value);
const same = (a, b) => canonical(a) === canonical(b);
const childCandidateId = (itemId, invocationId, targetSha256) =>
  `child:${hexDigest(sharedCanonical({ item_id: itemId, invocation_id: invocationId,
    target_sha256: targetSha256 }))}`;
const parseLines = (text, path) => text.split(/\r?\n/).flatMap((line, index) => {
  if (!line.trim()) return [];
  try { return [{ value: JSON.parse(line), line: index + 1, raw: line }]; }
  catch (error) { throw new Error(`${path}:${index + 1}: invalid JSON: ${error.message}`); }
});
async function resultsFiles(dir) {
  const out = [];
  for (const item of await readdir(dir, { withFileTypes: true })) {
    const path = resolve(dir, item.name);
    if (item.isDirectory()) out.push(...await resultsFiles(path));
    else if (item.isFile() && item.name === 'results.jsonl') out.push(path);
  }
  return out.sort();
}
function effectiveFields(item, task) {
  // The item ID is the host-side key used to aggregate results; it does not
  // classify the task. A child target can be source-bound by the complete
  // visible state, question, and decision schema even when the controller
  // keeps its map key outside the inline function's scope.
  const fields = { state: item.state, question: item.question };
  for (const key of ['criteria', 'options', 'levels']) {
    if (Object.hasOwn(item, key)) fields[key] = item[key];
    else if (Object.hasOwn(task, key)) fields[key] = task[key];
  }
  return fields;
}
function sourceOutputReview(item, semantics, expected, actual, correctionsByItem = new Map()) {
  const spec = semantics.oracle;
  if (item.family === 'decision:review-satisfaction-estimate-v1' &&
      spec?.normalization === 'ordinal-distance-record') {
    return ordinalDistanceRecordReview({ [item.id]: actual }, { [item.id]: expected },
      spec.ordinal_levels, spec.ordinal_distance_max);
  }
  if (same(actual, expected)) return { accepted: true, errors: [], entries: [] };
  const correction = correctionsByItem.get(item.id);
  if (item.family === 'decision:toxicity-binary-threshold-v1' && correction &&
      correction.original_annotation === expected && correction.corrected_label === actual &&
      correction.actual_generated_value === actual && actual !== expected)
    return { accepted: true, errors: [], entries: [], source_label_correction: correction };
  return { accepted: false, errors: ['source_label_mismatch'], entries: [] };
}
const sourceCasesPath = resolve(values['source-cases']);
const sourceManifestPath = resolve(values['source-manifest']);
const sourceBytes = await readFile(sourceCasesPath);
const sourceText = sourceBytes.toString('utf8');
const sourceSha = sha(sourceBytes);
const sourceManifestBytes = await readFile(sourceManifestPath);
const sourceManifest = JSON.parse(sourceManifestBytes.toString('utf8'));
if (sourceManifest.selected_sha256 && sourceManifest.selected_sha256 !== sourceSha ||
    sourceManifest.cases_sha256 && sourceManifest.cases_sha256 !== sourceSha)
  throw new Error('source manifest does not bind the supplied source cases');
const sourceRows = new Map(parseLines(sourceText, sourceCasesPath).map(({ value: row }) => [row.id, row]));
const sourceRawById = new Map(parseLines(sourceText, sourceCasesPath).map(({ value: row, raw }) => [row.id, raw]));
const groupRoles = new Map();
for (const row of sourceRows.values()) {
  const roles = groupRoles.get(row.group) ?? new Set();
  roles.add(row.role ?? row.split);
  groupRoles.set(row.group, roles);
}
const canonicalHolds = JSON.parse(await readFile(resolve('training/decision_source_quality_holds.json'), 'utf8'));
if (canonicalHolds.schema !== 'natlang.decision-source-quality-holds/1' || !Array.isArray(canonicalHolds.excluded_items))
  throw new Error('unsupported canonical source-quality hold policy');
const heldIds = new Set(canonicalHolds.excluded_items.map(row => row.item_id));
const heldGroups = new Set(canonicalHolds.excluded_items.map(row => row.original_source_group));
const reservationPath = resolve('training/source-split-reservations.json');
const reservationBytes = await readFile(reservationPath);
const reservationPolicy = JSON.parse(reservationBytes.toString('utf8'));
if (reservationPolicy.schema !== 'natlang.source-split-reservations/1')
  throw new Error('unsupported source split reservation policy');
const reservationSourcePath = resolve(reservationPolicy.source.path);
const reservationSourceBytes = await readFile(reservationSourcePath);
if (sha(reservationSourceBytes) !== reservationPolicy.source.sha256)
  throw new Error('JeFF reservation source checksum mismatch');
const reservedIds = new Set(reservationPolicy.records.flatMap(item => item.source_ids ?? []));
let labelCorrectionSha = null;
const correctionsByItem = new Map();
if (values['source-label-corrections']) {
  const correctionBytes = await readFile(resolve(values['source-label-corrections']));
  const correctionDoc = JSON.parse(correctionBytes.toString('utf8'));
  labelCorrectionSha = sha(correctionBytes);
  if (correctionDoc.schema !== 'natlang.visible-decision-source-label-corrections/1' ||
      correctionDoc.source_cases?.sha256 !== sourceSha || !Array.isArray(correctionDoc.corrections))
    throw new Error('source-label corrections do not bind the supplied case snapshot');
  for (const correction of correctionDoc.corrections) {
    const source = sourceRows.get(correction.item_id);
    const raw = sourceRawById.get(correction.item_id);
    const input = source && { state: source.state, question: source.question, criteria: source.criteria ?? null,
      options: source.options ?? null, levels: source.levels ?? null };
    if (!source || correction.source_group !== source.group || correction.source_row_sha256 !== sha(raw) ||
        correction.input_sha256 !== hexDigest(sharedCanonical(input)) ||
        correction.source_revision !== sourceManifest.source_sha256 ||
        correction.original_annotation !== source.answer || typeof correction.corrected_label !== 'boolean' ||
        correction.actual_generated_value !== correction.corrected_label ||
        correction.corrected_label === correction.original_annotation || !correction.rubric_reason?.trim() ||
        correctionsByItem.has(correction.item_id))
      throw new Error(`invalid source-label correction binding: ${correction.item_id ?? '(missing ID)'}`);
    correctionsByItem.set(correction.item_id, { ...correction, evidence_sha256: labelCorrectionSha });
  }
}
const verifiedLabelCorrections = labelCorrectionSha ? {
  schema: 'natlang.verified-source-label-corrections/1', evidence_sha256: labelCorrectionSha,
  source_cases_sha256: sourceSha, evidence_files_verified: true, corrections: [...correctionsByItem.values()],
} : undefined;
const reservedNeedles = [];
for (const { value } of parseLines(reservationSourceBytes.toString('utf8'), reservationSourcePath)) {
  if (!reservedIds.has(value.id) || typeof value.state !== 'string') continue;
  const normalized = value.state.replace(/\\/g, ' ').replace(/\s+/g, ' ').trim();
  if (normalized.length >= 40) reservedNeedles.push({ id: value.id, text: normalized.slice(0, 60) });
}
function stringLeaves(value, out = []) {
  if (typeof value === 'string') out.push(value);
  else if (Array.isArray(value)) for (const child of value) stringLeaves(child, out);
  else if (value && typeof value === 'object') for (const child of Object.values(value)) stringLeaves(child, out);
  return out;
}
function jeffReservationHit(row) {
  const haystack = stringLeaves([row.task, (row.messages ?? [])
    .filter(message => message.role === 'user').map(message => message.content ?? '')])
    .join(' ').replace(/\\/g, ' ').replace(/\s+/g, ' ').trim();
  return reservedNeedles.find(needle => haystack.includes(needle.text))?.id ?? null;
}

let review = null, reviewBytes = null, reviewSha = null;
if (values.review) {
  reviewBytes = await readFile(resolve(values.review));
  reviewSha = sha(reviewBytes);
  review = JSON.parse(reviewBytes.toString('utf8'));
  if (review.schema !== 'natlang.provider-child-action-review/1' || !Array.isArray(review.approvals))
    throw new Error('review must use natlang.provider-child-action-review/1 with approvals[]');
}
let semanticReview = null, semanticReviewBytes = null, semanticReviewSha = null;
if (values['semantic-review']) {
  semanticReviewBytes = await readFile(resolve(values['semantic-review']));
  semanticReviewSha = sha(semanticReviewBytes);
  semanticReview = JSON.parse(semanticReviewBytes.toString('utf8'));
  if (semanticReview.schema !== 'natlang.provider-child-semantic-review/1' ||
      semanticReview.training_admission !== false || semanticReview.trace_admission !== false ||
      semanticReview.source_cases_sha256 !== sourceSha || !Array.isArray(semanticReview.rows))
    throw new Error('semantic review must bind the supplied source cases and remain non-admission evidence');
}
let semanticSourceAuditBytes = null, semanticSourceAuditSha = null;
if (values['semantic-source-audit']) {
  semanticSourceAuditBytes = await readFile(resolve(values['semantic-source-audit']));
  semanticSourceAuditSha = sha(semanticSourceAuditBytes);
  if (!semanticReview || semanticReview.candidate_audit_sha256 !== semanticSourceAuditSha)
    throw new Error('semantic source audit does not match the exact candidate audit pinned by semantic review');
}
let selectionPinCorrectionBytes = null, selectionPinCorrectionSha = null, selectionPinCorrection = null;
if (values['selection-pin-correction']) {
  selectionPinCorrectionBytes = await readFile(resolve(values['selection-pin-correction']));
  selectionPinCorrectionSha = sha(selectionPinCorrectionBytes);
  selectionPinCorrection = JSON.parse(selectionPinCorrectionBytes.toString('utf8'));
  if (!review || !semanticReview || !semanticSourceAuditSha ||
      selectionPinCorrection.schema !== 'natlang.provider-child-review-input-pin-correction/1' ||
      selectionPinCorrection.corrected_review_sha256 !== reviewSha ||
      selectionPinCorrection.corrected_candidate_manifest_sha256 !== semanticSourceAuditSha ||
      selectionPinCorrection.selected_approval_count !== review.approvals.length ||
      selectionPinCorrection.selected_action_count !== review.selected_child_count ||
      review.supersedes_review_sha256 !== selectionPinCorrection.prior_review_sha256 ||
      review.candidate_manifest_sha256 !== semanticSourceAuditSha ||
      review.correction_reason !== selectionPinCorrection.reason ||
      !selectionPinCorrection.selected_approvals_unchanged)
    throw new Error('selection pin correction does not exactly bind the reviewed selection and audit');
}
let sourceQualityClearance = null;
const sourceQualityReviewPath = values['source-quality-review'] ? resolve(values['source-quality-review']) : null;
if (sourceQualityReviewPath) {
  const reviewBytes = await readFile(sourceQualityReviewPath);
  const sourceQualityReview = JSON.parse(reviewBytes.toString('utf8'));
  const problems = sourceQualityClearanceProblems(sourceQualityReview);
  if (problems.length) throw new Error(`invalid source-quality clearance: ${problems.join(', ')}`);
  const policyPath = resolve(sourceQualityReview.source_policy.path);
  if (policyPath !== resolve('training/decision_source_quality_holds.json'))
    throw new Error('source-quality clearance must bind the current canonical policy');
  const policyBytes = await readFile(policyPath);
  if (sha(policyBytes) !== sourceQualityReview.source_policy.sha256)
    throw new Error('source-quality clearance policy hash mismatch');
  const policy = JSON.parse(policyBytes.toString('utf8'));
  const heldIds = [...new Set(policy.excluded_items.map(item => item.item_id))].sort();
  const heldGroups = [...new Set(policy.excluded_items.map(item => item.original_source_group))].sort();
  if (!same(heldIds, [...sourceQualityReview.source_policy.held_source_ids].sort()) ||
      !same(heldGroups, [...sourceQualityReview.source_policy.held_source_groups].sort()))
    throw new Error('source-quality clearance held ID/group census mismatch');
  for (const evidence of sourceQualityReview.evidence) {
    const evidenceBytes = await readFile(resolve(evidence.path));
    if (sha(evidenceBytes) !== evidence.sha256) throw new Error(`source-quality evidence hash mismatch: ${evidence.path}`);
  }
  const reviewIds = new Set(sourceQualityReview.programs.map(program => program.program_id));
  if (reviewIds.size !== sourceQualityReview.programs.length) throw new Error('duplicate source-quality program binding');
  sourceQualityClearance = { schema: 'natlang.verified-source-quality-clearance/1',
    review_sha256: sha(reviewBytes), review_canonical_sha256: hexDigest(sharedCanonical(sourceQualityReview)),
    evidence_files_verified: true, review: sourceQualityReview };
  const clearanceProblems = sourceQualityClearanceProblems(sourceQualityClearance);
  if (clearanceProblems.length) throw new Error(`invalid verified source-quality clearance: ${clearanceProblems.join(', ')}`);
}
const approvalsByKey = new Map();
for (const approval of review?.approvals ?? []) {
  if (approval.approved !== true || !approval.item_id || !/^[a-f0-9]{64}$/.test(approval.item_sha256 ?? '') ||
      !approval.group || !approval.reason?.trim() || !Array.isArray(approval.evidence) || !approval.evidence.length ||
      approval.evidence.some(item => typeof item !== 'string' || !item.trim()))
    throw new Error(`invalid reviewed source item: ${approval.item_id ?? '(missing ID)'}`);
  if (approval.candidate_id !== undefined && typeof approval.candidate_id !== 'string')
    throw new Error(`invalid candidate-specific approval ID: ${approval.item_id}`);
  const key = approval.candidate_id
    ? `${approval.program_id}\0candidate\0${approval.candidate_id}`
    : `${approval.program_id}\0item\0${approval.item_id}`;
  if (approvalsByKey.has(key)) throw new Error(`duplicate review approval: ${approval.item_id}`);
  approvalsByKey.set(key, approval);
}

const candidates = [], prepared = [], rejected = [];
for (const resultsPath of await resultsFiles(resolve(values['results-dir']))) {
  const text = await readFile(resultsPath, 'utf8');
  for (const { value: row, line } of parseLines(text, resultsPath)) {
    const program = row.task?.program_ir;
    const semantics = program?.semantics;
    const taskText = semantics?.folder_files?.['task.json'];
    if (!program || !semantics || typeof taskText !== 'string') { rejected.push({ resultsPath, line, reason: 'missing-program-or-task-contract' }); continue; }
    const sourceBound = program.generation?.source_cases_sha256 === sourceSha ||
      program.source_revisions?.includes(sourceSha) === true;
    if (!sourceBound) throw new Error(`${row.id}: source cases hash mismatch`);
    const task = JSON.parse(taskText), originalDigest = nativeRowDigest(row);
    const reservedSourceId = jeffReservationHit(row);
    const rootInvocation = (row.outcome?.invocation_ledger ?? []).find(item => item.parent_invocation_id == null)?.invocation_id;
    if (!rootInvocation) { rejected.push({ id: row.id, reason: 'root-invocation-missing' }); continue; }
    const scopedQuality = sourceQualityClearance?.review?.target_scope === 'exact-selected-actions-only';
    const qualityBinding = sourceQualityClearance && !scopedQuality
      ? sourceQualityClearanceForProgram(sourceQualityClearance, program) : undefined;
    if (sourceQualityClearance && !qualityBinding && !scopedQuality)
      throw new Error(`source-quality clearance does not bind exact program: ${program.id}`);
    if (scopedQuality && !sourceQualityClearance.review.programs.some(entry => entry.program_id === program.id)) {
      if ([...approvalsByKey.values()].some(approval => approval.program_id === program.id))
        throw new Error(`source-quality clearance does not authorize selected program: ${program.id}`);
      continue;
    }
    if (sourceQualityClearance && (program.external_source?.source_path !== sourceCasesPath ||
        program.external_source?.snapshot_sha256 !== sourceSha ||
        program.external_source?.source_manifest_sha256 !== sha(sourceManifestBytes)))
      throw new Error(`source-quality clearance is bound to different source bytes: ${program.id}`);
    const materialized = materializeNativeRows([row], { failedRuns: true, pendingDecisionCandidates: true,
      candidateSelectionOnly: true,
      ...(verifiedLabelCorrections ? { sourceLabelCorrections: verifiedLabelCorrections } : {}),
      ...(sourceQualityClearance ? { sourceQualityClearance } : {}) });
    if (materialized.unlinked.length) { rejected.push({ id: row.id, reason: 'action-ledger-unlinked', unlinked: materialized.unlinked }); continue; }
    const ledger = row.outcome?.invocation_ledger ?? [];
    for (const [path, raw] of Object.entries(semantics.folder_files)) {
      if (!path.startsWith('items/') || !path.endsWith('.json')) continue;
      const item = JSON.parse(raw), itemSha = sha(raw), id = item.id;
      const source = sourceRows.get(id);
      const itemRole = item.role ?? item.split;
      const itemGroup = item.group ?? item.group_id;
      if (!source || source.role !== 'train' || itemRole !== 'train' || source.group !== itemGroup ||
          (source.split !== undefined && source.split !== 'train')) {
        rejected.push({ program_id: program.id, item_id: id, item_sha256: itemSha, reason: 'source-row-or-train-split-binding-failed' });
        continue;
      }
      const roles = groupRoles.get(source.group) ?? new Set();
      if (roles.size !== 1 || !roles.has('train') || heldIds.has(id) || heldGroups.has(source.group)) {
        rejected.push({ program_id: program.id, item_id: id, item_sha256: itemSha, group: source.group,
          reason: 'source-group-crosses-split-or-canonical-hold' });
        continue;
      }
      if (reservedSourceId) {
        rejected.push({ program_id: program.id, item_id: id, item_sha256: itemSha, group: source.group,
          reason: 'jeff-test-reservation-overlap', reserved_source_id: reservedSourceId });
        continue;
      }
      const expected = semantics.expected?.[id], observed = row.outcome?.value?.[id];
      const sourceReview = sourceOutputReview(item, semantics, expected, observed, correctionsByItem);
      if (['decision:toxicity', 'decision:app-stars', 'decision:yelp-stars'].includes(item.family)) {
        rejected.push({ program_id: program.id, item_id: id, item_sha256: itemSha, group: source.group,
          reason: `adapter-contract-repair-required:${item.family}` });
        continue;
      }
      if (expected === undefined) continue;
      const fields = effectiveFields(item, task);
      let matches = [];
      const bindingGaps = [];
      for (const entry of ledger) {
        if (entry.completion_status !== 'done' || entry.host_result?.complete !== true) continue;
        const declaredReturnType = entry.inline_instruction_site?.returns?.text;
        const parentChain = parentEvalChain(row, ledger, entry);
        const parentCodes = parentChain.flatMap(parent => [parent.code, ...(parent.followup_codes ?? [])]);
        const projection = projectDeclaredChildResult(declaredReturnType, entry.host_result.value,
          parentCodes, entry.inline_instruction_site);
        const childReview = reviewDecisionChildOutput({ expected, parentObserved: observed,
          childObserved: projection.value,
          review: (expectedValue, actualValue) => sourceOutputReview(item, semantics,
            expectedValue, actualValue, correctionsByItem) });
        if (!projection.accepted || !childReview.accepted) continue;
        if (childReview.child_output_review.source_label_correction && !labelCorrectionSha)
          throw new Error(`source-label correction lacks immutable evidence file: ${id}`);
        // An intermediate caller may return the typed value unchanged before
        // an enclosing action projects it (for example `return selected;`
        // followed by a per-item `answer[0]`). Require the captured result at
        // every such step to equal the child result and the exact action to
        // return its own typed value; never infer a projection from labels.
        const passthrough = projection.causal_aggregate || parentChain.slice(0, projection.projection_code_index).every(parent => {
          const returned = parent.code.match(/\breturn\s+([A-Za-z_$][\w$]*)\s*;?\s*$/);
          const sameValue = parent.entry?.host_result?.complete === true &&
            parent.entry.completion_status === 'done' && same(JSON.stringify(parent.entry.host_result.value),
              JSON.stringify(entry.host_result.value));
          const sameType = parent.entry?.inline_instruction_site?.returns?.text === declaredReturnType;
          return returned && sameValue && sameType && /\bnl\.with\s*</.test(parent.code);
        });
        if (!passthrough) continue;
        const invocationTurns = materialized.turns.filter(turn => turn.source_ref?.invocation_id === entry.invocation_id);
        const turn = invocationTurns.at(-1);
        if (!turn) { bindingGaps.push({ invocation_id: entry.invocation_id, reason: 'materializer_has_no_child_turn' }); continue; }
        const providerView = providerRenderedTurn(turn, row);
        if (!providerView.text) {
          bindingGaps.push({ invocation_id: entry.invocation_id, reason: providerView.reason });
          continue;
        }
        const text = providerView.text;
        const snapshots = entry.inline_instruction_site?.runtime_capture_snapshots?.captures ?? [];
        const scopeValues = Object.fromEntries(snapshots.map(snapshot => [snapshot.name, snapshot.value]));
        const scopeBound = Object.fromEntries(Object.entries(fields).map(([key, value]) =>
          [key, Object.hasOwn(scopeValues, key) && same(scopeValues[key], value)]));
        const visible = Object.fromEntries(Object.entries(fields).map(([key, value]) =>
          [key, visibleValue(value, text)]));
        // caseId is provenance rather than task content, but when the runtime
        // recorded it, it must bind this exact invocation to this item. This
        // prevents equal labels from associating another item's child result.
        const caseIdBound = Object.hasOwn(scopeValues, 'caseId')
          ? same(scopeValues.caseId, id) : true;
        if (!caseIdBound) {
          bindingGaps.push({ invocation_id: entry.invocation_id, reason: 'runtime_case_id_mismatch',
            case_id_sha256: sha(String(scopeValues.caseId)),
            visible_fields: Object.keys(visible).filter(key => visible[key]) });
          continue;
        }
        if (Object.values(visible).every(Boolean)) matches.push({ entry, turn, visible, caseIdBound,
          scopeBoundFields: Object.keys(scopeBound).filter(key => scopeBound[key]), providerView,
          childValue: projection.value,
          childReview, projection: {
            kind: projection.projection, declared_return_type: declaredReturnType,
            result_binding: projection.result_binding, parent_invocation_id: entry.parent_invocation_id,
            parent_action_sha256: sha(parentCodes[projection.projection_code_index] ?? ''),
            parent_chain_sha256: sha(JSON.stringify(parentChain.map(parent => ({
              invocation_id: parent.invocation_id, code_sha256: sha(parent.code),
              followup_code_sha256: (parent.followup_codes ?? []).map(sha),
              host_value_sha256: sha(JSON.stringify(parent.entry?.host_result?.value ?? null)),
            })))), projected_value_sha256: sha(JSON.stringify(projection.value)) } });
        else bindingGaps.push({ invocation_id: entry.invocation_id, reason: 'source_fields_not_fully_bound',
          missing_fields: Object.keys(visible).filter(key => !visible[key]),
          scope_capture_names: Object.keys(scopeValues) });
      }
      const itemApprovalKey = `${program.id}\0item\0${id}`;
      const itemApproval = approvalsByKey.get(itemApprovalKey);
      const matchingCandidates = matches.map(match => ({ ...match, targetSha256:
        nativeDecisionTargetDigest(match.turn.target), candidateId: childCandidateId(id,
          match.entry.invocation_id, nativeDecisionTargetDigest(match.turn.target)) }));
      let itemApprovalCandidateId = null;
      if (itemApproval && matchingCandidates.length > 1) {
        const selected = matchingCandidates.filter(match => itemApproval.decision_index === match.turn.decision.index &&
          itemApproval.target_sha256 === match.targetSha256);
        if (selected.length === 1) itemApprovalCandidateId = selected[0].candidateId;
        else throw new Error(`item-scoped approval does not uniquely bind one child candidate: ${program.id}/${id}`);
      } else if (itemApproval && matchingCandidates.length === 1)
        itemApprovalCandidateId = matchingCandidates[0].candidateId;
      if (!matchingCandidates.length) {
        rejected.push({ program_id: program.id, item_id: id, item_sha256: itemSha, group: source.group,
          reason: 'no-child-with-complete-exact-visible-inputs',
          parent_observed_value: observed,
          parent_output_review: sourceReview,
          matching_children: 0,
          expected_value_has_completed_child: bindingGaps.length > 0,
          binding_gaps: bindingGaps.slice(0, 8) });
        continue;
      }
      for (const match of matchingCandidates) {
      const { entry, turn, visible, caseIdBound, scopeBoundFields, providerView } = match;
      const candidate = { candidate_id: match.candidateId, program_id: program.id,
        trajectory_id: row.id, trajectory_sha256: originalDigest,
        item_id: id, item_sha256: itemSha, source_group: source.group, source_role: source.role,
        source_split: source.role ?? source.split,
        expected_value: expected, observed_value: match.childValue,
        parent_observed_value: observed,
        parent_output_review: match.childReview.parent_output_review,
        invocation_id: entry.invocation_id,
        decision_index: turn.decision.index, target_sha256: match.targetSha256,
        target_tool: turn.target?.tool_calls?.[0]?.function?.name ?? null,
        provider_request_sha256: providerView.requestSha256,
        provider_rendered_request_sha256: providerView.renderedRequestSha256,
        provider_rendered_messages_sha256: sha(JSON.stringify(providerView.renderedMessages)),
        visible_fields: Object.keys(visible), scope_bound_fields: scopeBoundFields,
        source_item_id_runtime_bound: caseIdBound,
        typed_output_projection: match.projection,
        source_reference_review: match.childReview.child_output_review,
        source_quality_review_only: turn.candidate_selection?.source_quality_review_only === true,
        source_quality_hold_reasons: turn.candidate_selection?.source_quality_hold_reasons ?? [],
        scope_snapshot_sha256: sha(JSON.stringify(entry.inline_instruction_site?.runtime_capture_snapshots ?? null)),
        group_role_closure: [...roles] };
      candidates.push(candidate);
      const candidateApprovalKey = `${program.id}\0candidate\0${match.candidateId}`;
      const specificApproval = approvalsByKey.get(candidateApprovalKey);
      const approval = specificApproval ?? (itemApprovalCandidateId === match.candidateId ? itemApproval : null);
      if (!approval) continue;
      if (approval.item_sha256 !== itemSha || approval.group !== source.group ||
          approval.decision_index !== candidate.decision_index || approval.target_sha256 !== candidate.target_sha256 ||
          (approval.candidate_id && approval.candidate_id !== candidate.candidate_id))
        throw new Error(`review/source/target binding mismatch for ${program.id}/${id}/${candidate.candidate_id}`);
      if (semanticReview) {
        const exactSemantic = semanticReview.rows.filter(value => value.candidate_id === candidate.candidate_id);
        const semanticMatches = exactSemantic.length ? exactSemantic : semanticReview.rows.filter(value =>
          value.candidate_id === undefined && value.item_id === id &&
          value.target_sha256 === candidate.target_sha256 && value.decision_index === candidate.decision_index &&
          value.invocation_id === candidate.invocation_id);
        const semantic = semanticMatches.length === 1 ? semanticMatches[0] : null;
        if (!semantic || semantic.semantic_review !== 'clear' || semantic.source_group !== source.group ||
            semantic.source_row_sha256 !== itemSha || semantic.target_sha256 !== candidate.target_sha256 ||
            semantic.decision_index !== candidate.decision_index || semantic.invocation_id !== candidate.invocation_id ||
            semantic.generated_label !== observed || semantic.source_annotation !== expected)
          throw new Error(`semantic review/source/target binding mismatch for ${program.id}/${id}/${candidate.candidate_id}`);
      }
      const correction = sourceReview.source_label_correction;
      if (correction) {
        if (approval.source_label_correction?.evidence_sha256 !== correction.evidence_sha256 ||
            approval.source_label_correction?.item_id !== id ||
            approval.source_label_correction?.item_sha256 !== itemSha ||
            approval.source_label_correction?.source_row_sha256 !== correction.source_row_sha256 ||
            approval.source_label_correction?.input_sha256 !== correction.input_sha256 ||
            approval.source_label_correction?.original_annotation !== expected ||
            approval.source_label_correction?.corrected_label !== observed ||
            approval.source_label_correction?.actual_generated_value !== observed)
          throw new Error(`review is missing exact source-label correction binding for ${program.id}/${id}`);
      } else if (approval.source_label_correction)
        throw new Error(`review claims an unverified source-label correction for ${program.id}/${id}`);
      const decisionApproval = { schema: 'natlang.native-decision-approval/1', trajectory_id: row.id,
        source_row_sha256: originalDigest, decision_index: turn.decision.index,
        target_sha256: candidate.target_sha256,
        review_sha256: review.supersedes_review_sha256 ?? reviewSha,
        reason: approval.reason, evidence: approval.evidence,
        ...(correction ? { source_label_correction: approval.source_label_correction } : {}) };
      const selectedAction = { trajectory_id: row.id, trajectory_sha256: originalDigest,
        decision_index: turn.decision.index, invocation_id: candidate.invocation_id,
        target_sha256: candidate.target_sha256 };
      const selectedQualityBinding = sourceQualityClearance
        ? sourceQualityClearanceForProgram(sourceQualityClearance, program,
          scopedQuality ? selectedAction : undefined) : undefined;
      if (sourceQualityClearance && !selectedQualityBinding)
        throw new Error(`source-quality clearance does not bind exact selected child action: ${program.id}/${candidate.candidate_id}`);
      const approved = materializeNativeRows([row], { failedRuns: true, decisionApprovals: [decisionApproval],
        candidateSelectionOnly: true,
        ...(selectedQualityBinding ? { sourceQualityClearanceRef: { sidecar_path: 'source-quality-clearance.json',
          file_sha256: sourceQualityClearance.review_sha256 } } : {}),
        ...(verifiedLabelCorrections ? { sourceLabelCorrections: verifiedLabelCorrections } : {}),
        ...(selectedQualityBinding ? { sourceQualityClearance } : {}) });
      const selected = approved.turns.filter(value => value.decision.index === turn.decision.index &&
        value.candidate_selection?.approved === true && value.training_admission?.approved === false &&
        value.trace_admission?.admitted === false);
      if (selected.length !== 1) throw new Error(`selected target was not admitted exactly once: ${program.id}/${id}/${candidate.candidate_id}`);
      selected[0].sampling_group = `source-item:${id}`;
      prepared.push({ candidate, turn: selected[0], approval: decisionApproval });
      approvalsByKey.delete(specificApproval ? candidateApprovalKey : itemApprovalKey);
      }
    }
  }
}
if (approvalsByKey.size) throw new Error(`review names source items without an exact, visible successful child action: ${[...approvalsByKey.values()].map(item => item.item_id).join(', ')}`);

const candidateIds = candidates.map(candidate => candidate.candidate_id);
if (new Set(candidateIds).size !== candidateIds.length)
  throw new Error('candidate_id collision across exact child invocations');

const candidatePath = resolve(values['candidate-out']);
await mkdir(dirname(candidatePath), { recursive: true });
const candidateHandle = await open(candidatePath, 'wx');
try { await candidateHandle.writeFile(JSON.stringify({ schema: 'natlang.provider-child-action-candidates/2',
  candidate_selection: true, training_admission: false, trace_admission: false, created_at: new Date().toISOString(),
  candidate_id_rule: 'child:<sha256(canonical({item_id, invocation_id, target_sha256}))>',
  source_cases: { path: 'source-cases.jsonl', sha256: sourceSha },
  source_manifest: { path: 'source-manifest.json', sha256: sha(sourceManifestBytes) },
  source_split_reservations: { path: 'source-split-reservations.json', sha256: sha(reservationBytes),
    test_source_path: 'reservation-source.json', test_source_sha256: sha(reservationSourceBytes),
    matching_rule: 'case-sensitive; backslashes to spaces; whitespace collapsed; first 60 characters of reserved state searched in full program IR and user context' },
  review_sha256: reviewSha,
  candidates,
  rejected: rejected.map(({ resultsPath, ...entry }) => ({ ...entry,
    ...(resultsPath ? { results_file: packetResultsPath(resultsPath) } : {}) })),
  counts: { candidate_actions: candidates.length, rejected_items: rejected.length, selected_children: prepared.length } }, null, 2) + '\n'); }
finally { await candidateHandle.close(); }

if (values.out) {
  const out = resolve(values.out);
  await mkdir(out, { recursive: true });
  if ((await readdir(out)).length || !(await stat(out)).isDirectory()) throw new Error(`output directory must be empty: ${out}`);
  if (sourceQualityReviewPath) {
    const clearanceBytes = await readFile(sourceQualityReviewPath);
    const clearanceHandle = await open(resolve(out, 'source-quality-clearance.json'), 'wx');
    try { await clearanceHandle.writeFile(clearanceBytes); }
    finally { await clearanceHandle.close(); }
  }
  const sourceCasesCopy = await open(resolve(out, 'source-cases.jsonl'), 'wx');
  try { await sourceCasesCopy.writeFile(sourceBytes); }
  finally { await sourceCasesCopy.close(); }
  const sourceManifestCopy = await open(resolve(out, 'source-manifest.json'), 'wx');
  try { await sourceManifestCopy.writeFile(sourceManifestBytes); }
  finally { await sourceManifestCopy.close(); }
  if (reviewBytes) {
    const candidateReviewCopy = await open(resolve(out, 'candidate-selection-review.json'), 'wx');
    try { await candidateReviewCopy.writeFile(reviewBytes); }
    finally { await candidateReviewCopy.close(); }
  }
  if (semanticReviewBytes) {
    const semanticReviewCopy = await open(resolve(out, 'semantic-child-review.json'), 'wx');
    try { await semanticReviewCopy.writeFile(semanticReviewBytes); }
    finally { await semanticReviewCopy.close(); }
  }
  const clearanceEvidence = [];
  if (sourceQualityClearance) {
    for (const [index, evidence] of sourceQualityClearance.review.evidence.entries()) {
      const filename = evidence.role === 'source_quality_review' ? 'source-quality-review.json' :
        evidence.role === 'source_closure' ? 'source-closure.json' : `source-quality-evidence-${index + 1}.bin`;
      const bytes = await readFile(resolve(evidence.path));
      if (sha(bytes) !== evidence.sha256) throw new Error(`source-quality evidence hash mismatch during packet copy: ${evidence.path}`);
      const handle = await open(resolve(out, filename), 'wx');
      try { await handle.writeFile(bytes); }
      finally { await handle.close(); }
      clearanceEvidence.push({ role: evidence.role, path: filename, sha256: evidence.sha256 });
    }
  }
  const reservationCopy = await open(resolve(out, 'source-split-reservations.json'), 'wx');
  try { await reservationCopy.writeFile(reservationBytes); }
  finally { await reservationCopy.close(); }
  const reservationSourceCopy = await open(resolve(out, 'reservation-source.json'), 'wx');
  try { await reservationSourceCopy.writeFile(reservationSourceBytes); }
  finally { await reservationSourceCopy.close(); }
  const outputHandle = await open(resolve(out, 'turns.jsonl'), 'wx');
  try { for (const item of prepared) await outputHandle.writeFile(JSON.stringify(item.turn) + '\n'); }
  finally { await outputHandle.close(); }
  const candidateBytes = await readFile(candidatePath);
  const discoveryCopy = await open(resolve(out, 'candidate-discovery-audit.json'), 'wx');
  try { await discoveryCopy.writeFile(candidateBytes); }
  finally { await discoveryCopy.close(); }
  if (semanticSourceAuditBytes) {
    const semanticSourceAuditCopy = await open(resolve(out, 'semantic-source-candidate-audit.json'), 'wx');
    try { await semanticSourceAuditCopy.writeFile(semanticSourceAuditBytes); }
    finally { await semanticSourceAuditCopy.close(); }
  }
  if (selectionPinCorrectionBytes) {
    const correctionCopy = await open(resolve(out, 'selection-review-pin-correction.json'), 'wx');
    try { await correctionCopy.writeFile(selectionPinCorrectionBytes); }
    finally { await correctionCopy.close(); }
  }
  const manifest = { schema: 'natlang.provider-child-action-packet/2', candidate_selection: true,
    training_admission: false, trace_admission: false, created_at: new Date().toISOString(), review_sha256: reviewSha,
    candidate_selection_review: reviewBytes ? { path: 'candidate-selection-review.json', sha256: reviewSha } : null,
    semantic_review: semanticReviewBytes ? { path: 'semantic-child-review.json', sha256: semanticReviewSha,
      candidate_audit_sha256: semanticReview.candidate_audit_sha256,
      ...(semanticSourceAuditBytes ? { candidate_audit_path: 'semantic-source-candidate-audit.json',
        candidate_audit_file_sha256: semanticSourceAuditSha } : {}),
      counts: semanticReview.counts } : null,
    ...(semanticSourceAuditBytes ? { semantic_review_input_audit: {
      path: 'semantic-source-candidate-audit.json', sha256: semanticSourceAuditSha } } : {}),
    ...(selectionPinCorrectionBytes ? { selection_review_pin_correction: {
      path: 'selection-review-pin-correction.json', sha256: selectionPinCorrectionSha } } : {}),
    source_cases: { path: 'source-cases.jsonl', sha256: sourceSha },
    source_manifest: { path: 'source-manifest.json', sha256: sha(sourceManifestBytes) },
    source_quality_clearance: sourceQualityReviewPath ? { path: 'source-quality-clearance.json',
      file_sha256: sourceQualityClearance.review_sha256,
      review_canonical_sha256: sourceQualityClearance.review_canonical_sha256, evidence: clearanceEvidence } : null,
    source_split_reservations: { path: 'source-split-reservations.json', sha256: sha(reservationBytes),
      test_source_path: 'reservation-source.json', test_source_sha256: sha(reservationSourceBytes) },
    candidate_manifest: 'candidate-discovery-audit.json', candidate_manifest_sha256: sha(await readFile(candidatePath)),
    turns: prepared.map(item => ({ id: item.turn.id, candidate_id: item.candidate.candidate_id,
      program_id: item.candidate.program_id,
      trajectory_id: item.candidate.trajectory_id, trajectory_sha256: item.candidate.trajectory_sha256,
      item_id: item.candidate.item_id, item_sha256: item.candidate.item_sha256, source_group: item.candidate.source_group,
      sampling_group: `source-item:${item.candidate.item_id}`,
      invocation_id: item.candidate.invocation_id, decision_index: item.candidate.decision_index,
      target_sha256: item.candidate.target_sha256, visible_fields: item.candidate.visible_fields,
      approval: item.approval })) };
  const manifestHandle = await open(resolve(out, 'materialized-manifest.json'), 'wx');
  try { await manifestHandle.writeFile(JSON.stringify(manifest, null, 2) + '\n'); }
  finally { await manifestHandle.close(); }
}
console.log(JSON.stringify({ candidates: candidates.length, rejected: rejected.length, approved_turns: prepared.length,
  candidate_manifest: candidatePath, ...(values.out ? { output: resolve(values.out) } : {}) }));
