#!/usr/bin/env node
/** Read-only independent verifier for the held five-world Luna action export. */
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { materializeNativeRows } from '../ts-host/dist/teacher/native-materializer.js';

const ROOT = 'runs/semantic-lambda-topology-v17-v8-corrected-review-v1/authored-root-source-expansion-v1';
const CAMPAIGN = `${ROOT}/luna/campaign-v1`;
const CASES_PATH = `${ROOT}/source-cases-v1.jsonl`;
const CASES_SHA = 'b2c7c7659c3d355e0e7d0183873938ce3bc54341e2fabfca6e612fc50605076f';
const PROPOSAL_PATH = `${CAMPAIGN}/native-sft-action-candidate-proposal-v1.jsonl`;
const NATIVE_PATH = `${CAMPAIGN}/native-sft-materialized-38-held-v1.jsonl`;
const MAP_PATH = `${CAMPAIGN}/native-sft-materialized-38-id-map-v1.json`;
const REVIEW_PATH = `${CAMPAIGN}/independent-semantic-action-review-v2.json`;
const INCIDENT_PATH = `${CAMPAIGN}/independent-semantic-action-review-v2-overwrite-incident-v1.json`;
const CALLSTORE_DIR = `${CAMPAIGN}/callstore-exact-five-world-children-v1`;
const CALLSTORE_CALLS = `${CALLSTORE_DIR}/calls.jsonl`;
const CALLSTORE_PROVENANCE = `${CALLSTORE_DIR}/provenance.json`;
const REVIEW_SHA = '275e6ab47804406bcbef7261a6cf41a33b6f5178888af8b21f879988062be6e1';
const PROPOSAL_SHA = '72ffdb3208e29933ebe5ddbfbba431414e7508729d15d784f437756b28f39fed';
const NATIVE_SHA = '6f2559d202c169a06fbebd89491086d9e4a4ffba62673a05ffade79aaa5fb1c9';
const MAP_SHA = 'c9e668204de0cdef21bee8fefa1d04ce9cd1ca64f9090768e36b10be19bdf44a';
const CALLSTORE_CALLS_SHA = 'd6c0af5bb4e1aa055fcb37c7478c9b7d5ba1678ca783cc6321299434a210fb6a';
const CALLSTORE_PROVENANCE_SHA = '7eb1240ab54d2a53d2ac389272bcedd0cfb88b7511e3d9474191a3b78f104c11';

const bytes = path => readFileSync(path);
const text = path => bytes(path).toString('utf8');
const sha = value => createHash('sha256').update(value).digest('hex');
const jsonSha = value => sha(Buffer.from(JSON.stringify(value)));
const stableJson = value => {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`;
  return JSON.stringify(value);
};
const stableSha = value => sha(Buffer.from(stableJson(value)));
const lines = path => text(path).trimEnd().split(/\r?\n/).filter(Boolean).map(JSON.parse);
const requireThat = (condition, message) => { if (!condition) throw new Error(message); };
const sameJson = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const sourceCases = lines(CASES_PATH);
requireThat(sha(bytes(CASES_PATH)) === CASES_SHA, 'source-cases-v1 pin mismatch');
const sourceCaseById = new Map(sourceCases.map(row => [row.id, row]));
const proposals = lines(PROPOSAL_PATH);
const nativeRows = lines(NATIVE_PATH);
const idMap = JSON.parse(text(MAP_PATH));
const callStoreRows = lines(CALLSTORE_CALLS);
const resultCache = new Map();
const materializedCache = new Map();
const proposalSha = sha(bytes(PROPOSAL_PATH));
const nativeSha = sha(bytes(NATIVE_PATH));
const mapSha = sha(bytes(MAP_PATH));
const callStoreCallsSha = sha(bytes(CALLSTORE_CALLS));
const callStoreProvenanceSha = sha(bytes(CALLSTORE_PROVENANCE));

requireThat(proposalSha === PROPOSAL_SHA, `proposal pin mismatch: ${proposalSha}`);
requireThat(nativeSha === NATIVE_SHA, `native export pin mismatch: ${nativeSha}`);
requireThat(mapSha === MAP_SHA, `ID-map pin mismatch: ${mapSha}`);
requireThat(sha(bytes(REVIEW_PATH)) === REVIEW_SHA, 'review-v2 pin mismatch');
requireThat(callStoreCallsSha === CALLSTORE_CALLS_SHA && callStoreProvenanceSha === CALLSTORE_PROVENANCE_SHA,
  'exact CallStore export/provenance pin mismatch');
requireThat(proposals.length === 38 && nativeRows.length === 38, 'expected 38 proposal and native rows');
requireThat(idMap.row_count === 38 && idMap.bijection.length === 38, 'ID map count mismatch');
requireThat(idMap.proposal_sha256 === proposalSha && idMap.native_row_file_sha256 === nativeSha,
  'ID map does not pin proposal/export bytes');

const proposalIds = new Set(), nativeIds = new Set(), sourceGroups = new Set();
let train = 0, test = 0, nativeApprovedBeforeHold = 0, statusOnlyProofCount = 0;
const checked = [];
for (let i = 0; i < proposals.length; i++) {
  const p = proposals[i], mapping = idMap.bijection[i], exported = nativeRows[i];
  const tag = `candidate ${i} (${p.id})`;
  requireThat(!proposalIds.has(p.id), `${tag}: duplicate proposal ID`); proposalIds.add(p.id);
  requireThat(mapping.proposal_id === p.id && mapping.proposal_row_sha256 === jsonSha(p), `${tag}: proposal map mismatch`);
  requireThat(mapping.native_record_id === exported.id && mapping.native_record_version === exported.version,
    `${tag}: native ID/version map mismatch`);
  requireThat(!nativeIds.has(exported.id), `${tag}: duplicate native ID`); nativeIds.add(exported.id);
  requireThat(mapping.exported_native_row_sha256 === jsonSha(exported), `${tag}: exported row hash mismatch`);
  requireThat(exported.training_admission?.approved === false && exported.decision?.training_approved === false,
    `${tag}: export is not held`);
  requireThat(exported.split === p.source.split && mapping.split === p.source.split, `${tag}: split mismatch`);
  requireThat(exported.source_groups?.includes(p.source.source_id) && exported.source_groups?.includes(p.source.source_group)
    && exported.source_ids?.includes(p.source.source_group), `${tag}: native provenance groups mismatch`);
  if (exported.split === 'train') train++; else if (exported.split === 'test') test++;
  sourceGroups.add(p.source.source_group);

  const sourceCase = sourceCaseById.get(p.source.source_id);
  requireThat(sourceCase, `${tag}: source case ID absent from pinned source-cases file`);
  requireThat(sourceCase.split === p.source.split && sourceCase.source_ids?.includes(p.source.source_group)
    && sourceCase.source_groups?.includes(p.source.source_group), `${tag}: source-case lineage/split mismatch`);

  let raw = resultCache.get(p.lineage.result_path);
  if (!raw) {
    const resultBytes = bytes(p.lineage.result_path);
    requireThat(sha(resultBytes) === p.lineage.result_sha256, `${tag}: raw result file pin mismatch`);
    raw = JSON.parse(resultBytes);
    resultCache.set(p.lineage.result_path, raw);
    materializedCache.set(p.lineage.result_path, materializeNativeRows([raw], { failedRuns: true }).turns);
  }
  const traceBytes = bytes(p.lineage.trace_path);
  requireThat(sha(traceBytes) === p.lineage.trace_sha256, `${tag}: trace file pin mismatch`);
  const program = raw.task?.program_ir;
  const folderFiles = program?.semantics?.folder_files;
  requireThat(folderFiles && Object.hasOwn(folderFiles, p.source.source_file), `${tag}: source missing from task.program_ir.semantics.folder_files`);
  requireThat(sameJson(folderFiles, sourceCase.semantics?.folder_files), `${tag}: raw/source-case folder_files differ`);
  const sourceText = folderFiles[p.source.source_file];
  requireThat(typeof sourceText === 'string', `${tag}: folder file is not exact text`);
  const sourceBytes = Buffer.from(sourceText, 'utf8');
  requireThat(sha(sourceBytes) === p.source.source_file_sha256 && sourceBytes.length === p.source.source_file_bytes
    && sourceText === p.source.source_file_text, `${tag}: exact folder-file bytes/text mismatch`);
  const sourceFact = JSON.parse(sourceText);
  const expectedReadChildren = sourceCase.curriculum?.reference?.children?.filter(child =>
    Array.isArray(child.expected_reads) && child.expected_reads.length === 1
    && child.expected_reads[0] === p.source.source_file) || [];
  requireThat(expectedReadChildren.length === 1, `${tag}: source reference must have one exact expected_reads child`);
  const expectedReturnCalls = (expectedReadChildren[0].calls || []).filter(call =>
    Array.isArray(call) && call[0] === 'return_result' && call[1]?.status === 'success');
  requireThat(expectedReturnCalls.length === 1, `${tag}: expected exactly one successful typed source return`);
  const expectedTypedFact = expectedReturnCalls[0][1].value;
  requireThat(expectedTypedFact?.candidateId === sourceFact.candidateId,
    `${tag}: source reference candidate ID differs from exact folder file`);
  if (p.selected_action.result_type === 'ScoreFact') {
    requireThat(Number.isFinite(sourceFact.score) && expectedTypedFact?.score === sourceFact.score,
      `${tag}: reference ScoreFact does not match source JSON score`);
  } else if (p.selected_action.result_type === 'EligibilityFact') {
    requireThat(typeof expectedTypedFact?.eligible === 'boolean' && expectedTypedFact.evidence === sourceFact.evidence,
      `${tag}: reference EligibilityFact differs from exact source file facts`);
  } else throw new Error(`${tag}: unsupported selected typed return ${p.selected_action.result_type}`);
  requireThat(sameJson(expectedTypedFact, p.selected_action.host_result_value),
    `${tag}: sampled typed return differs from the source reference child fact`);
  requireThat(sourceFact.candidateId && p.selected_action.host_result_value?.candidateId === sourceFact.candidateId,
    `${tag}: typed return candidate ID differs from folder file`);
  requireThat(p.selected_action.host_result_value?.evidence === sourceFact.evidence,
    `${tag}: typed return evidence differs from exact folder-file fact`);
  requireThat(p.selected_action.source_reference_match === true
    && sameJson(p.selected_action.host_result_value, p.selected_action.source_reference_value),
    `${tag}: host/reference typed result mismatch`);
  const ledgerEntry = raw.outcome?.invocation_ledger?.find(row => row.invocation_id === p.selected_action.invocation_id);
  requireThat(ledgerEntry?.host_result?.complete === true
    && sameJson(ledgerEntry.host_result.value, p.selected_action.host_result_value)
    && ledgerEntry.host_result.result_type === p.selected_action.result_type,
    `${tag}: raw typed host-result capture mismatch`);

  const sampledTurn = raw.trajectory?.[p.selected_action.turn_index];
  requireThat(sampledTurn?.invocation_id === p.selected_action.invocation_id,
    `${tag}: sampled turn index/invocation mismatch`);
  requireThat(sampledTurn.raw_response_sha256 === p.selected_action.provider_raw_response_sha256,
    `${tag}: sampled raw response digest mismatch`);
  requireThat(sampledTurn.request_sha256 === p.selected_action.provider_request_sha256,
    `${tag}: sampled request digest mismatch`);
  requireThat(sameJson(sampledTurn.context, p.provider_context) && sameJson(p.messages, p.provider_context),
    `${tag}: full saved request context differs from sampled turn`);
  requireThat(stableSha(sampledTurn.context) === p.selected_action.provider_context_sha256
    && stableSha(p.messages) === p.selected_action.native_messages_sha256,
    `${tag}: canonical sampled context/message hash mismatch`);
  requireThat(sameJson(p.target, exported.target), `${tag}: proposal target differs from native materialized target`);
  requireThat(stableSha(p.target) === p.selected_action.native_target_sha256,
    `${tag}: canonical native target hash mismatch`);
  const terminalAction = raw.outcome?.action_ledger?.find(action => action.call_id === p.selected_action.invocation_id
    && action.seq === p.selected_action.terminal_action_seq);
  requireThat(terminalAction && terminalAction.name === p.selected_action.terminal_tool
    && (p.selected_action.terminal_tool_call_id === null
      ? terminalAction.tool_call_id === undefined
      : terminalAction.tool_call_id === p.selected_action.terminal_tool_call_id),
    `${tag}: terminal action ledger linkage mismatch`);
  const targetCall = p.target.tool_calls?.[0];
  requireThat(targetCall?.function?.name === terminalAction.name
    && sameJson(JSON.parse(targetCall.function.arguments), terminalAction.arguments),
    `${tag}: sampled terminal action arguments differ from target`);
  const matches = materializedCache.get(p.lineage.result_path).filter(row =>
    row.source_ref?.invocation_id === p.selected_action.invocation_id
    && row.decision?.source_raw_response_sha256 === p.selected_action.provider_raw_response_sha256);
  requireThat(matches.length === 1, `${tag}: materializer match count ${matches.length}`);
  const untouched = matches[0];
  requireThat(untouched.id === exported.id && untouched.version === exported.version,
    `${tag}: native materializer ID/version differs from held export`);
  requireThat(jsonSha(untouched) === mapping.materializer_untouched_row_sha256,
    `${tag}: untouched materializer row hash mismatch`);
  const normalizedExport = structuredClone(exported);
  normalizedExport.training_admission = structuredClone(untouched.training_admission);
  normalizedExport.decision.training_approved = untouched.decision.training_approved;
  requireThat(sameJson(normalizedExport, untouched), `${tag}: export changed native row beyond held approval flags`);
  if (untouched.training_admission?.approved === true) nativeApprovedBeforeHold++;

  const proofList = exported.decision?.status_only_success_validation || [];
  for (const entry of proofList) {
    requireThat(entry.valid === true && entry.proof?.schema === 'natlang.status-only-success-proof/1'
      && entry.proof.basis === 'same-invocation-staged-result'
      && entry.proof.invocation_id === exported.source_ref.invocation_id
      && entry.proof.terminal_action_seq === p.selected_action.terminal_action_seq,
      `${tag}: invalid status-only proof linkage`);
    statusOnlyProofCount++;
  }
  requireThat(mapping.invocation_id === p.selected_action.invocation_id
    && mapping.provider_raw_response_sha256 === p.selected_action.provider_raw_response_sha256,
    `${tag}: ID map invocation/raw response mismatch`);
  const callRow = callStoreRows.find(row => row.call_id === p.lineage.callstore_call_id);
  requireThat(callRow && callRow.parent_call_id === p.lineage.callstore_parent_call_id
    && callRow.record_hash === p.lineage.callstore_record_hash
    && callRow.events_hash === p.lineage.callstore_events_hash,
    `${tag}: exact child CallStore record/events binding mismatch`);
  for (const [kind, hash] of [['record', p.lineage.callstore_record_hash], ['events', p.lineage.callstore_events_hash]]) {
    requireThat(callRow.call_blobs?.some(blob => blob.kind === kind && blob.hash === hash),
      `${tag}: CallStore ${kind} ref absent`);
    const blobPath = `${CALLSTORE_DIR}/blobs/${hash.slice(0, 2)}/${hash.slice(2)}`;
    requireThat(sha(bytes(blobPath)) === hash, `${tag}: CallStore ${kind} blob bytes/hash mismatch`);
  }
  checked.push({ proposal_id:p.id,native_id:exported.id,source_group:p.source.source_group,
    folder_file:p.source.source_file,typed_return_type:p.selected_action.result_type,
    invocation_id:p.selected_action.invocation_id,split:exported.split });
}
requireThat(proposalIds.size === 38 && nativeIds.size === 38, 'proposal/native ID bijection is not unique');
requireThat(statusOnlyProofCount === 1, `expected one status-only proof, found ${statusOnlyProofCount}`);
const proofRow = nativeRows.find(row => row.decision?.status_only_success_validation?.length);
const proof = proofRow.decision.status_only_success_validation.find(row => row.proof)?.proof;
requireThat(proof?.invocation_id === 'task-1-dzdjbx/14'
  && proof?.staged_output_sha256 === '92422bf33bc4f2efadc538ecd3132aa6829345dafb207be7c92620810d091e75',
  'status-only proof does not match the independently pinned one-row proof');

const incident = JSON.parse(text(INCIDENT_PATH));
requireThat(incident.status.includes('unrecoverable') && incident.current_file_sha256 === REVIEW_SHA,
  'review overwrite incident is not consistent with the current immutable review pin');
console.log(JSON.stringify({
  status:'verified', proposal_sha256:proposalSha, review_v2_sha256:sha(bytes(REVIEW_PATH)),
  native_rows_sha256:nativeSha, id_map_sha256:mapSha, row_count:checked.length,
  unique_proposal_ids:proposalIds.size, unique_native_ids:nativeIds.size,
  splits:{train,test}, distinct_source_groups:[...sourceGroups].sort(),
  source_result_files:resultCache.size, materializer_approved_before_hold:nativeApprovedBeforeHold,
  materializer_held_before_hold:checked.length - nativeApprovedBeforeHold,
  source_cases_sha256:sha(bytes(CASES_PATH)), callstore_calls_sha256:callStoreCallsSha,
  callstore_provenance_sha256:callStoreProvenanceSha,
  all_exports_held:true, status_only_proof_count:statusOnlyProofCount, status_only_proof:proof,
  incident_receipt_sha256:sha(bytes(INCIDENT_PATH)), checked
}, null, 2));
