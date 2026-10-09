#!/usr/bin/env node
/**
 * Build one explicitly derived writer row from a root-admitted typed-text
 * supervision target. The original sampled action remains nested as provenance;
 * no trajectory or runtime qualification is inferred.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { verifyPinnedCodeClosure } from './verified_code_closure.mjs';

const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const hashJson = value => sha(Buffer.from(JSON.stringify(value)));
function parseArgs(argv) {
  const out = {};
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i];
    if (!key?.startsWith('--') || !argv[i + 1] || key in out) throw new Error(`invalid argument near ${key}`);
    out[key.slice(2)] = argv[i + 1];
  }
  for (const key of ['candidate', 'admission', 'result', 'materializer', 'code-manifest', 'output'])
    if (!out[key]) throw new Error(`missing --${key}`);
  return out;
}
const options = parseArgs(process.argv);
const paths = Object.fromEntries(Object.entries(options).map(([k, v]) => [k, resolve(v)]));
const readJson = path => JSON.parse(readFileSync(path));
const candidateBytes = readFileSync(paths.candidate), candidate = JSON.parse(candidateBytes);
const admissionBytes = readFileSync(paths.admission), admission = JSON.parse(admissionBytes);
const rawBytes = readFileSync(paths.result), raw = JSON.parse(rawBytes);
const manifestBytes = readFileSync(paths['code-manifest']);
const verifiedClosure = verifyPinnedCodeClosure(paths['code-manifest'], paths.materializer);
if (candidate.schema !== 'natlang.derived-observed-text-writer-candidate/1' ||
    candidate.conversion_provenance?.verified_code_closure?.manifest_sha256 !== sha(manifestBytes) ||
    candidate.conversion_provenance?.verified_code_closure?.artifact?.sha256 !== sha(readFileSync(paths.materializer)) ||
    candidate.conversion_provenance?.verified_code_closure?.closure_sha256 !== verifiedClosure.closure_sha256)
  throw new Error('candidate does not bind the current verified executable closure');
if (candidate.source_result.sha256 !== sha(rawBytes)) throw new Error('raw source result does not match candidate pin');
if (admission.schema !== 'natlang.root-derived-observed-text-writer-admission/1' ||
    admission.decision !== 'admit-derived-observed-text-writer-body' || admission.training_admission !== true ||
    admission.original_action_admission !== false || admission.limits?.whole_trajectory_admission !== false ||
    admission.limits?.runtime_gradient_qualification !== false || admission.limits?.active_training_inputs_changed !== false)
  throw new Error('root receipt does not authorize only the derived writer body');
for (const [rel, pin] of Object.entries(admission.input_pins ?? {})) {
  const path = resolve(process.cwd(), rel);
  if (!pin || !/^[a-f0-9]{64}$/.test(pin.sha256) || sha(readFileSync(path)) !== pin.sha256 ||
      readFileSync(path).length !== pin.bytes)
    throw new Error(`root admission input pin mismatch: ${rel}`);
}
const supervision = candidate.derived_supervision;
if (!supervision || supervision.role !== 'derived_sft_target_not_original_assistant_action' ||
    supervision.training_admission !== false || supervision.trajectory_admission !== false ||
    supervision.original_eval_hidden_states_equivalent !== false ||
    supervision.runtime_gradient_qualification !== false ||
    hashJson(supervision.target) !== admission.derived_target_sha256 ||
    candidate.proposal_id !== admission.proposal_id ||
    candidate.source_result.native_record_id !== admission.source_native_record_id ||
    candidate.source_result.split !== admission.split ||
    !candidate.source_result.source_groups?.includes(admission.source_group) ||
    candidate.exact_body_sha256 !== admission.exact_body_sha256)
  throw new Error('candidate target/source bindings disagree with the root approval');

const materializer = await import(pathToFileURL(paths.materializer).href);
if (typeof materializer.materializeNativeRows !== 'function') throw new Error('pinned materializer lacks materializeNativeRows');
const nativeRows = materializer.materializeNativeRows([raw], { failedRuns: true }).turns;
const matches = nativeRows.filter(row => row.id === candidate.source_result.native_record_id &&
  row.source_ref?.invocation_id === candidate.source_result.invocation_id &&
  row.decision?.index === candidate.source_result.decision_index &&
  row.decision?.source_raw_response_sha256 === candidate.source_result.raw_response_sha256);
if (matches.length !== 1) throw new Error(`expected one exact source native row; got ${matches.length}`);
const native = matches[0];
if (hashJson(native.target) !== candidate.source_result.native_target_sha256 ||
    hashJson(native.messages) !== candidate.source_result.native_messages_sha256 ||
    native.source_ref.source_row_sha256 !== candidate.source_result.source_row_sha256)
  throw new Error('rematerialized native target/context differs from the candidate');

const row = structuredClone(native);
row.id = candidate.proposal_id;
row.target = structuredClone(supervision.target);
row.training_admission = {
  kind: 'root-derived-text-writer-body-admission', approved: true,
  receipt_sha256: sha(admissionBytes), exact_target_sha256: admission.derived_target_sha256
};
row.trace_admission = {
  admitted: false, kind: 'derived-writer-target-only',
  reason: 'root admission covers this derived typed-text writer target only; no whole trajectory or recurrence admission'
};
row.review_disposition = 'admitted-derived-text-writer-body-only';
row.source_trace_provenance = {
  original_native_trace_admission: structuredClone(native.trace_admission ?? null),
  original_parent_outcome: structuredClone(native.outcome ?? null),
  original_native_action_training_admission: false,
  scope: 'original raw action/result/context are provenance; the admitted target is the separately labeled derived typed-text writer target'
};
row.derived_target = {
  schema: 'natlang.root-admitted-derived-text-writer-target/1',
  proposal_id: candidate.proposal_id,
  transformation: supervision.transformation,
  derivation_role: supervision.role,
  exact_body_sha256: candidate.exact_body_sha256,
  target_sha256: admission.derived_target_sha256,
  original_native_record_id: native.id,
  original_target: structuredClone(native.target),
  original_target_sha256: hashJson(native.target),
  original_messages_sha256: hashJson(native.messages),
  root_admission_path: paths.admission,
  root_admission_sha256: sha(admissionBytes),
  conversion_candidate_path: paths.candidate,
  conversion_candidate_sha256: sha(candidateBytes),
  converter_path: candidate.conversion_provenance.converter_path,
  converter_sha256: candidate.conversion_provenance.converter_sha256,
  closure_verifier_path: candidate.conversion_provenance.closure_verifier_path,
  closure_verifier_sha256: candidate.conversion_provenance.closure_verifier_sha256,
  verified_code_closure: verifiedClosure,
  original_eval_hidden_states_equivalent: false,
  recurrence_admission: false,
  runtime_gradient_qualification: false
};
row.provenance = { ...row.provenance,
  derived_supervision: 'root-admitted derived text writer body; original eval action remains held' };
writeFileSync(paths.output, JSON.stringify(row, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ output: paths.output, sha256: sha(readFileSync(paths.output)),
  id: row.id, source_native_record_id: native.id, target_sha256: hashJson(row.target),
  body_sha256: candidate.exact_body_sha256, split: row.split, source_groups: row.source_groups,
  closure_files_verified: verifiedClosure.closure_files_verified,
  training_admission: row.training_admission, trace_admission: row.trace_admission }, null, 2));
