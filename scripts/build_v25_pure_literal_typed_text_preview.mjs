#!/usr/bin/env node
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const outDir = resolve(root, process.argv[2] ?? 'runs/neuralese-v25-pure-literal-derived-text-preview-20261008-v2');
const materializerPath = process.env.NATLANG_TS_HOST_MATERIALIZER ?? resolve(root, 'ts-host/dist/teacher/native-materializer.js');
const { derivePureLiteralTypedTextTarget } = await import(pathToFileURL(resolve(materializerPath)).href);
const conversionPath = process.env.NATLANG_TS_HOST_CONVERSION ?? resolve(root, 'ts-host/dist/compiler/neuralese-conversion.js');
const { pureLiteralEvalReturn } = await import(pathToFileURL(resolve(conversionPath)).href);
const inputPaths = {
  nativePreview: 'runs/neuralese-v25-actual-action-preview-20261008-v3/native-preview.jsonl',
  roleAudit: 'runs/neuralese-v25-target-output-role-audit-20261008-v1/roles.jsonl',
  sourceReview: 'runs/neuralese-v25-source-action-review-20261008-v8/review.json',
};
const digest = value => createHash('sha256').update(value).digest('hex');
const rows = async path => (await readFile(resolve(root, path), 'utf8')).trim().split('\n').filter(Boolean).map(line => JSON.parse(line));
const [nativeRows, roleRows, sourceReview] = await Promise.all([
  rows(inputPaths.nativePreview), rows(inputPaths.roleAudit), readFile(resolve(root, inputPaths.sourceReview), 'utf8').then(JSON.parse),
]);
const nativeByEvent = new Map();
for (const row of nativeRows) {
  const event = row.preview_source_selection?.target_generation_turn;
  if (!event) continue;
  nativeByEvent.set(`${row.family ?? row.task_family ?? ''}\0${event.invocation_id}\0${event.request_sha256}\0${event.raw_response_sha256}`, row);
}
const reviewByEvent = new Map(sourceReview.selected_action_recommendations.map(item =>
  [`${item.case_id}\0${item.request_sha256}\0${item.raw_response_sha256}`, item]));
const sourceReviewSha = digest(await readFile(resolve(root, inputPaths.sourceReview)));
const roleSha = digest(await readFile(resolve(root, inputPaths.roleAudit)));
const nativePreviewSha = digest(await readFile(resolve(root, inputPaths.nativePreview)));
const sourceResultFileSha = new Map();
for (const item of roleRows) sourceResultFileSha.set(item.source_result_path, item.source_result_file_sha256);

const literalRoles = roleRows.filter(item => item.split === 'train' &&
  item.target_role_class === 'literal_output_body_in_selected_eval_code');
const pureLiteralRoles = literalRoles.filter(item => pureLiteralEvalReturn(item.target_code));
const selected = pureLiteralRoles.filter(item => item.typed_output.writer_source === 'eval-finish');
const nonterminalLiteralHolds = pureLiteralRoles.filter(item => item.typed_output.writer_source !== 'eval-finish').map(item => ({
  case_id: item.case_id, invocation_id: item.generation_event.invocation_id,
  reason: 'eval-return only stages a value; later completion actions could intervene, so no terminal equivalence is asserted',
}));
const effectfulLiteralHolds = literalRoles.filter(item => !pureLiteralEvalReturn(item.target_code)).map(item => ({
  case_id: item.case_id, invocation_id: item.generation_event.invocation_id,
  reason: 'selected eval executes an effect before returning its literal; no derived typed-text target',
}));
const transformed = [];
const selection = [];
const holds = [];
for (const role of selected) {
  const event = role.generation_event;
  const review = reviewByEvent.get(`${role.case_id}\0${event.request_sha256}\0${event.raw_response_sha256}`);
  if (!review || review.target_body_sha256 !== role.typed_output.body_sha256 ||
      review.writer_block_id !== role.typed_output.writer_block_id || review.writer_node !== role.typed_output.writer_node ||
      review.writer_call_id !== event.invocation_id || review.split !== role.split ||
      !['source_grounded_phase_candidate',
        'source_grounded_main_chain_candidate; contradictory sibling branches held'].includes(review.candidate_disposition)) {
    holds.push({ case_id: role.case_id, invocation_id: event.invocation_id, reason: 'missing or mismatched source action review binding' });
    continue;
  }
  const row = nativeByEvent.get(`${role.case_id}\0${event.invocation_id}\0${event.request_sha256}\0${event.raw_response_sha256}`) ??
    nativeRows.find(candidate => candidate.preview_source_selection?.target_generation_turn?.invocation_id === event.invocation_id &&
      candidate.preview_source_selection?.target_generation_turn?.request_sha256 === event.request_sha256 &&
      candidate.preview_source_selection?.target_generation_turn?.raw_response_sha256 === event.raw_response_sha256);
  if (!row) {
    holds.push({ case_id: role.case_id, invocation_id: event.invocation_id, reason: 'no exact materialized row for selected generation turn' });
    continue;
  }
  const witness = {
    trajectory_id: row.source_ref.trajectory_id,
    source_row_sha256: row.source_ref.source_row_sha256,
    source_result_row_sha256: role.source_result_row_sha256,
    generation_turn: event,
    target_code_sha256: role.target_code_sha256,
    writer_call_id: event.invocation_id,
    writer_node: role.typed_output.writer_node,
    block_id: role.typed_output.writer_block_id,
    body_source: role.typed_output.body_text,
    body_sha256: role.typed_output.body_sha256,
    result_type: role.typed_output.result_type,
    source: role.typed_output.writer_source,
    marker_context: 'return-result',
    source_kind: 'typed-text-result',
  };
  const derived = derivePureLiteralTypedTextTarget(row, witness);
  if (!derived) {
    holds.push({ case_id: role.case_id, invocation_id: event.invocation_id,
      reason: 'strict pure-literal transform or authenticated typed-write binding rejected the row' });
    continue;
  }
  const inputText = await readFile(resolve(root, role.source_result_path), 'utf8');
  const inputDigest = digest(inputText);
  if (inputDigest !== role.source_result_file_sha256 || sourceResultFileSha.get(role.source_result_path) !== inputDigest) {
    holds.push({ case_id: role.case_id, invocation_id: event.invocation_id, reason: 'source-result file hash mismatch' });
    continue;
  }
  transformed.push(derived);
  selection.push({
    schema: 'natlang.v25-pure-literal-derived-target-receipt/1',
    derived_row_id: derived.id,
    source_row_id: row.id,
    case_id: role.case_id,
    split: role.split,
    source_group: role.source_group,
    source_result_path: role.source_result_path,
    source_result_file_sha256: inputDigest,
    source_result_row_sha256: role.source_result_row_sha256,
    source_action_review_sha256: sourceReviewSha,
    source_action_review: {
      request_sha256: review.request_sha256,
      raw_response_sha256: review.raw_response_sha256,
      pass_file: review.pass_file,
      target_body_sha256: review.target_body_sha256,
      writer_call_id: review.writer_call_id,
      writer_node: review.writer_node,
      writer_block_id: review.writer_block_id,
      on_selected_output_dependency_path: review.on_selected_output_dependency_path,
      candidate_disposition: review.candidate_disposition,
    },
    source_visible_fact_review: {
      basis: 'exact-selected-request-history-only',
      note: role.case_id === 'FISH-731' && event.invocation_id.endsWith('/16')
        ? 'The exact request contains the prior metrics and the pass-03 condition source used in the literal.'
        : role.case_id === 'VAX-721' && event.invocation_id.endsWith('/7')
          ? 'The exact request contains prior metrics and pass-03 conditions used in the literal.'
          : 'The literal restates the request identity, rules, placeholder, or explicit no-evidence status present in this request.' ,
      later_context_added: false,
      effects_executed_by_selected_eval: false,
    },
    native_preview_sha256: nativePreviewSha,
    role_audit_sha256: roleSha,
    original_target_sha256: derived.derived_target.original_target_sha256,
    derived_target_sha256: derived.derived_target.derived_target_sha256,
  });
}

if (transformed.length !== 6) throw new Error(`Expected six source-reviewed terminal pure literals, transformed ${transformed.length}; holds=${JSON.stringify(holds)}`);
await mkdir(outDir, { recursive: true });
const outputs = {
  'derived-rows.jsonl': transformed.map(row => JSON.stringify(row)).join('\n') + '\n',
  'selection-receipts.jsonl': selection.map(item => JSON.stringify(item)).join('\n') + '\n',
  'README.md': `# V25 pure-literal derived typed-text preview\n\nHeld review-only transformation of six train-split pure terminal eval-finish literals. Every original eval target remains preserved in its source row and is separately hash-bound. The transformed target is an equivalent typed \`return_result\` call, not the original assistant action and not a runtime-gradient qualification.\n\nThe transform accepts only one immutable string literal or no-substitution template literal followed by \`return sameBinding\`, backed by one exact terminal \`eval-finish\` typed-result receipt. It rejects reads, calls, branches, interpolation, mutation, extra statements, mismatched graph writes, and nonmatching request/response bindings. Exact request history is unchanged.\n\nThe pure VAX-720 \`eval-return\` stage is held because it is not terminal; later completion actions could intervene. Two other train literals execute \`source.readText()\` and are also excluded. Source action and visible-fact checks are recorded per row in \`selection-receipts.jsonl\`. All other V25 role classes remain untransformed. No training admission is granted.\n`,
};
for (const [name, content] of Object.entries(outputs)) await writeFile(resolve(outDir, name), content);
const manifest = {
  schema: 'natlang.immutable-held-preview-manifest/1',
  corpus_id: basename(outDir),
  admission: 'held-derived-target-preview-no-training-admission',
  inputs: Object.fromEntries(Object.entries(inputPaths).map(([key, path]) => [key, { path, sha256:
    key === 'nativePreview' ? nativePreviewSha : key === 'roleAudit' ? roleSha : sourceReviewSha }])),
  counts: { train_literal_output_roles: literalRoles.length, pure_literal_candidates: pureLiteralRoles.length,
    terminal_pure_literal_candidates: selected.length, transformed: transformed.length,
    held: holds.length + effectfulLiteralHolds.length + nonterminalLiteralHolds.length,
    train: transformed.filter(row => row.split === 'train').length, test: 0 },
  output_files: Object.fromEntries(Object.entries(outputs).map(([name, content]) => [name,
    { bytes: Buffer.byteLength(content), sha256: digest(content) }])),
  holds: [...holds, ...effectfulLiteralHolds, ...nonterminalLiteralHolds],
  supersedes: 'neuralese-v25-pure-literal-derived-text-preview-20261008-v1',
  transform: 'pure-terminal-eval-finish-to-typed-return/2',
};
await writeFile(resolve(outDir, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ outDir, manifest: resolve(outDir, 'manifest.json'), counts: manifest.counts,
  manifest_sha256: digest(await readFile(resolve(outDir, 'manifest.json'))), holds }, null, 2));
