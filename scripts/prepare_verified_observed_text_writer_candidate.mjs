#!/usr/bin/env node
/**
 * Prepare a held derived text-writer target from an authenticated Neuralese<string>
 * runtime output and its same-run downstream read receipts. The original sampled
 * assistant target and messages are preserved verbatim. The derived target is a
 * separate proposal; this script never approves training or trajectory admission.
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { verifyPinnedCodeClosure } from './verified_code_closure.mjs';

function args(argv) {
  const values = {};
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i];
    if (!key?.startsWith('--') || !argv[i + 1] || key in values) throw new Error(`invalid argument near ${key}`);
    values[key.slice(2)] = argv[i + 1];
  }
  for (const key of ['result', 'materializer', 'code-manifest', 'writer-invocation', 'decision-index', 'output'])
    if (!values[key]) throw new Error(`missing --${key}`);
  if (!/^\d+$/.test(values['decision-index'])) throw new Error('--decision-index must be a nonnegative integer');
  return values;
}

const options = args(process.argv);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const bytes = value => Buffer.from(JSON.stringify(value));
const digest = value => sha(bytes(value));
const resultPath = resolve(options.result), outputPath = resolve(options.output);
const modulePath = pathToFileURL(resolve(options.materializer));
const manifestPath = resolve(options['code-manifest']);
const manifestBytes = readFileSync(manifestPath), manifest = JSON.parse(manifestBytes);
const materializerBytes = readFileSync(resolve(options.materializer));
const materializerSha = sha(materializerBytes);
const verifiedClosure = verifyPinnedCodeClosure(manifestPath, options.materializer);
if (verifiedClosure.artifact.sha256 !== materializerSha)
  throw new Error('code manifest does not pin the exact materializer module');
const { materializeNativeRows } = await import(modulePath.href);
const rawBytes = readFileSync(resultPath), raw = JSON.parse(rawBytes);
const rows = materializeNativeRows([raw], { failedRuns: true }).turns;
const writer = rows.filter(row => row.source_ref?.invocation_id === options['writer-invocation'] &&
  row.decision?.index === Number(options['decision-index']));
if (writer.length !== 1) throw new Error(`expected one exact materialized writer row; found ${writer.length}`);
const source = writer[0];
const parentInvocation = source.source_ref.invocation_id;
const writerLedgerEntries = (raw.outcome?.invocation_ledger ?? []).filter(item =>
  item.invocation_id === source.source_ref.invocation_id);
if (writerLedgerEntries.length !== 1) throw new Error('writer invocation ledger entry is not unique');
const writerLedger = writerLedgerEntries[0];
const capture = source.source_ref?.host_result_capture?.capture;
const blockRef = capture?.value?.$neuralese;
const invocationEntries = (raw.outcome?.invocation_ledger ?? []).filter(item =>
  item.invocation_id === source.source_ref.invocation_id);
const hostCapture = invocationEntries[0]?.host_result;
if (!capture || capture.kind !== 'host_capture' || capture.complete !== true ||
    capture.call_id !== source.source_ref.invocation_id || capture.result_type !== 'Neuralese<string>' ||
    blockRef?.type !== 'Neuralese<string>' || typeof blockRef.id !== 'string' ||
    invocationEntries.length !== 1 || hostCapture?.kind !== 'host_capture' || hostCapture.complete !== true ||
    hostCapture.call_id !== capture.call_id || hostCapture.value_sha256 !== capture.value_sha256 ||
    hostCapture.terminal_action_seq !== capture.terminal_action_seq)
  throw new Error('writer host capture does not bind a completed Neuralese<string> block');
const terminal = source.decision?.assistant?.calls?.filter(call =>
  call.outcome?.trace_seq === capture.terminal_action_seq);
if (!Array.isArray(terminal) || terminal.length !== 1) throw new Error('native terminal action does not match host capture');
const graph = Array.isArray(raw.outcome?.execution_graph) ? raw.outcome.execution_graph : [];
const writerEvents = graph.filter(event => event.kind === 'block_write' &&
  event.call_id === source.source_ref.invocation_id && event.block === blockRef.id &&
  event.source === 'eval-finish' && event.result_type === 'Neuralese<string>' && event.truncated === false);
if (writerEvents.length !== 1) throw new Error(`expected one exact eval-finish graph writer; found ${writerEvents.length}`);
const event = writerEvents[0];
const readerRows = [];
for (const row of rows) {
  for (const receipt of row.source_ref?.provider_expanded_read_contexts ?? []) {
    const producer = receipt.producer_write ?? {};
    const block = receipt.block ?? {};
    if (receipt.schema !== 'natlang.provider-expanded-read-context/2' ||
        receipt.origin !== 'same-run-producer' || receipt.writer_target_selected !== false ||
        producer.call_id !== source.source_ref.invocation_id || producer.node !== event.node ||
        block.id !== blockRef.id || block.type !== 'Neuralese<string>' ||
        typeof block.body !== 'string' || sha(Buffer.from(block.body)) !== event.text_body_sha256 ||
        producer.text_body_sha256 !== event.text_body_sha256 || producer.source !== 'eval-finish') continue;
    const read = receipt.block_read, turn = receipt.model_turn;
    if (!read || !turn || read.call_id !== row.source_ref.invocation_id || turn.call_id !== row.source_ref.invocation_id ||
        !read.inputs?.some(input => input.node === event.node && input.block === blockRef.id) ||
        !turn.inputs?.some(input => input.node === read.node && input.block === blockRef.id))
      throw new Error(`incomplete producer-to-reader graph for ${row.source_ref.invocation_id}`);
    readerRows.push({ row, receipt });
  }
}
if (readerRows.length === 0) throw new Error('no authenticated downstream provider read found');
const body = readerRows[0].receipt.block.body;
if (readerRows.some(({ receipt }) => receipt.block.body !== body)) throw new Error('read receipts disagree on body');
const turns = raw.trajectory.map((turn, index) => ({ turn, index })).filter(({ turn }) =>
  turn.invocation_id === source.source_ref.invocation_id &&
  turn.raw_response_sha256 === source.decision.source_raw_response_sha256);
if (turns.length !== 1) throw new Error(`expected one raw sampled writer response; found ${turns.length}`);
const { turn, index } = turns[0];
const sameRunScoreFacts = (raw.outcome.invocation_ledger ?? []).filter(item =>
  item.parent_invocation_id === source.source_ref.parent_invocation_id &&
  item.inline_instruction_site?.returns?.natlang === 'ScoreFact' &&
  item.host_result?.result_type === 'ScoreFact' && item.completion_status === 'done')
  .map(item => {
    const nativeFact = rows.find(row => row.source_ref?.invocation_id === item.invocation_id &&
      row.decision?.source_raw_response_sha256);
    const factTurn = raw.trajectory.find(t => t.invocation_id === item.invocation_id &&
      t.raw_response_sha256 === nativeFact?.decision?.source_raw_response_sha256);
    return { invocation_id: item.invocation_id, value: item.host_result.value,
      host_result_value_sha256: item.host_result.value_sha256,
      native_record_id: nativeFact?.id ?? null,
      request_sha256: factTurn?.request_sha256 ?? null,
      raw_response_sha256: factTurn?.raw_response_sha256 ?? null,
      context_sha256: factTurn ? digest(factTurn.context) : null };
  });
const callId = `derived_${sha(Buffer.from(`${source.id}\0${event.node}\0${event.text_body_sha256}`)).slice(0, 24)}`;
const derivedTarget = { role: 'assistant', content: '', tool_calls: [{ id: callId, type: 'function',
  function: { name: 'return_result', arguments: JSON.stringify({ status: 'success', value: body }) } }] };
const candidate = {
  schema: 'natlang.derived-observed-text-writer-candidate/1',
  proposal_id: `derived-writer:${source.id}:${event.text_body_sha256.slice(0, 16)}`,
  conversion_provenance: {
    converter_path: resolve(process.argv[1]),
    converter_sha256: sha(readFileSync(resolve(process.argv[1]))),
    closure_verifier_path: resolve(new URL('./verified_code_closure.mjs', import.meta.url).pathname),
    closure_verifier_sha256: sha(readFileSync(new URL('./verified_code_closure.mjs', import.meta.url))),
    verified_code_closure: verifiedClosure
  },
  source_result: { path: resultPath, sha256: sha(rawBytes), trace_sha256: source.provenance.trace_sha256,
    code_manifest_path: manifestPath, code_manifest_sha256: sha(manifestBytes),
    materializer_module_path: resolve(options.materializer), materializer_module_sha256: materializerSha,
    trajectory_index: index, invocation_id: turn.invocation_id, request_sha256: turn.request_sha256,
    raw_response_sha256: turn.raw_response_sha256, raw_model_action: turn.model_response.calls,
    native_record_id: source.id, decision_index: source.decision.index,
    native_target: source.target, native_target_sha256: digest(source.target),
    native_messages: source.messages, native_messages_sha256: digest(source.messages),
    source_row_sha256: source.source_ref.source_row_sha256, split: source.split,
    source_ids: source.source_ids, source_groups: source.source_groups,
    inline_instruction_site: writerLedger.inline_instruction_site,
    raw_sampled_context: turn.context, raw_sampled_context_sha256: digest(turn.context),
    terminal_host_capture: capture, terminal_native_action: terminal[0] },
  delegated_typed_score_fact_returns: sameRunScoreFacts,
  writer_graph_event: event,
  exact_body: body,
  exact_body_sha256: sha(Buffer.from(body)),
  downstream_reader_bindings: readerRows.map(({ row, receipt }) => ({
    invocation_id: receipt.invocation_id, native_record_id: row.id,
    source_action_target_sha256: receipt.source_action_target_sha256,
    source_response_sha256: receipt.source_response_sha256,
    read_node: receipt.block_read.node, read_inputs: receipt.block_read.inputs,
    model_turn_node: receipt.model_turn.node, model_turn_inputs: receipt.model_turn.inputs })),
  derived_supervision: { role: 'derived_sft_target_not_original_assistant_action',
    transformation: 'authenticated-observed-eval-finish-Neuralese-string-to-direct-return-result/1',
    original_target: source.target, original_target_sha256: digest(source.target),
    original_messages: source.messages, original_messages_sha256: digest(source.messages),
    target: derivedTarget, target_sha256: digest(derivedTarget), derived_call_id: callId,
    training_admission: false, trajectory_admission: false, learned_vectors: false,
    qualification_certificate: false, original_eval_hidden_states_equivalent: false,
    runtime_gradient_qualification: false },
  status: 'held-for-independent-per-body-review'
};
writeFileSync(outputPath, JSON.stringify(candidate, null, 2) + '\n', { flag: 'wx' });
console.log(JSON.stringify({ output: outputPath, sha256: sha(Buffer.from(JSON.stringify(candidate, null, 2) + '\n')),
  source_record: source.id, body_sha256: candidate.exact_body_sha256, readers: readerRows.length }));
