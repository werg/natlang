import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { signatureHasExactArgumentPath, signatureHasExactParameter, validateSoftStateEdge } from './soft-state-proof.mjs';
import { materializeNativeRows, nativeRowDigest } from '../../dist/teacher/native-materializer.js';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const fail = message => { throw new Error(`soft-state evidence rejected: ${message}`); };

export function materializedActionProjection(row) {
  const firstUser = row.messages?.find(message => message.role === 'user');
  const scopeIndex = row.messages?.findIndex(message => message.role === 'assistant' &&
    message.tool_calls?.some(call => call.id === 'scope_0' && call.function?.name === 'eval')) ?? -1;
  const scopeCall = scopeIndex >= 0 ? row.messages[scopeIndex] : undefined;
  const scopeResult = scopeIndex >= 0 ? row.messages.find((message, index) => index > scopeIndex &&
    message.role === 'tool' && message.tool_call_id === 'scope_0') : undefined;
  return { id: row.id, source_ref: row.source_ref, outcome: row.outcome, training_admission: row.training_admission,
    provenance: row.provenance, task: row.task, split: row.split, source_groups: row.source_groups,
    decision: row.decision, target: row.target,
    messages: row.messages?.filter(message => message === firstUser || message === scopeCall || message === scopeResult) ?? [] };
}

const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}` : JSON.stringify(value);

function readSingleJsonLine(bytes, label) {
  const lines = bytes.toString('utf8').split(/\r?\n/).filter(line => line.trim());
  if (lines.length !== 1) fail(`${label} must contain exactly one JSON record`);
  try { return JSON.parse(lines[0]); } catch { fail(`${label} is not valid JSON`); }
}

function hostValue(result, callId) {
  const entries = (result.outcome?.invocation_ledger ?? []).filter(item => item.invocation_id === callId);
  if (entries.length !== 1) fail(`expected one invocation ledger entry for ${callId}`);
  const capture = entries[0].host_result;
  if (capture?.capture_kind !== 'invocation_output' || capture.complete !== true ||
      capture.origin !== 'observed-host-result; not a model-generated writer target')
    fail(`no complete observed typed host output for ${callId}`);
  return capture.value;
}

/** Recover only the exact complete text body that the model action returned. */
export function writerActionBody(row, expectedBodySha256, { allowStagedEvalCode = false } = {}) {
  const calls = row?.target?.tool_calls ?? [];
  const bodies = [];
  for (const call of calls) {
    if (call?.function?.name === 'return_result') {
      try {
        const args = JSON.parse(call.function.arguments), value = args?.value;
        const open = '<|neuralese|>', close = '<|/neuralese|>';
        if (args?.status === 'success' && typeof value === 'string' && value.startsWith(open) && value.endsWith(close)) {
          const body = value.slice(open.length, -close.length);
          if (sha256(Buffer.from(body, 'utf8')) === expectedBodySha256) bodies.push(body);
        }
      } catch { /* invalid marker is not a writer target */ }
    } else if (call?.function?.name === 'eval') {
      try {
        const args = JSON.parse(call.function.arguments), code = args?.code;
        // A typed value can be staged by a nonterminal eval and returned later.
        // The caller may accept that form only after independently validating
        // the exact eval-code block_write, typed host reference, and reader edge.
        if ((args?.finish !== true && !allowStagedEvalCode) || typeof code !== 'string') continue;
        const markers = [...code.matchAll(/<\|neuralese\|>([\s\S]*?)<\|\/neuralese\|>/g)];
        if (markers.length !== 1) continue;
        const body = markers[0][1];
        if (typeof body === 'string' && sha256(Buffer.from(body, 'utf8')) === expectedBodySha256) bodies.push(body);
      } catch { /* invalid source code is not a writer target */ }
    }
  }
  return bodies.length === 1 ? bodies[0] : undefined;
}

function actionForCall(rows, callId, trajectoryId, sourceRowSha, role, edge, bodySha) {
  let matches = rows.filter(row => row?.source_ref?.trajectory_id === trajectoryId &&
    row.source_ref.source_row_sha256 === sourceRowSha && row.source_ref.invocation_id === callId &&
    row.decision?.training_approved === true && row.decision?.failed_action === false);
  if (role === 'writer') {
    matches = matches.filter(row => writerActionBody(row, bodySha,
      { allowStagedEvalCode: edge?.marker_context === 'eval-code' }) !== undefined);
  } else {
    matches = matches.filter(row => {
      const opening = row.messages?.find(message => message.role === 'user')?.content;
      const signature = typeof opening === 'string' ? /^You are inside this call: ([^\n]+)/.exec(opening)?.[1] ?? '' : '';
      const scope = row.messages?.find(message => message.role === 'tool' && message.tool_call_id === 'scope_0');
      const blocks = Array.isArray(scope?.content) ? scope.content.filter(part => part?.type === 'neuralese' && part.id === edge.block_id) : [];
      return signatureHasExactArgumentPath(signature, edge.consumer_argument, 'Neuralese<string>') && blocks.length === 1;
    }).sort((a, b) => (a.decision?.index ?? Infinity) - (b.decision?.index ?? Infinity));
  }
  if (!matches.length || (role === 'writer' && matches.length !== 1))
    fail(`expected ${role === 'writer' ? 'one approved nonfailed producer target' : 'an approved nonfailed exact typed consumer'} row for invocation ${callId}, found ${matches.length}`);
  const row = matches[0];
  const index = row.decision?.index;
  const idIndex = /:decision:(\d+)$/.exec(String(row.id ?? ''));
  if (!Number.isSafeInteger(index) || !idIndex || Number(idIndex[1]) !== index)
    fail(`materialized row ID/index mismatch for ${callId}`);
  return { row, index };
}

function markerBody(row, blockId, bodySha, markerContext) {
  const body = writerActionBody(row, bodySha, { allowStagedEvalCode: markerContext === 'eval-code' });
  if (body === undefined) fail(`producer ${row.id} lacks one exact complete text Neuralese writer action`);
  if (!body.length && !blockId) fail(`empty producer marker for ${row.id}`);
  return body;
}

function providerExpansion(result, readerCallId, blockId) {
  const blocks = [];
  for (const turn of result.trajectory ?? []) {
    if (turn.invocation_id !== readerCallId) continue;
    const transport = turn.model_response?.transport_provenance;
    if (!transport) continue;
    if (transport.version !== 'text-marker-standin/2' || transport.learned_vectors !== false ||
        transport.qualification_certificate !== false || transport.training_admission !== false)
      fail(`reader ${readerCallId} has unqualified or unexpected provider transport provenance`);
    for (const block of transport.expanded_input_blocks ?? []) if (block.id === blockId) blocks.push(block);
  }
  if (!blocks.length) fail(`no provider-visible expansion for ${blockId} at reader ${readerCallId}`);
  const first = blocks[0];
  if ((first.type !== 'Neuralese<string>' && first.type !== null) || first.learned_vectors !== false || typeof first.body !== 'string' ||
      sha256(Buffer.from(first.body, 'utf8')) !== first.body_sha256)
    fail(`invalid provider-visible body expansion for ${blockId}`);
  if (blocks.some(block => block.type !== first.type || block.body !== first.body ||
      block.body_sha256 !== first.body_sha256 || block.learned_vectors !== false))
    fail(`provider-visible expansions disagree for ${blockId}`);
  // Historical transports omitted the label for typed eval-source writes. Keep
  // that raw provider fact intact; the caller may infer the stored type only
  // from the independently validated typed writer/reader graph and matching body.
  return first;
}

function providerMarkerOutput(result, writerCallId, bodySha) {
  const outputs = [];
  for (const turn of result.trajectory ?? []) {
    if (turn.invocation_id !== writerCallId) continue;
    const transport = turn.model_response?.transport_provenance;
    if (!transport) continue;
    if (transport.version !== 'text-marker-standin/2' || transport.learned_vectors !== false ||
        transport.qualification_certificate !== false || transport.training_admission !== false)
      fail(`writer ${writerCallId} has unqualified or unexpected provider transport provenance`);
    outputs.push(...(transport.marker_outputs ?? []).filter(item => item.body_sha256 === bodySha));
  }
  if (!outputs.length) fail(`no provider marker output matches graph writer body ${bodySha}`);
}

function checkReaderAction(row, edge) {
  const opening = row.messages?.find(message => message.role === 'user')?.content;
  if (typeof opening !== 'string') fail(`reader ${row.id} lacks a string call opening`);
  const signature = /^You are inside this call: ([^\n]+)/.exec(opening)?.[1] ?? '';
  if (!signatureHasExactArgumentPath(signature, edge.consumer_argument, 'Neuralese<string>'))
    fail(`materialized reader ${row.id} does not expose typed ${edge.consumer_argument}`);
  const scope = row.messages?.find(message => message.role === 'tool' && message.tool_call_id === 'scope_0');
  const content = Array.isArray(scope?.content) ? scope.content : [];
  const blocks = content.filter(part => part?.type === 'neuralese' && part.id === edge.block_id);
  const path = edge.consumer_argument.split('.');
  const blockIndex = content.findIndex(part => part?.type === 'neuralese' && part.id === edge.block_id);
  const beforeBlock = content.slice(0, blockIndex).filter(part => typeof part?.text === 'string').map(part => part.text).join('');
  const typedBinding = path.length === 1
    ? beforeBlock.endsWith(`${path[0]}: Neuralese<string> = `)
    : path.length === 2 && new RegExp(`${path[0]}: \\{\\s*${path[1]}: Neuralese<string>[^}]*\\}\\s*=\\s*\\{\\s*${path[1]}: $`).test(beforeBlock);
  if (blocks.length !== 1 || !typedBinding)
    fail(`materialized reader ${row.id} lacks the exact typed block in scope_0`);
}

/** Rebuild a conversion receipt from the actual result, independent review, and materialized action rows. */
export function validateSoftStateConversionEvidence({ resultPath, reviewPath, actionRows }) {
  if (!resultPath || !reviewPath || !Array.isArray(actionRows))
    fail('actual result path, review path, and materialized action rows are required');
  const resultBytes = readFileSync(resultPath);
  const reviewBytes = readFileSync(reviewPath);
  const resultSha = sha256(resultBytes), reviewSha = sha256(reviewBytes);
  const result = readSingleJsonLine(resultBytes, 'actual result');
  let review;
  try { review = JSON.parse(reviewBytes.toString('utf8')); } catch { fail('review is not valid JSON'); }
  if (review.status !== 'held; no training admission' || review.result?.sha256 !== resultSha ||
      review.result?.accepted !== true || review.result?.outcome_status !== 'done')
    fail('review does not bind this accepted completed result or does not remain held');
  if (result.id !== review.result?.trajectory_id && review.result?.trajectory_id !== undefined)
    fail('review trajectory identity does not match the result');
  if (result.provenance?.text_neuralese_transport?.mode !== 'text-marker-standin/2' ||
      result.outcome?.accepted !== true || result.outcome?.status !== 'done')
    fail('result is not an accepted text-marker-stand-in trajectory');
  const graph = result.outcome?.execution_graph;
  const reviewedEdges = review.actual_graph?.edges;
  if (!Array.isArray(graph) || !Array.isArray(reviewedEdges) || !reviewedEdges.length ||
      review.actual_graph.validated_true_soft_edges !== reviewedEdges.length)
    fail('actual graph or reviewed edge list is missing/inconsistent');

  const firstAction = actionRows[0];
  const trajectoryId = firstAction?.source_ref?.trajectory_id;
  const sourceRowSha = firstAction?.source_ref?.source_row_sha256;
  const ir = firstAction?.task?.program_ir;
  const sourceIr = result.task?.program_ir;
  const recomputedSourceRowSha = nativeRowDigest(result);
  if (!trajectoryId || !sourceRowSha || sourceRowSha !== recomputedSourceRowSha || trajectoryId !== result.id ||
      !ir?.split || !Array.isArray(ir.source_groups) || !sourceIr?.split || !Array.isArray(sourceIr.source_groups))
    fail('materialized actions do not bind to the result trajectory and source partition');
  if (ir.split !== sourceIr.split || canonical(ir.source_groups) !== canonical(sourceIr.source_groups))
    fail('materialized source partition differs from the actual result program IR');
  if (actionRows.some(row => row.source_ref?.trajectory_id !== trajectoryId ||
      row.source_ref?.source_row_sha256 !== sourceRowSha || row.task?.program_ir?.split !== ir.split ||
      JSON.stringify(row.task?.program_ir?.source_groups) !== JSON.stringify(ir.source_groups) ||
      canonical(row.provenance?.text_neuralese_transport) !== canonical(result.provenance?.text_neuralese_transport)))
    fail('materialized action rows mix trajectory, source-row, split, or source-group identities');
  let regenerated;
  try { regenerated = materializeNativeRows([result]).turns; }
  catch (error) { fail(`actual result cannot be rematerialized: ${error.message}`); }
  const regeneratedById = new Map(regenerated.map(row => [row.id, materializedActionProjection(row)]));
  for (const row of actionRows) {
    const expected = regeneratedById.get(row.id);
    if (!expected || canonical(expected.source_ref) !== canonical(row.source_ref) ||
        canonical(expected.target) !== canonical(row.target) || expected.decision?.index !== row.decision?.index)
      fail(`action row ${row.id} source or target differs from exact materializer output for the actual result`);
  }

  const edges = [];
  const blockIds = new Set();
  for (const reviewed of reviewedEdges) {
    const blockId = reviewed.block;
    if (typeof blockId !== 'string' || blockIds.has(blockId)) fail('duplicate or invalid reviewed block ID');
    blockIds.add(blockId);
    const writerCallId = reviewed.writer_call, readerCallId = reviewed.reader_call, argument = reviewed.arg;
    if (typeof writerCallId !== 'string' || typeof readerCallId !== 'string' || typeof argument !== 'string')
      fail('review edge lacks exact writer/reader call IDs or consumer argument');
    const actualValue = hostValue(result, writerCallId);
    const validated = validateSoftStateEdge({ graph, actualValue, expectedType: 'Neuralese<string>',
      writerCallId, consumerCallId: readerCallId, consumerArgument: argument, writerNode: reviewed.writer_node,
      expectedBodySha256: reviewed.body_sha256 });
    if (validated.block !== blockId || validated.writer_node !== reviewed.writer_node ||
        validated.block_read_node !== reviewed.reader_node || validated.consumer_signature !== reviewed.signature ||
        validated.invocation_input_port !== `arg:${argument}` ||
        (argument.includes('.') && validated.capture_input_port !== `capture:${argument.split('.').at(-1)}`))
      fail(`shared graph validation does not reproduce reviewed edge ${blockId}`);

    const writes = graph.filter(event => event.kind === 'block_write' && event.call_id === writerCallId &&
      event.node === validated.writer_node && event.block === blockId);
    if (writes.length !== 1) fail(`expected one selected graph writer event for ${blockId}`);
    const write = writes[0];
    if (write.emulation_version !== 'text-marker-standin/2' || write.learned_vectors !== false ||
        !['return-result', 'eval-code'].includes(write.marker_context) || write.result_type !== 'Neuralese<string>' ||
        !/^[0-9a-f]{64}$/.test(write.text_body_sha256 ?? ''))
      fail(`graph writer ${blockId} lacks text stand-in body provenance`);
    const expansion = providerExpansion(result, readerCallId, blockId);
    if (write.marker_context === 'return-result')
      providerMarkerOutput(result, writerCallId, write.text_body_sha256);
    if (expansion.body_sha256 !== write.text_body_sha256)
      fail(`provider-visible body digest differs from graph writer for ${blockId}`);
    if (expansion.type === null && write.result_type !== 'Neuralese<string>')
      fail(`cannot infer omitted provider type for ${blockId} without an exact typed graph writer`);

    const actionEdge = { ...reviewed, block_id: blockId, consumer_argument: argument,
      marker_context: write.marker_context };
    const writer = actionForCall(actionRows, writerCallId, trajectoryId, sourceRowSha, 'writer', actionEdge, write.text_body_sha256);
    const reader = actionForCall(actionRows, readerCallId, trajectoryId, sourceRowSha, 'reader', actionEdge);
    if (writer.row.task?.program_ir?.split !== ir.split || reader.row.task?.program_ir?.split !== ir.split ||
        JSON.stringify(writer.row.task?.program_ir?.source_groups) !== JSON.stringify(ir.source_groups) ||
        JSON.stringify(reader.row.task?.program_ir?.source_groups) !== JSON.stringify(ir.source_groups))
      fail(`materialized edge actions changed source partition for ${blockId}`);
    const body = markerBody(writer.row, blockId, write.text_body_sha256, write.marker_context);
    if (body !== expansion.body) fail(`producer marker and provider-visible body differ for ${blockId}`);
    const derived = { block_id: blockId, writer_call_id: writerCallId, writer_node: validated.writer_node,
      writer_record_id: writer.row.id, writer_decision_index: writer.index,
      reader_call_id: readerCallId, reader_node: validated.block_read_node, reader_record_id: reader.row.id,
      reader_decision_index: reader.index, consumer_argument: argument,
      consumer_signature: validated.consumer_signature, expected_type: 'Neuralese<string>',
      provider_expansion_type: expansion.type,
      ...(expansion.type === null ? { provider_type_inference: {
        inferred_type: 'Neuralese<string>', basis: 'authenticated-typed-writer-and-reader-contract-with-matching-body-sha256',
        provider_saw_type_label: false,
      } } : {}),
      ...(validated.capture_input_port ? { capture_input_port: validated.capture_input_port } : {}),
      body_sha256: expansion.body_sha256, body_source: expansion.body };
    checkReaderAction(reader.row, derived);
    edges.push(derived);
  }

  return { schema: 'natlang.validated-runtime-soft-state-edges/1', status: 'passed',
    validation: { validator: 'validateSoftStateEdge', review_sha256: reviewSha, result_sha256: resultSha,
      actual_result_path: String(resultPath), actual_review_path: String(reviewPath), revalidated_edges: edges.length },
    source: { trajectory_id: trajectoryId, source_row_sha256: sourceRowSha, split: ir.split,
      source_groups: ir.source_groups, transport_mode: 'text-marker-standin/2', learned_vectors: false,
      qualification_certificate: false, training_admission: false }, edges };
}
