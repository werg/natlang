import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';

const distRoot = process.env.NATLANG_TS_HOST_TEST_DIST;
const distModule = distRoot
  ? pathToFileURL(resolve(distRoot, 'teacher/native-materializer.js')).href
  : new URL('../dist/teacher/native-materializer.js', import.meta.url).href;
const conversionModule = distRoot
  ? pathToFileURL(resolve(distRoot, 'compiler/neuralese-conversion.js')).href
  : new URL('../dist/compiler/neuralese-conversion.js', import.meta.url).href;
const { derivePureLiteralTypedTextTarget, nativeRowDigest } = await import(distModule);
const { pureLiteralEvalReturn, convertTrajectory } = await import(conversionModule);
const sha = value => createHash('sha256').update(value).digest('hex');

test('pure-literal eval parser accepts one immutable literal and exact terminal return only', () => {
  assert.deepEqual(pureLiteralEvalReturn('const note: Neuralese<string> = "grounded note"; return note;'), {
    binding_name: 'note', value: 'grounded note', literal_kind: 'string',
  });
  assert.deepEqual(pureLiteralEvalReturn('const note = `grounded note`; return note;'), {
    binding_name: 'note', value: 'grounded note', literal_kind: 'no-substitution-template',
  });
  for (const code of [
    'const note = `value: ${source}`; return note;',
    'const note = await source.readText(); return note;',
    'const note = nl`write a note`; return note;',
    'const note = "first"; const suffix = "second"; return note + suffix;',
    'let note = "mutable"; return note;',
    'const note = "first"; note = "changed"; return note;',
    'const note = "first"; if (ok) return note; return "other";',
    'const note = "first"; return other;',
    'const note: Neuralese<number> = "wrong declaration"; return note;',
    'const note = "unterminated; return note;',
  ]) assert.equal(pureLiteralEvalReturn(code), undefined, code);
});

function sourceRow(code = 'const note: Neuralese<string> = "Supported by the request."; return note;') {
  const row = {
    version: 'natlang.teacher_trajectory.native/1', id: 'source-row-1',
    source_ref: {
      trajectory_id: 'trajectory-1', source_row_sha256: 'a'.repeat(64), invocation_id: 'invocation/2',
      inline_instruction_site: { site: { returns: { natlang: 'Neuralese<string>' } } },
    },
    messages: [{ role: 'user', content: 'Visible source facts supporting this exact note.' }],
    tools: [{ function: { name: 'return_result' } }],
    target: { role: 'assistant', content: '', tool_calls: [{ id: 'eval-call', type: 'function', function: {
      name: 'eval', arguments: JSON.stringify({ code, finish: true }),
    } }] },
    decision: { index: 3, source_raw_response_sha256: 'c'.repeat(64), assistant: { calls: [] } },
    preview_source_selection: { target_generation_turn: { trajectory_index: 3, invocation_id: 'invocation/2',
      turn: 1, request_sha256: 'b'.repeat(64), raw_response_sha256: 'c'.repeat(64) } },
    training_admission: { approved: false },
  };
  return row;
}

function witness(row, overrides = {}) {
  const code = JSON.parse(row.target.tool_calls[0].function.arguments).code;
  const body = pureLiteralEvalReturn(code).value;
  const proof = {
    trajectory_id: 'trajectory-1', source_row_sha256: 'a'.repeat(64), source_result_row_sha256: 'd'.repeat(64),
    generation_turn: { trajectory_index: 3, invocation_id: 'invocation/2', turn: 1,
      request_sha256: 'b'.repeat(64), raw_response_sha256: 'c'.repeat(64) },
    target_code_sha256: sha(code), writer_call_id: 'invocation/2', writer_node: 'invocation/2#18',
    block_id: 'nz1_' + 'e'.repeat(32), body_source: body, body_sha256: sha(body),
    result_type: 'Neuralese<string>', source: 'eval-finish', marker_context: 'return-result',
    source_kind: 'typed-text-result', ...overrides,
  };
  row.decision.assistant.calls = [{ source_tool: 'eval', arguments: { code, finish: true }, outcome: {
    name: 'eval', arguments: { code, finish: true }, tool_call_id: 'actual-provider-eval-call', typed_result_writes: [{
    schema: 'natlang.typed-result-write/1', trajectory_id: proof.trajectory_id,
    source_row_sha256: proof.source_row_sha256, invocation_id: proof.generation_turn.invocation_id,
    writer_call_id: proof.writer_call_id, writer_node: proof.writer_node, block_id: proof.block_id,
    source_kind: proof.source_kind, source: proof.source, result_type: proof.result_type,
    marker_context: proof.marker_context, body_source: proof.body_source, body_sha256: proof.body_sha256,
    request_sha256: proof.generation_turn.request_sha256,
    raw_response_sha256: proof.generation_turn.raw_response_sha256,
  }] } }];
  return proof;
}

test('derived typed text target preserves source action and exact provenance while staying held', () => {
  const row = sourceRow();
  const originalTarget = structuredClone(row.target);
  const derived = derivePureLiteralTypedTextTarget(row, witness(row));
  assert.ok(derived);
  assert.deepEqual(row.target, originalTarget, 'source action remains unchanged');
  assert.deepEqual(derived.messages, row.messages, 'no context is inserted');
  assert.equal(derived.source_ref.source_row_sha256, row.source_ref.source_row_sha256);
  assert.equal(derived.derived_target.original_messages_sha256, nativeRowDigest(row.messages));
  assert.deepEqual(derived.target.tool_calls[0].function, { name: 'return_result',
    arguments: JSON.stringify({ status: 'success', value: 'Supported by the request.' }) });
  assert.equal(derived.training_admission.approved, false);
  assert.equal(derived.derived_target.derivation_role, 'derived_target_not_original_assistant_action');
  assert.equal(derived.derived_target.original_target_sha256, nativeRowDigest(originalTarget));
  assert.deepEqual(derived.derived_target.original_target, originalTarget);
  assert.equal(derived.derived_target.original_runtime_target_preserved_in_source_row, true);
  assert.equal(derived.derived_target.runtime_gradient_qualification, false);
});

test('shared conversion emits only a derivation-qualified typed text writer for the transformed target', () => {
  const row = sourceRow();
  const proof = witness(row);
  const derived = derivePureLiteralTypedTextTarget(row, proof);
  assert.ok(derived);
  const converted = convertTrajectory(derived).record;
  const args = JSON.parse(converted.target.tool_calls[0].function.arguments);
  assert.equal(args.value.$write.source, proof.body_source);
  assert.equal(args.value.$write.type, 'Neuralese<string>');
  assert.match(args.value.$write.name, /^soft-state:nz1_[a-z2-7]+@derived:/);
  assert.equal(converted.neuralese_conversion.selected_runtime_result_writes, undefined);
  assert.deepEqual(converted.neuralese_conversion.derived_semantic_text_writes.map(write => write.role),
    ['derived-equivalent-pure-terminal-eval-finish-target']);
  const receipt = converted.neuralese_conversion.derived_semantic_text_writes[0];
  assert.equal(receipt.runtime_gradient_qualification, false);
  assert.equal(receipt.original_eval_hidden_states_equivalent, false);

  for (const tamper of [
    value => { value.derived_target.original_target.tool_calls[0].id = 'wrong-original-call'; },
    value => { value.messages[0].content = 'changed context'; },
    value => { value.target.tool_calls[0].function.name = 'eval'; },
    value => { value.target.tool_calls[0].function.arguments = JSON.stringify({ status: 'success', value: 'different' }); },
    value => { value.decision.assistant.calls[0].outcome.typed_result_writes[0].body_sha256 = 'f'.repeat(64); },
  ]) {
    const changed = structuredClone(derived);
    tamper(changed);
    assert.throws(() => convertTrajectory(changed), /derived typed-text target proof mismatch|exact terminal output receipt|exactly one validated derived writer/);
  }
});

test('derived typed text target rejects mismatched write, source, turn, type, or target evidence', () => {
  const row = sourceRow();
  const valid = witness(row);
  for (const change of [
    { body_source: 'different body' },
    { body_sha256: 'f'.repeat(64) },
    { result_type: 'Neuralese<object>' },
    { source: 'return-result' },
    { marker_context: 'eval-code' },
    { source_kind: 'legacy-text-marker-standin' },
    { writer_call_id: 'invocation/9' },
    { writer_node: 'invocation/9#18' },
    { target_code_sha256: 'f'.repeat(64) },
    { source_row_sha256: 'f'.repeat(64) },
    { trajectory_id: 'other-trajectory' },
    { source_result_row_sha256: 'not-a-sha' },
    { generation_turn: { ...valid.generation_turn, request_sha256: 'f'.repeat(64) } },
  ]) assert.equal(derivePureLiteralTypedTextTarget(row, { ...valid, ...change }), undefined);
  const noReturnTool = { ...row, tools: [] };
  assert.equal(derivePureLiteralTypedTextTarget(noReturnTool, valid), undefined);
  const approved = { ...row, training_admission: { approved: true } };
  assert.equal(derivePureLiteralTypedTextTarget(approved, valid), undefined);
  const tamperedReceipt = structuredClone(row);
  tamperedReceipt.decision.assistant.calls[0].outcome.typed_result_writes[0].body_sha256 = 'f'.repeat(64);
  assert.equal(derivePureLiteralTypedTextTarget(tamperedReceipt, valid), undefined);
  const duplicateReceipt = structuredClone(row);
  duplicateReceipt.decision.assistant.calls[0].outcome.typed_result_writes.push(
    structuredClone(duplicateReceipt.decision.assistant.calls[0].outcome.typed_result_writes[0]));
  assert.equal(derivePureLiteralTypedTextTarget(duplicateReceipt, valid), undefined);
});

test('derived typed text target rejects any effectful or nonliteral selected eval', () => {
  for (const code of [
    'const note = await source.readText(); return note;',
    'const note = `text ${await source.readText()}`; return note;',
    'const note = transform("text"); return note;',
  ]) {
    const row = sourceRow(code);
    assert.equal(derivePureLiteralTypedTextTarget(row, { ...witness(sourceRow()), target_code_sha256: sha(code) }), undefined);
  }
});

test('a nonterminal eval-return stage cannot be converted as an equivalent terminal result', () => {
  const row = sourceRow();
  row.target.tool_calls[0].function.arguments = JSON.stringify({
    code: JSON.parse(row.target.tool_calls[0].function.arguments).code,
  });
  const proof = witness(row);
  const code = JSON.parse(row.target.tool_calls[0].function.arguments).code;
  const marker = `${proof.block_id}`;
  proof.source = 'eval-return';
  proof.marker_context = 'return-result';
  row.decision.assistant.calls = [{ outcome: { name: 'eval', arguments: { code },
    result: `Staged ${marker} as the result.` } }];
  assert.equal(derivePureLiteralTypedTextTarget(row, proof), undefined);
});
