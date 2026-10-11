import assert from 'node:assert/strict';
import { createHash as cryptoCreateHash } from 'node:crypto';
import { test } from 'node:test';
import { canonical } from '../dist/adaptation/identity.js';
import { ChildResultIndexBuilder, childCallIds, childFunctionNames, childReturn, convertTrajectory, instructionsDigest, invocationOf, printedResults } from '../dist/compiler/neuralese-conversion.js';
import { COMPACTION_NOTICE, GENERATION_GUIDANCE, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN, TOOLS_PROMPT } from '../dist/native/prompt.js';
import { programGuidance } from '../dist/adaptation/prompts.js';

const note = 'Checked lines 1-3; line 2 is a fee with no PO line. Left: prior invoices.';
const compact = (id, text) => ({ role: 'assistant', content: '', tool_calls: [{ id, type: 'function',
  function: { name: 'compact_history', arguments: JSON.stringify({ note: text }) } }] });
const record = () => ({ id: 'r1', messages: [
  { role: 'system', content: TOOLS_PROMPT + GENERATION_GUIDANCE + programGuidance('Prefer the contract over emails.') },
  { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nIs line 2 a fee?\n\nIn eval you can use state.' },
  { role: 'user', content: HANDOVER_NOTE_OPEN + note + HANDOVER_NOTE_CLOSE },
  compact('c1', note),
  { role: 'tool', tool_call_id: 'c1', content: 'Compacted.' },
  { role: 'assistant', content: '', tool_calls: [{ id: 'e1', type: 'function', function: { name: 'eval',
    arguments: JSON.stringify({ code: 'const v = await nl<boolean>`Is it a fee?`(state);' }) } }] },
  { role: 'tool', tool_call_id: 'e1', content: 'console:\ntrue' + COMPACTION_NOTICE },
], target: compact('c2', 'Line 2 is a fee. Return true.') });

test('prompts, guidance and handover notes become Neuralese; the rest is counted', () => {
  const { record: out, pieces } = convertTrajectory(record());
  const system = out.messages[0].content;
  assert.deepEqual(system.filter(p => p.type === 'soft').map(p => p.name).slice(0, 2),
    [`prompt:interpreter@${createHash(TOOLS_PROMPT)}`, `prompt:generation-guidance@${createHash(GENERATION_GUIDANCE)}`]);
  assert.ok(system.some(p => p.type === 'soft' && p.name.startsWith('guidance@')), 'program guidance is its own soft parameter');
  assert.ok(!JSON.stringify(system).includes('You are running one call'));

  const pinned = out.messages[2].content;
  assert.deepEqual(pinned.map(p => p.type), ['soft', 'read', 'soft']);
  const write = JSON.parse(out.messages[3].tool_calls[0].function.arguments).note.$write;
  assert.equal(write.name, pinned[1].name, 'the pinned note reads the block the compaction call writes');
  assert.equal(write.source, note);
  assert.equal(write.type, 'Neuralese<HandoverNote>');
  assert.ok(JSON.parse(out.target.tool_calls[0].function.arguments).note.$write, 'a compaction target is a write');

  assert.ok(out.messages[6].content.some(p => p.type === 'soft' && p.name === `prompt:compaction-notice@${createHash(COMPACTION_NOTICE)}`));
  assert.equal(typeof out.messages[1].content, 'string', 'single-use instructions stay text');
  const { sites } = out.neuralese_conversion;
  assert.equal(sites['handover-write'].converted, 2);
  assert.equal(sites['handover-read'].converted, 1);
  assert.equal(sites['nl-literal'].exact['later-curriculum-step'], 1);
  assert.equal(sites['tool-output'].exact['single-use'], 1);
  assert.equal(sites['child-result'].exact['producer-missing'], 1, 'without the run\'s child returns a printed result stays exact');
  assert.equal(sites.instructions.exact['single-use'], 1);
  assert.ok(pieces.some(p => p.name === `prompt:interpreter@${createHash(TOOLS_PROMPT)}` && p.text === TOOLS_PROMPT), 'pieces carry their initial text');
});

test('runtime soft-state conversion supports an exact nested typed member and rejects a wrong member path', () => {
  const body = 'observed note body';
  const block = `nz1_${'a'.repeat(52)}`;
  const edge = { block_id: block, writer_call_id: 'writer', writer_node: 'writer#1', writer_record_id: 'writer-row',
    writer_decision_index: 1, reader_call_id: 'reader', reader_node: 'reader#5', reader_record_id: 'reader-row',
    reader_decision_index: 2, consumer_argument: 'input.notes',
    consumer_signature: '(input: { notes: Neuralese<string>, outputContract: string }) => Draft',
    expected_type: 'Neuralese<string>', body_sha256: cryptoCreateHash('sha256').update(body).digest('hex'), body_source: body };
  const reader = { id: 'reader-row', decision: { index: 2 }, source_ref: { invocation_id: 'reader', trajectory_id: 't', source_row_sha256: 'x' },
    task: { program_ir: { split: 'train', source_groups: ['g'] } }, messages: [
    { role: 'user', content: 'You are inside this call: nl@eval:1(input: { notes: Neuralese<string>, outputContract: string }): Draft' },
    { role: 'tool', tool_call_id: 'scope_0', content: [
      { type: 'text', text: 'input: { notes: Neuralese<string>, outputContract: string } = { notes: ' },
      { type: 'neuralese', id: block }, { type: 'text', text: ', outputContract: "contract" }' }
    ] }
  ] };
  const proof = { schema: 'natlang.validated-runtime-soft-state-edges/1', status: 'passed',
    validation: { validator: 'validateSoftStateEdge', review_sha256: 'r', result_sha256: 's' },
    source: { trajectory_id: 't', source_row_sha256: 'x', split: 'train', source_groups: ['g'],
      transport_mode: 'text-marker-standin/2', learned_vectors: false, qualification_certificate: false, training_admission: false }, edges: [edge] };
  const converted = convertTrajectory(reader, { softStateEdges: proof }).record;
  assert.ok(converted.messages[1].content.some(part => part.type === 'read' && part.name === `soft-state:${block}`));
  assert.throws(() => convertTrajectory(reader, { softStateEdges: { ...proof, edges: [{ ...edge, consumer_argument: 'input.outputContract' }] } }),
    /does not expose exact input.outputContract/);
});

test('typed final-result receipts become exact nested native/R write leaves', () => {
  const body = 'one exact generated note';
  const block = `nz1_${'c'.repeat(52)}`;
  const bodySha = cryptoCreateHash('sha256').update(body).digest('hex');
  const args = { status: 'success', value: { notes: body, other: body } };
  const receipts = ['notes', 'other'].map((field, index) => ({
    schema: 'natlang.typed-result-write/1', trajectory_id: 'typed-run', source_row_sha256: 'typed-row-sha',
    invocation_id: 'typed-call', writer_call_id: 'typed-call', writer_node: `typed-call#${index + 4}`,
    block_id: block, source_kind: 'typed-text-result-field', source: 'return_result', result_type: 'Neuralese<string>',
    result_path: ['return', field], model_turn_node: 'typed-call#2', action_seq: 8,
    body_sha256: bodySha, body_source: body, body_source_basis: 'exact-raw-model-result-field',
  }));
  const input = record();
  input.id = 'typed-row';
  input.source_ref = { trajectory_id: 'typed-run', invocation_id: 'typed-call', source_row_sha256: 'typed-row-sha' };
  input.decision = { index: 3, assistant: { calls: [{ source_tool: 'return_result', arguments: args,
    outcome: { name: 'return_result', arguments: args, typed_result_writes: receipts } }] } };
  input.target = { role: 'assistant', tool_calls: [{ id: 'return-1', type: 'function',
    function: { name: 'return_result', arguments: JSON.stringify(args) } }] };
  const converted = convertTrajectory(input).record;
  const target = JSON.parse(converted.target.tool_calls[0].function.arguments);
  assert.deepEqual(target.value.notes.$write, { name: `typed-result:${block}:${createHash(JSON.stringify([receipts[0].writer_node, receipts[0].result_path]))}`,
    block_id: block, type: 'Neuralese<string>', source: body });
  assert.deepEqual(target.value.other.$write, { name: `typed-result:${block}:${createHash(JSON.stringify([receipts[1].writer_node, receipts[1].result_path]))}`,
    block_id: block, type: 'Neuralese<string>', source: body });
  assert.equal(converted.neuralese_conversion.sites['typed-result-write'].converted, 2);

  const jsonText = '{"x":1}', canonicalBody = '{"x":1}', jsonTextBlock = `nz1_${'e'.repeat(52)}`;
  const jsonTextArgs = { status: 'success', value: jsonText };
  const jsonTextReceipt = { schema: 'natlang.typed-result-write/1', trajectory_id: 'json-run',
    source_row_sha256: 'json-row-sha', invocation_id: 'json-call', writer_call_id: 'json-call',
    writer_node: 'json-call#8', block_id: jsonTextBlock, source_kind: 'typed-json-result', source: 'return_result',
    result_type: 'Neuralese<{ x: number }>', result_path: ['return'], model_turn_node: 'json-call#2', action_seq: 9,
    body_sha256: cryptoCreateHash('sha256').update(canonicalBody).digest('hex'), body_source: canonicalBody,
    body_source_basis: 'parsed-json-string-for-concrete-neuralese-result-type',
    raw_model_value_sha256: cryptoCreateHash('sha256').update(JSON.stringify(jsonText)).digest('hex') };
  const jsonTextRow = record();
  jsonTextRow.id = 'json-row';
  jsonTextRow.source_ref = { trajectory_id: 'json-run', invocation_id: 'json-call', source_row_sha256: 'json-row-sha' };
  jsonTextRow.decision = { index: 4, assistant: { calls: [{ source_tool: 'return_result', arguments: jsonTextArgs,
    outcome: { name: 'return_result', arguments: jsonTextArgs, typed_result_writes: [jsonTextReceipt] } }] } };
  jsonTextRow.target = { role: 'assistant', tool_calls: [{ id: 'json-return', type: 'function',
    function: { name: 'return_result', arguments: JSON.stringify(jsonTextArgs) } }] };
  const normalizedJson = convertTrajectory(jsonTextRow).record;
  assert.equal(JSON.parse(normalizedJson.target.tool_calls[0].function.arguments).value.$write.source, canonicalBody,
    'the authenticated normalized body is trained as a write while the receipt retains the raw JSON text hash');

  const stringRow = structuredClone(jsonTextRow);
  stringRow.id = 'string-row';
  stringRow.source_ref = { ...jsonTextRow.source_ref, trajectory_id: 'string-run', invocation_id: 'string-call', source_row_sha256: 'string-row-sha' };
  stringRow.decision.assistant.calls[0].arguments = { status: 'success', value: jsonText };
  stringRow.decision.assistant.calls[0].outcome.arguments = { status: 'success', value: jsonText };
  stringRow.decision.assistant.calls[0].outcome.typed_result_writes = [{ ...jsonTextReceipt,
    trajectory_id: 'string-run', source_row_sha256: 'string-row-sha', invocation_id: 'string-call', writer_call_id: 'string-call',
    source_kind: 'typed-text-result', result_type: 'Neuralese<string>', body_sha256: cryptoCreateHash('sha256').update(jsonText).digest('hex'),
    body_source: jsonText, body_source_basis: 'exact-raw-model-result-string', raw_model_value_sha256: undefined }];
  stringRow.target.tool_calls[0].function.arguments = JSON.stringify({ status: 'success', value: jsonText });
  const literalJsonText = convertTrajectory(stringRow).record;
  assert.equal(JSON.parse(literalJsonText.target.tool_calls[0].function.arguments).value.$write.source, jsonText,
    'a JSON-looking Neuralese<string> value stays an exact literal string');

  const tampered = structuredClone(input);
  tampered.target.tool_calls[0].function.arguments = JSON.stringify({ ...args, value: { ...args.value, notes: 'different' } });
  const held = convertTrajectory(tampered).record;
  assert.equal(JSON.parse(held.target.tool_calls[0].function.arguments).value.notes, 'different',
    'a raw target value that no longer matches its receipt remains unchanged');

  const evalRow = record();
  evalRow.id = 'eval-row';
  evalRow.source_ref = { trajectory_id: 'eval-run', invocation_id: 'eval-call', source_row_sha256: 'eval-row-sha' };
  const evalArgs = { code: 'return await buildResult();', finish: true };
  evalRow.decision = { index: 4, assistant: { calls: [{ source_tool: 'eval', arguments: evalArgs,
    outcome: { name: 'eval', arguments: evalArgs, typed_result_writes: [{
      schema: 'natlang.typed-result-write/1', trajectory_id: 'eval-run', source_row_sha256: 'eval-row-sha',
      invocation_id: 'eval-call', writer_call_id: 'eval-call', writer_node: 'eval-call#12',
      block_id: `nz1_${'d'.repeat(52)}`, source_kind: 'typed-text-result-field', source: 'eval-finish',
      result_type: 'Neuralese<string>', result_path: ['return', 'note'], body_sha256: bodySha,
      body_source_basis: 'authenticated-final-host-output-reference',
    }] } }] } };
  evalRow.target = { role: 'assistant', tool_calls: [{ id: 'eval-1', type: 'function',
    function: { name: 'eval', arguments: JSON.stringify(evalArgs) } }] };
  const evalConverted = convertTrajectory(evalRow).record;
  assert.equal(evalConverted.target.tool_calls[0].function.arguments, JSON.stringify(evalArgs),
    'eval source remains exact because no validated code span corresponds to the returned value');
  assert.equal(evalConverted.neuralese_conversion.sites['typed-result-write'].exact['eval-result-kept-without-source-span'], 1);
});

test('an exact direct typed prose return is event-qualified as a semantic recurrence writer', () => {
  const body = 'The approved note preserves the verified deadline and keeps the hold.';
  const block = `nz1_${'f'.repeat(52)}`;
  const eventName = `soft-state:${block}@${createHash(JSON.stringify(['semantic-run', 'semantic-call', 'semantic-call#9']))}`;
  const args = { status: 'success', value: body };
  const receipt = { schema: 'natlang.typed-result-write/1', trajectory_id: 'semantic-run',
    source_row_sha256: 'semantic-row-sha', invocation_id: 'semantic-call', writer_call_id: 'semantic-call',
    writer_node: 'semantic-call#9', block_id: block, source_kind: 'typed-text-result',
    source: 'return_result', result_type: 'Neuralese<string>', result_path: ['return'],
    model_turn_node: 'semantic-call#3', action_seq: 11,
    body_sha256: cryptoCreateHash('sha256').update(body).digest('hex'), body_source: body,
    body_source_basis: 'exact-raw-model-result-string' };
  const input = record();
  input.id = 'semantic-row';
  input.source_ref = { trajectory_id: 'semantic-run', invocation_id: 'semantic-call', source_row_sha256: 'semantic-row-sha' };
  input.decision = { index: 0, assistant: { calls: [{ source_tool: 'return_result', arguments: args,
    outcome: { name: 'return_result', arguments: args, typed_result_writes: [receipt] } }] } };
  input.target = { role: 'assistant', tool_calls: [{ id: 'semantic-return', type: 'function',
    function: { name: 'return_result', arguments: JSON.stringify(args) } }] };
  const converted = convertTrajectory(input).record;
  const value = JSON.parse(converted.target.tool_calls[0].function.arguments).value.$write;
  assert.deepEqual(value, { name: eventName, block_id: block, type: 'Neuralese<string>', source: body });
  assert.deepEqual(converted.neuralese_conversion.selected_runtime_result_writes, [{
    schema: 'natlang.selected-runtime-result-write/1',
    role: 'selected-direct-typed-text-semantic-writer', trajectory_id: 'semantic-run',
    source_row_sha256: 'semantic-row-sha', invocation_id: 'semantic-call', writer_call_id: 'semantic-call',
    writer_node: 'semantic-call#9', block_id: block, result_type: 'Neuralese<string>',
    body_sha256: receipt.body_sha256, result_path: ['return'], action_seq: 11,
    typed_result_receipt_sha256: cryptoCreateHash('sha256').update(canonical(receipt)).digest('hex'),
    action_target_call_id: 'semantic-return',
    action_arguments_sha256: cryptoCreateHash('sha256').update(JSON.stringify(args)).digest('hex'),
    target_write_name: eventName,
    target_write_sha256: cryptoCreateHash('sha256').update(JSON.stringify({
      $write: { block_id: block, name: eventName, source: body, type: 'Neuralese<string>' },
    })).digest('hex'),
    body_source_basis: 'exact-raw-model-result-string', learned_vectors: false,
    qualification_certificate: false, training_admission: false,
  }]);
  const reader = record();
  reader.id = 'semantic-reader';
  reader.source_ref = { trajectory_id: 'semantic-run', invocation_id: 'semantic-reader-call',
    source_row_sha256: 'reader-row-sha', provider_expanded_read_contexts: [{
      schema: 'natlang.provider-expanded-read-context/2', origin: 'same-run-producer',
      invocation_id: 'semantic-reader-call', parent_invocation_id: null, source_row_sha256: 'reader-row-sha',
      trace_sha256: '1'.repeat(64), transport_provenance_sha256: '2'.repeat(64),
      raw_request_sha256: '3'.repeat(64), rendered_request_sha256: '4'.repeat(64),
      block: { id: block, type: 'Neuralese<string>', body, body_sha256: receipt.body_sha256 },
      producer_write: { kind: 'block_write', producer: 'text-marker-emulation', source_kind: 'typed-text-result',
        source: 'return_result', marker_context: 'return-result', block, result_type: 'Neuralese<string>',
        text_body_sha256: receipt.body_sha256, call_id: 'semantic-call', node: 'semantic-call#9', truncated: false },
      block_read: { kind: 'block_read', block, call_id: 'semantic-reader-call', turn: 'semantic-reader-call#turn1',
        node: 'semantic-reader-call#4', seq: 1, inputs: [{ node: 'semantic-call#9', block }] },
      model_turn: { kind: 'model_turn', call_id: 'semantic-reader-call', node: 'semantic-reader-call#turn1', seq: 2,
        inputs: [{ node: 'semantic-reader-call#4', port: 'read', block }] },
      writer_target_selected: false, writer_source_class: 'modern-typed-text-result',
      context_occurrences: 1,
      learned_vectors: false, qualification_certificate: false, training_admission: false,
    }] };
  reader.provenance = { trace_sha256: '1'.repeat(64) };
  reader.messages = [{ role: 'user', content: [{ type: 'neuralese', id: block }] }];
  reader.target = { role: 'assistant', content: 'A grounded follow-up.' };
  const convertedReader = convertTrajectory(reader).record;
  assert.deepEqual(convertedReader.messages[0].content[0], { type: 'read', name: eventName, source: body });
  assert.equal(convertedReader.neuralese_conversion.external_context_inputs[0].target_write_name, eventName);
  const computed = structuredClone(input);
  computed.decision.assistant.calls[0].outcome.typed_result_writes[0].source = 'eval-finish';
  computed.decision.assistant.calls[0].outcome.typed_result_writes[0].body_source_basis = 'authenticated-final-host-output-reference';
  computed.decision.assistant.calls[0].source_tool = 'eval';
  computed.target.tool_calls[0].function.name = 'eval';
  const notSemantic = convertTrajectory(computed).record;
  assert.equal(notSemantic.neuralese_conversion.selected_runtime_result_writes, undefined,
    'computed eval output does not inherit semantic direct-return status');
});

test('an opaque Neuralese handle return is recorded as identity forwarding only', () => {
  const block = `nz1_${'a'.repeat(52)}`;
  const args = { status: 'success', value: { $neuralese: { id: block, type: 'Neuralese<string>' } } };
  const input = record();
  input.id = 'forward-row';
  input.source_ref = { trajectory_id: 'forward-run', invocation_id: 'forward-call', source_row_sha256: 'forward-row-sha' };
  input.decision = { index: 1, assistant: { calls: [{ source_tool: 'return_result', arguments: args,
    outcome: { name: 'return_result', arguments: args } }] } };
  input.target = { role: 'assistant', tool_calls: [{ id: 'forward-return', type: 'function',
    function: { name: 'return_result', arguments: JSON.stringify(args) } }] };

  const converted = convertTrajectory(input).record;
  assert.equal(converted.target.tool_calls[0].function.arguments, JSON.stringify(args),
    'the existing opaque reference remains unchanged');
  assert.deepEqual(converted.neuralese_conversion.runtime_result_forwardings, [{
    schema: 'natlang.runtime-result-forwarding/1', role: 'identity-forwarding-existing-neuralese-reference',
    trajectory_id: 'forward-run', source_row_sha256: 'forward-row-sha', invocation_id: 'forward-call',
    action_target_call_id: 'forward-return', block_id: block, result_type: 'Neuralese<string>',
    action_arguments_sha256: cryptoCreateHash('sha256').update(JSON.stringify(args)).digest('hex'),
    creates_model_writer_target: false, recurrence_edge: false, learned_vectors: false,
    qualification_certificate: false, training_admission: false,
  }]);
  assert.equal(converted.neuralese_conversion.selected_runtime_result_writes, undefined);
});

const createHash = text => {
  // Match the converter's stable 12-hex content identity without depending on implementation exports.
  return cryptoCreateHash('sha256').update(text).digest('hex').slice(0, 12);
};

test('prompt piece identity preserves exact context across collected prompt variants', () => {
  const variants = ['Generation guidance from runtime A.', 'Generation guidance from runtime B with a newer rule.'];
  const expand = (content, pieces) => content.map(part => part.type === 'soft' ? pieces.find(piece => piece.name === part.name).text : part.text).join('');
  const converted = variants.map(text => {
    const input = record();
    input.messages[0].content = `Header\n${text}\nFooter`;
    const result = convertTrajectory(input, { pieces: [{ id: 'generation-guidance', text }] });
    return { result, expected: input.messages[0].content };
  });
  const [a, b] = converted;
  const aContent = a.result.record.messages[0].content;
  const bContent = b.result.record.messages[0].content;
  const aPiece = aContent.find(part => part.type === 'soft' && part.name.startsWith('prompt:generation-guidance'));
  const bPiece = bContent.find(part => part.type === 'soft' && part.name.startsWith('prompt:generation-guidance'));
  assert.notEqual(aPiece.name, bPiece.name, 'the same runtime prompt ID with different text gets distinct piece identities');
  assert.equal(expand(aContent, a.result.pieces), a.expected, 'variant A expands to the exact captured context');
  assert.equal(expand(bContent, b.result.pieces), b.expected, 'variant B expands to the exact captured context');
});

test('instructions used by several calls become one shared soft parameter; unregistered system text is versioned', () => {
  const input = record();
  input.messages[0].content = 'An older runtime prompt.';
  const digest = instructionsDigest('Is line 2 a fee?');
  const { record: out, pieces } = convertTrajectory(input, { instructionCalls: new Map([[digest, 3]]), instructionsShare: 0 });
  assert.match(out.messages[0].content[0].name, /^prompt:system@[0-9a-f]{12}$/);
  const body = out.messages[1].content.find(p => p.type === 'soft');
  assert.equal(body.name, `instructions@${digest}`);
  assert.equal(pieces.find(p => p.name === body.name).text, 'Is line 2 a fee?');
  assert.equal(out.neuralese_conversion.sites['instructions-reused'].converted, 1);
  const single = convertTrajectory(record(), { instructionCalls: new Map([[digest, 1]]), instructionsShare: 1 }).record;
  assert.equal(single.neuralese_conversion.sites['instructions-coverage'].converted, 1, 'a share of single-use instructions converts for coverage');
});

test('a cut-off value in the root call\'s opening listing becomes a view site of the full value', () => {
  const full = { task: 'Review the packet.', lines: Array.from({ length: 50 }, (_, i) => ({ id: `L${i}`, amount: i * 10 })) };
  const input = { id: 'r3', task: { program_ir: { semantics: { root: 'judge.nl', inputs: { state: full } } } }, messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nDecide.\n\nIn eval you can use state.' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'scope_0', type: 'function', function: { name: 'eval', arguments: '{"code":"const state = read_inputs().state;"}' } }] },
    { role: 'tool', tool_call_id: 'scope_0', content: 'state: unknown = { task: "Review the packet.", <<cut off: 1 of 2 fields not shown; state holds all of it>> }\nDeclared state for the rest of this call.' },
  ], target: { role: 'assistant', content: 'true' } };
  const { record: out } = convertTrajectory(input);
  const parts = out.messages[3].content;
  const view = parts.find(p => p.type === 'view');
  assert.deepEqual(JSON.parse(view.source), full);
  assert.match(view.preview, /cut off/);
  assert.match(view.name, /^view:[0-9a-f]{12}$/);
  assert.equal(parts[0].text, 'state: unknown = ');
  assert.equal(out.neuralese_conversion.sites.view.converted, 1);
  assert.ok(!parts.some(p => p.type === 'digest'));
  const nested = convertTrajectory({ ...input, task: undefined }).record;
  assert.equal(nested.neuralese_conversion.sites.view.exact['full-value-unavailable'], 1);
});

test('a child call\'s returned value that its caller prints becomes a write in the child and a read in the caller', () => {
  const summary = 'Line 2 bills 7.5 hours of on-site work at the contract rate.';
  const run = { source_ref: { trajectory_id: 'run-1' }, task: { program_ir: { semantics: { root: 'judge.nl' } } } };
  const child = { ...run, id: 'child', messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: summarise(line: string): string\n\nInstructions:\nSummarise the line.\n\nIn eval you can use line.' },
  ], target: { role: 'assistant', content: '', tool_calls: [{ id: 'r1', type: 'function', function: { name: 'return_result',
    arguments: JSON.stringify({ status: 'success', value: summary }) } }] } };
  const caller = { ...run, id: 'caller', messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nIs line 2 a fee?\n\nIn eval you can use state.' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'e1', type: 'function', function: { name: 'eval',
      arguments: JSON.stringify({ code: 'const s = await nl`Summarise the line.`(state.lines[1]); console.log(s)' }) } }] },
    { role: 'tool', tool_call_id: 'e1', content: `console:\n${summary}\nStored local s.` },
  ], target: { role: 'assistant', content: 'true' } };
  assert.equal(childReturn(child), summary);
  assert.equal(childReturn({ ...child, messages: [child.messages[0], caller.messages[1]] }), undefined, 'the root call returns no child result');
  const multipartRoot = { ...caller, target: { role: 'assistant', tool_calls: [{ id: 'root-return', function: {
    name: 'return_result', arguments: JSON.stringify({ status: 'success', value: summary }),
  } }] }, messages: [
    { role: 'user', content: [
      { type: 'text', text: 'You are inside this call: judge(state: unknown): string\n\nInstructions:\n' },
      { type: 'neuralese', id: 'root-body' },
      { type: 'text', text: '\n\nIn eval you can use state.' },
    ] },
  ] };
  assert.equal(childReturn(multipartRoot), undefined, 'multipart root openings still cannot become child producers');
  assert.deepEqual([...childCallIds(caller.messages)], ['e1']);
  const read = new Set(printedResults(caller.messages[3].content, [summary, 'true']));
  const childResults = new Map([['run-1', { returned: [summary, 'true'], read }]]);

  const written = convertTrajectory(child, { childResults }).record;
  const write = JSON.parse(written.target.tool_calls[0].function.arguments);
  assert.equal(write.status, 'success');
  assert.equal(write.value.$write.source, summary);
  assert.equal(write.value.$write.type, 'Neuralese<string>');

  const reading = convertTrajectory(caller, { childResults }).record;
  const parts = reading.messages[3].content;
  assert.deepEqual(parts.map(p => p.type), ['text', 'read', 'text']);
  assert.equal(parts[1].name, write.value.$write.name, 'the caller reads the block the child writes');
  assert.equal(parts.map(p => p.type === 'read' ? p.source : p.text).join(''), caller.messages[3].content);
  assert.equal(reading.neuralese_conversion.sites['child-result'].converted, 1);

  const short = convertTrajectory(caller, { childResults: new Map([['run-1', { returned: ['true'], read: new Set() }]]) }).record;
  assert.equal(short.neuralese_conversion.sites['child-result'].exact['crisp-value'], 1, 'a short value is its exact form');
});

test('a parent eval console log links an exact returned child array before the parent aggregates it', () => {
  const value = '["ARC-201","ARC-215"]';
  const run = 'folder-reducer-run';
  const producer = { id: 'teacher:decision:0004', decision: { index: 4 },
    source_ref: { trajectory_id: run, invocation_id: 'folder-reducer', parent_invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } },
    messages: [{ role: 'user', content: [
      { type: 'text', text: 'You are inside this call: soft@eval:1(folder: Folder): string[]\n\nInstructions:\n' },
      { type: 'neuralese', id: 'writer-body' },
      { type: 'text', text: '\n\nIn eval you can use folder.' },
    ] }],
    target: { role: 'assistant', tool_calls: [{ id: 'result', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value: JSON.parse(value) }) } }] } };
  const caller = { id: 'teacher:decision:0007', decision: { index: 7 },
    source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string[]' },
      { role: 'assistant', tool_calls: [{ id: 'aggregate', function: { name: 'eval',
        arguments: JSON.stringify({ code: "const perFolder = await reduceFolders(); console.log('Per-folder reducer results', perFolder); return perFolder.flat();" }) } }] },
      { role: 'tool', tool_call_id: 'aggregate', content: `console:\nPer-folder reducer results ${JSON.stringify([JSON.parse(value)])}\nStaged ["ARC-201", "ARC-215"] as the result.` },
    ], target: { role: 'assistant', content: '["ARC-201","ARC-215"]' } };
  const index = new ChildResultIndexBuilder();
  index.add(caller);
  index.add(producer);
  const childResults = index.finish().get(run);
  assert.deepEqual(childResults.readers.map(({ tool_call_id, value: read }) => [tool_call_id, read]), [['aggregate', value]],
    'the eval console is a real parent observation and authoritative decision indexes establish producer order');
  const written = convertTrajectory(producer, { childResults: new Map([[run, childResults]]) }).record;
  const write = JSON.parse(written.target.tool_calls[0].function.arguments).value.$write;
  assert.equal(write.source, value);
  const reading = convertTrajectory(caller, { childResults: new Map([[run, childResults]]) }).record;
  assert.ok(reading.messages[2].content.some(part => part.type === 'read' && part.source === value));
});

test('eval output text alone does not link future, unrelated-parent, or ambiguous child returns', () => {
  const value = 'the exact returned value is visible here';
  const makeProducer = (id, index, invocation, parent = 'root') => ({ id, decision: { index },
    source_ref: { trajectory_id: 'same-run', invocation_id: invocation, parent_invocation_id: parent },
    task: { program_ir: { semantics: { root: 'root.nl' } } },
    messages: [{ role: 'user', content: `You are inside this call: child${index}(): string\n\nInstructions:\nReturn the value.\n\nIn eval you can use it.` }],
    target: { role: 'assistant', tool_calls: [{ id: `return-${id}`, function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value }) } }] } });
  const makeCaller = index => ({ id: `caller-${index}`, decision: { index },
    source_ref: { trajectory_id: 'same-run', invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string' },
      { role: 'assistant', tool_calls: [{ id: 'root-eval', function: { name: 'eval',
        arguments: JSON.stringify({ code: 'console.log("unrelated computation");' }) } }] },
      { role: 'tool', tool_call_id: 'root-eval', content: `console:\n${value}` },
    ], target: { role: 'assistant', content: value } });
  const cases = [
    { name: 'future producer', callerIndex: 4, producers: [makeProducer('future', 7, 'child-future')] },
    { name: 'different parent', callerIndex: 7, producers: [makeProducer('elsewhere', 4, 'child-other', 'other-root')] },
    { name: 'ambiguous identical producers', callerIndex: 7, producers: [
      makeProducer('same-a', 4, 'child-a'), makeProducer('same-b', 5, 'child-b'),
    ] },
  ];
  for (const scenario of cases) {
    const index = new ChildResultIndexBuilder();
    const caller = makeCaller(scenario.callerIndex);
    index.add(caller);
    for (const producer of scenario.producers) index.add(producer);
    const childResults = index.finish().get('same-run');
    assert.deepEqual(childResults.readers, [], `${scenario.name} must stay unlinked`);
    const result = convertTrajectory(caller, { childResults: new Map([['same-run', childResults]]) }).record;
    assert.equal(result.messages[2].content, `console:\n${value}`, `${scenario.name} remains crisp text`);
    if (scenario.name === 'ambiguous identical producers')
      assert.equal(result.neuralese_conversion.sites['child-result'].exact['ambiguous-producer'], 1);
  }
});


test('a structured result\'s text field passed into another call is written field by field and read from its argument listing', () => {
  const facts = 'It was co-founded in 1973 by former astronaut Edgar Mitchell.';
  const run = { source_ref: { trajectory_id: 'run-2' }, task: { program_ir: { semantics: { root: 'research.nl' } } } };
  const opening = (signature, scope) => [{ role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: `You are inside this call: ${signature}\n\nInstructions:\nRead it.\n\nIn eval you can use x.` },
    { role: 'assistant', content: '', tool_calls: [{ id: 'scope_0', type: 'function', function: { name: 'eval', arguments: '{"code":"x"}' } }] },
    { role: 'tool', tool_call_id: 'scope_0', content: scope }];
  const judge = { ...run, id: 'judge', messages: opening('nl@eval:2(file: unknown): { relevant: boolean, facts: string }',
    `file: unknown = folder.file("a.md")\n  // File contents:\n"# IONS\\n\\nThe institute. ${facts} More."`),
    target: { role: 'assistant', content: '', tool_calls: [{ id: 'r', type: 'function', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value: { relevant: true, facts } }) } }] } };
  const answer = { ...run, id: 'answer', messages: opening('nl@eval:8(notes: string[]): string', `notes: string[] = ["${facts}"]`),
    target: { role: 'assistant', content: '1973' } };
  const producer = { id: 'judge#facts', invocation: invocationOf(judge), field: 'facts', value: facts, renderings: [facts] };
  const childResults = new Map([['run-2', { returned: [JSON.stringify({ relevant: true, facts }), facts], read: new Set([facts]),
    producers: [producer], readers: [{ invocation: invocationOf(answer), value: facts, producer_id: producer.id }] }]]);
  assert.equal(invocationOf(judge), invocationOf({ ...judge, id: 'judge-turn-2' }), 'every turn of one call shares its invocation');
  assert.notEqual(invocationOf(judge), invocationOf(answer));

  const written = JSON.parse(convertTrajectory(judge, { childResults }).record.target.tool_calls[0].function.arguments);
  assert.equal(written.value.relevant, true, 'the branch field stays exact');
  assert.equal(written.value.facts.$write.source, facts);
  assert.equal(written.value.facts.$write.type, 'Neuralese<string>');

  const read = convertTrajectory(answer, { childResults }).record;
  const parts = read.messages[3].content;
  assert.deepEqual(parts.map(p => p.type), ['text', 'read', 'text']);
  assert.equal(parts[1].name, written.value.facts.$write.name, 'the consumer reads the block the producer writes');
  assert.equal(read.neuralese_conversion.sites['argument-read'].converted, 1);

  // The producer's own listing shows the quoted text inside its file, not as a passed value: no read.
  const own = convertTrajectory(judge, { childResults: new Map([['run-2', { ...childResults.get('run-2'),
    readers: [{ invocation: invocationOf(judge), value: facts, producer_id: producer.id }] }]]) }).record;
  assert.equal(typeof own.messages[3].content, 'string');
});

test('named file calls are recognized and equal returns in independent runs stay separate', () => {
  const value = 'A private observation supports the contractual claim.';
  const make = run => ({ source_ref: { trajectory_id: run }, task: { program_ir: { semantics: {
    root: 'folder/review.nl', files: { 'folder/review.nl': '', 'folder/review/assess.nl': '' } } } },
    messages: [{ role: 'user', content: 'You are inside this call: review(): string' },
      { role: 'assistant', tool_calls: [{ id: 'e1', function: { name: 'eval', arguments: JSON.stringify({ code: 'await assess()' }) } }] },
      { role: 'tool', tool_call_id: 'e1', content: value }] });
  const a = make('run-a'), b = make('run-b');
  assert.deepEqual([...childCallIds(a.messages, childFunctionNames(a))], ['e1']);
  const childResults = new Map(['run-a', 'run-b'].map(id => [id, { returned: [value], read: new Set([value]) }]));
  const read = row => convertTrajectory(row, { childResults }).record.messages.at(-1).content.find(p => p.type === 'read');
  assert.equal(read(a).source, value);
  assert.notEqual(read(a).name, read(b).name, 'equal text in different executions is not one producer');
  assert.equal(childReturn({ ...a, target: { role: 'assistant', tool_calls: [{ function: {
    name: 'return_result', arguments: JSON.stringify({ status: 'success', value }) } }] } }), undefined,
    'a root inside a folder is still a root');
});

test('invocation-indexed results preserve ambiguous equal returns as text and never turn a root into a writer', () => {
  const value = 'This sufficiently detailed conclusion is shared by two independent invocations.';
  const root = {id:'reader', source_ref:{trajectory_id:'run',invocation_id:'root'},
    messages:[{role:'user',content:'You are inside this call: root(): string'},
      {role:'assistant',tool_calls:[{id:'e',function:{name:'eval',arguments:JSON.stringify({code:'return await nl`Assess item.`();'})}}]},
      {role:'tool',tool_call_id:'e',content:value}],
    target:{role:'assistant',tool_calls:[{id:'r',function:{name:'return_result',arguments:JSON.stringify({status:'success',value})}}]}};
  const producers=[{id:'child-a',invocation:'a',value},{id:'child-b',invocation:'b',value}];
  const ambiguous=new Map([['run',{returned:[value],read:new Set(),producers}]]);
  const held=convertTrajectory(root,{childResults:ambiguous}).record;
  assert.equal(held.messages[2].content,value);
  assert.equal(held.neuralese_conversion.sites['child-result'].exact['ambiguous-producer'],1);
  const unique=new Map([['run',{returned:[value],read:new Set([value]),producers:producers.slice(0,1)}]]);
  const reader=convertTrajectory(root,{childResults:unique}).record;
  assert.equal(reader.messages[2].content[0].type,'read');
  assert.equal(JSON.parse(reader.target.tool_calls[0].function.arguments).value,value,'root cannot write a child result');
  const child=convertTrajectory({...root,id:'child-a',source_ref:{trajectory_id:'run',invocation_id:'a'}},{childResults:unique}).record;
  assert.equal(JSON.parse(child.target.tool_calls[0].function.arguments).value.$write.name,reader.messages[2].content[0].name);
});

test('observed parentage distinguishes equal returns along a nested chain', () => {
 const value='A long exact evidence record retained through a nested call chain.';
 const producers=[{id:'leaf',invocation:'leaf',parent:'middle',value},{id:'middle',invocation:'middle',parent:'root',value}];
 const readers=[{invocation:'middle',value,producer_id:'leaf'},{invocation:'root',value,producer_id:'middle'}];
 const childResults=new Map([['run',{returned:[value],read:new Set([value]),producers,readers}]]);
 const make=invocation=>({id:invocation,source_ref:{trajectory_id:'run',invocation_id:invocation},messages:[
   {role:'assistant',tool_calls:[{id:'e',function:{name:'eval',arguments:JSON.stringify({code:'return await nl`Read evidence.`();'})}}]},
   {role:'tool',tool_call_id:'e',content:value}],target:{role:'assistant',tool_calls:[{id:'r',function:{name:'return_result',arguments:JSON.stringify({status:'success',value})}}]}});
 const middle=convertTrajectory(make('middle'),{childResults}).record,root=convertTrajectory(make('root'),{childResults}).record;
 const written=JSON.parse(middle.target.tool_calls[0].function.arguments).value.$write;
 assert.equal(root.messages[1].content[0].name,written.name);
 assert.notEqual(middle.messages[1].content[0].name,written.name,'middle reads its child and writes its own distinct return');
 assert.equal(JSON.parse(root.target.tool_calls[0].function.arguments).value,value);
});

test('all exact printer renderings are replaced so a structured result cannot leak beside its block',()=>{
 const value='{"source":"packet_0","quote":"A current report establishes completion."}';
 const printed='{ source: "packet_0", quote: "A current report establishes completion." }';
 const producers=[{id:'child',invocation:'child',parent:'root',value,renderings:[printed]}];
 const childResults=new Map([['run',{returned:[value],read:new Set([value]),producers,readers:[{invocation:'root',value,producer_id:'child'}]}]]);
 const record={id:'root',source_ref:{trajectory_id:'run',invocation_id:'root'},messages:[
  {role:'assistant',tool_calls:[{id:'e',function:{name:'eval',arguments:JSON.stringify({code:'return await nl`Extract.`();'})}}]},
  {role:'tool',tool_call_id:'e',content:`console:\n${value}\nStaged ${printed} as the result.`}]};
 const result=convertTrajectory(record,{childResults}).record.messages[1].content;
 assert.equal(result.filter(p=>p.type==='read').length,2);
 assert.ok(!result.filter(p=>p.type==='text').map(p=>p.text).join('').includes('packet_0'));
});

test('child result indexing never links a future producer to an earlier scope listing', () => {
  const value = 'Which test summary proves that at least 85% of the 20 devices passed?';
  const run = 'causal-run';
  const scope = `objective: string = "${value}"`;
  const root = { id: 'root-at-5', decision: { index: 5 }, source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string' },
      { role: 'assistant', tool_calls: [{ id: 'scope_0', function: { name: 'eval', arguments: '{"code":"x"}' } }] },
      { role: 'tool', tool_call_id: 'scope_0', content: scope },
    ], target: { role: 'assistant', content: value } };
  const producer = { id: 'future-child', decision: { index: 11 },
    source_ref: { trajectory_id: run, invocation_id: 'child', parent_invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: find(): string' },
    ], target: { role: 'assistant', tool_calls: [{ id: 'ret', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value }) } }] } };
  const index = new ChildResultIndexBuilder();
  index.add(producer);
  index.add(root);
  const childResults = index.finish();
  assert.equal(childResults.get(run).read.size, 0);
  assert.equal(childResults.get(run).readers.length, 0);
  const convertedRoot = convertTrajectory(root, { childResults }).record;
  assert.equal(convertedRoot.messages[2].content, scope, 'the original scope input stays exact');
  const convertedProducer = convertTrajectory(producer, { childResults }).record;
  assert.equal(JSON.parse(convertedProducer.target.tool_calls[0].function.arguments).value, value,
    'an unread future result does not become a write');
});

test('child result indexing requires authoritative order when decision indexes are absent', () => {
  const value = 'A later child conclusion with enough detail to become a Neuralese value.';
  const run = 'unknown-order-run';
  const makeRoot = id => ({ id, source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string' },
      { role: 'assistant', tool_calls: [{ id: 'scope_0', function: { name: 'eval', arguments: '{"code":"x"}' } }] },
      { role: 'tool', tool_call_id: 'scope_0', content: `objective: string = "${value}"` },
    ], target: { role: 'assistant', content: value } });
  const makeChild = id => ({ id, source_ref: { trajectory_id: run, invocation_id: 'child', parent_invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl' } } }, messages: [
      { role: 'user', content: 'You are inside this call: find(): string' },
    ], target: { role: 'assistant', tool_calls: [{ id: 'ret', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value }) } }] } });
  const unknown = new ChildResultIndexBuilder();
  unknown.add(makeChild('imported-child'));
  unknown.add(makeRoot('imported-root'));
  const noOrder = unknown.finish().get(run);
  assert.equal(noOrder.readers.length, 0, 'file/ingestion order cannot prove a recurrence edge');
  assert.equal(noOrder.read.size, 0);

  const indexed = new ChildResultIndexBuilder();
  indexed.add(makeChild('teacher:decision:0011'));
  indexed.add(makeRoot('teacher:decision:0005'));
  assert.equal(indexed.finish().get(run).readers.length, 0,
    'an unambiguous decision suffix is authoritative even when rows are shuffled');
  const causal = new ChildResultIndexBuilder();
  causal.add(makeRoot('teacher:decision:0005'));
  causal.add(makeChild('teacher:decision:0004'));
  assert.equal(causal.finish().get(run).readers.length, 1,
    'the suffix recovers a genuinely earlier producer despite reverse ingestion');
});

test('authenticated host-return values stay visible crisp context without a model writer or recurrence edge', () => {
  const value = { reviewId: 'MSD-303', selectedId: 'none', eligibleIds: 'none', priorityScore: '0', decision: 'hold' };
  const encoded = JSON.stringify(value), valueSha = cryptoCreateHash('sha256').update(encoded).digest('hex');
  const run = 'observed-host-run';
  const child = { id: 'child-result', decision: { index: 7 },
    source_ref: { trajectory_id: run, invocation_id: 'child', parent_invocation_id: 'root',
      host_result_capture: { capture: { version: 'reduction-trace/1', kind: 'host_capture',
        capture_kind: 'invocation_output', complete: true,
        origin: 'observed-host-result; not a model-generated writer target', call_id: 'child',
        parent_call_id: 'root', result_type: 'Draft', value, value_sha256: valueSha } } },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } },
    messages: [{ role: 'user', content: 'You are inside this call: decide(): Draft' }],
    target: { role: 'assistant', tool_calls: [{ id: 'ret', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value }) } }] } };
  const parent = { id: 'parent-read', decision: { index: 8 }, source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): Draft' },
      { role: 'assistant', tool_calls: [{ id: 'e', function: { name: 'eval', arguments: '{"code":"const value = await nl`decide()`(); return value;"}' } }] },
      { role: 'tool', tool_call_id: 'e', content: `console:\n${encoded}\nnull` },
    ], target: { role: 'assistant', content: 'done' } };
  const index = new ChildResultIndexBuilder();
  index.add(child); index.add(parent);
  const indexed = index.finish().get(run);
  assert.equal(indexed.producers.length, 0, 'host capture is not model-authored producer evidence');
  assert.equal(indexed.readers.length, 0, 'host capture does not create a recurrence edge');
  assert.equal(indexed.observed_host_contexts.length, 1);
  const convertedChild = convertTrajectory(child, { childResults: index.finish() }).record;
  assert.deepEqual(JSON.parse(convertedChild.target.tool_calls[0].function.arguments).value, value);
  const convertedParent = convertTrajectory(parent, { childResults: index.finish() }).record;
  assert.equal(convertedParent.messages[2].content, parent.messages[2].content,
    'the exact already-visible tool transcript remains crisp and unchanged');
  assert.equal(convertedParent.neuralese_conversion.sites['child-result'].exact['observed-host-result'], 1);
  assert.deepEqual(convertedParent.neuralese_conversion.observed_host_result_contexts, [{
    schema: 'natlang.observed-host-result-context/1', origin: 'completed-child-invocation-output',
    invocation_id: 'root', tool_call_id: 'e', producer_record_id: 'child-result', result_type: 'Draft',
    body_sha256: cryptoCreateHash('sha256').update(encoded).digest('hex'),
    capture_sha256: cryptoCreateHash('sha256').update(JSON.stringify(child.source_ref.host_result_capture.capture)).digest('hex'),
    visible_as_crisp_context: true, model_writer_target: false, recurrence_edge: false,
  }]);
  const tampered = structuredClone(child);
  tampered.source_ref.host_result_capture.capture.value_sha256 = '0'.repeat(64);
  const rejected = new ChildResultIndexBuilder(); rejected.add(tampered); rejected.add(parent);
  assert.equal(rejected.finish().get(run).observed_host_contexts.length, 0,
    'a mismatched host capture digest cannot authorize context provenance');
});

test('reader links use tool message IDs and choose whole-object blocks over overlapping field blocks', () => {
  const facts = 'Eighteen of twenty devices passed the receiving test at the required threshold.';
  const structured = { facts, status: 'complete' };
  const value = JSON.stringify(structured);
  const run = 'field-whole-run';
  const root = { id: 'root', decision: { index: 3 }, source_ref: { trajectory_id: run, invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: root(): string' },
      { role: 'assistant', tool_calls: [
        { id: 'scope_0', function: { name: 'eval', arguments: '{"code":"x"}' } },
        { id: 'e_facts', function: { name: 'eval', arguments: '{"code":"await child()"}' } },
      ] },
      { role: 'tool', tool_call_id: 'scope_0', content: `evidence: unknown = "${value}"` },
      { role: 'tool', tool_call_id: 'e_facts', content: `console:\n${facts}` },
    ], target: { role: 'assistant', content: 'done' } };
  const child = { id: 'child', decision: { index: 1 },
    source_ref: { trajectory_id: run, invocation_id: 'child', parent_invocation_id: 'root' },
    task: { program_ir: { semantics: { root: 'root.nl', files: { 'root.nl': '', 'child.nl': '' } } } }, messages: [
      { role: 'user', content: 'You are inside this call: child(): Evidence' },
    ], target: { role: 'assistant', tool_calls: [{ id: 'ret', function: { name: 'return_result',
      arguments: JSON.stringify({ status: 'success', value: structured }) } }] } };
  const index = new ChildResultIndexBuilder();
  index.add(root);
  index.add(child);
  const childResults = index.finish().get(run);
  assert.deepEqual(childResults.readers.map(({ tool_call_id, value: read }) => [tool_call_id, read]), [['scope_0', value]],
    'the specific whole-object observation wins; the overlapping facts field is not an orphan read');
  const convertedChild = convertTrajectory(child, { childResults: new Map([[run, childResults]]) }).record;
  const result = JSON.parse(convertedChild.target.tool_calls[0].function.arguments).value;
  assert.ok(result.$write, 'the whole-object reader has a matching whole-object writer');
  assert.equal(result.$write.source, value, 'the whole-object writer owns the complete structured result');
  const convertedRoot = convertTrajectory(root, { childResults: new Map([[run, childResults]]) }).record;
  assert.ok(convertedRoot.messages[2].content.some(part => part.type === 'read'));
  assert.equal(convertedRoot.messages[3].content, `console:\n${facts}`, 'a separate tool message has no unrelated field link');
});

test('provider-expanded configured function and producer blocks become source-bound context, not new writes', () => {
  const digest = text => cryptoCreateHash('sha256').update(text).digest('hex');
  const functionId = `nz1_${'a'.repeat(32)}`, noteId = `nz1_${'b'.repeat(32)}`;
  const functionBody = 'Read the value held by v and return it exactly.';
  const noteBody = 'The note from the earlier producer.';
  const invocation = 'call-7', readNode = `${invocation}#3`, turnNode = `${invocation}#turn1`;
  const makeReceipt = (id, type, body, origin) => ({
    schema: origin === 'same-run-producer' ? 'natlang.provider-expanded-read-context/3' :
      'natlang.provider-expanded-read-context/1', origin, invocation_id: invocation,
    source_row_sha256: '1'.repeat(64), trace_sha256: '2'.repeat(64),
    transport_provenance_sha256: '3'.repeat(64), raw_request_sha256: '4'.repeat(64),
    rendered_request_sha256: '5'.repeat(64), learned_vectors: false,
    qualification_certificate: false, training_admission: false,
    block: { id, type, body, body_sha256: digest(body), learned_vectors: false },
    readout: origin === 'configured-function-definition' ? {
      schema: 'natlang.text-template-readout/1', call: 'return_result', value: 'decode', value_type: 'string',
      read_body_id: id, read_source_sha256: digest(body), learned_vectors: false,
      qualification_certificate: false, training_admission: false,
    } : null,
    definition: origin === 'configured-function-definition' ? { id: `nz-fn:${id}` } : null,
    block_read: { kind: 'block_read', call_id: invocation, block: id, node: readNode, seq: 3,
      inputs: origin === 'same-run-producer' ? [{ node: `${invocation}#2`, block: id, port: 'block' }] : [] },
    model_turn: { kind: 'model_turn', call_id: invocation, node: turnNode, seq: 4,
      inputs: [{ node: readNode, port: 'read', block: id }] },
    context_occurrences: 1,
    producer_write: origin === 'same-run-producer' ? { kind: 'block_write', call_id: invocation, seq: 2, block: id,
      node: `${invocation}#2`, truncated: false, learned_vectors: false, producer: 'text-marker-emulation',
      source_kind: 'typed-text-result', result_type: type, text_body_sha256: digest(body) } : null,
    ...(origin === 'same-run-producer' ? { writer_source_class: 'modern-typed-text-result', writer_target_selected: false } : {}),
  });
  const row = { id: 'row-7', decision: { index: 4 }, source_ref: { trajectory_id: 'run-1', invocation_id: invocation,
    source_row_sha256: '1'.repeat(64), provider_expanded_read_contexts: [
      makeReceipt(functionId, 'Neuralese<(v: Neuralese<unknown>) => unknown>', functionBody, 'configured-function-definition'),
      { ...makeReceipt(noteId, 'Neuralese<string>', noteBody, 'same-run-producer'), serialized_literal_id_mentions: 1 },
    ] }, provenance: { trace_sha256: '2'.repeat(64) }, messages: [
      { role: 'system', content: 'System.' }, { role: 'user', content: [{ type: 'text', text: 'Prompt: ' },
        { type: 'neuralese', id: functionId }] },
      { role: 'assistant', content: null, tool_calls: [{ id: 'eval-note', type: 'function',
        function: { name: 'eval', arguments: JSON.stringify({ code: 'const note = await nl<string>`Write the note.`;' }) } }] },
      { role: 'tool', tool_call_id: 'eval-note', content: 'stored note' },
      { role: 'tool', tool_call_id: 'scope_0', content: [{ type: 'text', text: 'note: ' },
        { type: 'neuralese', id: noteId }] },
      { role: 'tool', tool_call_id: 'debug', content: `console: {"$neuralese":{"type":"Neuralese<string>","id":"${noteId}"}}` },
    ], target: { role: 'assistant', content: 'done' } };
  const { record: out, pieces } = convertTrajectory(row);
  assert.equal(out.messages[0].role, 'system');
  assert.equal(out.messages[1].role, 'user');
  assert.deepEqual(out.messages[1].content, [{ type: 'text', text: 'Prompt: ' }, { type: 'text', text: functionBody }],
    'configured function body stays crisp at its exact request-context location');
  assert.equal(out.messages[2].tool_calls[0].function.name, 'eval',
    'the same invocation records the producer action before the context read');
  assert.equal(out.messages[3].tool_call_id, 'eval-note');
  assert.deepEqual(out.messages[4].content, [{ type: 'text', text: 'note: ' },
    { type: 'read', name: `soft-state:${noteId}`, source: noteBody }],
  'a graph-authenticated same-run producer remains a typed read of its original writer');
  assert.equal(out.neuralese_conversion.external_context_inputs.length, 2);
  assert.ok(out.neuralese_conversion.external_context_inputs.every(item => item.learned_vectors === false &&
    item.qualification_certificate === false && item.training_admission === false));
  assert.equal(out.neuralese_conversion.external_context_inputs.find(item => item.block_id === noteId).learner_representation,
    'typed-read-linked-to-existing-writer');
  assert.equal(out.neuralese_conversion.external_context_inputs.find(item => item.block_id === noteId).producer_write_node,
    `${invocation}#2`, 'the context read remains bound to the earlier same-invocation producer node');
  assert.equal(out.neuralese_conversion.external_context_inputs.find(item => item.block_id === noteId).context_occurrences, 1,
    'only the structured typed reference remains a Neuralese read');
  assert.equal(out.neuralese_conversion.external_context_inputs.find(item => item.block_id === noteId).serialized_literal_id_mentions, 1,
    'serialized debug ID mentions remain separately observable');
  assert.equal(out.messages[5].content, `console: {"$neuralese":{"type":"Neuralese<string>","id":"${noteId}"}}`,
    'quoted JSON text remains literal rather than becoming a vector read');
  assert.equal(out.neuralese_conversion.external_context_inputs.find(item => item.block_id === functionId).learner_representation,
    'crisp-external-function-context');
  assert.ok(pieces.every(piece => piece.kind !== 'function-body'), 'external context does not create soft function-body targets');
  const implicit = structuredClone(row);
  implicit.source_ref.provider_expanded_read_contexts[0].context_occurrences = 0;
  implicit.messages[1].content = 'Prompt.';
  const implicitOut = convertTrajectory(implicit).record;
  assert.equal(implicitOut.messages[1].content, functionBody,
    'a configured body omitted from canonical messages is carried as a crisp system context input');
  const corrupt = structuredClone(row);
  corrupt.source_ref.provider_expanded_read_contexts[0].block.body = 'changed';
  assert.throws(() => convertTrajectory(corrupt), /provider-expanded context provenance mismatch/);
  const duplicated = structuredClone(row);
  duplicated.messages.push({ role: 'user', content: [{ type: 'neuralese', id: functionId }] });
  assert.throws(() => convertTrajectory(duplicated), /configured function block has ambiguous message occurrences/);
});

test('generic provider-expanded same-run inputs hydrate as context-only typed reads with exact graph binding', () => {
  const row = { id: 'context-only-row', decision: { index: 2 },
    source_ref: { trajectory_id: 'context-run', invocation_id: 'reader-call', source_row_sha256: '1'.repeat(64) },
    provenance: { trace_sha256: '2'.repeat(64) },
    messages: [{ role: 'user', content: [{ type: 'text', text: 'Prior notes: ' },
      { type: 'neuralese', id: `nz1_${'f'.repeat(52)}` }] }],
    target: { role: 'assistant', content: 'Decision.' } };
  const id = `nz1_${'f'.repeat(52)}`, body = 'Exact provider-visible note body.';
  const bodySha = cryptoCreateHash('sha256').update(body).digest('hex');
  const write = { kind: 'block_write', call_id: 'writer-call', node: 'writer-call#8', seq: 8,
    block: id, truncated: false, producer: 'text-marker-emulation', source_kind: 'typed-text-result',
    result_type: 'Neuralese<string>', text_body_sha256: bodySha };
  const read = { kind: 'block_read', call_id: 'reader-call', node: 'reader-call#3', seq: 10,
    block: id, inputs: [{ node: write.node, block: id, port: 'block' }] };
  const turn = { kind: 'model_turn', call_id: 'reader-call', node: 'reader-call#turn1',
    inputs: [{ node: read.node, port: 'read', block: id }] };
  const read2 = { kind: 'block_read', call_id: 'reader-call', node: 'reader-call#7', block: id,
    inputs: [{ node: write.node, block: id, port: 'block' }] };
  const turn2 = { kind: 'model_turn', call_id: 'reader-call', node: 'reader-call#turn2',
    inputs: [{ node: read2.node, port: 'read', block: id }] };
  row.source_ref.parent_invocation_id = 'root-call';
  row.source_ref.provider_expanded_read_contexts = [{
    schema: 'natlang.provider-expanded-read-context/2', origin: 'same-run-producer',
    invocation_id: 'reader-call', parent_invocation_id: 'root-call', source_row_sha256: '1'.repeat(64),
    trace_sha256: '2'.repeat(64), transport_provenance_sha256: '3'.repeat(64),
    raw_request_sha256: '4'.repeat(64), rendered_request_sha256: '5'.repeat(64),
    block: { id, type: 'Neuralese<string>', body, body_sha256: bodySha, learned_vectors: false },
    block_read: read, model_turn: turn,
    additional_read_turn_pairs: [{ block_read: read2, model_turn: turn2 }],
    context_occurrences: 1, producer_write: write,
    writer_target_selected: false, writer_source_class: 'modern-typed-text-result',
    learned_vectors: false, qualification_certificate: false, training_admission: false,
  }];
  const converted = convertTrajectory(row).record;
  assert.deepEqual(converted.messages[0].content[1], { type: 'read', name: `soft-state:${id}`, source: body });
  const context = converted.neuralese_conversion.external_context_inputs[0];
  assert.equal(context.learner_representation, 'typed-read-from-authenticated-runtime-writer-event-context-only');
  assert.equal(context.writer_target_selected, false);
  assert.equal(context.writer_source_class, 'modern-typed-text-result');
  assert.deepEqual(context.additional_read_nodes, [read2.node]);
  assert.deepEqual(context.additional_model_turn_nodes, [turn2.node]);
  assert.equal(converted.neuralese_conversion.sites['typed-result-write'], undefined,
    'context hydration does not create a model writer target');
  const repeatedContext = structuredClone(row);
  repeatedContext.source_ref.provider_expanded_read_contexts[0].context_occurrences = 2;
  repeatedContext.messages.push({ role: 'assistant', content: null, tool_calls: [{ id: 'scope_0', type: 'function',
    function: { name: 'scope', arguments: JSON.stringify([{ type: 'neuralese', id }]) } }] });
  const repeatedConverted = convertTrajectory(repeatedContext).record;
  assert.equal(repeatedConverted.neuralese_conversion.external_context_inputs[0].context_occurrences, 2,
    'the receipt counts both the ordinary message reference and nested tool-argument reference');
  assert.deepEqual(repeatedConverted.messages[0].content[1], { type: 'read', name: `soft-state:${id}`, source: body });
  assert.match(repeatedConverted.messages[1].tool_calls[0].function.arguments, new RegExp(id),
    'native tool transcript remains raw for source faithful replay; text rendering hydrates the same bound occurrence');
  const corrupt = structuredClone(row);
  corrupt.source_ref.provider_expanded_read_contexts[0].block.body = 'changed';
  assert.throws(() => convertTrajectory(corrupt), /provider-expanded context provenance mismatch/);
  const badGraph = structuredClone(row);
  badGraph.source_ref.provider_expanded_read_contexts[0].additional_read_turn_pairs[0].model_turn.inputs = [];
  assert.throws(() => convertTrajectory(badGraph), /invalid repeated read\/turn binding/);

  const sidecarRow = structuredClone(row);
  const sidecarBlockId = `nz1_${'a'.repeat(52)}`;
  const sidecarCodeSource = `\uE000${sidecarBlockId}\uE001`;
  const sidecarPrefix = 'const result = nl.with({input})`';
  const sidecarSuffix = '`(input);';
  const sidecarCode = sidecarPrefix + sidecarCodeSource + sidecarSuffix;
  const sidecarName = 'inline-site:adapter-fixture';
  const sidecarHash = value => cryptoCreateHash('sha256').update(value).digest('hex');
  const sidecarSpan = { start: sidecarPrefix.length - 1,
    end: sidecarPrefix.length + sidecarCodeSource.length + 1 };
  const capturePlan = { capture_binding_plan: { schema: 'natlang.inline-capture-binding-plan/1', syntax: 'nl.with',
    body_block_id: sidecarBlockId, body_source_sha256: sidecarHash('approved'), parent_invocation_id: 'parent',
    parent_scope_sha256: '1'.repeat(64), child_scope_sha256: '2'.repeat(64),
    captures: [{ name: 'input', type: 'string', mode: 'snapshot', value: 'visible' }] } };
  const sidecar = { schema: 'natlang.inline-instruction-code/1', code_sha256: sidecarHash(sidecarCode),
    parts: [{ type: 'text', text: sidecarPrefix },
      { $write: { name: sidecarName, type: 'Neuralese<string>', source: 'approved', code_source: sidecarCodeSource } },
      { type: 'text', text: sidecarSuffix }],
    sites: [{ name: sidecarName, code_span: sidecarSpan, plan: capturePlan }] };
  const targetCall = { id: 'scope-sidecar', type: 'function', neuralese_code: sidecar,
    function: { name: 'scope', arguments: JSON.stringify({ code: sidecarCode }) } };
  sidecarRow.target = { role: 'assistant', content: null, tool_calls: [targetCall] };
  const projectedTarget = structuredClone(sidecarRow.target);
  delete projectedTarget.tool_calls[0].neuralese_code;
  const projectedHash = sidecarHash(canonical(projectedTarget));
  const protectedHash = sidecarHash(canonical(sidecarRow.target));
  const adapter = { schema: 'natlang.protected-target-inline-sidecar-equivalence/1',
    kind: 'remove-one-validated-neuralese-code-sidecar', protected_target_sha256: protectedHash,
    materializer_target_sha256: projectedHash, projection_sha256: projectedHash, call_id: targetCall.id,
    sidecar_code_sha256: sidecar.code_sha256, sidecar_sha256: sidecarHash(canonical(sidecar)) };
  const sidecarReceipt = sidecarRow.source_ref.provider_expanded_read_contexts[0];
  sidecarReceipt.target_binding_adapter = adapter;
  sidecarReceipt.source_trajectory_index = 2;
  sidecarReceipt.source_action_target_sha256 = projectedHash;
  sidecarReceipt.source_request_sha256 = '6'.repeat(64);
  sidecarReceipt.source_response_sha256 = '7'.repeat(64);
  sidecarRow.source_ref.provider_expanded_read_contexts = [sidecarReceipt];
  sidecarRow.source_ref.provider_expanded_read_contexts[0].context_occurrences = 1;
  // Preserve the rest of the receipt contract while binding both copies to this target adapter.
  const sidecarConverted = convertTrajectory(sidecarRow).record;
  assert.equal(sidecarConverted.neuralese_conversion.external_context_inputs[0].target_binding_adapter.projection_sha256,
    projectedHash);
  const badSidecarArgs = structuredClone(sidecarRow);
  badSidecarArgs.target.tool_calls[0].function.arguments = JSON.stringify({ code: 'return false' });
  assert.throws(() => convertTrajectory(badSidecarArgs), /selected-action binding mismatch/);
  const badSidecarCallId = structuredClone(sidecarRow);
  badSidecarCallId.source_ref.provider_expanded_read_contexts[0].target_binding_adapter.call_id = 'wrong-call';
  assert.throws(() => convertTrajectory(badSidecarCallId), /selected-action binding mismatch/);
  const badSidecarProjection = structuredClone(sidecarRow);
  badSidecarProjection.source_ref.provider_expanded_read_contexts[0].target_binding_adapter.projection_sha256 = '8'.repeat(64);
  assert.throws(() => convertTrajectory(badSidecarProjection), /selected-action binding mismatch/);
  const extraRemovedField = structuredClone(sidecarRow);
  extraRemovedField.target.tool_calls[0].unexpected = true;
  extraRemovedField.source_ref.provider_expanded_read_contexts[0].target_binding_adapter.protected_target_sha256 =
    sidecarHash(canonical(extraRemovedField.target));
  assert.throws(() => convertTrajectory(extraRemovedField), /selected-action binding mismatch/,
    'the adapter removes only neuralese_code; an extra target change is not projected away');
  for (const field of ['name', 'type', 'span']) {
    const tampered = structuredClone(sidecarRow);
    const changedSidecar = tampered.target.tool_calls[0].neuralese_code;
    if (field === 'name') changedSidecar.parts[1].$write.name = 'renamed-site';
    else if (field === 'type') changedSidecar.parts[1].$write.type = 'Neuralese<number>';
    else changedSidecar.sites[0].code_span.start += 1;
    const changedProjection = structuredClone(tampered.target);
    delete changedProjection.tool_calls[0].neuralese_code;
    const receipt = tampered.source_ref.provider_expanded_read_contexts[0];
    const changedAdapter = receipt.target_binding_adapter;
    changedAdapter.protected_target_sha256 = sidecarHash(canonical(tampered.target));
    changedAdapter.sidecar_sha256 = sidecarHash(canonical(changedSidecar));
    assert.throws(() => convertTrajectory(tampered), /selected-action binding mismatch/,
      `rehashed ${field} tampering must fail sidecar validation`);
  }

  const legacy = structuredClone(row);
  const legacyReceipt = legacy.source_ref.provider_expanded_read_contexts[0];
  legacyReceipt.writer_source_class = 'legacy-text-marker-standin-eval-code';
  delete legacyReceipt.producer_write.producer;
  delete legacyReceipt.producer_write.source_kind;
  legacyReceipt.producer_write.emulation_version = 'text-marker-standin/2';
  legacyReceipt.producer_write.marker_context = 'eval-code';
  legacyReceipt.producer_write.learned_vectors = false;
  const legacyConverted = convertTrajectory(legacy).record;
  assert.equal(legacyConverted.messages[0].content[1].source, body);
  assert.equal(legacyConverted.neuralese_conversion.external_context_inputs[0].writer_source_class,
    'legacy-text-marker-standin-eval-code');

  const legacyReturn = structuredClone(row);
  const returnReceipt = legacyReturn.source_ref.provider_expanded_read_contexts[0];
  Object.assign(returnReceipt.producer_write, { producer: 'text-marker-emulation', source_kind: 'typed-text-result',
    source: 'return_result', marker_context: 'return-result' });
  returnReceipt.writer_source_class = 'legacy-text-marker-standin-return-result';
  returnReceipt.writer_witness = { kind: 'raw-return-result-value-equals-expanded-body', source: 'return_result',
    host_result_call_id: write.call_id, host_result_type: 'Neuralese<string>',
    host_result_value_sha256: '6'.repeat(64), raw_response_sha256: '7'.repeat(64) };
  const returnConverted = convertTrajectory(legacyReturn).record;
  assert.equal(returnConverted.neuralese_conversion.external_context_inputs[0].writer_witness.kind,
    'raw-return-result-value-equals-expanded-body');
  const missingRawWitness = structuredClone(legacyReturn);
  missingRawWitness.source_ref.provider_expanded_read_contexts[0].writer_witness.raw_response_sha256 = null;
  assert.throws(() => convertTrajectory(missingRawWitness), /provider-expanded producer context lacks an earlier writer/);

  const legacyFinish = structuredClone(row);
  const finishReceipt = legacyFinish.source_ref.provider_expanded_read_contexts[0];
  Object.assign(finishReceipt.producer_write, { producer: 'text-marker-emulation', source_kind: 'typed-text-result',
    source: 'eval-finish', marker_context: 'return-result' });
  finishReceipt.writer_source_class = 'legacy-text-marker-standin-eval-finish';
  finishReceipt.writer_witness = { kind: 'completed-eval-finish-host-reference', source: 'eval-finish',
    host_result_call_id: write.call_id, host_result_type: 'Neuralese<string>', host_result_value_sha256: '8'.repeat(64) };
  assert.equal(convertTrajectory(legacyFinish).record.neuralese_conversion.external_context_inputs[0].writer_witness.kind,
    'completed-eval-finish-host-reference');
  const mismatchedFinish = structuredClone(legacyFinish);
  mismatchedFinish.source_ref.provider_expanded_read_contexts[0].writer_witness.host_result_call_id = 'other-call';
  assert.throws(() => convertTrajectory(mismatchedFinish), /provider-expanded producer context lacks an earlier writer/);

  const legacyEvalReturn = structuredClone(row);
  const evalReturnReceipt = legacyEvalReturn.source_ref.provider_expanded_read_contexts[0];
  Object.assign(evalReturnReceipt.producer_write, { producer: 'text-marker-emulation', source_kind: 'typed-text-result',
    source: 'eval-return', marker_context: 'return-result' });
  evalReturnReceipt.writer_source_class = 'legacy-text-marker-standin-eval-return';
  evalReturnReceipt.writer_witness = { kind: 'completed-eval-return-host-reference', source: 'eval-return',
    host_result_call_id: write.call_id, host_result_type: 'Neuralese<string>',
    host_result_value_sha256: '9'.repeat(64), completion_status: 'done', completion_source: 'execution_graph',
    completion_detail: `\uE000${id}\uE001` };
  const evalReturnConverted = convertTrajectory(legacyEvalReturn).record;
  assert.equal(evalReturnConverted.neuralese_conversion.external_context_inputs[0].writer_witness.kind,
    'completed-eval-return-host-reference');
  for (const patch of [
    { completion_status: 'failed' }, { completion_source: 'model' },
    { completion_detail: `\uE000nz1_${'e'.repeat(52)}\uE001` }, { host_result_call_id: 'other-call' },
  ]) {
    const mismatch = structuredClone(legacyEvalReturn);
    Object.assign(mismatch.source_ref.provider_expanded_read_contexts[0].writer_witness, patch);
    assert.throws(() => convertTrajectory(mismatch), /provider-expanded producer context lacks an earlier writer/);
  }
});

test('same-invocation earlier writer context remains context-only and requires ordered graph nodes', () => {
  const row = { id: 'same-invocation-context-row', decision: { index: 2 },
    source_ref: { trajectory_id: 'context-run', invocation_id: 'reader-call', source_row_sha256: '1'.repeat(64) },
    provenance: { trace_sha256: '2'.repeat(64) },
    messages: [{ role: 'user', content: [{ type: 'neuralese', id: `nz1_${'a'.repeat(52)}` }] }],
    target: { role: 'assistant', content: 'Continue.' } };
  const id = `nz1_${'a'.repeat(52)}`, body = 'Exact earlier same-call result.';
  const bodySha = cryptoCreateHash('sha256').update(body).digest('hex');
  const write = { kind: 'block_write', call_id: 'reader-call', node: 'reader-call#8', seq: 8,
    block: id, truncated: false, producer: 'text-marker-emulation', source_kind: 'typed-text-result',
    result_type: 'Neuralese<string>', text_body_sha256: bodySha };
  const read = { kind: 'block_read', call_id: 'reader-call', node: 'reader-call#10', seq: 10,
    block: id, inputs: [{ node: write.node, block: id, port: 'block' }] };
  const turn = { kind: 'model_turn', call_id: 'reader-call', node: 'reader-call#turn2',
    inputs: [{ node: read.node, port: 'read', block: id }] };
  row.source_ref.provider_expanded_read_contexts = [{
    schema: 'natlang.provider-expanded-read-context/2', origin: 'same-run-producer',
    invocation_id: 'reader-call', parent_invocation_id: null, source_row_sha256: '1'.repeat(64),
    trace_sha256: '2'.repeat(64), transport_provenance_sha256: '3'.repeat(64),
    raw_request_sha256: '4'.repeat(64), rendered_request_sha256: '5'.repeat(64),
    block: { id, type: 'Neuralese<string>', body, body_sha256: bodySha, learned_vectors: false },
    block_read: read, model_turn: turn, additional_read_turn_pairs: [], context_occurrences: 1,
    producer_write: write, writer_target_selected: false, writer_source_class: 'modern-typed-text-result',
    learned_vectors: false, qualification_certificate: false, training_admission: false,
  }];
  const converted = convertTrajectory(row).record;
  assert.deepEqual(converted.messages[0].content[0], { type: 'read', name: `soft-state:${id}`, source: body });
  const context = converted.neuralese_conversion.external_context_inputs[0];
  assert.equal(context.writer_target_selected, false);
  assert.equal(context.learner_representation, 'typed-read-from-authenticated-runtime-writer-event-context-only');
  assert.equal(converted.neuralese_conversion.sites['typed-result-write'], undefined);

  const unordered = structuredClone(row);
  unordered.source_ref.provider_expanded_read_contexts[0].producer_write.seq = 11;
  assert.throws(() => convertTrajectory(unordered), /provider-expanded producer context lacks an earlier writer/);
  const selected = structuredClone(row);
  selected.source_ref.provider_expanded_read_contexts[0].writer_target_selected = true;
  assert.throws(() => convertTrajectory(selected), /provider-expanded context-only writer receipt is incomplete/);
});
