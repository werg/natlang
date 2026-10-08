import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { nativeDecisionTargetDigest } from '../dist/native/decision-review.js';
import { markAuthoredStaticReferencePending, materializeNativeRows, nativeRowDigest } from '../dist/teacher/native-materializer.js';

const system = { role: 'system', content: 'Use the native scope tools.' };
const opening = { role: 'user', content: '1 [ ] Compute the result. Current inputs: {"n": 3}' };
const firstCall = { tool: 'write', source_tool: 'write', arguments: { path: 'return', type: 'number', value: 6 }, call_id: null };
const secondCall = { tool: 'mark_done', source_tool: 'mark_done', arguments: { start: 1, end: 1 }, call_id: null };
const schema = [{ type: 'function', function: { name: 'write', description: 'Write a value.',
  parameters: { type: 'object', properties: { path: { type: 'string' } } } } }];
const nativeRow = (id, accepted = true) => ({ version: 'natlang.teacher_trajectory.native/1', id,
  task: { kind: 'whole_program', program_ir: { id: 'program-1', version: 'natlang.program/2' }, source_program_ids: ['program-1'] },
  provenance: { model: 'fixture-teacher', tool_schema: 'scope-eval-v1', context_tokens: 16384 },
  outcome: { status: 'done', accepted, value: 6, action_ledger: [
    { seq: 12, name: 'write', arguments: firstCall.arguments, outcome: 'ok', result_text: 'stored value' },
    { seq: 17, name: 'mark_done', arguments: secondCall.arguments, outcome: 'ok', result_text: 'marked line done' },
  ] },
  trajectory: [
    { phase: 'action', context: [system, opening], tools_offered: schema,
      assistant: { content: '', reasoning: 'The result is twice n.', calls: [firstCall] }, raw_response_sha256: 'raw-1' },
    { phase: 'action', context: [system, opening,
      { role: 'assistant', content: '', tool_calls: [{ id: 'call-1', type: 'function',
        function: { name: 'write', arguments: JSON.stringify(firstCall.arguments) } }] },
      { role: 'tool', tool_call_id: 'call-1', content: 'stored value' }],
      tools_offered: schema, assistant: { content: '', reasoning: 'The computation is complete.', calls: [secondCall] },
      raw_response_sha256: 'raw-2' },
  ], capture_limits: [] });

const softBlock = `nz1_${'s'.repeat(32)}`;
const softCallId = 'child-soft-1';
const softMarker = '<|neuralese|>A constructed soft note.<|/neuralese|>';
const softSentinel = `${softBlock}`;
function softEvalRow({ id = 'soft-eval-linked', marker = '<|neuralese|>note body<|/neuralese|>',
  actionCode = `const note: Neuralese<string> = ${softSentinel};`, callId = 'soft-eval-1' } = {}) {
  const row = nativeRow(id);
  const sourceCode = `const note: Neuralese<string> = ${marker}${marker.endsWith('<|/neuralese|>') ? ';' : ''}`;
  const arguments_ = { code: sourceCode, finish: true };
  const actionArguments = { code: actionCode, finish: true };
  row.outcome.action_ledger = [{ seq: 4, call_id: callId, name: 'eval', arguments: actionArguments,
    outcome: 'ok', result_text: 'evaluated' }];
  row.trajectory = [{ phase: 'action', invocation_id: callId, context: [system, opening], tools_offered: schema,
    assistant: { content: '', reasoning: 'Record the note.', calls: [{ tool: 'eval', source_tool: 'eval',
      arguments: arguments_, call_id: 'raw-eval-1' }] }, model_response: { raw_calls: [{ function: { name: 'eval',
      arguments: JSON.stringify(arguments_) } }] }, raw_response_sha256: 'raw-soft-eval' }];
  return row;
}

function softReturnRow(id = 'soft-return-linked') {
  const row = nativeRow(id);
  const arguments_ = { status: 'success', value: softMarker };
  const actionArguments = { status: 'success', value: softSentinel };
  const turnNode = `${softCallId}#turn1`;
  row.outcome.action_ledger = [{ seq: 9, call_id: softCallId, name: 'return_result', arguments: actionArguments,
    outcome: 'completed', result_text: `Returned ${softSentinel}.` }];
  row.outcome.execution_graph = [
    { kind: 'model_turn', seq: 5, call_id: softCallId, node: turnNode, turn: 1,
      inputs: [{ node: `call:${softCallId}`, port: 'invocation' }] },
    { kind: 'block_write', seq: 6, call_id: softCallId, block: softBlock, node: `${softCallId}#6`,
      turn: turnNode, inputs: [{ node: turnNode, port: 'turn' }] }
  ];
  row.outcome.invocation_ledger = [{ invocation_id: softCallId,
    inline_instruction_site: { returns: { natlang: 'Neuralese<string>' } },
    host_result: { kind: 'host_capture', capture_kind: 'invocation_output', call_id: softCallId,
      complete: true, result_type: 'Neuralese<string>',
      value: { $neuralese: { type: 'Neuralese<string>', id: softBlock } }, terminal_action_seq: 9 } }];
  row.trajectory = [{ phase: 'action', invocation_id: softCallId, context: [system, opening], tools_offered: schema,
    assistant: { content: '', reasoning: 'Return the typed soft note.', calls: [{ tool: 'return_result', source_tool: 'return_result',
      arguments: arguments_, call_id: null }] }, model_response: { raw_calls: [{ function: { name: 'return_result',
      arguments: JSON.stringify(arguments_) } }] }, raw_response_sha256: 'raw-soft-return' }];
  return row;
}

test('native lineage hashes the saved JSON representation', () => {
  const row = nativeRow('json-lineage');
  row.provenance.optional = undefined;
  row.outcome.optional = { omitted: undefined, array: [undefined] };
  const saved = JSON.parse(JSON.stringify(row));
  assert.equal(nativeRowDigest(row), nativeRowDigest(saved));
  const turns = materializeNativeRows([row], { directAnswers: true }).turns;
  assert.ok(turns.length);
  assert.ok(turns.every(t => t.source_ref.source_row_sha256 === nativeRowDigest(saved)));
});

test('status-only success requires a same-invocation linked staged typed result unless the return type is void', () => {
  const stageCallId = 'stage-return';
  const stagedText = 'Staged { answer: "ready" } as the result. If this is the result of the task you were given and you are satisfied with it, reply done to return exactly this value without a tool call, or call return_result with status "success" and omit value to finish using this exact stored result. You can keep working and return a different value later.';
  const stageCall = { id: stageCallId, type: 'function', function: { name: 'eval', arguments: '{"code":"draft"}' } };
  const make = ({ withStage = true, linked = true, stageInvocation = 'same-invocation', returnType = '{ answer: string }',
    resultArgs = { status: 'success' }, stageOutput = stagedText, ledgerToolCallId = true } = {}) => {
    const row = nativeRow(`status-only-${withStage}-${linked}-${returnType}`);
    const user = { role: 'user', content: `You are inside this call: root(): ${returnType}\n\nInstructions:\nReturn the staged value.` };
    const context = [system, user];
    if (withStage) context.push({ role: 'assistant', content: '', tool_calls: [stageCall] },
      { role: 'tool', tool_call_id: linked ? stageCallId : 'other-eval', content: stageOutput });
    row.outcome.action_ledger = [
      ...(withStage ? [{ seq: 1, call_id: stageInvocation, ...(ledgerToolCallId ? { tool_call_id: stageCallId } : {}),
        name: 'eval', arguments: { code: 'draft' }, outcome: 'ok', result_text: stageOutput }] : []),
      { seq: 2, call_id: 'same-invocation', name: 'return_result', arguments: resultArgs, outcome: 'completed', result_text: 'Returned staged result.' },
    ];
    row.trajectory = [
      ...(withStage ? [{ phase: 'action', invocation_id: 'same-invocation', context: [system, user], tools_offered: schema,
        assistant: { content: '', reasoning: 'Evaluate the result.', calls: [{ tool: 'eval', source_tool: 'eval',
          arguments: { code: 'draft' }, call_id: stageCallId }] }, raw_response_sha256: 'status-only-stage-raw' }] : []),
      { phase: 'action', invocation_id: 'same-invocation', context, tools_offered: schema,
        assistant: { content: '', reasoning: 'Finish with the staged result.', calls: [{ tool: 'return_result', source_tool: 'return_result',
          arguments: resultArgs, call_id: null }] }, raw_response_sha256: 'status-only-raw' },
    ];
    return row;
  };

  const valid = materializeNativeRows([make()]).turns.at(-1);
  assert.equal(valid.training_admission.approved, true);
  assert.deepEqual(valid.decision.status_only_success_validation[0].proof, {
    schema: 'natlang.status-only-success-proof/1', basis: 'same-invocation-staged-result',
    invocation_id: 'same-invocation', staged_call_id: stageCallId, staged_action_linkage: 'tool-call-id',
    staged_action_seq: 1, terminal_action_seq: 2,
    staged_output_sha256: createHash('sha256').update(stagedText).digest('hex'), declared_return_type: '{ answer: string }',
  });
  const legacyLinked = materializeNativeRows([make({ ledgerToolCallId: false })]).turns.at(-1);
  assert.equal(legacyLinked.training_admission.approved, true);
  assert.equal(legacyLinked.decision.status_only_success_validation[0].proof.staged_action_linkage,
    'unique-invocation-arguments-output');

  const afterInspection = make();
  const stageTurn = afterInspection.trajectory[0];
  const stageContext = [...stageTurn.context, { role: 'assistant', content: '', tool_calls: [stageCall] },
    { role: 'tool', tool_call_id: stageCallId, content: stagedText }];
  const inspectionCall = { id: 'inspect-after-stage', type: 'function',
    function: { name: 'write', arguments: '{"path":"inspect"}' } };
  afterInspection.trajectory.splice(1, 0, { phase: 'action', invocation_id: 'same-invocation', context: stageContext,
    tools_offered: schema, assistant: { content: '', reasoning: 'Inspect after staging.', calls: [{ tool: 'write',
      source_tool: 'write', arguments: { path: 'inspect' }, call_id: inspectionCall.id }] },
    raw_response_sha256: 'inspection-after-stage' });
  afterInspection.trajectory.at(-1).context.push({ role: 'assistant', content: '', tool_calls: [inspectionCall] },
    { role: 'tool', tool_call_id: inspectionCall.id, content: 'The return_result contract.' });
  const terminalEvent = { ...afterInspection.outcome.action_ledger[1], seq: 3 };
  afterInspection.outcome.action_ledger[1] = { seq: 2, call_id: 'same-invocation', name: 'write',
    arguments: { path: 'inspect' }, outcome: 'ok', result_text: 'The return_result contract.' };
  afterInspection.outcome.action_ledger.push(terminalEvent);
  const inspectedTurn = materializeNativeRows([afterInspection]).turns.at(-1);
  assert.equal(inspectedTurn.training_admission.approved, true);
  assert.equal(inspectedTurn.decision.status_only_success_validation[0].proof.staged_call_id, stageCallId);

  const segmented = make();
  const markerId = `nz1_${'a'.repeat(26)}`;
  const segmentedText = stagedText.replace('{ answer: "ready" }', `${markerId}`);
  const finalMessage = segmented.trajectory.at(-1).context.at(-1);
  finalMessage.content = [
    { type: 'text', text: 'Staged ' },
    { type: 'neuralese', id: markerId },
    { type: 'text', text: segmentedText.slice(segmentedText.indexOf(' as the result.')) },
  ];
  segmented.outcome.action_ledger[0].result_text = segmentedText;
  const segmentedTurn = materializeNativeRows([segmented]).turns.at(-1);
  assert.equal(segmentedTurn.training_admission.approved, true);
  assert.equal(segmentedTurn.decision.status_only_success_validation[0].proof.staged_output_sha256,
    createHash('sha256').update(segmentedText).digest('hex'));

  for (const row of [make({ withStage: false }), make({ linked: false })]) {
    const turn = materializeNativeRows([row]).turns.at(-1);
    assert.ok(turn);
    assert.equal(turn.training_admission.approved, false);
    assert.match(turn.training_admission.reason, /same-invocation staged typed result/);
  }
  const voidTurn = materializeNativeRows([make({ withStage: false, returnType: 'void' })]).turns.at(-1);
  assert.equal(voidTurn.training_admission.approved, true);
  assert.equal(voidTurn.decision.status_only_success_validation[0].proof.basis, 'declared-void-return');
  assert.equal(materializeNativeRows([make({ resultArgs: {} })]).turns.at(-1).training_admission.approved, true);

  const imitated = `console:\n${stagedText}\nundefined\nNo result of type { answer: string } was returned.`;
  const consoleImitation = make({ stageOutput: imitated });
  const fake = materializeNativeRows([consoleImitation]).turns.at(-1);
  assert.equal(fake.training_admission.approved, false);
});

test('direct and failed-run exports cannot bypass source-review or retired-contract holds', () => {
  for (const curriculum of [
    { family: 'folio_batch', shape: 'story337' },
    { family: 'commaqa_numeric', family_version: 1 },
    { family: 'inline_type_repair' },
  ]) {
    const row = nativeRow('held');
    row.task.program_ir.curriculum = curriculum;
    row.task.program_ir.semantics = {};
    for (const options of [{}, { failedRuns: true }, { directAnswers: true }]) {
      const result = materializeNativeRows([row], options);
      assert.equal(result.rejectedRows, 1);
      assert.deepEqual(result.turns, []);
    }
  }
});

test('scripted authored references keep runtime success separate from pending training admission', () => {
  const row = nativeRow('authored-static-reference');
  row.provenance = { ...row.provenance, collection_role: 'authored-source-static-reference' };
  const runtimeTurns = materializeNativeRows([row], { directAnswers: true }).turns;
  assert.equal(runtimeTurns.length, 2);
  assert.equal(runtimeTurns[0].outcome.accepted, true);
  assert.equal(runtimeTurns[0].training_admission.approved, true);

  const turns = markAuthoredStaticReferencePending(runtimeTurns);
  assert.equal(turns[0].outcome.accepted, true, 'the successful runtime outcome remains available');
  assert.equal(turns[0].training_admission.approved, false);
  assert.equal(turns[0].training_admission.kind, 'authored-static-reference-pending-review');
  assert.equal(turns[0].trace_admission.admitted, false);
  assert.equal(turns[0].trace_admission.kind, 'authored-static-reference-not-teacher-trace');
  assert.equal(turns[0].decision.training_approved, false);
  assert.equal(turns[0].provenance.training_provenance,
    'authored-source-static-reference;zero-provider-calls');
  assert.equal(runtimeTurns[0].training_admission.approved, true,
    'the helper leaves the original materializer object untouched for separate attestation');
});

test('legacy highlighter gold is held even when the old outcome accepted it', () => {
  const row = nativeRow('legacy-highlighter');
  row.task.program_ir.family = 'cb_highlighter';
  row.task.program_ir.semantics = {};
  assert.equal(materializeNativeRows([row]).rejectedRows, 1);
  row.task.program_ir.generation = { highlighter_quality_version: 2 };
  assert.equal(materializeNativeRows([row]).acceptedRows, 1);
});

test('accepted native rows become linked template neutral decisions with their exact contexts', () => {
  const accepted = nativeRow('teacher-1'), rejected = nativeRow('teacher-rejected', false);
  accepted.task.program_ir.source_groups = ['original-program'];
  const result = materializeNativeRows([accepted, rejected]);
  assert.equal(result.acceptedRows, 1);
  assert.equal(result.rejectedRows, 1);
  assert.equal(result.turns.length, 2);
  const [actionTurn, nextTurn] = result.turns;
  assert.deepEqual(actionTurn.outcome, { accepted: true, status: 'done' });
  assert.deepEqual(actionTurn.source_groups, ['program-1', 'original-program']);
  const action = actionTurn.decision, next = nextTurn.decision;
  assert.equal(action.assistant.reasoning, 'The result is twice n.');
  assert.deepEqual(action.tool_schemas, [{ name: 'write', description: 'Write a value.',
    parameters: { type: 'object', properties: { path: { type: 'string' } } } }]);
  assert.deepEqual(action.assistant.calls[0].outcome, { event_index: 0, trace_seq: 12,
    name: 'write', arguments: firstCall.arguments, status: 'ok', result: 'stored value', diagnostics: [] });
  assert.equal(next.assistant.calls[0].outcome.trace_seq, 17);
  // The conversation continues: the next decision sees the earlier call and its result.
  assert.deepEqual(nextTurn.messages.map(message => message.role), ['system', 'user', 'assistant', 'tool']);
  assert.deepEqual(next.durable_opening.map(message => message.role), ['system', 'user']);
  assert.equal(actionTurn.source_ref.trajectory_id, 'teacher-1');
  assert.deepEqual(actionTurn.provenance, accepted.provenance);
  assert.deepEqual(actionTurn.messages, [system, opening]);
  assert.deepEqual(actionTurn.tools, schema);
  assert.equal(actionTurn.skill, 'write');
  assert.equal(actionTurn.teacher_reasoning, 'The result is twice n.');
  assert.deepEqual(actionTurn.target, { role: 'assistant', content: '', tool_calls: [{
    id: 'teacher_0_0', type: 'function', function: {
      name: 'write', arguments: JSON.stringify(firstCall.arguments),
    },
  }] });
});

test('materialized turns retain case oracle evidence levels as metadata', () => {
  for (const level of ['exact', 'normalized', 'span', 'judged']) {
    const row = nativeRow(`oracle-${level}`);
    row.task.program_ir.semantics = { oracle: level };
    const turn = materializeNativeRows([row]).turns[0];
    assert.equal(turn.oracle_level, level);
    assert.equal(turn.training_admission.oracle_level, level);
  }
  const row = nativeRow('oracle-object');
  row.task.program_ir.semantics = { oracle: { level: 'normalized', method: 'trim-and-casefold' } };
  assert.equal(materializeNativeRows([row]).turns[0].oracle_level, 'normalized');
});

test('a captured execution plan is the training reasoning instead of provider reasoning', () => {
  const row = nativeRow('planned');
  row.trajectory[0].assistant.execution_plan = 'Compute twice n, then store it.';
  row.trajectory[0].assistant.reasoning = 'opaque provider reasoning';
  const [turn] = materializeNativeRows([row]).turns;
  assert.equal(turn.teacher_execution_plan, 'Compute twice n, then store it.');
  assert.equal(turn.teacher_reasoning, 'Compute twice n, then store it.');
  assert.equal(turn.decision.assistant.execution_plan, 'Compute twice n, then store it.');
  assert.equal(turn.decision.assistant.reasoning, 'Compute twice n, then store it.');
});

test('an attempted but missing execution plan does not fall back to provider reasoning', () => {
  const row = nativeRow('plan-unsupported');
  row.trajectory[0].assistant.execution_plan = null;
  row.trajectory[0].assistant.reasoning = 'opaque provider reasoning';
  const [turn] = materializeNativeRows([row]).turns;
  assert.equal(turn.teacher_execution_plan, null);
  assert.equal(turn.teacher_reasoning, null);
  assert.equal(turn.decision.assistant.reasoning, null);
});

test('rows collected with conversation rollover are rejected', () => {
  const row = nativeRow('rolled-over');
  row.trajectory.splice(1, 0, { phase: 'checkpoint', context: [system, opening], tools_offered: [],
    assistant: { content: 'A working note.', reasoning: null, calls: [] }, raw_response_sha256: 'raw-note' });
  const result = materializeNativeRows([row]);
  assert.equal(result.acceptedRows, 0); assert.equal(result.rejectedRows, 1); assert.equal(result.turns.length, 0);
});

test('accepted rows with an unlinked or reordered action outcome are rejected', () => {
  const row = nativeRow('bad-link');
  row.outcome.action_ledger[0].arguments = { path: 'return', value: 99 };
  const result = materializeNativeRows([row, nativeRow('good-link')]);
  assert.equal(result.acceptedRows, 1); assert.equal(result.rejectedRows, 1);
  assert.deepEqual(result.unlinked, [{ id: row.id, outcomes: row.outcome.action_ledger.length }]);
  assert.ok(result.turns.every(turn => turn.teacher_trajectory_id !== row.id), 'no turn of the unlinked row is used');
});

test('diagnostic preview arguments are unlinked unless an exact raw model call contains them', () => {
  const preview = { $diagnostic_preview: 'largeData holds all of it', complete: false, holder: 'largeData' };
  const projected = nativeRow('projected-only');
  projected.trajectory = projected.trajectory.slice(0, 1);
  projected.outcome.action_ledger = projected.outcome.action_ledger.slice(0, 1);
  projected.trajectory[0].assistant.calls[0].arguments = preview;
  projected.outcome.action_ledger[0].arguments = preview;
  let result = materializeNativeRows([projected]);
  assert.equal(result.acceptedRows, 0);
  assert.deepEqual(result.unlinked, [{ id: 'projected-only', outcomes: 1,
    reason: 'incomplete_diagnostic_arguments_without_exact_raw_model_call' }]);
  assert.ok(result.turns.every(turn => turn.teacher_trajectory_id !== projected.id),
    'a stored diagnostic preview alone cannot become a training turn');

  const projectedOutcome = nativeRow('projected-outcome-only');
  projectedOutcome.trajectory = projectedOutcome.trajectory.slice(0, 1);
  projectedOutcome.outcome.action_ledger = projectedOutcome.outcome.action_ledger.slice(0, 1);
  projectedOutcome.outcome.action_ledger[0].arguments = preview;
  result = materializeNativeRows([projectedOutcome]);
  assert.equal(result.acceptedRows, 0);
  assert.deepEqual(result.unlinked, [{ id: 'projected-outcome-only', outcomes: 1,
    reason: 'incomplete_diagnostic_arguments_without_exact_raw_model_call' }]);

  const actualCall = nativeRow('raw-preview-call');
  actualCall.trajectory = actualCall.trajectory.slice(0, 1);
  actualCall.outcome.action_ledger = actualCall.outcome.action_ledger.slice(0, 1);
  const rawArguments = { path: 'return', type: 'number', value: 6 };
  actualCall.trajectory[0].assistant.calls[0].arguments = rawArguments;
  actualCall.outcome.action_ledger[0].arguments = preview;
  actualCall.trajectory[0].model_response = { raw_calls: [{ type: 'function', function: {
    name: 'write', arguments: JSON.stringify(rawArguments),
  } }] };
  result = materializeNativeRows([actualCall]);
  assert.equal(result.acceptedRows, 1, 'an exact raw model call remains auditable as the source of the argument');
  assert.deepEqual(result.unlinked, []);
  assert.equal(result.turns[0].training_admission.approved, true);
  assert.equal(result.turns[0].target.tool_calls[0].function.arguments, JSON.stringify(rawArguments));
  assert.deepEqual(result.turns[0].decision.assistant.calls[0].outcome.arguments, rawArguments);
  assert.equal(result.turns[0].decision.assistant.calls[0].outcome.arguments_source, 'exact_raw_model_call');
});

test('links a soft return marker only to its exact typed runtime block write and return action', () => {
  const result = materializeNativeRows([softReturnRow()], { directAnswers: true });
  assert.equal(result.acceptedRows, 1);
  assert.deepEqual(result.unlinked, []);
  assert.equal(result.turns.length, 1);
  const call = result.turns[0].decision.assistant.calls[0];
  assert.equal(call.outcome.status, 'completed');
  assert.equal(JSON.parse(result.turns[0].target.tool_calls[0].function.arguments).value, softMarker);
});

test('retains exact typed final-result write receipts for native and gold-text consumers', () => {
  const make = ({ id, tool = 'return_result', args, runtimeArgs = args, writes, hostValue, terminal = 9 }) => {
    const row = nativeRow(id);
    const callId = `call-${id}`, turnNode = `${callId}#turn1`;
    row.outcome.action_ledger = [{ seq: terminal, call_id: callId, name: tool, arguments: runtimeArgs,
      outcome: tool === 'return_result' ? 'completed' : 'ok', result_text: 'completed' }];
    row.outcome.execution_graph = [
      { kind: 'model_turn', seq: 5, call_id: callId, node: turnNode, turn: 1,
        inputs: [{ node: `call:${callId}`, port: 'invocation' }] },
      ...writes.map((write, index) => ({ kind: 'block_write', seq: 6 + index, call_id: callId,
        node: `${callId}#${6 + index}`, turn: turnNode,
        inputs: [{ node: turnNode, port: 'result-source' }], truncated: false, ...write })),
    ];
    row.outcome.invocation_ledger = [{ invocation_id: callId, host_result: { kind: 'host_capture',
      capture_kind: 'invocation_output', call_id: callId, complete: true, value: hostValue,
      terminal_action_seq: terminal } }];
    row.trajectory = [{ phase: 'action', invocation_id: callId, context: [system, opening], tools_offered: schema,
      assistant: { content: '', reasoning: 'Return the typed result.', calls: [{ tool, source_tool: tool,
        arguments: args, call_id: 'raw-result' }] },
      model_response: { raw_calls: [{ function: { name: tool, arguments: JSON.stringify(args) } }] },
      raw_response_sha256: `raw-${id}` }];
    return row;
  };
  const sha = value => createHash('sha256').update(value).digest('hex');
  const block = `nz1_${'a'.repeat(32)}`;
  const scalar = make({ id: 'typed-text-result', args: { status: 'success', value: 'exact final note' },
    writes: [{ block, source_kind: 'typed-text-result', source: 'return_result', marker_context: 'return-result',
      result_type: 'Neuralese<string>', text_body_sha256: sha('exact final note') }],
    hostValue: { $neuralese: { id: block, type: 'Neuralese<string>' } } });
  const scalarTurn = materializeNativeRows([scalar]).turns[0];
  const scalarCall = scalarTurn.decision.assistant.calls[0];
  assert.equal(scalarTurn.target.tool_calls[0].function.arguments,
    JSON.stringify({ status: 'success', value: 'exact final note' }), 'provider target remains its actual plain text');
  assert.deepEqual(scalarCall.outcome.typed_result_writes.map(({ body_source, body_source_basis, result_path }) =>
    ({ body_source, body_source_basis, result_path })), [{ body_source: 'exact final note',
      body_source_basis: 'exact-raw-model-result-string', result_path: ['return'] }]);

  const jsonBody = '{"count":4,"flag":true}', jsonBlock = `nz1_${'b'.repeat(32)}`;
  const json = make({ id: 'typed-json-result', args: { status: 'success', value: { flag: true, count: 4 } },
    writes: [{ block: jsonBlock, source_kind: 'typed-json-result', source: 'return_result', marker_context: 'return-result',
      result_type: 'Neuralese<{ count: number, flag: boolean }>', text_body_sha256: sha(jsonBody) }],
    hostValue: { $neuralese: { id: jsonBlock, type: 'Neuralese<{ count: number, flag: boolean }>' } } });
  const jsonCall = materializeNativeRows([json]).turns[0].decision.assistant.calls[0];
  assert.equal(jsonCall.outcome.typed_result_writes[0].body_source, jsonBody,
    'canonical JSON body is recomputed from the exact provider value and checked against runtime SHA');

  const jsonTextRaw = '{"count":4,"flag":true}', jsonTextBlock = `nz1_${'f'.repeat(32)}`;
  const parsedJson = make({ id: 'typed-json-text-result', args: { status: 'success', value: jsonTextRaw },
    writes: [{ block: jsonTextBlock, source_kind: 'typed-json-result', source: 'return_result', marker_context: 'return-result',
      result_type: 'Neuralese<{ count: number, flag: boolean }>', text_body_sha256: sha(jsonBody) }],
    hostValue: { $neuralese: { id: jsonTextBlock, type: 'Neuralese<{ count: number, flag: boolean }>' } } });
  const parsedReceipt = materializeNativeRows([parsedJson]).turns[0].decision.assistant.calls[0].outcome.typed_result_writes[0];
  assert.equal(parsedReceipt.body_source, jsonBody,
    'JSON text is normalized only when a concrete non-string Neuralese result type and runtime body digest prove parsing');
  assert.equal(parsedReceipt.body_source_basis, 'parsed-json-string-for-concrete-neuralese-result-type');
  assert.equal(parsedReceipt.raw_model_value_sha256, sha(JSON.stringify(jsonTextRaw)),
    'the raw provider text remains independently hash-bound');

  const jsonStringBlock = `nz1_${'e'.repeat(32)}`;
  const jsonString = make({ id: 'typed-string-containing-json', args: { status: 'success', value: jsonTextRaw },
    writes: [{ block: jsonStringBlock, source_kind: 'typed-text-result', source: 'return_result', marker_context: 'return-result',
      result_type: 'Neuralese<string>', text_body_sha256: sha(jsonTextRaw) }],
    hostValue: { $neuralese: { id: jsonStringBlock, type: 'Neuralese<string>' } } });
  const jsonStringReceipt = materializeNativeRows([jsonString]).turns[0].decision.assistant.calls[0].outcome.typed_result_writes[0];
  assert.equal(jsonStringReceipt.body_source, jsonTextRaw, 'JSON-looking string values remain literal for Neuralese<string>');
  assert.equal(jsonStringReceipt.body_source_basis, 'exact-raw-model-result-string');

  const repeatedBlock = `nz1_${'c'.repeat(32)}`, repeatedBody = 'same emitted note';
  const nestedArgs = { status: 'success', value: { nested: { first: repeatedBody }, final: repeatedBody } };
  const nested = make({ id: 'typed-text-fields', args: nestedArgs, writes: [
    { block: repeatedBlock, source_kind: 'typed-text-result-field', source: 'return_result',
      marker_context: 'return-result', result_type: 'Neuralese<string>', result_path: ['return', 'nested', 'first'],
      text_body_sha256: sha(repeatedBody) },
    { block: repeatedBlock, source_kind: 'typed-text-result-field', source: 'return_result',
      marker_context: 'return-result', result_type: 'Neuralese<string>', result_path: ['return', 'final'],
      text_body_sha256: sha(repeatedBody) },
  ], hostValue: { nested: { first: { $neuralese: { id: repeatedBlock, type: 'Neuralese<string>' } } },
    final: { $neuralese: { id: repeatedBlock, type: 'Neuralese<string>' } } } });
  const nestedCall = materializeNativeRows([nested]).turns[0].decision.assistant.calls[0];
  assert.deepEqual(nestedCall.outcome.typed_result_writes.map(write => write.result_path),
    [['return', 'nested', 'first'], ['return', 'final']], 'same-body writes at distinct paths stay distinct');
  assert.equal(new Set(nestedCall.outcome.typed_result_writes.map(write => write.writer_node)).size, 2);

  const evalBlock = `nz1_${'d'.repeat(32)}`;
  const evalRow = make({ id: 'eval-finish-result', tool: 'eval', args: { code: 'return { note: buildNote() };', finish: true },
    writes: [{ block: evalBlock, source_kind: 'typed-text-result-field', source: 'eval-finish',
      marker_context: 'return-result', result_type: 'Neuralese<string>', result_path: ['return', 'note'],
      text_body_sha256: sha('runtime-authored note') }],
    hostValue: { note: { $neuralese: { id: evalBlock, type: 'Neuralese<string>' } } } });
  const evalReceipt = materializeNativeRows([evalRow]).turns[0].decision.assistant.calls[0].outcome.typed_result_writes[0];
  assert.equal(evalReceipt.body_source, undefined, 'an eval expression is not misrepresented as literal provider output');
  assert.equal(evalReceipt.body_source_basis, 'authenticated-final-host-output-reference');
  assert.equal(evalReceipt.source, 'eval-finish');

  for (const mutate of [
    row => { row.outcome.execution_graph[1].text_body_sha256 = 'f'.repeat(64); },
    row => { row.outcome.execution_graph[1].result_path = ['return', 'wrong']; },
    row => { row.outcome.execution_graph[1].inputs[0].port = 'turn'; },
    row => { row.outcome.invocation_ledger[0].host_result.value.$neuralese.id = `nz1_${'e'.repeat(32)}`; },
  ]) {
    const invalid = structuredClone(scalar); mutate(invalid);
    const result = materializeNativeRows([invalid]);
    assert.equal(result.turns[0].decision.assistant.calls[0].outcome.typed_result_writes, undefined,
      'unbound writer/body/path evidence is not promoted to a portable write receipt');
  }
});

test('links runtime-truncated Neuralese markers only by the exact EOF rewrite and keeps that malformed action held', () => {
  const malformed = '<|neuralese|>note body|neuralese.textReadSource(prior); return note;';
  const row = softEvalRow({ marker: malformed, actionCode: `const note: Neuralese<string> = ${softSentinel}` });
  const readerId = 'soft-eval-reader-1', readerArgs = { code: 'console.log(String(prior));' };
  row.outcome.action_ledger.push({ seq: 8, call_id: readerId, name: 'eval', arguments: readerArgs,
    outcome: 'ok', result_text: 'read prior note' });
  row.outcome.execution_graph = [
    { kind: 'block_write', call_id: 'soft-eval-1', block: softBlock, truncated: true,
      text_body_sha256: createHash('sha256').update(malformed.slice('<|neuralese|>'.length)).digest('hex') },
    { kind: 'block_read', call_id: readerId, block: softBlock },
  ];
  row.trajectory.push({ phase: 'action', invocation_id: readerId, context: [system, opening], tools_offered: schema,
    assistant: { content: '', reasoning: 'Use the existing note.', calls: [{ tool: 'eval', source_tool: 'eval',
      arguments: readerArgs, call_id: 'raw-eval-reader' }] }, model_response: { raw_calls: [{ function: { name: 'eval',
      arguments: JSON.stringify(readerArgs) } }] }, raw_response_sha256: 'raw-soft-reader' });
  const result = materializeNativeRows([row], { directAnswers: true });
  assert.equal(result.acceptedRows, 1, JSON.stringify(result.unlinked));
  assert.deepEqual(result.unlinked, []);
  const outcome = result.turns[0].decision.assistant.calls[0].outcome;
  assert.equal(outcome.status, 'ok');
  assert.ok(outcome.diagnostics.includes('coerced-truncated-neuralese-marker'));
  assert.equal(result.turns[0].training_admission.approved, false,
    'the exact malformed action remains visible but cannot become a clean training target');
  const dependent = result.turns[1].decision.assistant.calls[0].outcome;
  assert.ok(dependent.diagnostics.includes('unreviewed-truncated-neuralese-dependency'));
  assert.equal(result.turns[1].training_admission.approved, false,
    'an observed reader of a truncated block stays held for factual review');

  const wrongRewrite = softEvalRow({ marker: malformed,
    actionCode: `const note: Neuralese<string> = nz1_${'t'.repeat(32)}` });
  wrongRewrite.outcome.execution_graph = [{ kind: 'block_write', call_id: 'soft-eval-1', block: softBlock,
    truncated: true, text_body_sha256: createHash('sha256').update(malformed.slice('<|neuralese|>'.length)).digest('hex') }];
  const rejected = materializeNativeRows([wrongRewrite], { directAnswers: true });
  assert.equal(rejected.acceptedRows, 0);
  assert.deepEqual(rejected.unlinked, [{ id: wrongRewrite.id, outcomes: 1 }]);
});

test('keeps soft return markers unlinked when runtime writer evidence disagrees or is incomplete', () => {
  const mutations = [
    row => { row.outcome.execution_graph[1].block = `nz1_${'x'.repeat(32)}`; },
    row => { row.outcome.execution_graph = []; },
    row => { row.outcome.execution_graph[0].call_id = 'unrelated-child'; },
    row => { row.outcome.execution_graph[1].inputs[0].node = 'unrelated-child#turn1'; },
    row => {
      row.outcome.execution_graph.push({ kind: 'model_turn', seq: 10, call_id: softCallId,
        node: `${softCallId}#turn2`, turn: 2, inputs: [{ node: `call:${softCallId}`, port: 'invocation' }] });
      row.outcome.action_ledger[0].seq = 12;
      row.outcome.invocation_ledger[0].host_result.terminal_action_seq = 12;
    },
    row => { row.outcome.invocation_ledger[0].host_result.value.$neuralese.id = `nz1_${'x'.repeat(32)}`; },
    row => { row.outcome.invocation_ledger[0].host_result.complete = false; },
    row => { row.outcome.invocation_ledger[0].host_result.result_type = 'Neuralese<number>'; },
    row => { row.outcome.invocation_ledger[0].inline_instruction_site.returns.natlang = 'string'; },
    row => { row.outcome.invocation_ledger[0].host_result.terminal_action_seq = 8; },
    row => { row.outcome.action_ledger[0].seq = 10; },
    row => { row.trajectory[0].assistant.calls[0].arguments.value =
      '<|neuralese|>one<|/neuralese|><|neuralese|>two<|/neuralese|>'; },
  ];
  for (const [index, mutate] of mutations.entries()) {
    const row = softReturnRow(`soft-return-held-${index}`);
    mutate(row);
    const result = materializeNativeRows([row], { directAnswers: true });
    assert.equal(result.acceptedRows, 0, `mutation ${index} unexpectedly linked`);
    assert.deepEqual(result.unlinked, [{ id: row.id, outcomes: 1 }]);
    assert.deepEqual(result.turns, []);
  }
});

test('small terminal eval and return_result turns remain materializable after large-scope continuation', () => {
  const row = nativeRow('small-terminal-after-large-scope');
  const evalArgs = { code: 'return allSubsets.length + allSubsets[199999][0]', finish: true };
  const returnArgs = { status: 'success', value: 399999 };
  row.trajectory[0].assistant.calls = [{ tool: 'eval', source_tool: 'eval', arguments: evalArgs, call_id: null }];
  row.trajectory[0].model_response = { raw_calls: [{ type: 'function', function: {
    name: 'eval', arguments: JSON.stringify(evalArgs),
  } }] };
  row.trajectory[1].assistant.calls = [{ tool: 'return_result', source_tool: 'return_result', arguments: returnArgs, call_id: null }];
  row.trajectory[1].model_response = { raw_calls: [{ type: 'function', function: {
    name: 'return_result', arguments: JSON.stringify(returnArgs),
  } }] };
  row.outcome.action_ledger = [
    { seq: 1, name: 'eval', arguments: evalArgs, outcome: 'ok', result_text: 'Staged 399999 as the result.' },
    { seq: 2, name: 'return_result', arguments: returnArgs, outcome: 'completed', result_text: 'Returned 399999.' },
  ];
  const result = materializeNativeRows([row]);
  assert.equal(result.acceptedRows, 1);
  assert.equal(result.turns.length, 2);
  assert.ok(result.turns.every(turn => turn.training_admission.approved));
  assert.doesNotMatch(JSON.stringify(result.turns), /\$diagnostic_preview/);
  assert.deepEqual(JSON.parse(result.turns[1].target.tool_calls[0].function.arguments), returnArgs);
  assert.deepEqual(result.unlinked, []);
});

test('failed and unexecuted proposals remain in IR but are excluded from SFT admission', () => {
  const row = nativeRow('negative-decisions');
  row.outcome.action_ledger[0].outcome = 'rejected';
  row.trajectory[0].assistant.calls.push({ tool: 'read_page', source_tool: 'read_page',
    arguments: { id: 'amber', page: 2 }, call_id: null });
  const result = materializeNativeRows([row]);
  assert.equal(result.turns[0].training_admission.approved, false);
  assert.equal(result.turns[0].decision.assistant.calls[0].outcome.status, 'rejected');
  assert.equal(result.turns[0].decision.assistant.calls[1].outcome.status, 'not_executed');
  assert.equal(result.turns[1].training_admission.approved, true);
});

test('coerced proposals are retained but denied positive SFT admission', () => {
  const row = nativeRow('coerced-decision');
  row.outcome.action_ledger[0].diagnostics = ['coerced-redundant-self-alias'];
  const result = materializeNativeRows([row]);
  assert.equal(result.turns[0].training_admission.approved, false);
});

test('unsupported row versions and missing admission decisions fail closed', () => {
  const row = nativeRow('bad-version');
  row.version = 'natlang.teacher_trajectory/1';
  assert.throws(() => materializeNativeRows([row]), /unsupported native teacher row version/);
  const missing = nativeRow('no-accepted-flag');
  delete missing.outcome.accepted;
  assert.throws(() => materializeNativeRows([missing]), /outcome.accepted must be boolean/);
});

test('oracle-accepted student decisions keep student provenance and can be rehearsed', () => {
  const row = nativeRow('student-success');
  row.provenance.collection_role = 'student';
  const result = materializeNativeRows([row]);
  assert.equal(result.turns[0].source, 'student-native');
  assert.deepEqual(result.turns[0].gold_sources, ['checked-student-trajectory', 'exact-runtime-oracle']);
  assert.equal(result.turns[0].training_admission.approved, true);
});

test('each call links to its own actions when the trajectory interleaves a root call with its nl children', () => {
  const judge = { role: 'user', content: 'You are inside this call: nl@eval:1(item): boolean' };
  const childCall = { tool: 'return_result', source_tool: 'return_result', arguments: { status: 'success', value: true }, call_id: null };
  const rootFinish = { tool: 'return_result', source_tool: 'return_result', arguments: { status: 'success', value: true }, call_id: null };
  const row = nativeRow('interleaved');
  row.outcome.action_ledger = [
    { seq: 12, call_id: 'root', name: 'write', arguments: firstCall.arguments, outcome: 'ok', result_text: 'stored value' },
    { seq: 30, call_id: 'root', name: 'return_result', arguments: rootFinish.arguments, outcome: 'ok', result_text: 'done' },
    { seq: 21, call_id: 'child', name: 'return_result', arguments: childCall.arguments, outcome: 'ok', result_text: 'done' },
  ];
  // Root, then its child (whose call looks exactly like the root's own finish), then the root again.
  row.trajectory = [row.trajectory[0],
    { phase: 'action', context: [system, judge], tools_offered: schema, assistant: { content: '', calls: [childCall] }, raw_response_sha256: 'raw-c' },
    { phase: 'action', context: row.trajectory[1].context, tools_offered: schema, assistant: { content: '', calls: [rootFinish] }, raw_response_sha256: 'raw-f' }];
  const [write, child, finish] = materializeNativeRows([row]).turns.map(turn => turn.decision.assistant.calls[0].outcome);
  assert.equal(write.trace_seq, 12);
  assert.equal(child.trace_seq, 21, 'the child links to its own log, not to the root finish it resembles');
  assert.equal(finish.trace_seq, 30);
});

test('a call identical to an earlier one, the same instructions on the same input, links to its own actions', () => {
  const judge = { role: 'user', content: 'You are inside this call: nl@eval:1(item): boolean' };
  const childCall = { tool: 'return_result', source_tool: 'return_result', arguments: { status: 'success', value: true }, call_id: null };
  const row = nativeRow('repeated');
  row.outcome.action_ledger = [...row.outcome.action_ledger,
    { seq: 13, call_id: 'first', name: 'return_result', arguments: childCall.arguments, outcome: 'ok', result_text: 'done' },
    { seq: 15, call_id: 'second', name: 'return_result', arguments: childCall.arguments, outcome: 'ok', result_text: 'done' }];
  const child = raw => ({ phase: 'action', context: [system, judge], tools_offered: schema,
    assistant: { content: '', calls: [childCall] }, raw_response_sha256: raw });
  row.trajectory.splice(1, 0, child('raw-a'), child('raw-b'));
  const seqs = materializeNativeRows([row]).turns.map(turn => turn.decision.assistant.calls[0].outcome.trace_seq);
  assert.deepEqual(seqs, [12, 13, 15, 17]);
});

test('a call with no action log is marked not recorded and kept out of training, not treated as unexecuted', () => {
  const judge = { role: 'user', content: 'You are inside this call: nl@eval:1(item): boolean' };
  const row = nativeRow('unrecorded-child');
  row.trajectory.splice(1, 0, { phase: 'action', context: [system, judge], tools_offered: schema,
    assistant: { content: '', calls: [{ tool: 'return_result', source_tool: 'return_result', arguments: { status: 'success', value: true }, call_id: null }] },
    raw_response_sha256: 'raw-c' });
  const child = materializeNativeRows([row]).turns[1];
  assert.equal(child.decision.assistant.calls[0].outcome.status, 'not_recorded');
  assert.equal(child.training_admission.approved, false);
  assert.match(child.training_admission.reason, /not recorded/);
});

test('a repeat with the same result and an attempt the checker refused are context, not training targets', () => {
  const verify = { tool: 'eval', source_tool: 'eval', arguments: { code: 'proof.verify(steps)' }, call_id: null };
  const refused = '{"ok":false,"certificate":null,"problem":"step 2: F1 does not apply"}';
  const row = nativeRow('checker');
  row.outcome.action_ledger = [
    { seq: 1, name: 'eval', arguments: verify.arguments, outcome: 'ok', result_text: refused },
    { seq: 2, name: 'eval', arguments: verify.arguments, outcome: 'ok', result_text: refused },
    { seq: 3, name: 'write', arguments: firstCall.arguments, outcome: 'ok', result_text: 'stored value' },
    { seq: 4, name: 'write', arguments: firstCall.arguments, outcome: 'ok', result_text: 'stored value' },
    { seq: 5, name: 'write', arguments: firstCall.arguments, outcome: 'ok', result_text: 'stored again' },
  ];
  const turn = (calls, raw) => ({ phase: 'action', context: row.trajectory[1].context, tools_offered: schema,
    assistant: { content: '', calls }, raw_response_sha256: raw });
  row.trajectory = [{ ...row.trajectory[0], assistant: { content: '', calls: [verify] } },
    turn([verify], 'r2'), turn([firstCall], 'r3'), turn([firstCall], 'r4'), turn([firstCall], 'r5')];
  const admissions = materializeNativeRows([row]).turns.map(t => t.training_admission);
  assert.deepEqual(admissions.map(a => a.approved), [false, false, true, false, true]);
  assert.equal(admissions[0].reason, "the task's checker rejected this attempt");
  assert.equal(admissions[3].reason, 'repeats an earlier call of this call with the same result');
});

test('a direct answer and synthetic reasoning are marked for training as the student calls for', () => {
  const row = nativeRow('reference');
  row.provenance = { ...row.provenance, collection_role: 'reference', synthetic_reasoning: 'action-notes/1' };
  row.trajectory[0].assistant.direct_answer = true;
  const [held, other] = materializeNativeRows([row]).turns;
  assert.equal(held.training_admission.approved, false);
  assert.equal(held.training_admission.reason, 'an answer given without reasoning towards it');
  assert.equal(other.training_admission.approved, true);
  assert.equal(held.teacher_reasoning_trained, false);
  assert.equal(held.family, 'reference_program');
  const [trained] = materializeNativeRows([row], { directAnswers: true }).turns;
  assert.equal(trained.training_admission.approved, true);
});

test('an unchanged skill reread with a reminder remains context rather than a target', () => {
  const row = nativeRow('skill-reread');
  const read = { tool: 'read_code', source_tool: 'read_code', arguments: { name: 'skills.exact-computation' }, call_id: null };
  row.outcome.action_ledger = [
    { seq: 1, name: 'read_code', arguments: read.arguments, outcome: 'ok', result_text: 'Calculate using eval.' },
    { seq: 2, name: 'read_code', arguments: read.arguments, outcome: 'ok',
      result_text: 'Unchanged instructions. Calculate using eval.', diagnostics: ['skill-unchanged-read'] },
  ];
  row.trajectory = [0, 1].map(index => ({ ...row.trajectory[index], assistant: { content: '', calls: [read] } }));
  const turns = materializeNativeRows([row]).turns;
  assert.equal(turns[0].training_admission.approved, true);
  assert.equal(turns[1].training_admission.approved, false);
  assert.equal(turns[1].training_admission.reason, 'retrieves unchanged skill instructions again');
});

test('invocation identity links interleaved calls with identical openings and action histories', () => {
  const row = nativeRow('identical-parallel');
  const first = structuredClone(row.trajectory[0]), next = structuredClone(row.trajectory[1]);
  row.outcome.action_ledger = [
    { ...row.outcome.action_ledger[0], call_id: 'a', seq: 12 },
    { ...row.outcome.action_ledger[0], call_id: 'b', seq: 14 },
    { ...row.outcome.action_ledger[1], call_id: 'a', seq: 17 },
    { ...row.outcome.action_ledger[1], call_id: 'b', seq: 19 },
  ];
  row.trajectory = [{ ...first, invocation_id: 'a' }, { ...first, invocation_id: 'b' },
    { ...next, invocation_id: 'a' }, { ...next, invocation_id: 'b' }];
  const linked = materializeNativeRows([row]);
  assert.deepEqual(linked.unlinked, []);
  assert.deepEqual(linked.turns.map(turn => turn.decision.assistant.calls[0].outcome.trace_seq), [12, 14, 17, 19]);
  row.trajectory[2].invocation_id = 'wrong';
  assert.equal(materializeNativeRows([row]).acceptedRows, 0, 'bad explicit identities must never fall back to another call');
});


test('correct final outcomes do not bypass reviewed intermediate-decision holds', () => {
  const row = nativeRow('reviewed-recovery');
  row.provenance.trace_sha256 = '63975b4cef1a6e5cf1d1d1ee02928875cd136617c97670857f2a231bbb125b75';
  for (const options of [{}, { failedRuns: true }, { directAnswers: true }]) {
    const result = materializeNativeRows([row], options);
    assert.equal(result.rejectedRows, 1);
    assert.deepEqual(result.turns, []);
  }
  row.provenance.reused_from = { provenance: { trace_sha256: row.provenance.trace_sha256 } };
  delete row.provenance.trace_sha256;
  assert.equal(materializeNativeRows([row]).acceptedRows, 0, 'reuse preserves the original review hold');
  delete row.provenance.reused_from;
  row.provenance.trace_sha256 = 'a-new-attempt-on-the-same-source';
  assert.equal(materializeNativeRows([row]).acceptedRows, 1);
});


test('partial label agreement and pending file quality stay held; accepted free-text spans remain admissible', () => {
  for (const evidence of [
    { oracle: { level: 'agreement', accepted: true, score: 0.946 } },
    { files_check: { failed: ['one-message'], score: 0.95 } },
    { quality_pending: ['independent review needed'] },
  ]) {
    const row = nativeRow('partial-correctness');
    Object.assign(row.outcome, evidence);
    assert.equal(materializeNativeRows([row]).acceptedRows, 0);
    assert.equal(materializeNativeRows([row], { directAnswers: true }).turns.length, 0);
  }
  const span = nativeRow('accepted-span');
  span.outcome.oracle = { level: 'span', accepted: true, score: .95 };
  assert.equal(materializeNativeRows([span]).acceptedRows, 1);
  const exact = nativeRow('full-agreement');
  exact.outcome.oracle = { level: 'agreement', accepted: true, score: 1 };
  assert.equal(materializeNativeRows([exact]).acceptedRows, 1);
});

test('chunk prefix supervision survives re-materialization of its full validated trajectory', () => {
  const row = nativeRow('chunk-prefix');
  row.provenance.student_chunk_rewrite = {supervision_cutoff_decision:0};
  const turns = materializeNativeRows([row], {directAnswers:true}).turns;
  assert.equal(turns[0].training_admission.approved, true);
  assert.equal(turns[1].training_admission.approved, false);
  assert.match(turns[1].training_admission.reason, /supervision cutoff/);
  for (const cutoff of [-1, .5, 2]) {
    row.provenance.student_chunk_rewrite.supervision_cutoff_decision=cutoff;
    assert.throws(() => materializeNativeRows([row], {directAnswers:true}), /invalid chunk-rewrite/);
  }
});

test('template reasoning (authored plans, action notes) stays context; only model-written rationales train', () => {
  const row = nativeRow('authored-plan');
  row.provenance.synthetic_reasoning = 'action-notes/1';
  assert.equal(materializeNativeRows([row]).turns[0].teacher_reasoning_trained, false);
  row.provenance.synthetic_reasoning = 'authored-action-plans/1';
  row.provenance.reasoning_supervision = 'authored-action-plan';
  assert.equal(materializeNativeRows([row]).turns[0].teacher_reasoning_trained, false);
  row.provenance.synthetic_reasoning = 'rationalized-actions/1';
  delete row.provenance.reasoning_supervision;
  assert.equal(materializeNativeRows([row]).turns[0].teacher_reasoning_trained, true);
});


test('semantic review holds a wrong decision in an accepted run without deleting evidence', () => {
  const row = nativeRow('accepted-with-wrong-child');
  const hold = { trajectory_id: row.id, source_row_sha256: nativeRowDigest(row), decision_index: 0,
    reason: 'child decision contradicts the source temperature log', evidence: ['source log: 4 degrees; child: false'] };
  const result = materializeNativeRows([row], { directAnswers: true, decisionHolds: [hold] });
  assert.equal(result.turns.length, 2);
  assert.equal(result.turns[0].outcome.accepted, true);
  assert.equal(result.turns[0].training_admission.approved, false);
  assert.deepEqual(result.turns[0].training_admission.semantic_review, hold);
  assert.equal(result.turns[1].training_admission.approved, true);
  assert.deepEqual(result.turns[0].target, materializeNativeRows([row], {directAnswers:true}).turns[0].target);
  assert.throws(() => materializeNativeRows([row], {decisionHolds:[{...hold,source_row_sha256:'bad'}]}), /source hash mismatch/);
  assert.throws(() => materializeNativeRows([row], {decisionHolds:[{...hold,decision_index:99}]}), /invalid semantic/);
  assert.throws(() => materializeNativeRows([row], {decisionHolds:[hold,hold]}), /duplicate semantic/);
});


test('exact reviewed decisions in a failed run are admitted without changing its parent grade', () => {
  const row = nativeRow('failed-parent-valid-action', false);
  const raw = materializeNativeRows([row], { directAnswers: true, failedRuns: true });
  const approval = { schema: 'natlang.native-decision-approval/1', trajectory_id: row.id,
    source_row_sha256: nativeRowDigest(row), decision_index: 0,
    target_sha256: nativeDecisionTargetDigest(raw.turns[0].target), review_sha256: 'a'.repeat(64),
    reason: 'exact action computes the correct value despite a later failed decision',
    evidence: ['input n=3; authored action writes6; observed clean write'] };
  const result = materializeNativeRows([row], { directAnswers: true, failedRuns: true, decisionApprovals: [approval] });
  assert.equal(result.turns[0].outcome.accepted, false);
  assert.equal(result.turns[0].training_admission.approved, true);
  assert.equal(result.turns[0].decision.training_approved, true);
  assert.equal(result.turns[0].training_admission.kind, 'reviewed-native-decision');
  assert.equal(result.turns[0].source_ref.native_target_sha256, approval.target_sha256);
  assert.equal(result.turns[1].training_admission.approved, false);
  assert.deepEqual(result.turns[0].target, raw.turns[0].target);
  for (const changed of [{ ...approval, target_sha256: 'b'.repeat(64) },
    { ...approval, review_sha256: 'not-a-review-hash' }, { ...approval, evidence: [] }])
    assert.throws(() => materializeNativeRows([row], { failedRuns: true, decisionApprovals: [changed] }), /target or evidence mismatch/);
  assert.throws(() => materializeNativeRows([row], { failedRuns: true, decisionApprovals: [approval, approval] }), /duplicate or conflicting/);
  const hold = { trajectory_id: row.id, source_row_sha256: nativeRowDigest(row), decision_index: 0,
    reason: 'review conflict', evidence: ['conflicting hold'] };
  assert.throws(() => materializeNativeRows([row], { failedRuns: true, decisionApprovals: [approval], decisionHolds: [hold] }), /duplicate or conflicting/);
  const failed = structuredClone(row); failed.outcome.action_ledger[0].outcome = 'error';
  const failedTarget = materializeNativeRows([failed], { failedRuns: true, directAnswers: true }).turns[0];
  const failedApproval = { ...approval, source_row_sha256: nativeRowDigest(failed), target_sha256: nativeDecisionTargetDigest(failedTarget.target) };
  assert.equal(materializeNativeRows([failed], { failedRuns: true, decisionApprovals: [failedApproval] }).turns[0].training_admission.approved, false);
});

test('materialization preserves same-call provider-expanded read provenance with distinct root and producer origins', () => {
  const row = nativeRow('provider-expanded-context');
  const invocation = 'provider-call';
  const functionId = `nz1_${'a'.repeat(32)}`;
  const noteId = `nz1_${'b'.repeat(32)}`;
  const functionBody = 'Read the value held by v and return it exactly.';
  const noteBody = 'The note from the earlier producer.';
  const sha = value => createHash('sha256').update(value).digest('hex');
  const functionBlock = { id: functionId, type: 'Neuralese<(v: Neuralese<unknown>) => unknown>', body: functionBody,
    body_sha256: sha(functionBody), learned_vectors: false };
  const noteBlock = { id: noteId, type: 'Neuralese<string>', body: noteBody, body_sha256: sha(noteBody), learned_vectors: false };
  const source = row.trajectory[0];
  source.invocation_id = invocation;
  source.request_sha256 = 'a'.repeat(64);
  source.raw_response_sha256 = 'b'.repeat(64);
  row.trajectory = [source];
  row.outcome.action_ledger = [{ seq: 12, call_id: invocation, name: 'write', arguments: firstCall.arguments,
    outcome: 'ok', result_text: 'stored value' }];
  row.outcome.invocation_ledger = [{ invocation_id: invocation, parent_invocation_id: 'parent-call' }];
  source.context = [system, { ...opening, content: [{ type: 'text', text: opening.content }, { type: 'neuralese', id: noteId }] }];
  source.model_response = { calls: [['write', firstCall.arguments]],
    raw_calls: [{ function: { name: 'write', arguments: JSON.stringify(firstCall.arguments) } }],
    transport_provenance: { learned_vectors: false, qualification_certificate: false,
    training_admission: false, raw_request_sha256: 'c'.repeat(64), rendered_request_sha256: 'd'.repeat(64),
    expanded_input_blocks: [functionBlock, noteBlock],
    text_template_readout: { schema: 'natlang.text-template-readout/1', call: 'return_result', value: 'decode',
      value_type: 'string', read_body_id: functionId, read_source_sha256: functionBlock.body_sha256,
      learned_vectors: false, qualification_certificate: false, training_admission: false } } };
  row.provenance.trace_sha256 = 'e'.repeat(64);
  row.outcome.execution_graph = [
    { kind: 'invocation', phase: 'start', seq: 1, call_id: invocation,
      definition: { id: `nz-fn:${functionId}` }, signature: '(v: Neuralese<string>) => string' },
    { kind: 'block_write', seq: 2, call_id: 'producer-call', block: noteId, node: 'producer-call#5',
      truncated: false, producer: 'text-marker-emulation', source_kind: 'typed-text-result',
      result_type: noteBlock.type, text_body_sha256: noteBlock.body_sha256 },
    { kind: 'block_read', seq: 3, call_id: invocation, block: functionId, node: `${invocation}#6` },
    { kind: 'block_read', seq: 4, call_id: invocation, block: noteId, node: `${invocation}#7`,
      inputs: [{ node: 'producer-call#5', block: noteId, port: 'block' }] },
    { kind: 'model_turn', seq: 5, call_id: invocation, turn: 1, calls: ['write'], node: `${invocation}#turn1`, inputs: [
      { node: `${invocation}#6`, port: 'read', block: functionId }, { node: `${invocation}#7`, port: 'read', block: noteId }] },
  ];
  const turns = materializeNativeRows([row]).turns;
  const receipt = turns[0].source_ref.provider_expanded_read_contexts;
  assert.equal(receipt.length, 2);
  assert.deepEqual(receipt.map(item => [item.block.id, item.origin]), [
    [functionId, 'configured-function-definition'], [noteId, 'same-run-producer']]);
  assert.equal(receipt[1].schema, 'natlang.provider-expanded-read-context/2');
  assert.equal(receipt[1].parent_invocation_id, 'parent-call');
  assert.equal(receipt[1].source_trajectory_index, 0);
  assert.equal(receipt[1].source_request_sha256, 'a'.repeat(64));
  assert.equal(receipt[1].source_response_sha256, 'b'.repeat(64));
  assert.equal(receipt[1].model_turn.node, `${invocation}#turn1`,
    'the provider receipt binds to the exact model turn that made the selected action');
  assert.equal(receipt[1].writer_target_selected, false,
    'the raw writer event authenticates context without creating a selected writer target');
  for (const item of receipt) {
    assert.equal(item.learned_vectors, false);
    assert.equal(item.qualification_certificate, false);
    assert.equal(item.training_admission, false);
  }
  const corrupt = structuredClone(row);
  corrupt.trajectory[0].model_response.transport_provenance.expanded_input_blocks[1].body_sha256 = 'f'.repeat(64);
  const corruptReceipt = materializeNativeRows([corrupt]).turns[0].source_ref.provider_expanded_read_contexts;
  assert.equal(corruptReceipt.some(item => item.block.id === noteId), false,
    'a mismatched provider body digest cannot become context provenance');
  const wrongSelectedTurn = structuredClone(row);
  wrongSelectedTurn.outcome.execution_graph.find(event => event.kind === 'model_turn' && event.call_id === invocation).turn = 2;
  assert.equal(materializeNativeRows([wrongSelectedTurn]).turns[0].source_ref.provider_expanded_read_contexts, undefined,
    'a read from a different turn cannot be attached to the selected model action');
  const wrongRawAction = structuredClone(row);
  wrongRawAction.trajectory[0].model_response.raw_calls[0].function.arguments = JSON.stringify({ path: 'other' });
  assert.equal(materializeNativeRows([wrongRawAction]).turns[0].source_ref.provider_expanded_read_contexts, undefined,
    'the raw provider response must contain the exact selected action');
  const repeated = structuredClone(row);
  repeated.outcome.execution_graph.push(
    { kind: 'block_read', seq: 8, call_id: invocation, block: noteId, node: `${invocation}#8`,
      inputs: [{ node: 'producer-call#5', block: noteId, port: 'block' }] },
    { kind: 'model_turn', seq: 9, call_id: invocation, turn: 2, calls: ['write'], node: `${invocation}#turn2`,
      inputs: [{ node: `${invocation}#8`, port: 'read', block: noteId }] });
  const repeatedReceipt = materializeNativeRows([repeated]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(repeatedReceipt.additional_read_turn_pairs.length, 1,
    'repeated reads in one invocation retain separate per-turn graph bindings');
  const duplicateRequestRef = structuredClone(row);
  duplicateRequestRef.trajectory[0].context.push({ role: 'tool', content: [{ type: 'neuralese', id: noteId }] });
  const duplicateRequestReceipt = materializeNativeRows([duplicateRequestRef]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(duplicateRequestReceipt.context_occurrences, 2,
    'every exact same-run reference occurrence in the actual request is counted');

  const legacy = structuredClone(row);
  const legacyWrite = legacy.outcome.execution_graph.find(event => event.kind === 'block_write' && event.block === noteId);
  delete legacyWrite.producer;
  delete legacyWrite.source_kind;
  legacyWrite.emulation_version = 'text-marker-standin/2';
  legacyWrite.marker_context = 'eval-code';
  legacyWrite.learned_vectors = false;
  const legacyReceipt = materializeNativeRows([legacy]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(legacyReceipt.writer_source_class, 'legacy-text-marker-standin-eval-code');
  assert.equal(legacyReceipt.producer_write.producer, undefined,
    'legacy source events are classified in the receipt without rewriting raw graph fields');
  assert.equal(legacyReceipt.producer_write.source_kind, undefined);

  const withLegacyResult = (source, rawBody) => {
    const candidate = structuredClone(row);
    candidate.provenance.text_neuralese_transport = { mode: 'text-marker-standin/2' };
    const writer = candidate.outcome.execution_graph.find(event => event.kind === 'block_write' && event.block === noteId);
    Object.assign(writer, { source, marker_context: 'return-result' });
    const value = { $neuralese: { id: noteId, type: noteBlock.type } };
    candidate.outcome.invocation_ledger.push({ invocation_id: 'producer-call', completion_status: 'done',
      completion_source: 'execution_graph', completion_detail: `\uE000${noteId}\uE001`, host_result: {
        kind: 'host_capture', capture_kind: 'invocation_output', call_id: 'producer-call', complete: true,
        name: 'return', value, result_type: noteBlock.type, value_sha256: '8'.repeat(64) } });
    candidate.trajectory.push({ ...structuredClone(row.trajectory[0]), invocation_id: 'producer-call',
      raw_response_sha256: '7'.repeat(64),
      model_response: { raw_calls: rawBody ? [{ function: { name: 'return_result', arguments:
        JSON.stringify({ status: 'success', value: noteBody }) } }] : [] } });
    return candidate;
  };
  const rawReturn = withLegacyResult('return_result', true);
  const rawReturnReceipt = materializeNativeRows([rawReturn]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(rawReturnReceipt.writer_source_class, 'legacy-text-marker-standin-return-result');
  assert.equal(rawReturnReceipt.writer_witness.kind, 'raw-return-result-value-equals-expanded-body');
  const badRawReturn = withLegacyResult('return_result', false);
  const badRawReturnReceipt = materializeNativeRows([badRawReturn]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(badRawReturnReceipt, undefined, 'a result marker without exact raw return body is held');
  const evalFinish = withLegacyResult('eval-finish', false);
  const evalFinishReceipt = materializeNativeRows([evalFinish]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(evalFinishReceipt.writer_source_class, 'legacy-text-marker-standin-eval-finish');
  assert.equal(evalFinishReceipt.writer_witness.kind, 'completed-eval-finish-host-reference');
  const unfinished = withLegacyResult('eval-finish', false);
  unfinished.outcome.invocation_ledger.at(-1).completion_status = 'failed';
  const unfinishedReceipt = materializeNativeRows([unfinished]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(unfinishedReceipt, undefined, 'failed finish cannot authenticate a legacy context root');

  const evalReturn = withLegacyResult('eval-return', false);
  const evalReturnReceipt = materializeNativeRows([evalReturn]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(evalReturnReceipt.writer_source_class, 'legacy-text-marker-standin-eval-return');
  assert.equal(evalReturnReceipt.writer_witness.kind, 'completed-eval-return-host-reference');
  assert.equal(evalReturnReceipt.writer_witness.completion_status, 'done');
  assert.equal(evalReturnReceipt.writer_witness.completion_source, 'execution_graph');
  assert.equal(evalReturnReceipt.writer_witness.completion_detail, `\uE000${noteId}\uE001`);

  const failedEvalReturn = withLegacyResult('eval-return', false);
  failedEvalReturn.outcome.invocation_ledger.at(-1).completion_status = 'failed';
  assert.equal(materializeNativeRows([failedEvalReturn]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId), undefined, 'failed eval-return cannot authenticate a context');
  const wrongEvalReturnHost = withLegacyResult('eval-return', false);
  wrongEvalReturnHost.outcome.invocation_ledger.at(-1).host_result.call_id = 'different-call';
  assert.equal(materializeNativeRows([wrongEvalReturnHost]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId), undefined, 'eval-return host identity must match the writer call');
  const wrongEvalReturnRef = withLegacyResult('eval-return', false);
  wrongEvalReturnRef.outcome.invocation_ledger.at(-1).host_result.value.$neuralese.id = `nz1_${'c'.repeat(32)}`;
  assert.equal(materializeNativeRows([wrongEvalReturnRef]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId), undefined, 'eval-return host reference must match the written block');
  const wrongEvalReturnType = withLegacyResult('eval-return', false);
  wrongEvalReturnType.outcome.invocation_ledger.at(-1).host_result.value.$neuralese.type = 'Neuralese<number>';
  assert.equal(materializeNativeRows([wrongEvalReturnType]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId), undefined, 'eval-return host type must match the written block');

  const returnReemission = withLegacyResult('return_result', false);
  const writes = returnReemission.outcome.execution_graph.filter(event => event.kind === 'block_write' && event.block === noteId);
  const upstreamEvalOutput = structuredClone(writes[0]);
  Object.assign(upstreamEvalOutput, { seq: 2, node: 'producer-call#5', source: 'eval-return' });
  Object.assign(writes[0], { seq: 8, node: 'producer-call#9', source: 'return_result' });
  const consumerRead = returnReemission.outcome.execution_graph.find(event => event.kind === 'block_read' && event.block === noteId);
  consumerRead.inputs = [{ node: 'producer-call#9', block: noteId, port: 'block' }];
  returnReemission.outcome.execution_graph.push(upstreamEvalOutput);
  const reemissionReceipt = materializeNativeRows([returnReemission]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId);
  assert.equal(reemissionReceipt.writer_source_class, 'legacy-text-marker-standin-return-result-host-reference');
  assert.equal(reemissionReceipt.producer_write.node, 'producer-call#9',
    'context is bound to the exact re-emission node consumed by the read');
  assert.equal(reemissionReceipt.writer_witness.kind, 'completed-return-result-host-reference-reemission');
  assert.equal(reemissionReceipt.writer_witness.upstream_typed_output_node, 'producer-call#5');
  const failedReemission = structuredClone(returnReemission);
  failedReemission.outcome.invocation_ledger.at(-1).completion_status = 'failed';
  assert.equal(materializeNativeRows([failedReemission]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId), undefined, 'failed host completion cannot authenticate a re-emission');
  const mismatchedUpstream = structuredClone(returnReemission);
  const upstreamWrite = mismatchedUpstream.outcome.execution_graph.find(event => event.kind === 'block_write' &&
    event.node === 'producer-call#5');
  upstreamWrite.text_body_sha256 = sha('different body');
  assert.equal(materializeNativeRows([mismatchedUpstream]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId), undefined, 'upstream typed output must match the exact expanded body digest');

  const wrongEvalReturnBody = withLegacyResult('eval-return', false);
  wrongEvalReturnBody.trajectory[0].model_response.transport_provenance.expanded_input_blocks[1].body = 'other note';
  wrongEvalReturnBody.trajectory[0].model_response.transport_provenance.expanded_input_blocks[1].body_sha256 = sha('other note');
  assert.equal(materializeNativeRows([wrongEvalReturnBody]).turns[0].source_ref.provider_expanded_read_contexts
    .find(item => item.block.id === noteId), undefined, 'eval-return body must match the graph write digest');
});
