import assert from 'node:assert/strict';
import { test } from 'node:test';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';

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


test('tolerance-based acceptance and pending file quality cannot become positive training data', () => {
  for (const evidence of [
    { oracle: { level: 'agreement', accepted: true, score: 0.946 } },
    { oracle: { level: 'span', accepted: true, score: 0.95 } },
    { files_check: { failed: ['one-message'], score: 0.95 } },
    { quality_pending: ['independent review needed'] },
  ]) {
    const row = nativeRow('partial-correctness');
    Object.assign(row.outcome, evidence);
    assert.equal(materializeNativeRows([row]).acceptedRows, 0);
    assert.equal(materializeNativeRows([row], { directAnswers: true }).turns.length, 0);
  }
  const exact = nativeRow('full-agreement');
  exact.outcome.oracle = { level: 'agreement', accepted: true, score: 1 };
  assert.equal(materializeNativeRows([exact]).acceptedRows, 1);
});
