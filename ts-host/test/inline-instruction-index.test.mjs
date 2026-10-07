import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { nativeDecisionTargetDigest } from '../dist/native/decision-review.js';
import { convertTrajectory } from '../dist/compiler/neuralese-conversion.js';
import { buildInlineInstructionIndex } from '../dist/compiler/inline-instruction-index.js';
import { decisionExtractChain } from '../scripts/inline-curriculum/decision-rich.mjs';
import { referenceRow } from '../scripts/inline-curriculum/references.mjs';
import { curriculumCase, evalCall, returnCall } from '../scripts/inline-curriculum/lib.mjs';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const sha = value => createHash('sha256').update(value).digest('hex');

function makeSite({ code, template, parent = 'parent-call', tool = 'eval-call', definition = 'def-a',
  captures = [], interpolations = [], valid = true, span, explicitCaptures = false } = {}) {
  const start = span?.start ?? code.indexOf(template);
  const end = span?.end ?? start + template.length;
  const literal = template.slice(1, -1);
  return { schema: 'natlang.inline_instruction_site/1', definition_id: definition,
    template_span: { start, end }, template_segments: [literal],
    interpolations, parameters: [{ name: 'item', type: 'string' }], returns: { text: 'boolean' },
    captures, explicit_captures: explicitCaptures, realized_instruction: literal.endsWith('\n') ? literal : `${literal}\n`,
    origin: { parentInvocationId: parent, toolCallId: tool, actionOrdinal: 1,
      writtenCodeSha256: sha(code), checkedCodeSha256: sha(code) },
    __valid: valid };
}

function parentRow({ code, trajectory = 'run-1', invocation = 'parent-call', tool = 'eval-call',
  index = 0, status = 'ok', accepted = true, approved = true } = {}) {
  return { id: `${trajectory}:decision:${String(index).padStart(4, '0')}`,
    source_ref: { trajectory_id: trajectory, invocation_id: invocation },
    outcome: { accepted }, training_admission: { approved },
    target: { role: 'assistant', tool_calls: [{ id: `teacher_${index}_0`, type: 'function',
      function: { name: 'eval', arguments: JSON.stringify({ code }) } }] },
    decision: { index, training_approved: approved,
      assistant: { calls: [{ call_id: null, source_tool: 'eval', arguments: { code },
        outcome: { status, tool_call_id: tool } }] } }, messages: [] };
}

function childRow({ code, template, trajectory = 'run-1', invocation = 'child-call', parent = 'parent-call',
  tool = 'eval-call', definition = 'def-a', index = 1, opening, siteOptions = {} } = {}) {
  const site = makeSite({ code, template, parent, tool, definition, ...siteOptions });
  const { __valid, ...savedSite } = site;
  return { id: `${trajectory}:decision:${String(index).padStart(4, '0')}`,
    source_ref: { trajectory_id: trajectory, invocation_id: invocation, parent_invocation_id: parent,
      inline_instruction_site: { site: savedSite, validation: { valid: __valid, reasons: __valid ? [] : ['test-invalid'] } } },
    outcome: { accepted: true }, decision: { index, training_approved: true, assistant: { calls: [] } },
    messages: [{ role: 'user', content: opening ?? `Instructions:\n${savedSite.realized_instruction}\n\nIn eval you can use item.` }] };
}

test('indexes linked sites when rows are shuffled and repeated children share one action-scoped writer', () => {
  const code = 'const check = nl`Check each complete note.`;';
  const template = '`Check each complete note.`';
  const parent = parentRow({ code });
  const childA = childRow({ code, template, invocation: 'child-a', index: 3 });
  const childB = childRow({ code, template, invocation: 'child-b', index: 5 });
  const index = buildInlineInstructionIndex([childB, parent, childA]);
  assert.equal(index.writers.length, 1);
  assert.equal(index.reads.length, 2);
  assert.equal(index.held.length, 0);
  assert.equal(index.writers[0].template_source, template);
  assert.equal(index.writers[0].code, code);
  assert.equal(index.writers[0].tool_call_id, 'eval-call');
  assert.equal(index.writers[0].target_tool_call_id, 'teacher_0_0');
  assert.equal(index.reads[0].writer_id, index.reads[1].writer_id);
  assert.deepEqual(index.reads.map(read => read.invocation_id).sort(), ['child-a', 'child-b']);
  assert.equal(index.reads[0].realized_instruction, 'Check each complete note.\n');
});

test('identical realized text at distinct compiler sites gets distinct writers', () => {
  const code = 'const a = nl`Same instruction.`; const b = nl`Same instruction.`;';
  const t = '`Same instruction.`';
  const first = code.indexOf(t), second = code.indexOf(t, first + 1);
  const parentA = parentRow({ code, invocation: 'parent-call', tool: 'eval-a', index: 0 });
  const parentB = parentRow({ code, invocation: 'parent-call', tool: 'eval-b', index: 1 });
  // The distinct target action IDs are part of the origin, even when text is equal.
  const childA = childRow({ code, template: t, tool: 'eval-a', definition: 'def-a', index: 2, invocation: 'child-a',
    siteOptions: { span: { start: first, end: first + t.length } } });
  const childB = childRow({ code, template: t, tool: 'eval-b', definition: 'def-b', index: 3, invocation: 'child-b',
    siteOptions: { span: { start: second, end: second + t.length } } });
  const index = buildInlineInstructionIndex([childA, parentB, childB, parentA]);
  assert.equal(index.writers.length, 2);
  assert.equal(new Set(index.writers.map(item => item.writer_id)).size, 2);
  assert.equal(index.reads.length, 2);
  assert.notEqual(index.reads[0].writer_id, index.reads[1].writer_id);
});

test('tampered hash and span fail closed', () => {
  const code = 'const check = nl`Check note.`;';
  const template = '`Check note.`';
  const hashTampered = childRow({ code, template, invocation: 'child-hash', index: 1 });
  hashTampered.source_ref.inline_instruction_site.site.origin.writtenCodeSha256 = '0'.repeat(64);
  const spanTampered = childRow({ code, template, invocation: 'child-span', index: 2,
    siteOptions: { span: { start: 0, end: template.length } } });
  const result = buildInlineInstructionIndex([parentRow({ code }), hashTampered, spanTampered]);
  assert.equal(result.writers.length, 0);
  assert.equal(result.reads.length, 0);
  assert(result.held.some(item => item.reason === 'checked-source-hash-mismatch'));
  assert(result.held.some(item => item.reason === 'template-span-or-cooked-text-mismatch'));
});

test('target action name, code, and training approval must match the linked runtime action', () => {
  const code = 'const check = nl`Check note.`;';
  const template = '`Check note.`';
  const badTargetName = parentRow({ code });
  badTargetName.target.tool_calls[0].function.name = 'return_result';
  const badTargetCode = parentRow({ code, tool: 'eval-code-mismatch' });
  badTargetCode.target.tool_calls[0].function.arguments = JSON.stringify({ code: 'const other = 1;' });
  const heldAdmission = parentRow({ code, tool: 'eval-held', approved: true });
  heldAdmission.training_admission.approved = false;
  const result = buildInlineInstructionIndex([badTargetName,
    childRow({ code, template, invocation: 'child-name', index: 1 }),
    badTargetCode, childRow({ code, template, tool: 'eval-code-mismatch', invocation: 'child-code', index: 2 }),
    heldAdmission, childRow({ code, template, tool: 'eval-held', invocation: 'child-held', index: 3 })]);
  assert.equal(result.writers.length, 0);
  assert.equal(result.reads.length, 0);
  assert.equal(result.held.filter(item => item.reason === 'target-action-identity-or-arguments-mismatch').length, 2);
  assert(result.held.some(item => item.reason === 'parent-action-not-successful-and-approved'));
});

test('invalid, failed, unapproved, and absent parent actions create no edge', () => {
  const code = 'const check = nl`Check note.`;';
  const template = '`Check note.`';
  const invalid = childRow({ code, template, invocation: 'child-invalid', index: 1, siteOptions: { valid: false } });
  const failed = childRow({ code, template, invocation: 'child-failed', index: 2, tool: 'failed-eval' });
  const unapproved = childRow({ code, template, invocation: 'child-unapproved', index: 3, tool: 'unapproved-eval' });
  const absent = childRow({ code, template, invocation: 'child-absent', index: 4, tool: 'absent-eval' });
  const result = buildInlineInstructionIndex([invalid, failed, unapproved, absent,
    parentRow({ code, tool: 'failed-eval', index: 0, status: 'error' }),
    parentRow({ code, tool: 'unapproved-eval', index: 0, approved: false })]);
  assert.equal(result.writers.length, 0);
  assert.equal(result.reads.length, 0);
  assert.equal(result.held.length, 4);
  assert.deepEqual(new Set(result.held.map(item => item.reason)), new Set([
    'invalid-compiler-site', 'parent-action-not-successful-and-approved', 'missing-parent-action',
  ]));
});

test('unsupported interpolation and captures remain exact', () => {
  const code = 'const check = nl`Check ${limit}.`;';
  const template = '`Check ${limit}.`';
  const holes = childRow({ code, template, invocation: 'child-hole', index: 1,
    siteOptions: { interpolations: [{ expression: 'limit', rendered: '17' }] } });
  const captures = childRow({ code: 'const check = nl`Check note.`;', template: '`Check note.`', invocation: 'child-capture', index: 2,
    siteOptions: { captures: [{ name: 'criterion', type: 'string' }] } });
  const result = buildInlineInstructionIndex([parentRow({ code }), holes,
    parentRow({ code: 'const check = nl`Check note.`;', tool: 'other-eval', index: 0 }), captures]);
  assert.equal(result.writers.length, 0);
  assert.equal(result.reads.length, 0);
  assert.deepEqual(new Set(result.held.map(item => item.reason)), new Set(['unsupported-interpolation', 'unsupported-capture-contract']));
});

test('a child must occur after its actual eval writer in decision order', () => {
  const code = 'const check = nl`Check note.`;';
  const result = buildInlineInstructionIndex([parentRow({ code, index: 4 }),
    childRow({ code, template: '`Check note.`', index: 3 })]);
  assert.equal(result.writers.length, 0);
  assert.equal(result.reads.length, 0);
  assert.equal(result.held[0].reason, 'source-order-invalid-or-unknown');
});

test('duplicate parent origins and conflicting repeated site metadata fail closed', () => {
  const code = 'const check = nl`Check note.`;';
  const template = '`Check note.`';
  const duplicateOriginChild = childRow({ code, template, invocation: 'child-dup-origin', index: 2 });
  const duplicateParentA = parentRow({ code, invocation: 'parent-call', tool: 'eval-call', index: 0 });
  const duplicateParentB = parentRow({ code, invocation: 'parent-call', tool: 'eval-call', index: 1 });
  const conflictingA = childRow({ code, template, invocation: 'child-conflict', index: 3 });
  const conflictingB = childRow({ code, template: '`Different text.`', invocation: 'child-conflict', index: 4,
    siteOptions: { span: { start: code.indexOf(template), end: code.indexOf(template) + template.length } } });
  const result = buildInlineInstructionIndex([duplicateOriginChild, duplicateParentA, duplicateParentB,
    conflictingA, conflictingB]);
  assert.equal(result.writers.length, 0);
  assert.equal(result.reads.length, 0);
  assert(result.held.some(item => item.reason === 'duplicate-origin-action'));
  assert(result.held.some(item => item.reason === 'duplicate-site-metadata-conflict'));
});

test('typed metadata conflict across children blocks that entire site while an unrelated site remains usable', () => {
  const code = 'const a = nl<boolean>`First site.`; const b = nl<boolean>`Second site.`;';
  const first = '`First site.`', second = '`Second site.`';
  const firstStart = code.indexOf(first), secondStart = code.indexOf(second);
  const parent = parentRow({ code });
  const firstChildA = childRow({ code, template: first, invocation: 'child-a', index: 1, definition: 'def-a',
    siteOptions: { span: { start: firstStart, end: firstStart + first.length } } });
  const firstChildB = childRow({ code, template: first, invocation: 'child-b', index: 2, definition: 'def-a',
    siteOptions: { span: { start: firstStart, end: firstStart + first.length } } });
  firstChildB.source_ref.inline_instruction_site.site.returns = { text: 'string' };
  const independentChild = childRow({ code, template: second, invocation: 'child-c', index: 3, definition: 'def-b',
    siteOptions: { span: { start: secondStart, end: secondStart + second.length } } });
  const result = buildInlineInstructionIndex([independentChild, firstChildB, parent, firstChildA]);
  assert.deepEqual(result.writers.map(item => item.definition_id), ['def-b']);
  assert.deepEqual(result.reads.map(item => item.invocation_id), ['child-c']);
  assert.equal(result.held.filter(item => item.reason === 'conflicting-writer-site-metadata').length, 2);
});

test('equal realized text outside the actual Instructions section does not establish a read', () => {
  const code = 'const check = nl`Check the current state.`;';
  const child = childRow({ code, template: '`Check the current state.`', invocation: 'quoted-only', index: 1,
    opening: 'You are inside this call: child()\n\nInstructions:\nRead the report and decide.\n\nIn eval this text appears in the quoted input: Check the current state.\n' });
  const result = buildInlineInstructionIndex([parentRow({ code }), child]);
  assert.equal(result.writers.length, 0);
  assert.equal(result.reads.length, 0);
  assert.equal(result.held[0].reason, 'realized-instruction-not-visible-in-child');
});

test('materialized decision-rich trajectory keeps interpolation sites held and joins real tool identities', async () => {
  const [program] = decisionExtractChain(7202, 72);
  const nativeRow = await referenceRow(program, 0, { modelId: 'static-proof', rootSeed: 7202,
    systemPrompt: TOOLS_PROMPT, contextTokens: 65536, maxTurns: 60, collectionRole: 'reference',
    authoredActionPlans: true, followCutoffPages: true, followEvalCutoffPages: true });
  assert.equal(nativeRow.outcome.accepted, true);
  const turns = materializeNativeRows([nativeRow], { directAnswers: true }).turns;
  const index = buildInlineInstructionIndex(turns);
  assert.equal(index.writers.length, 0, 'this fixture interpolates the dynamic question and stays exact in Stage 2');
  assert.equal(index.reads.length, 0);
  assert(index.held.some(item => item.reason === 'unsupported-interpolation'));

  const parent = turns.find(turn => turn.decision.assistant.calls.some(call => call.source_tool === 'eval'));
  const parentCallIndex = parent.decision.assistant.calls.findIndex(call => call.source_tool === 'eval');
  const call = parent.decision.assistant.calls[parentCallIndex];
  const target = parent.target.tool_calls[parentCallIndex];
  assert.equal(call.call_id, null, 'collector call_id is not used as the origin identity');
  assert.equal(typeof call.outcome.tool_call_id, 'string');
  assert.equal(target.function.name, 'eval');
  assert.equal(JSON.parse(target.function.arguments).code, call.arguments.code);
});

test('a real materialized no-interpolation inline site yields one writer and its child read', async () => {
  const code = 'const judge = nl<boolean>`Does the report describe an active problem now?`; const answer = await judge(report); return answer;';
  const program = curriculumCase({ family: 'inline_instruction_index_test', shape: 'e2e-no-hole',
    splitGroup: 'inline_instruction_index_test:e2e-no-hole', split: 'train', slice: 'single', domain: 'other',
    mode: 'single_call', inline: 'required',
    root: { name: 'review', args: { report: 'string' }, returns: 'boolean',
      instructions: 'Decide whether the report describes an active problem now.' },
    inputs: { report: 'Users cannot submit orders today.' }, expected: true,
    reference: { root: [evalCall(code), returnCall(true)],
      children: [{ match: 'You are inside this call: nl@eval:', calls: [returnCall(true)] }] } });
  const collected = await referenceRow(program, 0, { modelId: 'static-proof', rootSeed: 8123,
    systemPrompt: TOOLS_PROMPT, contextTokens: 65536, maxTurns: 40, collectionRole: 'reference',
    authoredActionPlans: true });
  assert.equal(collected.outcome.accepted, true);
  const turns = materializeNativeRows([collected], { directAnswers: true }).turns;
  const index = buildInlineInstructionIndex(turns);
  assert.deepEqual(index.held, [], JSON.stringify(index.held));
  assert.equal(index.writers.length, 1);
  assert.equal(index.reads.length, 1);
  assert.equal(index.held.length, 0);
  const writer = index.writers[0];
  assert.equal(writer.template_source, '`Does the report describe an active problem now?`');
  assert.equal(writer.tool_call_id, collected.outcome.invocation_ledger.find(item => item.inline_instruction_site)?.inline_instruction_site.origin.toolCallId);
  assert.match(writer.target_tool_call_id, /^teacher_\d+_\d+$/);
  assert.equal(index.reads[0].writer_id, writer.writer_id);
  const converted=turns.map(turn=>convertTrajectory(turn,{inlineInstructions:index}).record);
  const parent=converted.find(turn=>turn.id===writer.decision_id);
  assert.equal(parent.target.tool_calls[0].neuralese_code.parts.filter(part=>part.$write).length,1);
  const child=converted.find(turn=>turn.id===index.reads[0].decision_id);
  assert.ok(child.messages.some(message=>Array.isArray(message.content) && message.content.some(part=>part.type==='read'&&part.name===writer.writer_id)));
});

test('host-attested nl.with aliases and primitive literal captures preserve child bindings', async () => {
  const code = 'const judge = nl.with<boolean>({ note: report, minimum: 3 as const, active: true as const, label: "approved" as const })`Does the note describe an active problem lasting at least the captured minimum days?`; const answer = await judge(report); return answer;';
  const program = curriculumCase({ family: 'inline_instruction_index_test', shape: 'e2e-capture-alias',
    splitGroup: 'inline_instruction_index_test:e2e-capture-alias', split: 'train', slice: 'single', domain: 'other',
    mode: 'single_call', inline: 'required',
    root: { name: 'review', args: { report: 'string' }, returns: 'boolean',
      instructions: 'Decide whether the report describes an active problem.' },
    inputs: { report: 'Orders cannot be submitted today.' }, expected: true,
    reference: { root: [evalCall(code), returnCall(true)],
      children: [{ match: 'You are inside this call: nl@eval:', calls: [returnCall(true)] }] } });
  const collected = await referenceRow(program, 0, { modelId: 'static-proof', rootSeed: 8124,
    systemPrompt: TOOLS_PROMPT, contextTokens: 65536, maxTurns: 40, collectionRole: 'reference',
    authoredActionPlans: true });
  assert.equal(collected.outcome.accepted, true);
  const turns = materializeNativeRows([collected], { directAnswers: true }).turns;
  const index = buildInlineInstructionIndex(turns);
  assert.deepEqual(index.held, [], JSON.stringify(index.held));
  assert.equal(index.writers.length, 1);
  assert.equal(index.reads.length, 1);
  assert.equal(index.held.length, 0);
  assert.deepEqual(index.writers[0].plan.capture_binding_plan.captures.map(capture => capture.name), ['note', 'minimum', 'active', 'label']);
  assert.equal(index.writers[0].plan.capture_binding_plan.captures[0].value, 'Orders cannot be submitted today.');
  assert.equal(index.writers[0].plan.capture_binding_plan.captures[1].value, 3);
  assert.equal(index.writers[0].plan.capture_binding_plan.captures[1].declared_type, '3');
  assert.equal(index.writers[0].plan.capture_binding_plan.captures[2].value, true);
  assert.equal(index.writers[0].plan.capture_binding_plan.captures[2].declared_type, undefined);
  assert.equal(index.writers[0].plan.capture_binding_plan.captures[3].value, 'approved');
  assert.equal(index.writers[0].plan.capture_binding_plan.captures[3].declared_type, '"approved"');
});


test('conversion preserves exact eval arguments and links multiple code writers to their own children', () => {
  const template='`Check the report carefully.`';
  const code=`const a=nl<boolean>${template}; const b=nl<boolean>${template}; return await a(report) && await b(report);`;
  const parent=parentRow({code});
  const a=childRow({code,template,invocation:'child-a',definition:'a',index:1});
  const start=code.indexOf(template,code.indexOf(template)+1);
  const b=childRow({code,template,invocation:'child-b',definition:'b',index:2,siteOptions:{span:{start,end:start+template.length}}});
  const inlineInstructions=buildInlineInstructionIndex([b,parent,a]);
  assert.equal(inlineInstructions.writers.length,2);
  const written=convertTrajectory(parent,{inlineInstructions}).record;
  const call=written.target.tool_calls[0];
  assert.equal(call.function.arguments,parent.target.tool_calls[0].function.arguments);
  assert.equal(call.neuralese_code.schema,'natlang.inline-instruction-code/1');
  assert.equal(call.neuralese_code.parts.map(part=>part.type==='text'?part.text:part.$write.source).join(''),code);
  assert.equal(call.neuralese_code.parts.filter(part=>part.$write).length,2);
  for (const child of [a,b]) {
    const converted=convertTrajectory(child,{inlineInstructions,instructionsShare:1}).record;
    const opening=converted.messages[0].content;
    assert.ok(Array.isArray(opening));
    const read=opening.find(part=>part.type==='read');
    assert.equal(read.name,inlineInstructions.reads.find(item=>item.invocation_id===child.source_ref.invocation_id).writer_id);
    assert.equal(opening.map(part=>part.type==='text'?part.text:part.source).join(''),child.messages[0].content);
    assert.equal(opening.some(part=>part.type==='soft'),false,'no detached shared instructions compete with the read');
  }
});


test('an exact reviewed writer in a failed parent is eligible; arbitrary approval cannot override it', () => {
  const code = 'const check = nl`Check each complete note.`;'; const template = '`Check each complete note.`';
  const parent = parentRow({ code, accepted: false });
  const child = childRow({ code, template });
  assert.equal(buildInlineInstructionIndex([parent, child]).writers.length, 0);
  parent.source_ref.source_row_sha256 = 'a'.repeat(64);
  const approval = { schema: 'natlang.native-decision-approval/1', trajectory_id: 'run-1',
    source_row_sha256: 'a'.repeat(64), decision_index: 0, target_sha256: nativeDecisionTargetDigest(parent.target),
    review_sha256: 'b'.repeat(64), reason: 'correct exact writer; later parent answer failed', evidence: ['source and actual action inspected'] };
  parent.training_admission = { kind: 'reviewed-native-decision', approved: true, semantic_review: approval };
  assert.equal(buildInlineInstructionIndex([parent, child]).writers.length, 1);
  approval.target_sha256 = 'c'.repeat(64);
  assert.equal(buildInlineInstructionIndex([parent, child]).writers.length, 0);
});
