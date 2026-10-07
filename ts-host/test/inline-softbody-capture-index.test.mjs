import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { convertTrajectory } from '../dist/compiler/neuralese-conversion.js';
import { buildInlineInstructionIndex } from '../dist/compiler/inline-instruction-index.js';
import { sourceWithLiteralCalls } from '../dist/native/neuralese.js';
import { desugarNlCalls } from '../dist/compiler/nl-call.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { executeProgram, expectedProvenance, programRow, programRunId, trajectoryTurn, defaultToolSurfaceHash } from '../dist/teacher/collector.js';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { inline } from '../dist/runtime/lowered.js';
import { callableMeta } from '../dist/runtime/callable.js';
import { NATLANG_COMPILE_VERSION } from '../dist/compiler/intrinsics.js';
import { curriculumCase } from '../scripts/inline-curriculum/lib.mjs';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

const sha = value => createHash('sha256').update(value).digest('hex');
const block = 'nz1_abcdefghijklmnopqrstuv';
const body = 'Return the supplied policySnapshot exactly as written.';
const policy = 'Only use details stated in the note.';

function fixture({ childPolicy = policy, capture = {}, codeOverride } = {}) {
  const sentinel = `\uE000${block}\uE001`;
  const code = codeOverride ?? `const checker: Neuralese<(note: string) => Promise<boolean>> = nl.with({ policy })\`${sentinel}\`; return await checker(note);`;
  const rawSpanStart = code.indexOf(`\`${sentinel}\``);
  const rawSpan = { start: rawSpanStart, end: rawSpanStart + sentinel.length + 2 };
  const checked = desugarNlCalls(sourceWithLiteralCalls(code));
  const checkedStart = checked.indexOf(`\`\${__neuralese.body(${JSON.stringify(block)})}\``);
  const checkedSpan = { start: checkedStart, end: checkedStart + `\`\${__neuralese.body(${JSON.stringify(block)})}\``.length };
  const capturePlan = { name: 'policy', source: 'input', mutable: false, mode: 'snapshot',
    type: { text: 'string', natlang: 'string' } };
  const site = { schema: 'natlang.inline_instruction_site/1', definition_id: 'checker-definition',
    template_span: rawSpan, checked_template_span: checkedSpan, template_segments: [''], interpolations: [],
    parameters: [{ name: 'note', type: { text: 'string', natlang: 'string' } }],
    returns: { text: 'boolean', natlang: 'boolean' }, captures: [{ ...capturePlan, ...capture }],
    explicit_captures: true, soft_body_id: block, raw_body_source: body, raw_body_source_sha256: sha(body),
    runtime_captures: { policy: { mode: 'snapshot', type: 'string' } },
    origin: { parentInvocationId: 'parent', toolCallId: 'eval-call', actionOrdinal: 0,
      writtenCodeSha256: sha(code), checkedCodeSha256: sha(checked), sourceTemplateSpan: rawSpan } };
  const parent = { id: 'trajectory:decision:0000', source_ref: { trajectory_id: 'trajectory', invocation_id: 'parent' },
    outcome: { accepted: true }, training_admission: { approved: true }, decision: { index: 0, training_approved: true,
      assistant: { calls: [{ source_tool: 'eval', arguments: { code }, outcome: { status: 'ok', tool_call_id: 'eval-call' } }] } },
    target: { tool_calls: [{ id: 'teacher_eval', function: { name: 'eval', arguments: JSON.stringify({ code }) } }] },
    messages: [{ role: 'assistant', tool_calls: [{ id: 'scope_0', function: { name: 'eval', arguments:
        JSON.stringify({ code: 'const inputs = read_inputs();\nconst policy: string = "Only use details stated in the note.";' }) } }] },
      { role: 'tool', tool_call_id: 'scope_0', content: `note: string = "A note."\npolicy: string = ${JSON.stringify(policy)}\nDeclared note, policy for the rest of this call.` }] };
  const child = { id: 'trajectory:decision:0001', source_ref: { trajectory_id: 'trajectory', invocation_id: 'child',
      parent_invocation_id: 'parent', inline_instruction_site: { site, validation: { valid: true, reasons: [] } } },
    outcome: { accepted: true }, training_admission: { approved: true }, decision: { index: 1, training_approved: true,
      assistant: { calls: [] } }, messages: [{ role: 'user', content: [
      { type: 'text', text: `Instructions:\n` }, { type: 'neuralese', id: block },
      { type: 'text', text: `\n\nIn eval you can use inputs.\nconst policy: string = ${JSON.stringify(childPolicy)};` },
    ] }, { role: 'assistant', tool_calls: [{ id: 'scope_0', function: { name: 'eval', arguments:
        JSON.stringify({ code: `const policy: string = ${JSON.stringify(childPolicy)};` }) } }] }] };
  return { parent, child };
}

test('indexes a compiler-checked soft body with an explicit primitive snapshot capture', () => {
  const { parent, child } = fixture();
  const result = buildInlineInstructionIndex([child, parent]);
  assert.equal(result.held.length, 0);
  assert.equal(result.writers.length, 1);
  assert.equal(result.reads.length, 1);
  assert.equal(result.writers[0].body_source, body);
  assert.equal(result.writers[0].body_code_source, `\uE000${block}\uE001`);
  assert.equal(result.writers[0].plan.capture_binding_plan.captures[0].value, policy);
  assert.match(result.reads[0].capture_binding_plan.parent_scope_sha256, /^[a-f0-9]{64}$/);
});

test('holds snapshot captures when visibility, binding contract, or checked span is changed', () => {
  const wrongValue = fixture({ childPolicy: 'A different policy.' });
  const wrongType = fixture({ capture: { type: { text: 'Live<string>', natlang: 'Live<string>' } } });
  const wrongSpan = fixture();
  wrongSpan.child.source_ref.inline_instruction_site.site.checked_template_span.start += 1;
  for (const rows of [wrongValue, wrongType, wrongSpan]) {
    const result = buildInlineInstructionIndex([rows.child, rows.parent]);
    assert.equal(result.writers.length, 0);
    assert.equal(result.reads.length, 0);
    assert.equal(result.held.length, 1);
  }
});

test('quoted or unrelated scope text cannot substitute for parent input or child scope declarations', () => {
  const parentDecoy = fixture();
  parentDecoy.parent.messages = [{ role: 'user', content: `Example only: policy: string = ${JSON.stringify(policy)}` },
    { role: 'tool', tool_call_id: 'scope_1', content: `policy: string = ${JSON.stringify(policy)}` }];
  const childDecoy = fixture();
  childDecoy.child.messages = childDecoy.child.messages.filter(message =>
    !(message.role === 'assistant' && message.tool_calls?.some(call => call.id === 'scope_0')));
  childDecoy.child.messages.push({ role: 'tool', tool_call_id: 'call_1_0',
    content: `const policy: string = ${JSON.stringify(policy)};` });
  for (const rows of [parentDecoy, childDecoy]) {
    const result = buildInlineInstructionIndex([rows.child, rows.parent]);
    assert.equal(result.writers.length, 0);
    assert.equal(result.reads.length, 0);
    assert.equal(result.held.length, 1);
  }
});

test('actual runtime collection preserves a captured soft body through materialization', async () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const rawCode = `const policySnapshot: string = policy; const checker: Neuralese<(note: string) => Promise<string>> = nl.with({ policySnapshot })\`<|neuralese|>${body}<|/neuralese|>\`; const actualPolicy = await checker(note); return actualPolicy === policy;`;
  const program = curriculumCase({ family: 'inline_softbody_capture_runtime_test', shape: 'explicit-snapshot',
    splitGroup: 'inline_softbody_capture_runtime_test:explicit-snapshot', split: 'train', slice: 'single', domain: 'other',
    mode: 'single_call', inline: 'required', root: { name: 'review', args: { note: 'string', policy: 'string' },
      returns: 'boolean', instructions: 'Create the instructed checker and ask it to classify the note.' },
    inputs: { note: 'The package arrived damaged.', policy }, expected: true, reference: { root: [] } });
  const options = { modelId: 'softbody-runtime-fixture', rootSeed: 9127, systemPrompt: TOOLS_PROMPT,
    contextTokens: 65536, maxTurns: 20, toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference' };
  const expected = expectedProvenance(program, options);
  const runId = programRunId(0, expected);
  const trajectory = [], requests = [];
  const driver = Object.assign(async request => {
    requests.push(request);
    let response;
    if (requests.length === 1) {
      const args = { code: rawCode };
      response = { calls: [['eval', args]], raw_calls: [{ id: 'call_1_0', type: 'function',
        function: { name: 'eval', arguments: JSON.stringify(args) } }] };
    } else if (requests.length === 2) {
      const args = { code: 'return_result(policySnapshot);' };
      response = { calls: [['eval', args]], raw_calls: [{ id: 'call_1_0', type: 'function',
        function: { name: 'eval', arguments: JSON.stringify(args) } }] };
    } else response = { calls: [['return_result', { status: 'success' }]] };
    trajectory.push(trajectoryTurn(request, response));
    return response;
  }, { neuralese: true });
  const run = await executeProgram(program, driver, { ...options, runId, neuralese: { store, port } });
  const row = programRow(program, options.modelId, runId, expected, run, trajectory);
  const materialized = materializeNativeRows([row], { failedRuns: true });
  assert.ok(requests.length >= 4, 'the child read the capture and both invocations returned their real staged values');
  const child = materialized.turns.find(turn => turn.source_ref.parent_invocation_id);
  assert.ok(child, `materializer must retain the child invocation; rejected=${materialized.rejectedRows}, unlinked=${JSON.stringify(materialized.unlinked)}`);
  const site = child.source_ref.inline_instruction_site;
  assert.ok(site, 'compiler-authored soft-body site survives collection and materialization');
  assert.equal(site.validation.valid, true, JSON.stringify(site.validation.reasons));
  assert.equal(site.site.raw_body_source, body);
  assert.deepEqual(site.site.runtime_captures, { policySnapshot: { mode: 'snapshot', type: 'string' } });
  assert.deepEqual(site.site.runtime_capture_snapshots, { schema: 'natlang.runtime_capture_snapshots/1', captures: [{
    name: 'policySnapshot', type: 'string', source: 'local', mode: 'snapshot', value: policy,
    value_canonical: JSON.stringify({ type: 'string', value: policy }),
    value_sha256: sha(`natlang.inline-capture-snapshot/v1\0${JSON.stringify({ type: 'string', value: policy })}`),
    creation: { parentInvocationId: site.site.origin.parentInvocationId, toolCallId: site.site.origin.toolCallId,
      actionOrdinal: site.site.origin.actionOrdinal, writtenCodeSha256: site.site.origin.writtenCodeSha256,
      checkedCodeSha256: site.site.origin.checkedCodeSha256, definitionId: site.site.definition_id,
      sourceSpan: site.site.source_span, templateSpan: site.site.template_span,
      checkedTemplateSpan: site.site.checked_template_span },
  }] });
  const index = buildInlineInstructionIndex(materialized.turns);
  assert.equal(index.held.length, 0, JSON.stringify(index.held));
  assert.equal(index.writers.length, 1);
  assert.equal(index.reads.length, 1);
  assert.equal(index.writers[0].plan.capture_binding_plan.schema, 'natlang.inline-capture-binding-plan/2');
  assert.equal(index.writers[0].plan.capture_binding_plan.captures[0].value, policy);
  const parent = materialized.turns.find(turn => turn.id === index.writers[0].decision_id);
  const converted = convertTrajectory(parent, { inlineInstructions: index }).record;
  assert.equal(converted.target.tool_calls[0].function.arguments, parent.target.tool_calls[0].function.arguments);
  const sidecar = converted.target.tool_calls[0].neuralese_code;
  assert.ok(sidecar);
  assert.equal(sidecar.parts.find(part => part.$write).$write.code_source, `<|neuralese|>${body}<|/neuralese|>`);
  assert.equal(index.writers[0].code_sha256, sha(JSON.parse(parent.target.tool_calls[0].function.arguments).code));
  const corrupt = structuredClone(materialized.turns);
  const corruptParent = corrupt.find(turn => turn.id === parent.id);
  corruptParent.decision.assistant.calls[0].outcome.arguments.code += ' ';
  assert.equal(buildInlineInstructionIndex(corrupt).writers.length, 0);
  const extraHole = structuredClone(materialized.turns);
  for (const turn of extraHole) {
    const site = turn.source_ref.inline_instruction_site?.site;
    if (site) site.interpolations.push({ expression: 'inventedDynamicValue' });
  }
  assert.equal(buildInlineInstructionIndex(extraHole).writers.length, 0);
  assert.ok(buildInlineInstructionIndex(extraHole).held.every(hold => hold.reason === 'unsupported-interpolation'));


  const scopeCall = requests[1].messages.flatMap(message => message.tool_calls ?? []).find(call => call.id === 'scope_0');
  assert.ok(scopeCall, 'child opening ran its actual typed scope declaration');
  assert.match(JSON.parse(scopeCall.function.arguments).code, /const policySnapshot: string = "Only use details stated in the note\."/);
});

test('explicit snapshot attestation uses one read and excludes live or non-primitive captures', () => {
  const origin = { parentInvocationId: 'parent', toolCallId: 'eval-call', actionOrdinal: 2,
    writtenCodeSha256: sha('creator'), checkedCodeSha256: sha('checked') };
  const makePlan = capture => ({ definitionId: 'capture-test', sourceSpan: { file: 'eval', start: 0, end: 20 },
    templateSpan: { start: 10, end: 20 }, strings: ['literal instruction'], interpolations: [], instructions: 'literal instruction',
    parameters: [], returns: { text: 'string', natlang: 'string' }, captures: [capture], explicitCaptures: true });
  let reads = 0;
  const snapshot = inline(makePlan({ name: 'policy', type: { text: 'string', natlang: 'string' }, source: 'local',
    mutable: false, mode: 'snapshot' }), [], { policy: [() => { reads++; return policy; }] }, undefined,
    NATLANG_COMPILE_VERSION, undefined, origin);
  assert.equal(reads, 1);
  const snapshotSite = callableMeta(snapshot).options.manifest.inline_instruction_site;
  assert.equal(snapshotSite.runtime_capture_snapshots.captures[0].value, policy);
  assert.equal(snapshotSite.runtime_capture_snapshots.captures[0].source, 'local');

  for (const [capture, value] of [
    [{ name: 'policy', type: { text: 'string', natlang: 'string' }, source: 'local', mutable: true, mode: 'live' }, policy],
    [{ name: 'packet', type: { text: 'object', natlang: '{ id: string }' }, source: 'local', mutable: false, mode: 'snapshot' }, { id: 'opaque' }],
    [{ name: 'packetFile', type: { text: 'FileHandle', natlang: 'FileHandle' }, source: 'local', mutable: false, mode: 'snapshot' }, {}],
    [{ name: 'negativeZero', type: { text: 'number', natlang: 'number' }, source: 'local', mutable: false, mode: 'snapshot' }, -0],
    [{ name: 'notANumber', type: { text: 'number', natlang: 'number' }, source: 'local', mutable: false, mode: 'snapshot' }, Number.NaN],
    [{ name: 'infinity', type: { text: 'number', natlang: 'number' }, source: 'local', mutable: false, mode: 'snapshot' }, Number.POSITIVE_INFINITY],
  ]) {
    const fn = inline(makePlan(capture), [], { [capture.name]: [() => value] }, undefined,
      NATLANG_COMPILE_VERSION, undefined, origin);
    assert.equal(Object.hasOwn(callableMeta(fn).options.manifest.inline_instruction_site, 'runtime_capture_snapshots'), false);
  }
});

function attestedFixture({ literal = false, source = 'local' } = {}) {
  const rows = fixture({ capture: { source } });
  const site = rows.child.source_ref.inline_instruction_site.site;
  site.source_span = { file: 'eval', start: 0, end: JSON.parse(rows.parent.target.tool_calls[0].function.arguments).code.length };
  if (literal) {
    const old = JSON.parse(rows.parent.target.tool_calls[0].function.arguments).code;
    const code = old.replace(`\uE000${block}\uE001`, body);
    rows.parent.target.tool_calls[0].function.arguments = JSON.stringify({ code });
    rows.parent.decision.assistant.calls[0].arguments.code = code;
    const start = code.indexOf(`\`${body}\``);
    site.template_span = { start, end: start + body.length + 2 };
    site.checked_template_span = { ...site.template_span };
    site.source_span.end = code.length;
    site.template_segments = [body];
    site.realized_instruction = body + '\n';
    delete site.soft_body_id;
    delete site.raw_body_source;
    delete site.raw_body_source_sha256;
    site.origin = { ...site.origin, writtenCodeSha256: sha(code), checkedCodeSha256: sha(code), sourceTemplateSpan: site.template_span };
    rows.child.messages[0].content = [{ type: 'text', text: `Instructions:\n${body}\n\nIn eval you can use inputs.\nconst policy: string = ${JSON.stringify(policy)};` }];
  }
  const value_canonical = JSON.stringify({ type: 'string', value: policy });
  site.runtime_capture_snapshots = { schema: 'natlang.runtime_capture_snapshots/1', captures: [{
    name: 'policy', type: 'string', source, mode: 'snapshot', value: policy, value_canonical,
    value_sha256: sha(`natlang.inline-capture-snapshot/v1\0${value_canonical}`),
    creation: { parentInvocationId: 'parent', toolCallId: 'eval-call', actionOrdinal: 0,
      writtenCodeSha256: site.origin.writtenCodeSha256, checkedCodeSha256: site.origin.checkedCodeSha256,
      definitionId: site.definition_id, sourceSpan: site.source_span, templateSpan: site.template_span,
      checkedTemplateSpan: site.checked_template_span },
  }] };
  return rows;
}

test('creation-attested local snapshots index crisp and soft bodies without parent input invention', () => {
  for (const literal of [false, true]) {
    const rows = attestedFixture({ literal });
    rows.parent.messages = [];
    const result = buildInlineInstructionIndex([rows.parent, rows.child]);
    assert.equal(result.held.length, 0, JSON.stringify(result.held));
    assert.equal(result.writers.length, 1);
    const binding = result.writers[0].plan.capture_binding_plan;
    assert.equal(binding.schema, 'natlang.inline-capture-binding-plan/2');
    assert.equal(binding.body_kind, literal ? 'literal' : 'neuralese_block');
    assert.equal(binding.captures[0].source, 'local');
    assert.equal(binding.captures[0].value, policy);
    assert.deepEqual(binding.creation, binding.captures[0].host_snapshot.creation);
    assert.equal(result.writers[0].body_code_source, literal ? body : `\uE000${block}\uE001`);
  }
});

test('local snapshot proof rejects altered value, digest, creator, span and incomplete envelopes', () => {
  for (const mutate of [
    snapshot => { snapshot.value = 'fabricated'; },
    snapshot => { snapshot.value_sha256 = '0'.repeat(64); },
    snapshot => { snapshot.creation.toolCallId = 'other-call'; },
    snapshot => { snapshot.creation.actionOrdinal = 1; },
    snapshot => { snapshot.creation.writtenCodeSha256 = '0'.repeat(64); },
    snapshot => { snapshot.creation.definitionId = 'other-definition'; },
    snapshot => { snapshot.creation.templateSpan = { start: 0, end: 1 }; },
    snapshot => { snapshot.source = 'input'; },
  ]) {
    const rows = attestedFixture({ literal: true });
    mutate(rows.child.source_ref.inline_instruction_site.site.runtime_capture_snapshots.captures[0]);
    const result = buildInlineInstructionIndex([rows.parent, rows.child]);
    assert.equal(result.writers.length, 0);
    assert.equal(result.held[0].reason, 'runtime-capture-snapshot-origin-or-value-mismatch');
  }
  const rows = attestedFixture({ literal: true });
  rows.child.source_ref.inline_instruction_site.site.runtime_capture_snapshots.captures = [];
  assert.equal(buildInlineInstructionIndex([rows.parent, rows.child]).held[0].reason, 'runtime-capture-snapshots-incomplete-or-ambiguous');
});

test('snapshot metadata cannot omit the source span or replace visible child values', () => {
  const missing = attestedFixture({ literal: true });
  delete missing.child.source_ref.inline_instruction_site.site.source_span;
  delete missing.child.source_ref.inline_instruction_site.site.runtime_capture_snapshots.captures[0].creation.sourceSpan;
  assert.equal(buildInlineInstructionIndex([missing.parent, missing.child]).held[0].reason, 'capture-source-span-incomplete');
  const altered = attestedFixture({ literal: true });
  altered.child.messages[1].tool_calls[0].function.arguments = JSON.stringify({ code: 'const policy: string = "Not the captured value.";' });
  assert.equal(buildInlineInstructionIndex([altered.parent, altered.child]).held[0].reason, 'capture-snapshot-value-mismatch');
});
