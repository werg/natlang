import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { buildOpenCodeStructuredPrompt } from '../scripts/opencode-structured-turn.mjs';
import { buildAuditedCompletion, extractSessionText, parseOpenCodeEnvelope, readSessionText,
  rejectOpenCodePermission, auditCliEvents } from '../scripts/opencode-cli-chat-adapter.mjs';

test('session messages recover the final assistant text when CLI JSON events omit it', () => {
  const text = extractSessionText([
    { info: { role: 'assistant' }, parts: [{ type: 'text', text: 'intermediate' }] },
    { info: { role: 'assistant' }, parts: [{ type: 'reasoning', text: 'hidden' }, { type: 'text', text: 'final' }] }
  ]);
  assert.equal(text, 'final');
});

test('session text fallback uses the official SDK path parameter shape', async () => {
  let call;
  const text = await readSessionText({ session: { messages: async options => {
    call = options;
    return { data: [{ info: { role: 'assistant' }, parts: [{ type: 'text', text: 'recovered' }] }] };
  } } }, 'ses_example', '/isolated/scratch');
  assert.equal(text, 'recovered');
  assert.deepEqual(call, { path: { id: 'ses_example' }, query: { directory: '/isolated/scratch' } });
});

test('native permission rejection uses the installed official SDK route and confirms rejection', async () => {
  let options;
  const result = await rejectOpenCodePermission({ postSessionIdPermissionsPermissionId: async value => {
    options = value; return { data: true };
  } }, { v2: false, sessionID: 'ses_native', requestID: 'per_native' }, '/isolated/scratch');
  assert.deepEqual(options, { path: { id: 'ses_native', permissionID: 'per_native' },
    query: { directory: '/isolated/scratch' }, body: { response: 'reject' } });
  assert.deepEqual(result, { ok: true, status: 'rejected' });
});

test('native permission rejection fails closed when the SDK route or confirmation is unavailable', async () => {
  await assert.rejects(() => rejectOpenCodePermission({}, { v2: false, sessionID: 'ses_native', requestID: 'per_native' }, '/isolated'),
    /permission reply API is unavailable/);
  await assert.rejects(() => rejectOpenCodePermission({ postSessionIdPermissionsPermissionId: async () => ({ data: false }) },
    { v2: false, sessionID: 'ses_native', requestID: 'per_native' }, '/isolated'), /not confirmed/);
});

test('text envelope parser validates calls against declared tools', () => {
  const recorded = [{ name: 'probe_tool', arguments: { status: 'ready' } }];
  const parsed = parseOpenCodeEnvelope(JSON.stringify({
    content: 'Call recorded.', toolCalls: [{ name: 'probe_tool', arguments: { status: 'ready' } }]
  }), ['probe_tool']);
  assert.equal(parsed.content, 'Call recorded.');
  assert.throws(() => parseOpenCodeEnvelope(JSON.stringify({
    content: 'Call recorded.', toolCalls: [{ name: 'probe_tool', arguments: { status: 'forged' } }]
  }), ['other_tool']), /declared Natlang tool/);
  assert.equal(recorded.length, 1);
});

test('ordered duplicate actions are preserved for MCP and text envelope routes', () => {
  const repeated = [{ name: 'read_file', arguments: { path: 'source.json' } },
    { name: 'read_file', arguments: { path: 'source.json' } }];
  const envelope = parseOpenCodeEnvelope(JSON.stringify({ content: '', toolCalls: repeated }), ['read_file']);
  assert.deepEqual(envelope.calls, repeated);
  const completion = buildAuditedCompletion({ responseText: '', names: ['read_file'], recordedActions: repeated });
  assert.deepEqual(completion.calls.map(call => call.function.name), ['read_file', 'read_file']);
  assert.deepEqual(completion.calls.map(call => JSON.parse(call.function.arguments)), repeated.map(call => call.arguments));
  assert.notEqual(completion.calls[0].id, completion.calls[1].id);
});

test('validated audited MCP actions are authoritative with empty or plain text final output', () => {
  const recorded = [{ name: 'probe_tool', arguments: { status: 'ready' } }];
  for (const responseText of ['', 'I completed the action and here is extra prose.']) {
    const result = buildAuditedCompletion({ responseText, names: ['probe_tool'], recordedActions: recorded });
    assert.equal(result.content, null);
    assert.equal(result.calls.length, 1);
    assert.equal(result.calls[0].function.name, 'probe_tool');
    assert.deepEqual(JSON.parse(result.calls[0].function.arguments), { status: 'ready' });
    assert.match(result.finalTextStatus, /empty_final_text_ignored|invalid_json_final_text_ignored/);
  }
});

test('audited actions ignore inconsistent JSON echoes and retain a diagnostic status', () => {
  const result = buildAuditedCompletion({
    responseText: JSON.stringify({ content: 'not authoritative', toolCalls: [] }),
    names: ['probe_tool'], recordedActions: [{ name: 'probe_tool', arguments: { status: 'ready' } }]
  });
  assert.equal(result.content, null);
  assert.equal(result.finalTextStatus, 'inconsistent_echo_ignored');
});

test('invalid or undeclared MCP action records remain failures', () => {
  assert.throws(() => buildAuditedCompletion({ responseText: '', names: ['probe_tool'],
    recordedActions: [{ name: 'other_tool', arguments: {} }] }), /did not match a declared/);
  assert.throws(() => buildAuditedCompletion({ responseText: '', names: ['probe_tool'],
    recordedActions: [{ name: 'probe_tool', arguments: [] }] }), /did not match a declared/);
});

test('non-bridge tool use and failed CLI exit remain failures even when MCP action was recorded', () => {
  const recordedActions = [{ name: 'probe_tool', arguments: { status: 'ready' } }];
  const base = { responseText: 'extra prose', names: ['probe_tool'], recordedActions };
  assert.throws(() => buildAuditedCompletion({ ...base, nonBridgeToolUses: 1 }), /non-bridge/);
  assert.throws(() => buildAuditedCompletion({ ...base, cliExitCode: 1 }), /exited 1/);
});

function unavailableNatlangToolEvent(tool = 'eval') {
  const error = `Model tried to call unavailable tool '${tool}'. Available tools: bash, invalid.`;
  return { type: 'tool_use', sessionID: 'ses_case', part: { type: 'tool', tool: 'invalid', callID: 'call_invalid',
    state: { status: 'completed', title: 'Invalid Tool', input: { tool, error },
      output: `The arguments provided to the tool are invalid: ${error}`, metadata: { truncated: false },
      time: { start: 1, end: 2 } } } };
}

function auditedNatlangActionEvent() {
  return { type: 'tool_use', sessionID: 'ses_case', part: { type: 'tool', tool: 'natlang_action_bridge_submit_action',
    callID: 'call_action', state: { status: 'completed', input: { name: 'eval', arguments: { code: 'return 7;' } },
      output: 'ACTION_RECORDED' } } };
}

test('CLI audit retains exact no-effect unavailable Natlang attempts beside audited MCP actions', () => {
  const audit = auditCliEvents([unavailableNatlangToolEvent(), auditedNatlangActionEvent()],
    new Set(['natlang_action_bridge_submit_action']), new Set(['eval']));
  assert.equal(audit.rejectedNatlangAttempts.length, 1);
  assert.equal(audit.rejectedNatlangAttempts[0].rejected_tool_name, 'eval');
  assert.deepEqual(audit.toolUses.map(use => [use.bridge, use.rejected_natlang_attempt]), [[false, true], [true, false]]);
  assert.equal(audit.toolUses.filter(use => !use.bridge && !use.rejected_natlang_attempt).length, 0);
});

test('CLI audit still refuses actual native tools and invalid records outside the exact no-op contract', () => {
  const native = { type: 'tool_use', part: { type: 'tool', tool: 'bash', state: { status: 'completed', input: { command: 'echo' } } } };
  const malformed = unavailableNatlangToolEvent();
  malformed.part.state.metadata = { sideEffect: true };
  for (const event of [native, malformed]) {
    const audit = auditCliEvents([event], new Set(['natlang_action_bridge_submit_action']), new Set(['eval']));
    assert.equal(audit.rejectedNatlangAttempts.length, 0);
    assert.equal(audit.toolUses.filter(use => !use.bridge).length, 1);
  }
  const invalidOnly = auditCliEvents([unavailableNatlangToolEvent()],
    new Set(['natlang_action_bridge_submit_action']), new Set(['eval']));
  assert.equal(invalidOnly.rejectedNatlangAttempts.length, 1);
  assert.equal(invalidOnly.toolUses.filter(use => use.bridge).length, 0);
  assert.throws(() => buildAuditedCompletion({ responseText: 'not JSON', names: ['eval'], recordedActions: [] }), /invalid JSON text/);
});

test('zero audited actions keep the strict JSON envelope protocol', () => {
  assert.throws(() => buildAuditedCompletion({ responseText: 'plain text', names: ['probe_tool'], recordedActions: [] }), /invalid JSON text/);
  const result = buildAuditedCompletion({ responseText: JSON.stringify({ content: 'answer', toolCalls: [] }),
    names: ['probe_tool'], recordedActions: [] });
  assert.equal(result.content, 'answer');
  assert.deepEqual(result.calls, []);
  assert.equal(result.finalTextStatus, 'strict_json_envelope_no_actions');
});

const exactV8Text = readFileSync(fileURLToPath(new URL('./fixtures/opencode-step5-v8-text-envelope.json', import.meta.url)), 'utf8');
const exactV12Retry2Text = readFileSync(fileURLToPath(new URL('./fixtures/opencode-step5-v12-retry2-text-envelope.json', import.meta.url)), 'utf8');

test('exact v8 declared envelope is accepted as prompt-directed text with raw/context pins', () => {
  const rawHash = createHash('sha256').update(exactV8Text).digest('hex');
  const context = '{"messages":[{"role":"user","content":"case"}]}';
  const contextHash = createHash('sha256').update(context).digest('hex');
  const result = buildAuditedCompletion({ responseText: exactV8Text, names: ['execution_plan'], recordedActions: [] });
  assert.equal(result.actionRoute, 'prompt_directed_text_envelope');
  assert.equal(result.actionFidelity, 'prompt_directed_text');
  assert.equal(result.calls.length, 1);
  assert.equal(result.calls[0].function.name, 'execution_plan');
  assert.ok(JSON.parse(result.calls[0].function.arguments).plan.startsWith('Evidence:'));
  assert.equal(rawHash, '8421a21620e44563b0a9ac62c4db71eef5dc3d19981b381285139a0eedf993d4');
  assert.match(contextHash, /^[a-f0-9]{64}$/);
});

test('exact v12-retry2 malformed response remains rejected with its raw hash pinned', () => {
  const rawHash = createHash('sha256').update(exactV12Retry2Text).digest('hex');
  assert.equal(Buffer.byteLength(exactV12Retry2Text), 1034);
  assert.equal(rawHash, '99e6c41de48f1d7361fec153c76f9c5f934531478b2a6653eaf5256a288212db');
  assert.match(exactV12Retry2Text, /ScoreFact\.\"]}}\]}/);
  assert.throws(() => buildAuditedCompletion({ responseText: exactV12Retry2Text,
    names: ['execution_plan'], recordedActions: [] }), /invalid JSON text/);
});

test('raw LF, CR, and TAB inside JSON strings are escaped losslessly before strict validation', () => {
  const strict = JSON.stringify({ content: 'line1\nline2\r\tend', toolCalls: [
    { name: 'probe_tool', arguments: { value: 1 } }
  ] });
  const responseText = strict.replace('line1\\nline2\\r\\tend', 'line1\nline2\r\tend');
  const result = buildAuditedCompletion({ responseText, names: ['probe_tool'], recordedActions: [] });
  assert.equal(result.responseNormalization, 'raw_string_controls_escaped');
  assert.equal(result.finalTextStatus, 'validated_prompt_directed_text_actions_raw_controls_escaped');
  assert.equal(result.calls.length, 1);
  assert.equal(result.content, null);
});

test('raw controls outside strings and other malformed JSON structure are never repaired', () => {
  for (const responseText of [
    '{\u0000"content":"x","toolCalls":[]}',
    '{"content":"unterminated\n,"toolCalls":[]}',
    '{"content":"x","toolCalls":[{"name":"probe_tool","arguments":{}}]'
  ]) assert.throws(() => buildAuditedCompletion({ responseText, names: ['probe_tool'], recordedActions: [] }),
    /invalid JSON text/);
});

test('the structured prompt permits validated text actions without claiming MCP fidelity', () => {
  const prompt = buildOpenCodeStructuredPrompt({ messages: [{ role: 'user', content: 'case' }],
    tools: [{ type: 'function', function: { name: 'probe_tool', parameters: { type: 'object' } } }],
    tool_choice: 'required' });
  assert.match(prompt.body.system, /audit record is preferred and authoritative/);
  assert.match(prompt.body.system, /never as MCP\/provider-native calls/);
  assert.doesNotMatch(prompt.body.system, /toolCalls.*must be an empty array/);
  assert.equal(prompt.responseSchema.properties.toolCalls.minItems, 1);
});

test('text envelopes reject malformed, unknown, and undeclared actions', () => {
  const envelope = call => JSON.stringify({ content: '', toolCalls: [call] });
  for (const text of [
    '{bad',
    JSON.stringify({ content: '', toolCalls: [{ name: 'unknown', arguments: {} }] }),
    envelope({ name: 'probe_tool', arguments: [] }),
    envelope({ name: 'probe_tool', arguments: {}, extra: true })
  ]) assert.throws(() => buildAuditedCompletion({ responseText: text, names: ['probe_tool'], recordedActions: [] }));
});

test('audited MCP calls remain authoritative over text envelope actions', () => {
  const result = buildAuditedCompletion({ responseText: JSON.stringify({ content: '', toolCalls: [
    { name: 'probe_tool', arguments: { status: 'different' } }
  ] }), names: ['probe_tool'], recordedActions: [{ name: 'probe_tool', arguments: { status: 'ready' } }] });
  assert.equal(result.actionRoute, 'audited_mcp');
  assert.equal(result.actionFidelity, 'mcp_audit');
  assert.deepEqual(JSON.parse(result.calls[0].function.arguments), { status: 'ready' });
  assert.equal(result.finalTextStatus, 'inconsistent_echo_ignored');
});
