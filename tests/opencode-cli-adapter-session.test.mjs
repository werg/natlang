import assert from 'node:assert/strict';
import test from 'node:test';
import { buildAuditedCompletion, extractSessionText, parseOpenCodeEnvelope, readSessionText } from '../scripts/opencode-cli-chat-adapter.mjs';

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

test('an echoed tool call is accepted only when it matches the audited MCP action', () => {
  const recorded = [{ name: 'probe_tool', arguments: { status: 'ready' } }];
  const parsed = parseOpenCodeEnvelope(JSON.stringify({
    content: 'Call recorded.', toolCalls: [{ name: 'probe_tool', arguments: { status: 'ready' } }]
  }), ['probe_tool'], recorded);
  assert.equal(parsed.content, 'Call recorded.');
  assert.throws(() => parseOpenCodeEnvelope(JSON.stringify({
    content: 'Call recorded.', toolCalls: [{ name: 'probe_tool', arguments: { status: 'forged' } }]
  }), ['probe_tool'], recorded), /did not match audited MCP action records/);
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

test('zero audited actions keep the strict JSON envelope protocol', () => {
  assert.throws(() => buildAuditedCompletion({ responseText: 'plain text', names: ['probe_tool'], recordedActions: [] }), /invalid JSON text/);
  const result = buildAuditedCompletion({ responseText: JSON.stringify({ content: 'answer', toolCalls: [] }),
    names: ['probe_tool'], recordedActions: [] });
  assert.equal(result.content, 'answer');
  assert.deepEqual(result.calls, []);
  assert.equal(result.finalTextStatus, 'strict_json_envelope_no_actions');
});
