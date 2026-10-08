import assert from 'node:assert/strict';
import test from 'node:test';
import { extractSessionText, parseOpenCodeEnvelope, readSessionText } from '../scripts/opencode-cli-chat-adapter.mjs';

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
