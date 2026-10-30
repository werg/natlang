import test from 'node:test';
import assert from 'node:assert/strict';
import { interpretOutput, valueSummary, fieldLabel, messageSummary } from '../dashboard/presentation.js';
import { toolDraft } from '../dashboard/execution.js';

test('complete records, lists, scalar results and line-delimited records are recognized as data', () => {
  assert.deepEqual(interpretOutput('{"summary":"Ready","count":3}'), { kind: 'value', value: { summary: 'Ready', count: 3 } });
  assert.deepEqual(interpretOutput('[1,2]'), { kind: 'value', value: [1, 2] });
  assert.deepEqual(interpretOutput('"billing"'), { kind: 'value', value: 'billing' });
  assert.deepEqual(interpretOutput('{"count":1}\n{"count":2}'), { kind: 'value', value: [{ count: 1 }, { count: 2 }] });
  assert.equal(interpretOutput('A plain-language answer.').kind, 'text');
  assert.equal(interpretOutput('[INFO] Work started').kind, 'text');
});

test('serialized drafts never become display text, and incomplete final data is reported explicitly', () => {
  const draft = '{"summary":"Receiv';
  assert.deepEqual(interpretOutput(draft, true), { kind: 'pending' });
  assert.deepEqual(interpretOutput(draft, false), { kind: 'invalid' });
  assert.deepEqual(interpretOutput('[{"name":', true), { kind: 'pending' });
  assert.deepEqual(interpretOutput(''), { kind: 'empty' });
  assert.deepEqual(interpretOutput('{"id":1}\n{"id":', true), { kind: 'sequence', value: [{ id: 1 }], pending: true, incomplete: false });
  assert.equal(interpretOutput('{"id":1}\n{"id":', false).incomplete, true);
});

test('structured log messages retain their context without exposing the serialization envelope', () => {
  const result = interpretOutput('Starting\nService reply {"available":true,"count":2}');
  assert.equal(result.kind, 'entries');
  assert.deepEqual(result.entries[1], { prefix: 'Service reply', value: { available: true, count: 2 } });
  assert.deepEqual(interpretOutput('Reply {"available":', true).entries[0], { prefix: 'Reply', pending: true });
  assert.equal(messageSummary('{"error":{"message":"Please sign in"}}'), 'Please sign in');
  assert.equal(messageSummary({ error: 'Try another input' }), 'Try another input');
});

test('value summaries explain shape and labels without becoming shortened JSON dumps', () => {
  assert.equal(valueSummary({ billing: 2, technical: 1 }), '2 fields');
  assert.equal(valueSummary(['a', 'b']), '2 items');
  assert.equal(valueSummary('{"billing":2}'), '1 field');
  assert.equal(valueSummary('{"billing":'), 'Structured details');
  assert.equal(valueSummary(true), 'Yes');
  assert.equal(valueSummary(null), 'No value');
  assert.equal(fieldLabel('by_label'), 'By label');
  assert.equal(fieldLabel('contextTokens'), 'Context Tokens');
  assert.equal(valueSummary(1.23456789).replace(/[^\d]/g, ''), '123456789');
});

test('tool call envelopes produce source code or structured values, including incomplete escaped code', () => {
  assert.deepEqual(toolDraft({ name: 'eval', arguments: '{"code":"return ' }), { kind: 'code', code: 'return ' });
  assert.deepEqual(toolDraft({ name: 'eval', arguments: '{"code":"const name = \\' }), { kind: 'code', code: 'const name = ' });
  assert.deepEqual(toolDraft({ name: 'read_code', arguments: '{"name":' }), { kind: 'pending' });
  assert.deepEqual(toolDraft({ name: 'return_result', arguments: '{"status":"success","value":{"count":2}}' }), { kind: 'value', value: { count: 2 } });
});
