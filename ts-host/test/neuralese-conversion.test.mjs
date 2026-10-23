import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { convertTrajectory } from '../dist/compiler/neuralese-conversion.js';
import { canonical } from '../dist/adaptation/identity.js';
import { HANDOVER_NOTE_OPEN, HANDOVER_NOTE_CLOSE, TOOLS_PROMPT } from '../dist/native/prompt.js';

const digest = value => createHash('sha256').update(canonical(value)).digest('hex');
const contentText = value => typeof value === 'string' ? value : value.map(part => part.text ?? '').join('');
const assertNative = record => {
  assert.equal(record.native_source_conversion.schema, 'natlang.native-source-conversion/1');
  assert.equal(record.native_source_conversion.native_target_sha256, digest(record.target));
  assert.equal(record.native_source_conversion.source_messages_sha256,
    digest(record.native_source_conversion.source_messages));
  for (const message of record.messages) {
    const parts = Array.isArray(message.content) ? message.content : [];
    assert.ok(parts.every(part => part.type === 'text'), 'current messages use ordinary text parts');
    assert.ok(!JSON.stringify(message).match(/"type":"(?:neuralese|native_state|soft|read|view)"/));
  }
  return record.native_source_conversion;
};

test('current conversion retains exact source and emits ordinary text with native receipts', () => {
  const note = 'Use the fee line as the answer.';
  const source = { id: 'prompt-row', source_ref: { dataset: 'fixture', row: '17' },
    provenance: { actor: 'teacher-x', runtime: 'runtime-y' },
    messages: [{ role: 'system', content: TOOLS_PROMPT },
      { role: 'user', content: `${HANDOVER_NOTE_OPEN}${note}${HANDOVER_NOTE_CLOSE}` }],
    target: { role: 'assistant', content: 'The fee is on line 2.' } };
  const { record, pieces } = convertTrajectory(source);
  const ledger = assertNative(record);
  assert.deepEqual(pieces, []);
  assert.deepEqual(ledger.source_messages, source.messages);
  assert.deepEqual(ledger.source_target, source.target);
  assert.deepEqual(ledger.source_ref, source.source_ref);
  assert.deepEqual(ledger.source_provenance, source.provenance);
  assert.equal(contentText(record.messages[0].content), TOOLS_PROMPT);
  assert.match(contentText(record.messages[1].content), /Use the fee line as the answer/);
  assert.ok(Array.isArray(ledger.views));
  assert.ok(ledger.views.every(view => typeof view.source_sha256 === 'string'));
});

test('missing source bodies stay unresolved and are not silently admitted as empty content', () => {
  const id = `nzs1_${'b'.repeat(52)}`;
  const source = { id: 'unresolved-row', messages: [{ role: 'user', content: [{ type: 'neuralese', id }] }],
    target: { role: 'assistant', content: 'Answer.' } };
  const { record } = convertTrajectory(source);
  const ledger = assertNative(record);
  assert.equal(ledger.unresolved_references.length, 1);
  assert.equal(ledger.unresolved_references[0].id, id);
  assert.equal(ledger.unresolved_references[0].schema, 'natlang.unresolved-native-source-reference/1');
  assert.deepEqual(ledger.occurrences, []);
});
