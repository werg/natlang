import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { convertTrajectory } from '../dist/compiler/neuralese-conversion.js';

const fixture = JSON.parse(readFileSync(fileURLToPath(new URL('./fixtures/neuralese-v83-soft-edges-minimal.json', import.meta.url)), 'utf8'));
const canonical = value => JSON.stringify(value, (_key, item) => item && !Array.isArray(item) && typeof item === 'object'
  ? Object.fromEntries(Object.entries(item).sort(([left], [right]) => left.localeCompare(right))) : item);

test('validated V83 historical receipts become plain text plus exact native occurrences', () => {
  assert.equal(fixture.actions.length, 10);
  assert.equal(fixture.proof.edges.length, 5);
  for (const edge of fixture.proof.edges) {
    const producer = fixture.actions.find(item => item.role === 'producer' && item.record.id === edge.writer_record_id).record;
    const consumer = fixture.actions.find(item => item.role === 'consumer' && item.record.id === edge.reader_record_id).record;
    const written = convertTrajectory(producer, { softStateEdges: fixture.proof }).record;
    const producerOccurrences = written.native_source_conversion.occurrences.filter(item => item.role === 'producer');
    assert.equal(producerOccurrences.length, 1);
    assert.equal(producerOccurrences[0].body, edge.body_source);
    assert.equal(producerOccurrences[0].reference, `soft-state:${edge.block_id}@${createHash('sha256').update(JSON.stringify([
      fixture.proof.source.trajectory_id, edge.writer_call_id, edge.writer_node])).digest('hex').slice(0, 12)}`);
    assert.deepEqual(written.native_source_conversion.source_target, producer.target);
    assert.equal(written.native_source_conversion.source_target_sha256,
      createHash('sha256').update(canonical(producer.target)).digest('hex'));
    assert.equal(written.native_source_conversion.native_target_sha256,
      createHash('sha256').update(canonical(written.target)).digest('hex'));
    assert.equal(written.neuralese_conversion.soft_state_edges[0].raw_marker_value,
      `<|neuralese|>${edge.body_source}<|/neuralese|>`, 'the exact producer marker remains provenance');
    assert.equal(written.neuralese_conversion.soft_state_edges[0].qualification_certificate, false);
    assert.equal(written.neuralese_conversion.soft_state_edges[0].training_admission, false);

    const read = convertTrajectory(consumer, { softStateEdges: fixture.proof }).record;
    const scope = read.messages.find(message => message.role === 'tool' && message.tool_call_id === 'scope_0');
    const consumers = read.native_source_conversion.occurrences.filter(item => item.role === 'consumer');
    assert.equal(consumers.length, 1);
    assert.equal(consumers[0].reference, producerOccurrences[0].reference);
    assert.equal(consumers[0].body, edge.body_source);
    assert.equal(scope.content.some(part => part.type === 'read'), false);
    assert.doesNotMatch(JSON.stringify({ messages: read.messages, target: read.target }),
      /\"type\":\"(?:soft|read|neuralese|native_state|view)\"|\"\$write\"/);
    assert.equal(read.neuralese_conversion.soft_state_edges[0].role, 'reader');
    assert.equal(read.neuralese_conversion.soft_state_edges[0].learned_vectors, false);
  }
});

test('validated writer and reader endpoints share an event identity distinct from another event', () => {
  const edge = fixture.proof.edges[0];
  const sameContentOtherEvent = { ...edge, writer_call_id: `${edge.writer_call_id}:later`,
    writer_node: `${edge.writer_node}:later` };
  const firstName = convertTrajectory(
    fixture.actions.find(item => item.role === 'producer' && item.record.id === edge.writer_record_id).record,
    { softStateEdges: fixture.proof }).record.native_source_conversion.occurrences.find(item => item.role === 'producer').reference;
  const secondName = `soft-state:${edge.block_id}@${createHash('sha256').update(JSON.stringify([
    fixture.proof.source.trajectory_id, sameContentOtherEvent.writer_call_id,
    sameContentOtherEvent.writer_node])).digest('hex').slice(0, 12)}`;
  assert.notEqual(firstName, secondName);
  assert.equal(firstName, convertTrajectory(
    fixture.actions.find(item => item.role === 'producer' && item.record.id === edge.writer_record_id).record,
    { softStateEdges: fixture.proof }).record.native_source_conversion.occurrences.find(item => item.role === 'producer').reference);
  const reader = fixture.actions.find(item => item.role === 'consumer' && item.record.id === edge.reader_record_id).record;
  const convertedReader = convertTrajectory(reader, { softStateEdges: fixture.proof }).record;
  const scope = convertedReader.messages.find(message => message.role === 'tool' && message.tool_call_id === 'scope_0');
  assert.equal(convertedReader.native_source_conversion.occurrences.filter(item => item.role === 'consumer').length, 1);
  assert.equal(scope.content.some(part => part.type === 'read'), false);
});

test('validated edge conversion rejects mismatched block identities and body digests', () => {
  const edge = fixture.proof.edges[0];
  const consumer = fixture.actions.find(item => item.role === 'consumer' && item.record.id === edge.reader_record_id).record;
  const wrongBlock = structuredClone(fixture.proof);
  wrongBlock.edges[0].block_id = 'nz1_aaaaaaaaaaaaaaaaaaaa';
  assert.throws(() => convertTrajectory(consumer, { softStateEdges: wrongBlock }), /lacks one exact typed/);
  const wrongBody = structuredClone(fixture.proof);
  wrongBody.edges[0].body_sha256 = '0'.repeat(64);
  assert.throws(() => convertTrajectory(consumer, { softStateEdges: wrongBody }), /proof body digest mismatch/);
});

test('assistant text writer retains exact target and lowers only an authenticated body sidecar', () => {
  const edge = fixture.proof.edges[0];
  const original = fixture.actions.find(item => item.role === 'producer' && item.record.id === edge.writer_record_id).record;
  const raw = `\n\n${edge.body_source}\n`;
  const row = fixture.proof.source.source_row_sha256;
  const request = 'c'.repeat(64), response = 'd'.repeat(64);
  const record = structuredClone(original);
  record.source_ref = { ...record.source_ref, invocation_id: edge.writer_call_id,
    source_row_sha256: row };
  record.target = { role: 'assistant', content: raw };
  record.decision = { index: edge.writer_decision_index, source_raw_response_sha256: response,
    assistant: { content: raw, typed_result_writes: [{
      schema: 'natlang.typed-result-write/1', source: 'assistant-text-return',
      source_kind: 'typed-text-result', result_type: 'Neuralese<string>', result_path: ['return'],
      invocation_id: edge.writer_call_id, writer_call_id: edge.writer_call_id, writer_node: edge.writer_node,
      block_id: edge.block_id, source_row_sha256: row, body_source: edge.body_source,
      body_sha256: edge.body_sha256, raw_assistant_text_sha256: createHash('sha256').update(raw).digest('hex'),
      body_start: Array.from(raw).length - Array.from(raw.trimStart()).length,
      body_end: Array.from(raw).length - Array.from(raw.trimStart()).length + Array.from(edge.body_source).length,
      request_sha256: request, raw_response_sha256: response,
    }] } };
  const converted = convertTrajectory(record, { softStateEdges: fixture.proof }).record;
  assert.equal(converted.target.content, raw);
  assert.equal(converted.target.tool_calls, undefined, 'assistant result must not become a synthetic tool call');
  assert.equal(converted.target.assistant_text_result, undefined, 'source-only writer receipt is not a model target field');
  const proof = converted.native_source_conversion.source_receipts[0];
  assert.equal(proof.body_source, edge.body_source);
  assert.equal(raw.slice(proof.body_start, proof.body_end), edge.body_source);
  assert.equal(proof.request_sha256, request);
  assert.throws(() => convertTrajectory({ ...record, decision: { ...record.decision,
    assistant: { ...record.decision.assistant, typed_result_writes: [] } } }, { softStateEdges: fixture.proof }),
  /lacks one exact terminal writer receipt/);
});
