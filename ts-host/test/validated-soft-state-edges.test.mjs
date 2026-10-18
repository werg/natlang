import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';
import { convertTrajectory } from '../dist/compiler/neuralese-conversion.js';

const fixture = JSON.parse(readFileSync(fileURLToPath(new URL('./fixtures/neuralese-v83-soft-edges-minimal.json', import.meta.url)), 'utf8'));

test('validated V83 text-stand-in graph edges convert into exact native writes and typed argument reads', () => {
  assert.equal(fixture.actions.length, 10);
  assert.equal(fixture.proof.edges.length, 5);
  for (const edge of fixture.proof.edges) {
    const producer = fixture.actions.find(item => item.role === 'producer' && item.record.id === edge.writer_record_id).record;
    const consumer = fixture.actions.find(item => item.role === 'consumer' && item.record.id === edge.reader_record_id).record;
    const written = convertTrajectory(producer, { softStateEdges: fixture.proof }).record;
    const targetArgs = JSON.parse(written.target.tool_calls[0].function.arguments);
    assert.equal(targetArgs.status, 'success');
    assert.deepEqual(targetArgs.value.$write, {
      name: `soft-state:${edge.block_id}@${createHash('sha256').update(JSON.stringify([
        fixture.proof.source.trajectory_id, edge.writer_call_id, edge.writer_node])).digest('hex').slice(0, 12)}`,
      type: 'Neuralese<string>', source: edge.body_source,
    });
    assert.equal(written.neuralese_conversion.soft_state_edges[0].raw_marker_value,
      `<|neuralese|>${edge.body_source}<|/neuralese|>`, 'the exact producer marker remains provenance');
    assert.equal(written.neuralese_conversion.soft_state_edges[0].qualification_certificate, false);
    assert.equal(written.neuralese_conversion.soft_state_edges[0].training_admission, false);

    const read = convertTrajectory(consumer, { softStateEdges: fixture.proof }).record;
    const scope = read.messages.find(message => message.role === 'tool' && message.tool_call_id === 'scope_0');
    const reads = scope.content.filter(part => part.type === 'read');
    assert.equal(reads.length, 1);
    assert.deepEqual(reads[0], { type: 'read', name: targetArgs.value.$write.name, source: edge.body_source });
    assert.equal(read.neuralese_conversion.soft_state_edges[0].role, 'reader');
    assert.equal(read.neuralese_conversion.soft_state_edges[0].learned_vectors, false);
  }
});

test('validated writer and reader endpoints share an event identity distinct from another event', () => {
  const edge = fixture.proof.edges[0];
  const sameContentOtherEvent = { ...edge, writer_call_id: `${edge.writer_call_id}:later`,
    writer_node: `${edge.writer_node}:later` };
  const firstName = `soft-state:${edge.block_id}@${createHash('sha256').update(JSON.stringify([
    fixture.proof.source.trajectory_id, edge.writer_call_id, edge.writer_node])).digest('hex').slice(0, 12)}`;
  const secondName = `soft-state:${edge.block_id}@${createHash('sha256').update(JSON.stringify([
    fixture.proof.source.trajectory_id, sameContentOtherEvent.writer_call_id,
    sameContentOtherEvent.writer_node])).digest('hex').slice(0, 12)}`;
  assert.notEqual(firstName, secondName);
  assert.equal(firstName, JSON.parse(convertTrajectory(
    fixture.actions.find(item => item.role === 'producer' && item.record.id === edge.writer_record_id).record,
    { softStateEdges: fixture.proof }).record.target.tool_calls[0].function.arguments).value.$write.name);
  const reader = fixture.actions.find(item => item.role === 'consumer' && item.record.id === edge.reader_record_id).record;
  const convertedReader = convertTrajectory(reader, { softStateEdges: fixture.proof }).record;
  const scope = convertedReader.messages.find(message => message.role === 'tool' && message.tool_call_id === 'scope_0');
  assert.deepEqual(scope.content.filter(part => part.type === 'read'), [
    { type: 'read', name: firstName, source: edge.body_source },
  ]);
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
  const proof = converted.target.assistant_text_result;
  assert.equal(proof.body_source, edge.body_source);
  assert.equal(raw.slice(proof.body_start, proof.body_end), edge.body_source);
  assert.equal(proof.request_sha256, request);
  assert.throws(() => convertTrajectory({ ...record, decision: { ...record.decision,
    assistant: { ...record.decision.assistant, typed_result_writes: [] } } }, { softStateEdges: fixture.proof }),
  /lacks one exact terminal writer receipt/);
});
