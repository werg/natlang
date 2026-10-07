import assert from 'node:assert/strict';
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
      name: `soft-state:${edge.block_id}`, type: 'Neuralese<string>', source: edge.body_source,
    });
    assert.equal(written.neuralese_conversion.soft_state_edges[0].raw_marker_value,
      `<|neuralese|>${edge.body_source}<|/neuralese|>`, 'the exact producer marker remains provenance');
    assert.equal(written.neuralese_conversion.soft_state_edges[0].qualification_certificate, false);
    assert.equal(written.neuralese_conversion.soft_state_edges[0].training_admission, false);

    const read = convertTrajectory(consumer, { softStateEdges: fixture.proof }).record;
    const scope = read.messages.find(message => message.role === 'tool' && message.tool_call_id === 'scope_0');
    const reads = scope.content.filter(part => part.type === 'read');
    assert.equal(reads.length, 1);
    assert.deepEqual(reads[0], { type: 'read', name: `soft-state:${edge.block_id}`, source: edge.body_source });
    assert.equal(read.neuralese_conversion.soft_state_edges[0].role, 'reader');
    assert.equal(read.neuralese_conversion.soft_state_edges[0].learned_vectors, false);
  }
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
