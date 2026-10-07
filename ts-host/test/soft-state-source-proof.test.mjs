import assert from 'node:assert/strict';
import { test } from 'node:test';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { resolveSoftStateArgument, signatureHasExactArgumentPath, signatureHasExactParameter, summarizeSourceEvidence,
  validateExpectedReadCount, validateSoftStateEdge } from '../scripts/inline-curriculum/soft-state-proof.mjs';

const block = `nz1_${'a'.repeat(32)}`;
const writerCallId = 'seed-0';
const consumerCallId = 'step-1';
const writerNode = `${writerCallId}#7`;
const consumerSignature = 'Neuralese<(source: FileHandle, prior: Neuralese<string>) => Promise<Neuralese<string>>>';

function graphFixture() {
  return [
    { kind: 'block_write', call_id: writerCallId, block, node: writerNode, inputs: [] },
    { kind: 'invocation', call_id: consumerCallId, phase: 'start', signature: consumerSignature,
      inputs: [{ node: writerNode, block, port: 'arg:prior' }] },
    { kind: 'block_read', call_id: consumerCallId, block, node: `${consumerCallId}#3`,
      inputs: [{ node: writerNode, block, port: 'block' }] }
  ];
}

test('allows an authored seed child with an explicit empty source-read set', () => {
  assert.deepEqual(validateExpectedReadCount({ actualCount: 0, expectedPaths: [], childCallId: writerCallId }), {
    expected_count: 0, actual_count: 0, no_source_read_expected: true
  });
  assert.throws(() => validateExpectedReadCount({ actualCount: 1, expectedPaths: [], childCallId: writerCallId }),
    /1 source reads, expected 0/);
  assert.throws(() => validateExpectedReadCount({ actualCount: 0, expectedPaths: ['pass-01.md'], childCallId: 'step-1' }),
    /0 source reads, expected 1/);
});

test('resolves soft-state ports from matching producer and consumer declarations', () => {
  assert.equal(resolveSoftStateArgument({ producerNextArgument: 'notes', consumerExpectedArgument: 'notes' }), 'notes');
  assert.throws(() => resolveSoftStateArgument({ producerNextArgument: 'prior', consumerExpectedArgument: 'notes' }),
    /producer declares next argument prior, but consumer declares notes/);
});

test('counts FileHandle openings separately from source-read actions', () => {
  const summary = summarizeSourceEvidence([{ clean_child_reads: [
    { exact_complete_filehandle_opening: true },
    { source_reads: [{ source_path: 'folder/item.md' }], exact_complete_source_read: true },
    { source_reads: [], exact_complete_filehandle_opening: false }
  ] }]);
  assert.deepEqual(summary, { successful_source_reads: 1, complete_filehandle_openings: 1 });
});

test('requires exact FileHandle parameter types without accepting unions', () => {
  assert.equal(signatureHasExactParameter('(source: FileHandle, prior: Neuralese<string>) => Neuralese<string>',
    'source', 'FileHandle'), true);
  assert.equal(signatureHasExactParameter('(source: FileHandle | string, prior: Neuralese<string>) => Neuralese<string>',
    'source', 'FileHandle'), false);
});

test('resolves an exact typed member on a nested record argument', () => {
  const signature = '(input: { notes: Neuralese<string>, outputContract: string, decisionRule: unknown }) => Draft';
  assert.equal(signatureHasExactArgumentPath(signature, 'input.notes', 'Neuralese<string>'), true);
  assert.equal(signatureHasExactArgumentPath(signature, 'input.outputContract', 'Neuralese<string>'), false);
  assert.equal(signatureHasExactArgumentPath('(input: { notes: string }) => Draft', 'input.notes', 'Neuralese<string>'), false);
  assert.equal(signatureHasExactArgumentPath('(other: { notes: Neuralese<string> }) => Draft', 'input.notes', 'Neuralese<string>'), false);
  assert.equal(signatureHasExactArgumentPath('(input: { nested: { notes: Neuralese<string> } }) => Draft', 'input.notes', 'Neuralese<string>'), false);
});

test('accepts only the actual Neuralese writer to typed argument and block_read edge', () => {
  const receipt = validateSoftStateEdge({ graph: graphFixture(), actualValue: neuraleseRef('Neuralese<string>', block),
    expectedType: 'Neuralese<string>', writerCallId, consumerCallId, consumerArgument: 'prior' });
  assert.deepEqual(receipt, { block, writer_call_id: writerCallId, writer_node: writerNode,
    consumer_call_id: consumerCallId, consumer_signature: consumerSignature, invocation_input_port: 'arg:prior',
    block_read_node: `${consumerCallId}#3`, exact_runtime_writer_to_reader_link: true });
});

test('accepts the final interpreter’s exact notes parameter', () => {
  const graph = graphFixture().map(event => event.kind === 'invocation' ? {
    ...event,
    signature: 'Neuralese<(notes: Neuralese<string>) => Draft>',
    inputs: [{ node: writerNode, block, port: 'arg:notes' }]
  } : event.kind === 'block_read' ? {
    ...event, inputs: [{ node: writerNode, block, port: 'block' }]
  } : event);
  const receipt = validateSoftStateEdge({ graph, actualValue: neuraleseRef('Neuralese<string>', block),
    expectedType: 'Neuralese<string>', writerCallId, consumerCallId, consumerArgument: 'notes' });
  assert.equal(receipt.invocation_input_port, 'arg:notes');
});

test('validates nested argument path, exact writer, body digest, and matching read node', () => {
  const graph = [
    { kind: 'block_write', call_id: writerCallId, block, node: writerNode, text_body_sha256: 'a'.repeat(64) },
    { kind: 'invocation', call_id: consumerCallId, phase: 'start',
      signature: '(input: { notes: Neuralese<string>, outputContract: string }) => Draft',
      inputs: [{ node: writerNode, block, port: 'arg:input.notes' }, { node: writerNode, block, port: 'capture:notes' }] },
    { kind: 'block_read', call_id: consumerCallId, block, node: `${consumerCallId}#5`,
      inputs: [{ node: writerNode, block, port: 'block' }] }
  ];
  const validate = (events = graph, opts = {}) => validateSoftStateEdge({ graph: events,
    actualValue: neuraleseRef('Neuralese<string>', block), expectedType: 'Neuralese<string>', writerCallId,
    consumerCallId, consumerArgument: 'input.notes', writerNode, expectedBodySha256: 'a'.repeat(64), ...opts });
  const receipt = validate();
  assert.equal(receipt.invocation_input_port, 'arg:input.notes');
  assert.equal(receipt.capture_input_port, 'capture:notes');
  assert.equal(receipt.writer_body_sha256, 'a'.repeat(64));
  assert.equal(receipt.block_read_node, `${consumerCallId}#5`);
  assert.throws(() => validate(graph.map(event => event.kind === 'invocation' ? {
    ...event, inputs: [{ node: writerNode, block, port: 'arg:input.other' }]
  } : event)), /no exact input.notes invocation edge/);
  assert.throws(() => validate(graph.map(event => event.kind === 'invocation' ? {
    ...event, inputs: event.inputs.filter(input => input.port !== 'capture:notes')
  } : event)), /no exact capture:notes capture edge/);
  assert.throws(() => validate(graph.map(event => event.kind === 'invocation' ? {
    ...event, signature: '(input: { notes: string }) => Draft'
  } : event)), /does not declare input.notes/);
  assert.throws(() => validate(graph.map(event => event.kind === 'block_write' ? {
    ...event, text_body_sha256: 'b'.repeat(64)
  } : event)), /body digest does not match/);
  assert.throws(() => validate(graph, { writerNode: 'different#1' }), /no actual block_write/);
});

test('rejects a mismatched block, declared type, writer, reader, argument port, or absent graph node', () => {
  const actualValue = neuraleseRef('Neuralese<string>', block);
  const validate = (graph, options = {}) => validateSoftStateEdge({ graph, actualValue, expectedType: 'Neuralese<string>',
    writerCallId, consumerCallId, consumerArgument: 'prior', ...options });

  assert.throws(() => validate(graphFixture(), { actualValue: neuraleseRef('Neuralese<number>', block) }), /did not return/);
  assert.throws(() => validate(graphFixture(), { writerCallId: 'different-writer' }), /no actual block_write/);
  assert.throws(() => validate(graphFixture().filter(event => event.kind !== 'block_write')), /no actual block_write/);
  assert.throws(() => validate(graphFixture().filter(event => event.kind !== 'invocation')), /no exact prior invocation edge/);
  assert.throws(() => validate(graphFixture().filter(event => event.kind !== 'block_read')), /no block_read/);
  assert.throws(() => validate(graphFixture().map(event => event.kind === 'invocation' ? {
    ...event, inputs: [{ node: writerNode, block: 'nz1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', port: 'arg:prior' }]
  } : event)), /no exact prior invocation edge/);
  assert.throws(() => validate(graphFixture().map(event => event.kind === 'invocation' ? {
    ...event, inputs: [{ node: writerNode, block, port: 'arg:other' }]
  } : event)), /no exact prior invocation edge/);
  assert.throws(() => validate(graphFixture().map(event => event.kind === 'invocation' ? {
    ...event, signature: 'Neuralese<(source: FileHandle, prior: string) => Promise<Neuralese<string>>>'
  } : event)), /does not declare prior: Neuralese<string>/);
  assert.throws(() => validate(graphFixture().map(event => event.kind === 'invocation' ? {
    ...event, signature: 'Neuralese<(source: FileHandle, prior: Neuralese<string> | string) => Promise<Neuralese<string>>>'
  } : event)), /does not declare prior: Neuralese<string>/);
  assert.throws(() => validate(graphFixture().map(event => event.kind === 'block_read' ? {
    ...event, inputs: [{ node: 'unrelated#2', block, port: 'block' }]
  } : event)), /no block_read/);
  assert.throws(() => validate([]), /no actual block_write/);
});
