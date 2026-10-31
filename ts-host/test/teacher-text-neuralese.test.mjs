import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { pathToFileURL } from 'node:url';

const dist = process.env.NATLANG_TS_HOST_TEST_DIST;
const load = name => dist ? import(pathToFileURL(join(dist, name)).href) : import(new URL(`../dist/${name}`, import.meta.url));
const [{ definitionProject }, { collectBatch, nativeJobRunner, defaultToolSurfaceHash, expectedProvenance }] =
  await Promise.all([load('teacher/program.js'), load('teacher/collector.js')]);

const sha256 = value => createHash('sha256').update(value, 'utf8').digest('hex');
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}` : JSON.stringify(value);

test('teacher text Neuralese executes an exact producer→consumer edge and stays tensor-free', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'teacher-text-neuralese-'));
  const note = 'Inspection pattern: ORCHID-42; the latest visit had no score.';
  const calls = [
    ['eval', { code:
      'const memo: Neuralese<string> = await nl<Neuralese<string>>`Write one concise inspection note.`();\n' +
      'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = ' +
      'nl.with<string>({})`Use the supplied note. Return its exact token.`;\n' +
      'return await reader(memo);', finish: true }],
    ['return_result', { status: 'success', value: `<|neuralese|>${note}<|/neuralese|>` }],
    ['return_result', { status: 'success', value: 'ORCHID-42' }],
  ];
  const requests = [];
  const server = createServer((request, response) => {
    const chunks = [];
    request.on('data', chunk => chunks.push(chunk));
    request.on('end', () => {
      let body;
      try { body = JSON.parse(Buffer.concat(chunks).toString('utf8')); }
      catch (error) { response.writeHead(400); response.end(String(error)); return; }
      requests.push(body);
      const [name, args] = calls[requests.length - 1] ?? [];
      if (!name) { response.writeHead(500); response.end('unexpected additional request'); return; }
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ finish_reason: 'tool_calls', message: { role: 'assistant', content: '',
        tool_calls: [{ id: `fixture-${requests.length}`, type: 'function',
          function: { name, arguments: JSON.stringify(args) } }] } }],
        usage: { prompt_tokens: 32, completion_tokens: 8 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const project = definitionProject('inspectionNote', { returns: 'string',
      instructions: 'Write a reusable note for a follow-up reviewer, delegate reading it, and return that answer.' });
    const record = { version: 'natlang.program/2', id: 'teacher-text-inspection-flow', kind: 'lambda_source',
      source: 'teacher-text-integration-fixture', split: 'train', source_ids: ['teacher-text-integration-fixture'],
      source_groups: ['test:teacher-text-neuralese'], license: 'test', semantics: {
        root: project.root, files: project.files, inputs: {}, expected: 'ORCHID-42', operation: 'exact' } };
    const options = { jobs: join(directory, 'jobs'), output: join(directory, 'out.jsonl'), workers: 1,
      modelId: 'scripted-teacher-text-fixture', endpoint: `http://127.0.0.1:${server.address().port}/v1`,
      rootSeed: 20261031, systemPrompt: 'Use the declared task and typed call signatures.', contextTokens: 8192,
      toolSurfaceSha256: await defaultToolSurfaceHash(), maxTurns: 12, maxModelRequests: 12,
      modelConcurrency: 1, transportRetries: 0, requestRetries: 0, retryDelayMs: 0,
      collectionRole: 'teacher', textNeuraleseEmulation: true };
    const provenance = expectedProvenance(record, options);
    assert.deepEqual(provenance.neuralese_transport, {
      schema: 'natlang.teacher-text-neuralese-transport/1', mode: 'teacher-text/1',
      learned_vectors: false, qualification_certificate: false, training_admission: false,
      tokenizer: { kind: 'provider-native-unspecified', ids_claimed: false },
    });
    await collectBatch([{ index: 0, record }], options, nativeJobRunner(options));
    const row = JSON.parse((await readFile(options.output, 'utf8')).trim());
    assert.equal(requests.length, 3, 'root, typed producer, and typed consumer each reached the provider');
    assert.equal(row.outcome.value, 'ORCHID-42');
    assert.equal(row.outcome.accepted, true);

    const events = row.outcome.execution_graph;
    const writers = events.filter(event => event.kind === 'block_write' && event.format === 'teacher-text');
    assert.equal(writers.length, 1, 'one typed writer is recorded in the execution graph');
    const writer = writers[0];
    assert.equal(writer.producer, 'teacher-text-emulation');
    assert.equal(writer.source_kind, 'typed-text-result');
    assert.equal(writer.result_type, 'Neuralese<string>');
    assert.equal(writer.text_body, note);
    assert.equal(writer.text_body_sha256, sha256(note));
    assert.equal(writer.learned_vectors, false);
    assert.equal(writer.tokenizer_ids, false);
    assert.equal(writer.training_admission, false);
    assert.match(writer.block, /^nzt1_[a-z2-7]{52}$/);
    const producerCall = row.outcome.invocation_ledger.find(entry => entry.invocation_id === writer.call_id);
    const capturedRef = producerCall.host_result.value.$neuralese;
    assert.equal(capturedRef.id, writer.block);
    assert.equal(capturedRef.type, writer.result_type);
    assert.equal(capturedRef.teacher_text, true);
    assert.equal(capturedRef.source_occurrence, writer.node,
      'the typed host capture binds the teacher-text reference to the exact writer occurrence');

    const reads = events.filter(event => event.kind === 'block_read' && event.format === 'teacher-text');
    assert.equal(reads.length, 1, 'consumer reads the actual teacher text reference exactly once');
    const read = reads[0];
    assert.equal(read.block, writer.block);
    assert.equal(read.result_type, writer.result_type);
    assert.equal(read.text_body_sha256, writer.text_body_sha256);
    assert.equal(read.learned_vectors, false);
    assert.equal(read.tokenizer_ids, false);
    assert.ok(read.inputs.some(input => input.block === writer.block && input.node === writer.node),
      'the consumer graph edge names the exact producer occurrence');

    const transportTurns = row.trajectory.filter(turn => turn.model_response?.transport_provenance);
    assert.equal(transportTurns.length, 3);
    const producerTransport = transportTurns[1].model_response.transport_provenance;
    assert.equal(producerTransport.schema, 'natlang.teacher-text-neuralese-transport/1');
    assert.equal(producerTransport.mode, 'teacher-text/1');
    assert.equal(producerTransport.learned_vectors, false);
    assert.equal(producerTransport.qualification_certificate, false);
    assert.equal(producerTransport.training_admission, false);
    assert.match(producerTransport.raw_request_sha256, /^[0-9a-f]{64}$/);
    assert.match(producerTransport.rendered_request_sha256, /^[0-9a-f]{64}$/);
    const emitted = producerTransport.marker_outputs.find(output => output.body === note);
    assert.ok(emitted, 'the producing turn receipt captures the exact marker body');
    assert.equal(emitted.body_sha256, sha256(note));
    assert.equal(producerTransport.raw_response_sha256, transportTurns[1].raw_response_sha256);
    const writeReceipt = producerTransport.typed_result_writes?.find(write => write.body === note);
    assert.ok(writeReceipt, 'the producing turn binds the marker to its typed block and writer occurrence');
    assert.equal(writeReceipt.id, writer.block);
    assert.equal(writeReceipt.type, writer.result_type);
    assert.equal(writeReceipt.body_sha256, writer.text_body_sha256);
    assert.equal(writeReceipt.writer_node, writer.node);
    assert.equal(writeReceipt.source, 'return_result');
    assert.equal(writeReceipt.source_kind, 'typed-text-result');
    assert.equal(writeReceipt.marker_context, 'return-result');
    const consumerTransport = transportTurns[2].model_response.transport_provenance;
    assert.equal(consumerTransport.expanded_input_blocks.length, 1);
    assert.deepEqual(consumerTransport.expanded_input_blocks[0], {
      id: writer.block, type: writer.result_type, body: note, body_sha256: writer.text_body_sha256,
      producer_call_id: writer.call_id, writer_node: writer.node, format: 'teacher-text', teacher_text: true,
      learned_vectors: false, tokenizer_ids: false,
    });
    const consumerMessage = requests[2].messages.map(message => message.content ?? '').join('\n');
    assert.ok(consumerMessage.includes(
      `[[teacher-text Neuralese id=${writer.block} type=${writer.result_type} body=${JSON.stringify(note)}]]`),
    'the consumer receives the exact JSON-escaped body rather than an opaque ID or synthetic vector');
    assert.equal(JSON.stringify(row).includes('native_identity'), false,
      'teacher text transport does not manufacture a native-model identity');
    assert.equal(JSON.stringify(row).includes('token_ids'), false,
      'provider-native token IDs are not claimed');
  } finally {
    await new Promise(resolve => { server.closeAllConnections?.(); server.close(resolve); });
    await rm(directory, { recursive: true, force: true });
  }
});
