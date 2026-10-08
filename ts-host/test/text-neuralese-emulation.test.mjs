import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { mkdir, mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';
const distRoot = process.env.NATLANG_TS_HOST_TEST_DIST;
const loadDist = specifier => distRoot
  ? import(pathToFileURL(join(distRoot, specifier)).href)
  : import(new URL(`../dist/${specifier}`, import.meta.url).href);
const [program, collector, materializer, emulationModule, neuraleseModule, runtimeModule, storeModule, nzModule,
  combinatorModule, contextModule, traceModule, graphModule] = await Promise.all([
  loadDist('teacher/program.js'), loadDist('teacher/collector.js'), loadDist('teacher/native-materializer.js'),
  loadDist('model/text-neuralese-emulation.js'), loadDist('native/neuralese.js'), loadDist('index.js'),
  loadDist('native/neuralese-store.js'), loadDist('native/nz-file.js'), loadDist('neuralese/combinators.js'),
  loadDist('runtime/context.js'), loadDist('native/trace.js'), loadDist('native/graph.js'),
]);
const { definitionProject } = program;
const { defaultToolSurfaceHash, expectedProvenance, nativeJobRunner } = collector;
const { materializeNativeRows } = materializer;
const { createTextNeuraleseEmulation, TEXT_NEURALESE_EMULATION_PROMPT,
  TEXT_NEURALESE_PROMPT_REVISION } = emulationModule;
const { neuraleseSentinel, textToParts } = neuraleseModule;
const { createNatlangRuntime } = runtimeModule;
const { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } = storeModule;
const { saveNz } = nzModule;
const { COMBINATORS, createNeuraleseLibrary, loadStandardLibrary } = combinatorModule;
const { currentFrame, runInFrame } = contextModule;
const { NativeTraceRecorder } = traceModule;
const { registerTrace, releaseTrace } = graphModule;

const sha256 = value => createHash('sha256').update(value).digest('hex');

const NOTE = 'A certified copy is waiting at desk 4.';

test('native teacher collection emulates typed Neuralese markers through a recorded text transport', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-text-neuralese-'));
  const project = definitionProject('protocol', { returns: 'string', instructions: 'Pass a soft note to a reader child.' });
  const record = { version: 'natlang.program/2', id: 'text-neuralese-roundtrip', kind: 'lambda_source',
    source: 'focused-test', split: 'test', source_ids: ['text-neuralese-roundtrip'],
    source_groups: ['text-neuralese-roundtrip'], license: 'test', semantics: {
      root: project.root, files: project.files, inputs: {}, expected: NOTE, operation: 'exact' } };
  const requests = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      const parsed = JSON.parse(body);
      requests.push(parsed);
      const activePrompt = parsed.messages.filter(message => message.role === 'user' &&
        typeof message.content === 'string' && message.content.includes('You are inside this call')).at(-1)?.content ?? '';
      let reply;
      if (requests.length === 1) reply = ['eval', { code: 'const seed: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Write the handoff note.`;\n' +
        'const prior = await seed();\n' +
        'const byTemplate = `${prior}`;\n' +
        'let byAppend = ""; byAppend += prior;\n' +
        'const byPlus = "" + prior;\n' +
        'const byString = String(prior);\n' +
        'const byJson = JSON.stringify(prior);\n' +
        'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read the supplied handoff note and repeat its content exactly.`;\n' +
        'return await reader(prior);' }];
      else if (activePrompt.includes('Write the handoff note')) reply = ['return_result', {
        status: 'success', value: NOTE }];
      else if (activePrompt.includes('Read the supplied handoff note')) reply = ['return_result', { status: 'success', value: NOTE }];
      else reply = ['return_result', { status: 'success', value: NOTE }];
      const message = { role: 'assistant', content: '', tool_calls: [{ id: `r${requests.length}`,
        type: 'function', function: { name: reply[0], arguments: JSON.stringify(reply[1]) } }] };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 20, completion_tokens: 12 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const options = { jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'), workers: 1, modelId: 'mock-teacher',
      rootSeed: 21, systemPrompt: 'Test system prompt.', contextTokens: 8192,
      toolSurfaceSha256: await defaultToolSurfaceHash(), endpoint: `http://127.0.0.1:${server.address().port}`,
      collectionRole: 'teacher', textNeuraleseEmulation: true, transportRetries: 1, retryDelayMs: 0 };
    await mkdir(options.jobs, { recursive: true });
    const item = { index: 0, record };
    const enabledIdentity = expectedProvenance(record, options);
    const disabledIdentity = expectedProvenance(record, { ...options, textNeuraleseEmulation: false });
    assert.equal(enabledIdentity.text_neuralese_transport.mode, 'text-marker-standin/2');
    assert.equal(enabledIdentity.text_neuralese_transport.prompt_revision, 'text-marker-guidance/10');
    assert.equal(Object.hasOwn(disabledIdentity, 'text_neuralese_transport'), false);
    assert.notEqual(enabledIdentity.system_prompt_sha256, disabledIdentity.system_prompt_sha256);
    const row = await nativeJobRunner(options)(item, expectedProvenance(record, options));
    assert.equal(row.outcome.accepted, true, JSON.stringify(row.outcome.rejection_reasons));
    assert.equal(row.outcome.value, NOTE);
    assert.ok(requests.length >= 7 && requests.length <= 12, `unexpected provider turns: ${requests.length}`);
    assert.equal(row.request_telemetry.starts.reduce((sum, start) => sum + start.chat_transport_starts, 0), requests.length,
      'transport telemetry follows rendered requests through the text emulation wrapper');
    const renderedTexts = requests.flatMap(request => request.messages ?? []).flatMap(message => {
      const content = message.content;
      return typeof content === 'string' ? [content] : Array.isArray(content) ? content.map(part => part.text ?? '') : [];
    }).join('\n');
    assert.match(renderedTexts, /Neuralese text block id=nz1_/);
    assert.match(renderedTexts, /return the computed plain text as the result/);
    assert.match(renderedTexts, /stage it with return_result\(text\) inside eval/);
    assert.doesNotMatch(renderedTexts, /Do not call return_result from inside eval/);
    assert.match(renderedTexts, /Native JavaScript text-coercion contexts use typed readout/);
    assert.match(renderedTexts, /For an ordinary string result, any characters that look like Neuralese marker delimiters are literal string content/);
    assert.match(renderedTexts, /a quoted marker is an ordinary string/);
    assert.match(renderedTexts, /label is a human-readable preview/);
    assert.match(renderedTexts, /complete unchanged label may be resolved as a Neuralese result only when its ID, type, and exact body digest match a typed value visible in this call/);
    assert.ok(renderedTexts.includes(`exact JSON string body=${JSON.stringify(NOTE)}`),
      `provider did not receive the exact stored literal body: ${renderedTexts}`);
    const providerToolArguments = requests.flatMap(request => request.messages ?? []).flatMap(message =>
      (message.tool_calls ?? []).map(call => call.function?.arguments).filter(value => value !== undefined));
    assert.ok(providerToolArguments.every(value => typeof value === 'string'),
      'text emulation restores provider tool arguments to their required string representation');
    assert.ok(requests.flatMap(request => request.messages ?? []).every(message =>
      message.content === undefined || typeof message.content === 'string'),
      'the text-only provider sees string message content, not unsupported multipart content');
    const provenance = row.trajectory.flatMap(turn => turn.model_response.transport_provenance ?
      [turn.model_response.transport_provenance] : []);
    assert.ok(provenance.length >= 1);
    assert.ok(provenance.every(item => item.version === 'text-marker-standin/2' &&
      item.prompt_revision === 'text-marker-guidance/10' &&
      item.vector_semantics.includes('non-learned') && item.rendered_request_sha256));
    const typedReadouts = provenance.map(item => item.text_template_readout).filter(Boolean);
    assert.equal(typedReadouts.length, 5, 'template interpolation, +=, +, String, and JSON.stringify each use declared read');
    assert.ok(typedReadouts.every(readout => readout.learned_vectors === false && readout.read_body_id.startsWith('nz1_') &&
      readout.read_source_sha256 === sha256(COMBINATORS.read.text)));
    const read = provenance.flatMap(item => item.expanded_input_blocks ?? []);
    assert.ok(read.some(block => block.body === NOTE && block.learned_vectors === false),
      'the expanded provider text is recorded alongside the raw typed-part trajectory');
    assert.ok(JSON.stringify(row.trajectory).includes('"type":"neuralese"'),
      'the host-side typed reference remains in the raw trajectory request');
    const graph = row.outcome.execution_graph;
    const write = graph.find(node => node.kind === 'block_write' && node.source_kind === 'typed-text-result');
    assert.ok(write, 'direct prose return_result creates a typed text result write');
    assert.equal(write.source, 'return_result');
    assert.equal(write.result_type, 'Neuralese<string>');
    assert.equal(write.text_body_sha256, sha256(NOTE));
    assert.ok(graph.some(node => node.kind === 'block_read' && node.block === write.block),
      'the child reader is linked to the same actual written block ID');
    const native = materializeNativeRows([row]);
    assert.equal(native.unlinked.length, 0, JSON.stringify(native.unlinked));
    const writerTurn = native.turns.find(turn => turn.source_ref?.invocation_id === write.call_id);
    assert.ok(writerTurn, 'the selected direct typed result remains a native child action');
    assert.equal(writerTurn.split, 'test');
    assert.deepEqual(writerTurn.source_groups, ['text-neuralese-roundtrip']);
    const targetArguments = writerTurn.target.tool_calls.map(call => JSON.parse(call.function.arguments));
    assert.ok(targetArguments.some(args => args.status === 'success' && args.value === NOTE),
      'the supervised target is the newly authored plain prose, not an old typed reference');
    const writeReceipt = writerTurn.decision.assistant.calls.flatMap(call => call.outcome?.typed_result_writes ?? [])
      .find(receipt => receipt.block_id === write.block && receipt.body_sha256 === write.text_body_sha256 &&
        receipt.writer_node === write.node);
    assert.ok(writeReceipt, 'native materialization binds the selected target to its exact typed body receipt');
    assert.equal(writeReceipt.body_source, NOTE);
    assert.equal(writeReceipt.body_source_basis, 'exact-raw-model-result-string');
    const readerTurn = native.turns.find(turn => turn.source_ref?.provider_expanded_read_contexts?.some(context =>
      context.origin === 'same-run-producer' && context.writer_target_selected === false &&
      context.producer_write?.node === write.node && context.block?.id === write.block &&
      context.block?.body_sha256 === write.text_body_sha256));
    assert.ok(readerTurn, `the matching child reader keeps its exact provider-visible cross-child input binding: ${JSON.stringify(native.turns.map(turn => turn.source_ref?.provider_expanded_read_contexts).filter(Boolean))}`);
    assert.equal(readerTurn.split, writerTurn.split);
    assert.deepEqual(readerTurn.source_groups, writerTurn.source_groups);
    assert.equal(readerTurn.source_ref.trajectory_id, writerTurn.source_ref.trajectory_id);
    const graphReadouts = graph.filter(node => node.kind === 'readout');
    assert.equal(graphReadouts.length, 5, `five implicit conversions each record a readout graph edge; got ${graphReadouts.length}`);
    assert.ok(graphReadouts.every(node => node.inputs.some(input => input.block === write.block)),
      'each readout graph edge points to the actual handoff block');
    assert.equal(requests[0].messages[0].content.includes('Declared Neuralese text-channel emulation'), true);
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});

test('text transport prompt explains typed text result promotion and literal markers', () => {
  assert.equal(TEXT_NEURALESE_PROMPT_REVISION, 'text-marker-guidance/10');
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /marker is transport syntax, not a JavaScript string/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /return the computed plain text as the result/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /stage it with return_result\(text\) inside eval/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /When you can compose the requested prose directly, prefer that direct typed return/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /use eval when the answer needs real computation or runtime effects/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /For a nested child call whose declared result is exactly Neuralese<string>/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /after all required reads and computation are done/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /do not use eval with finish:true solely to wrap a fixed string literal/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /This route guidance applies only to nested Neuralese<string> results/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /For other result types, follow their declared contract/);
  assert.doesNotMatch(TEXT_NEURALESE_EMULATION_PROMPT, /Do not call return_result from inside eval/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /Native JavaScript text-coercion contexts use typed readout/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /other payloads receive JavaScript's normal conversion after readout/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /does not enable field inspection of opaque values or turn console display into readout/);
  assert.doesNotMatch(TEXT_NEURALESE_EMULATION_PROMPT, /No general text conversion applies to other Neuralese<T> types/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /A quoted occurrence is ordinary string content/);
  assert.doesNotMatch(TEXT_NEURALESE_EMULATION_PROMPT, /quote the marker as a JavaScript string/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /unquoted value in an explicitly typed Neuralese position/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /\$\{name\} is stored exactly as written, never evaluated or interpolated/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /return the variable itself from eval/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /String\(notes\), `\$\{notes\}`, string concatenation, or JSON\.stringify\(notes\)/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /text read conversion handles these string positions automatically/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /pass the original typed variable unchanged to preserve its soft type/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /readText and read_code do not unpack a displayed label/);
  assert.doesNotMatch(TEXT_NEURALESE_EMULATION_PROMPT, /another helper to unwrap it/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /label is a human-readable preview/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /complete unchanged label may be resolved as a Neuralese result only when its ID, type, and exact body digest match a typed value visible in this call/);
});

test('nested Neuralese text child reads and computes in eval, then returns a native typed writer target', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-text-neuralese-eval-return-'));
  const sourceText = 'A certified copy is waiting at desk 4.';
  const expected = sourceText.toUpperCase();
  const project = definitionProject('evalreturn', { returns: 'string',
    instructions: 'Pass a soft note through a reader child and return its transformed text.' });
  const record = { version: 'natlang.program/2', id: 'text-neuralese-eval-return', kind: 'lambda_source',
    source: 'focused-test', split: 'train', source_ids: ['text-neuralese-eval-return'],
    source_groups: ['fixture:typed-text-eval-return'], license: 'test', semantics: {
      root: project.root, files: project.files, inputs: {}, expected, operation: 'exact' } };
  const requests = [];
  const rootCode = [
    'const seed: Neuralese<string> = await nl<Neuralese<string>>`Write the exact source note.`();',
    'const reader: Neuralese<(prior: Neuralese<string>) => Promise<Neuralese<string>>> =',
    '  nl.with<Neuralese<string>>({})`Read the supplied prior note, uppercase it, and return the exact transformed text.`;',
    'return await reader(seed);',
  ].join('\n');
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      const parsed = JSON.parse(body);
      requests.push(parsed);
      const activePrompt = parsed.messages.filter(message => message.role === 'user' &&
        typeof message.content === 'string' && message.content.includes('You are inside this call')).at(-1)?.content ?? '';
      let call;
      if (requests.length === 1) call = ['eval', { code: rootCode, finish: true }];
      else if (requests.length === 2) call = ['return_result', { status: 'success', value: sourceText }];
      else if (requests.length === 3)
        call = ['eval', { code: 'const inputs = read_inputs(); const source = await neuralese.read(inputs.v); return source.toUpperCase();' }];
      else if (requests.length === 4) call = ['return_result', { status: 'success', value: expected }];
      else if (requests.length === 5) call = ['return_result', { status: 'success', value: expected }];
      else call = ['return_result', { status: 'failed', reason:
        `unexpected fake-provider request ${activePrompt.slice(-700)}` }];
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: '', tool_calls: [{
        id: `eval-return-${requests.length}`, type: 'function',
        function: { name: call[0], arguments: JSON.stringify(call[1]) },
      }] } }], usage: { prompt_tokens: 20, completion_tokens: 10 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const options = { jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'), workers: 1,
      modelId: 'fixture-teacher', endpoint: `http://127.0.0.1:${server.address().port}`, rootSeed: 23,
      systemPrompt: 'Fixture system prompt.', contextTokens: 8192,
      toolSurfaceSha256: await defaultToolSurfaceHash(), maxTurns: 60,
      collectionRole: 'teacher', textNeuraleseEmulation: true };
    await mkdir(options.jobs, { recursive: true });
    const row = await nativeJobRunner(options)({ index: 0, record }, expectedProvenance(record, options));
    assert.equal(row.outcome.accepted, true, JSON.stringify({ value: row.outcome.value,
      rejection_reasons: row.outcome.rejection_reasons, calls: row.trajectory.map(turn => turn.assistant?.calls) }));
    assert.equal(row.outcome.value, expected);
    assert.equal(requests.length, 5, 'the reader performs eval work then returns separately, before the root finishes');
    const nestedRequests = requests.filter(item => JSON.stringify(item.messages).includes('You are inside this call: nl@'));
    assert.equal(nestedRequests.length, 3);
    assert.ok(nestedRequests.every(item => JSON.stringify(item.messages).includes(
      'For a nested child call whose declared result is exactly Neuralese<string>')));
    const graph = row.outcome.execution_graph;
    const sourceWrite = graph.find(event => event.kind === 'block_write' && event.source_kind === 'typed-text-result' &&
      event.text_body_sha256 === sha256(sourceText));
    const resultWrite = graph.find(event => event.kind === 'block_write' && event.source === 'return_result' &&
      event.result_type === 'Neuralese<string>' && event.text_body_sha256 === sha256(expected));
    assert.ok(sourceWrite, 'the first child creates a real typed source writer');
    assert.ok(resultWrite, 'the computed result has a real terminal return_result writer');
    const read = graph.find(event => event.kind === 'block_read' && event.block === sourceWrite.block &&
      event.call_id === resultWrite.call_id);
    assert.ok(read, 'the reader child has an exact graph edge to the source writer block');
    const readerTurn = graph.find(event => event.kind === 'model_turn' && event.call_id === resultWrite.call_id &&
      event.inputs?.some(input => input.node === read.node && input.block === sourceWrite.block));
    assert.ok(readerTurn, 'the return_result writer is linked to the model turn that read the source block');
    const native = materializeNativeRows([row]);
    assert.equal(native.unlinked.length, 0, JSON.stringify(native.unlinked));
    const readerActionTurn = native.turns.find(turn => turn.source_ref?.invocation_id === resultWrite.call_id &&
      turn.target.tool_calls.some(tool => tool.function.name === 'eval'));
    const writerTurn = native.turns.find(turn => turn.source_ref?.invocation_id === resultWrite.call_id &&
      turn.target.tool_calls.some(tool => tool.function.name === 'return_result'));
    assert.ok(readerActionTurn && writerTurn, 'sampled read/computation and terminal result remain separate native rows');
    assert.equal(writerTurn.source_ref.invocation_id, resultWrite.call_id,
      'the native target remains bound to the provider writer invocation');
    assert.ok(writerTurn.target.tool_calls.some(tool => tool.function.name === 'return_result' &&
      JSON.parse(tool.function.arguments).value === expected),
      `native target is the sampled assistant return_result call: ${JSON.stringify(writerTurn.target)}`);
    const sourceEdge = readerActionTurn.source_ref.provider_expanded_read_contexts?.find(context =>
      context.origin === 'same-run-producer' && context.block?.id === sourceWrite.block &&
      context.producer_write?.node === sourceWrite.node);
    assert.ok(sourceEdge, 'materialized writer context preserves the actual same-run source edge');
    assert.equal(readerActionTurn.split, 'train');
    assert.deepEqual(readerActionTurn.source_groups,
      ['text-neuralese-eval-return', 'fixture:typed-text-eval-return']);
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});

test('text transport refuses a marker in an ordinary string return', async () => {
  const emulation = createTextNeuraleseEmulation();
  await assert.rejects(() => emulation.port.write('plain result', { producer: {
    marker_context: 'return-result', result_type: 'string' } }), /only in a declared Neuralese<T> result or typed eval source/);
});

test('text transport accepts declared soft final text and typed argument writes only', async () => {
  const emulation = createTextNeuraleseEmulation();
  const literalMarker = '<|neuralese|>literal content<|/neuralese|>';
  for (const marker_context of ['assistant-text', 'typed-argument']) {
    const block = await emulation.port.write(literalMarker, { producer: {
      marker_context, result_type: 'Neuralese<string>' } });
    assert.equal(block.producer.result_type, 'Neuralese<string>');
    assert.equal(block.producer.text_body_sha256, sha256(literalMarker));
  }
  await assert.rejects(() => emulation.port.write('plain string', { producer: {
    marker_context: 'assistant-text', result_type: 'string' } }), /only in a declared Neuralese<T> result/);
  await assert.rejects(() => emulation.port.write('wrong tool payload', { producer: {
    marker_context: 'other-tool-argument', result_type: 'Neuralese<string>' } }), /only in a declared Neuralese<T> result/);
});

test('typed block parts in a prior tool-call argument become provider-valid JSON text', async () => {
  const emulation = createTextNeuraleseEmulation();
  const meta = await emulation.port.write(NOTE, { producer: { marker_context: 'return-result', result_type: 'Neuralese<string>' } });
  let providerRequest;
  const send = emulation.wrap(async request => { providerRequest = request; return { calls: [] }; });
  const args = JSON.stringify({ code: `const memo: Neuralese<string> = ${neuraleseSentinel(meta.id)};` });
  const response = await send({ messages: [{ role: 'assistant', tool_calls: [{ type: 'function', function: {
    name: 'eval', arguments: textToParts(args) } }] }],
    tools: [], seed: 1, max_tokens: 100 });
  const argument = providerRequest.messages[0].tool_calls[0].function.arguments;
  assert.equal(typeof argument, 'string');
  assert.ok(JSON.parse(argument).code.includes(`exact JSON string body=${JSON.stringify(NOTE)}`));
  assert.equal(typeof response.transport_provenance.rendered_request_sha256, 'string');
});

test('typed eval-source writes preserve their authenticated result type in provider expansions', async () => {
  const emulation = createTextNeuraleseEmulation();
  const block = await emulation.port.write(NOTE, { producer: {
    marker_context: 'eval-code', result_type: 'Neuralese<string>' } });
  const send = emulation.wrap(async () => ({ calls: [] }));
  const response = await send({ messages: [{ role: 'user', content: textToParts(neuraleseSentinel(block.id)) }],
    tools: [], seed: 4, max_tokens: 100 });
  assert.deepEqual(response.transport_provenance.expanded_input_blocks.map(item => [item.id, item.type, item.body]),
    [[block.id, 'Neuralese<string>', NOTE]]);
});

test('plain text mentions of a block ID are not recorded as expanded input blocks', async () => {
  const emulation = createTextNeuraleseEmulation();
  const block = await emulation.port.write(NOTE, { producer: {
    marker_context: 'return-result', result_type: 'Neuralese<string>' } });
  const send = emulation.wrap(async request => ({ text: request.messages[0].content, calls: [] }));
  const response = await send({ messages: [{ role: 'user', content: `This is only a textual mention: ${block.id}` }],
    tools: [], seed: 2, max_tokens: 100 });
  assert.deepEqual(response.transport_provenance.expanded_input_blocks, []);
});

test('concurrent provider requests keep expanded-block provenance request-local', async () => {
  const emulation = createTextNeuraleseEmulation();
  const [a, b] = await Promise.all(['first independent note', 'second independent note'].map(body =>
    emulation.port.write(body, { producer: { marker_context: 'return-result', result_type: 'Neuralese<string>' } })));
  const send = emulation.wrap(async request => {
    await new Promise(resolve => setTimeout(resolve, 5));
    return { text: request.messages[0].content, calls: [] };
  });
  const responseFor = id => send({ messages: [{ role: 'user', content: textToParts(neuraleseSentinel(id)) }],
    tools: [], seed: 3, max_tokens: 100 });
  const [responseA, responseB] = await Promise.all([responseFor(a.id), responseFor(b.id)]);
  assert.deepEqual(responseA.transport_provenance.expanded_input_blocks.map(block => block.id), [a.id]);
  assert.deepEqual(responseB.transport_provenance.expanded_input_blocks.map(block => block.id), [b.id]);
});

test('a crisp-return root may write a typed inline soft literal for a child reader', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'teacher-text-neuralese-inline-'));
  const project = definitionProject('inline', { returns: 'string', instructions: 'Use a typed note in eval and pass it to a reader.' });
  const record = { version: 'natlang.program/2', id: 'text-neuralese-inline-source', kind: 'lambda_source',
    source: 'focused-test', split: 'test', source_ids: ['text-neuralese-inline-source'],
    source_groups: ['text-neuralese-inline-source'], license: 'test', semantics: {
      root: project.root, files: project.files, inputs: {}, expected: NOTE, operation: 'exact' } };
  const requests = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8'); request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests.push(JSON.parse(body));
      const reply = requests.length === 1 ? ['eval', { code: `const prior: Neuralese<string> = <|neuralese|>${NOTE}<|/neuralese|>;\n` +
        'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read the supplied note.`;\n' +
        'return await reader(prior);' }] : ['return_result', { status: 'success', value: NOTE }];
      const message = { role: 'assistant', content: '', tool_calls: [{ id: `i${requests.length}`,
        type: 'function', function: { name: reply[0], arguments: JSON.stringify(reply[1]) } }] };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 18, completion_tokens: 9 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const options = { jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'), workers: 1, modelId: 'mock-teacher',
      rootSeed: 22, systemPrompt: 'Test system prompt.', contextTokens: 8192,
      toolSurfaceSha256: await defaultToolSurfaceHash(), endpoint: `http://127.0.0.1:${server.address().port}`,
      collectionRole: 'teacher', textNeuraleseEmulation: true, transportRetries: 1, retryDelayMs: 0 };
    await mkdir(options.jobs, { recursive: true });
    const item = { index: 0, record };
    const row = await nativeJobRunner(options)(item, expectedProvenance(record, options));
    assert.equal(row.outcome.accepted, true, JSON.stringify(row.outcome.rejection_reasons));
    const renderedTexts = requests.flatMap(request => request.messages ?? []).flatMap(message => {
      const content = message.content;
      return typeof content === 'string' ? [content] : Array.isArray(content) ? content.map(part => part.text ?? '') : [];
    }).join('\n');
    assert.ok(renderedTexts.includes(`exact JSON string body=${JSON.stringify(NOTE)}`),
      'the eval-authored, explicitly typed inline literal is stored then expanded to the child reader');
    assert.ok(row.outcome.execution_graph.some(node => node.kind === 'block_write' && node.learned_vectors === false));
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});

test('explicit text read source uses the declared read body, typed block context, and template validation', async () => {
  const store = new MemoryNeuraleseStore();
  const sourcePort = new StandInNeuralesePort(store, hashingEmbedder(4), 4, 'nd:text-read-test/1');
  const exports = {};
  const bodies = {};
  for (const [name, entry] of Object.entries(COMBINATORS)) {
    const type = `Neuralese<${entry.type}>`;
    const block = await sourcePort.write(entry.text, { type });
    bodies[name] = block.id;
    exports[name] = { type, description: entry.text, value: { kind: 'soft-function', type, body: block.id, captures: {} } };
  }
  const textReadSource = { schema: 'natlang.text-read-source/1', export: 'read', bodyId: bodies.read,
    type: `Neuralese<${COMBINATORS.read.type}>`, source: COMBINATORS.read.text,
    sourceSha256: sha256(COMBINATORS.read.text), learnedVectors: false };
  const bytes = await saveNz(exports, { store, dialect: sourcePort.dialect,
    provenance: { kind: 'text-provider-test', text_read_source: textReadSource } });
  const library = await loadStandardLibrary(bytes, store);
  assert.deepEqual(library.textReadSource, textReadSource);

  const emulation = createTextNeuraleseEmulation({ store, standardLibrary: library });
  const input = await emulation.port.write(NOTE, { type: 'Neuralese<string>', producer: {
    marker_context: 'return-result', result_type: 'Neuralese<string>' } });
  const { neuraleseRef } = await import('../dist/native/neuralese.js');
  const value = neuraleseRef('Neuralese<string>', input.id);
  const providerResponses = [];
  const sendText = emulation.wrap(async request => {
    assert.equal(Object.hasOwn(request, 'template'), false);
    assert.equal(request.tool_choice, 'required');
    assert.deepEqual(request.tools.map(tool => tool.function.name), ['return_result']);
    const shown = JSON.stringify(request.messages);
    assert.ok(shown.includes(COMBINATORS.read.text), 'provider receives the authenticated declared read source');
    assert.ok(shown.includes(NOTE), 'provider receives the exact typed input body');
    return { calls: [['return_result', { status: 'success', value: NOTE }]] };
  });
  const send = Object.assign(async request => {
    const response = await sendText(request);
    providerResponses.push(response);
    return response;
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: send, neuralese: emulation.runtime,
    services: { neuralese: library } });
  const graphTrace = new NativeTraceRecorder({ run_id: 'call:text-read-test' });
  registerTrace('call:text-read-test', graphTrace);
  const result = await runtime.run(() => runInFrame({ ...currentFrame(), parentCallId: 'call:text-read-test', adHocDepth: 5 },
    () => createNeuraleseLibrary(library).read(value)));
  releaseTrace('call:text-read-test');
  assert.equal(result, NOTE);
  assert.equal(providerResponses.length, 1);
  const provenance = providerResponses[0].transport_provenance;
  assert.deepEqual(provenance.expanded_input_blocks.map(block => [block.id, block.type, block.learned_vectors]), [
    [library.bodies.read, textReadSource.type, false], [input.id, 'Neuralese<string>', false],
  ]);
  assert.equal(provenance.text_template_readout.read_body_id, library.bodies.read);
  assert.equal(provenance.text_template_readout.read_source_sha256, textReadSource.sourceSha256);
  assert.equal(provenance.text_template_readout.qualification_certificate, false);
  const readout = graphTrace.events.find(event => event.kind === 'readout');
  assert.ok(readout, 'the ordinary library readout graph edge remains present');
  assert.ok(readout.inputs.some(input => input.block === value.$neuralese.id));
});

test('inherited read template is removed from the execution-plan-only auxiliary turn', async () => {
  const emulation = createTextNeuraleseEmulation();
  const request = { template: { call: 'return_result', arguments: { status: 'success' }, value: 'decode', value_type: 'string' },
    messages: [{ role: 'user', content: 'Before taking the next action, make a concise execution plan. Call execution_plan exactly once with the plan.' }],
    tools: [{ type: 'function', function: { name: 'execution_plan', parameters: { type: 'object' } } }],
    tool_choice: 'required', seed: 1, max_tokens: 100 };
  let received;
  const response = await emulation.wrap(async rendered => {
    received = rendered;
    return { calls: [['execution_plan', { plan: 'Read the supplied value, then return the result.' }]] };
  })(request);
  assert.equal(Object.hasOwn(received, 'template'), false);
  assert.deepEqual(received.tools, request.tools);
  assert.deepEqual(received.messages, request.messages);
  assert.deepEqual(response.calls, [['execution_plan', { plan: 'Read the supplied value, then return the result.' }]]);
  assert.equal(Object.hasOwn(response.transport_provenance, 'text_template_readout'), false);
});

test('text readout rejects vector-only libraries, mismatched body IDs, invalid source, and wrong result types', async () => {
  const base = createTextNeuraleseEmulation();
  const templateRequest = { template: { call: 'return_result', arguments: { status: 'success' }, value: 'decode', value_type: 'string' },
    messages: [], tools: [{ type: 'function', function: { name: 'return_result' } }], seed: 1, max_tokens: 100 };
  await assert.rejects(() => base.wrap(async () => ({ calls: [] }))(templateRequest), /explicitly declared text read source/);

  const store = new MemoryNeuraleseStore();
  const idA = await base.port.write('read body', { producer: { marker_context: 'eval-code' } });
  const idB = await base.port.write('other body', { producer: { marker_context: 'eval-code' } });
  const descriptor = { schema: 'natlang.text-read-source/1', export: 'read', bodyId: idA.id,
    type: 'Neuralese<(v: Neuralese<unknown>) => unknown>', source: 'Read exactly.', sourceSha256: sha256('Read exactly.'), learnedVectors: false };
  const badLibrary = { dialect: 'd', width: 4, bodies: { read: idB.id }, textReadSource: descriptor };
  const mismatched = createTextNeuraleseEmulation({ store: base.store, standardLibrary: badLibrary });
  await assert.rejects(() => mismatched.wrap(async () => ({ calls: [] }))(templateRequest), /does not match the configured library read body/);

  const wrongType = createTextNeuraleseEmulation({ store: base.store, standardLibrary: { ...badLibrary,
    bodies: { read: idA.id } } });
  const wrongTypeSend = wrongType.wrap(async () => ({ calls: [['return_result', { status: 'success', value: 42 }]] }));
  const requestWithBody = { ...templateRequest, messages: [{ role: 'user', content: textToParts(neuraleseSentinel(idA.id)) }] };
  const badSource = createTextNeuraleseEmulation({ store: base.store, standardLibrary: { ...badLibrary,
    bodies: { read: idA.id }, textReadSource: { ...descriptor, sourceSha256: '0'.repeat(64) } } });
  await assert.rejects(() => badSource.wrap(async () => ({ calls: [] }))(requestWithBody), /does not match the configured library's declared read body/);
  await assert.rejects(() => wrongTypeSend(requestWithBody), /does not match the declared string result type/);
});
