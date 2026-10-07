import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { definitionProject } from '../dist/teacher/program.js';
import { defaultToolSurfaceHash, expectedProvenance, nativeJobRunner } from '../dist/teacher/collector.js';
import { createTextNeuraleseEmulation, TEXT_NEURALESE_EMULATION_PROMPT,
  TEXT_NEURALESE_PROMPT_REVISION } from '../dist/model/text-neuralese-emulation.js';
import { neuraleseSentinel, textToParts } from '../dist/native/neuralese.js';

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
        'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read the supplied handoff note and repeat its content exactly.`;\n' +
        'return await reader(prior);' }];
      else if (activePrompt.includes('Write the handoff note')) reply = ['return_result', {
        status: 'success', value: `<|neuralese|>${NOTE}<|/neuralese|>` }];
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
    const item = { index: 0, record };
    const enabledIdentity = expectedProvenance(record, options);
    const disabledIdentity = expectedProvenance(record, { ...options, textNeuraleseEmulation: false });
    assert.equal(enabledIdentity.text_neuralese_transport.mode, 'text-marker-standin/2');
    assert.equal(enabledIdentity.text_neuralese_transport.prompt_revision, 'text-marker-guidance/3');
    assert.equal(Object.hasOwn(disabledIdentity, 'text_neuralese_transport'), false);
    assert.notEqual(enabledIdentity.system_prompt_sha256, disabledIdentity.system_prompt_sha256);
    const row = await nativeJobRunner(options)(item, expectedProvenance(record, options));
    assert.equal(row.outcome.accepted, true, JSON.stringify(row.outcome.rejection_reasons));
    assert.equal(row.outcome.value, NOTE);
    assert.ok(requests.length >= 3 && requests.length <= 6, `unexpected provider turns: ${requests.length}`);
    const renderedTexts = requests.flatMap(request => request.messages ?? []).flatMap(message => {
      const content = message.content;
      return typeof content === 'string' ? [content] : Array.isArray(content) ? content.map(part => part.text ?? '') : [];
    }).join('\n');
    assert.match(renderedTexts, /Neuralese text block id=nz1_/);
    assert.ok(renderedTexts.includes('{"status":"success","value":"<|neuralese|>your actual prose answer<|/neuralese|>"}'));
    assert.match(renderedTexts, /Do not call return_result from inside eval/);
    assert.match(renderedTexts, /eval\(\{code, finish:true\}\).*quoted string.*is wrong/);
    assert.match(renderedTexts, /already the typed input/);
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
      item.prompt_revision === 'text-marker-guidance/3' &&
      item.vector_semantics.includes('non-learned') && item.rendered_request_sha256));
    const read = provenance.flatMap(item => item.expanded_input_blocks ?? []);
    assert.ok(read.some(block => block.body === NOTE && block.learned_vectors === false),
      'the expanded provider text is recorded alongside the raw typed-part trajectory');
    assert.ok(JSON.stringify(row.trajectory).includes('"type":"neuralese"'),
      'the host-side typed reference remains in the raw trajectory request');
    const graph = row.outcome.execution_graph;
    const write = graph.find(node => node.kind === 'block_write');
    assert.equal(write.learned_vectors, false);
    assert.equal(write.emulation_version, 'text-marker-standin/2');
    assert.equal(typeof write.text_body_sha256, 'string');
    assert.ok(graph.some(node => node.kind === 'block_read' && node.block === write.block),
      'the child reader is linked to the same actual written block ID');
    assert.equal(requests[0].messages[0].content.includes('Declared Neuralese text-channel emulation'), true);
  } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
});

test('text transport prompt distinguishes direct typed return from eval finish', () => {
  assert.equal(TEXT_NEURALESE_PROMPT_REVISION, 'text-marker-guidance/3');
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /marker is transport syntax, not a JavaScript string/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /invoke the return_result tool directly/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /never put it inside a quoted string/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /unquoted value in an explicitly typed Neuralese position/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /\$\{name\} is stored exactly as written, never evaluated or interpolated/);
  assert.match(TEXT_NEURALESE_EMULATION_PROMPT, /return the variable itself from eval/);
});

test('text transport refuses a marker in an ordinary string return', async () => {
  const emulation = createTextNeuraleseEmulation();
  await assert.rejects(() => emulation.port.write('plain result', { producer: {
    marker_context: 'return-result', result_type: 'string' } }), /only in a declared Neuralese<T> result or typed eval source/);
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
