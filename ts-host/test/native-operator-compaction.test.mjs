import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { OPERATOR_FRAMES, viewSiteMessages } from '../dist/native/operator-frames.js';
import { VIEW_PROMPT } from '../dist/builtin/index.js';
import { createNatlangRuntime } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

const ID = 'nzs1_' + 'a'.repeat(52);

async function run(compaction, temperature) {
  const captures = [], calls = [], operatorCalls = [];
  const capture = async (call, options) => {
    captures.push({ tokens: { prompt: call.prompt, reply: call.reply }, text: call.text, options });
    return { id: ID, dialect: 'native-state/test', type: 'Neuralese<string>', length: 3, width: 8,
      producer: { native_source_capture: { segments: [], payload_start: 5, payload_stop: 8 } } };
  };
  const driver = Object.assign(async request => {
    if (request.invocation_id?.includes('#operator-')) {
      operatorCalls.push(request);
      return { text: 'value: 20000 lines of "line of data"', completion_tokens: 4, raw_response: {
        x_natlang_rl_capture: { prompt: [{ kind: 'tokens', token_ids: [1, 2, 3, 4] }], generated_inputs: [
          { token_id: 9, input_role: 'ordinary' }, { token_id: 10, input_role: 'ordinary' }] } } };
    }
    calls.push(request);
    return { calls: [['return_result', { status: 'success', value: 7 }]], completion_tokens: 1 };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ calls: false, model: temperature === undefined ? driver : { driver, temperature },
    neuralese: { store: new MemoryNeuraleseStore(), dialect: 'native-state/test', capture, ...(compaction ? { compaction } : {}) } });
  const fn = loadVirtualNatlang({ 'run.nl': '---\nargs: {value: string}\nreturns: number\n---\nCount the lines of value.\n' }, 'run.nl');
  let error;
  try { assert.equal(await runtime.run(() => fn('line of data\n'.repeat(20000))), 7); } catch (caught) { error = caught; }
  return { captures, operatorCalls, error, request: calls.length ? JSON.stringify(calls[0].messages) : '' };
}

test('compact-params is an operator call at view\'s site plus the in-context capture of its reply', async () => {
  const { captures, operatorCalls, error } = await run('nl');
  assert.equal(operatorCalls.length, 1);
  const [call] = operatorCalls;
  assert.match(call.invocation_id, /#operator-1$/);
  assert.equal(call.max_tokens, 2048 + 384);
  // Its reply is captured from its exact ids, so it asks for them (a host outside collection gets them too).
  assert.equal(call.exact_ids, true);
  assert.deepEqual(call.tools, []);
  assert.equal(call.messages[0].content, VIEW_PROMPT);
  assert.match(call.messages[1].content, /^instructions:\nThe call that receives this argument as value \(string\) answers: Count the lines of value\.\n\nvalue:\n"line of data/);
  assert.deepEqual(captures, [{ tokens: { prompt: [1, 2, 3, 4], reply: [9, 10] }, text: 'value: 20000 lines of "line of data"',
    options: { type: 'Neuralese<string>' } }]);
  // The fake state has no archived artifact, so the model turn refuses it for provenance: it was placed in the request.
  assert.match(String(error?.message), new RegExp(`${ID} is unavailable for provenance`));
});

test('the operator call samples under the run\'s temperature, as the call\'s own turns do', async () => {
  const { operatorCalls } = await run('nl', 1);
  assert.equal(operatorCalls.length, 1);
  assert.equal(operatorCalls[0].temperature, 1);
});

test('a greedy host samples the operator at temperature 1, the law it is trained under', async () => {
  const { operatorCalls } = await run('nl');
  assert.equal(operatorCalls[0].temperature, 1);
});

test('crisp (the opt-out) keeps the text cut and never calls the operator', async () => {
  const { operatorCalls, captures, request, error } = await run('crisp');
  assert.equal(error, undefined);
  assert.equal(operatorCalls.length + captures.length, 0);
  assert.ok(!request.includes(ID));
  assert.match(request, /cut off/);
});

test('view\'s site equals the pinned write site the Python engine\'s view write renders (tests/fixtures/view-site.json)', async () => {
  const { readFileSync } = await import('node:fs');
  const fixture = JSON.parse(readFileSync(new URL('../../tests/fixtures/view-site.json', import.meta.url), 'utf8'));
  assert.deepEqual(viewSiteMessages(fixture.site.value, fixture.site.instructions), fixture.messages);
  assert.deepEqual(viewSiteMessages(fixture.site.value), fixture.faithful);
  assert.equal(fixture.messages[0].content, VIEW_PROMPT);
  assert.equal(OPERATOR_FRAMES['compact-history'].close, '\n\nContinue from the most recent steps.');
});

test('the grammar capture forces the exact reply ids around produce markers and records the source capture', async () => {
  const { grammarCapture } = await import('../dist/model/native-operator.js');
  const { MemoryNeuraleseStore: Store } = await import('../dist/native/neuralese-store.js');
  const requests = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const body = init?.body ? JSON.parse(init.body) : undefined;
    requests.push({ url: String(url), body });
    if (String(url).endsWith('/tokenize'))
      return new Response(JSON.stringify({ tokens: body.content === '<think>' ? [6] : body.content === '</think>' ? [7] : body.content === '<|im_end|>' ? [2] : [50, 51] }));
    return new Response(JSON.stringify({ error: 'stop here' }), { status: 500 });
  };
  try {
    const capture = grammarCapture({ endpoint: 'http://fork', store: new Store(), info: {
      native_state: { enabled: true, production: true, role_marker_ids: [100, 101, 102, 103] } } });
    await assert.rejects(capture({ messages: [{ role: 'user', content: 'x' }], tools: [], prompt: [1], reply: [6, 7, 12, 13, 2] }),
      /forced span failed \(500\)/);
  } finally { globalThis.fetch = realFetch; }
  const forced = requests.find(r => r.url.endsWith('/v1/chat/completions')).body;
  assert.equal(forced.grammar, 'root ::= <[6]> <[7]> <[100]> <[12]> <[13]> <[101]>');
  assert.equal(forced.temperature, 0);
  assert.equal(forced.x_natlang_rl_capture, undefined);
  assert.deepEqual(forced.messages, [{ role: 'user', content: 'x' }]);
});

test('a compact_history note carries the state of the note captured in place in the reply that wrote it', async () => {
  const captures = [];
  const capture = async (call, options) => {
    captures.push({ call, options });
    return { id: ID, dialect: 'native-state/test', type: 'Neuralese<string>', length: 3, width: 8,
      producer: { native_source_capture: { segments: [], payload_start: 5, payload_stop: 8 } } };
  };
  let turn = 0;
  const driver = Object.assign(async request => {
    turn++;
    if (turn === 1) return { calls: [['compact_history', { note: 'Found the bug in parse(); next: fix it.' }]], completion_tokens: 9,
      raw_response: { x_natlang_rl_capture: { prompt: [{ kind: 'tokens', token_ids: [1, 2] }],
        generated_inputs: [{ token_id: 30, input_role: 'ordinary' }, { token_id: 31, input_role: 'ordinary' }] } } };
    return { calls: [['return_result', { status: 'success', value: 7 }]], completion_tokens: 1 };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ calls: false, model: driver,
    neuralese: { store: new MemoryNeuraleseStore(), dialect: 'native-state/test', capture, compaction: 'nl' } });
  const fn = loadVirtualNatlang({ 'run.nl': '---\nargs: {value: string}\nreturns: number\n---\nCount the lines of value.\n' }, 'run.nl');
  try { await runtime.run(() => fn('short')); } catch { /* the fake state is not archived for provenance */ }
  assert.equal(captures.length, 1);
  const [{ call }] = captures;
  assert.equal(call.select, 'Found the bug in parse(); next: fix it.');
  assert.deepEqual([call.prompt, call.reply], [[1, 2], [30, 31]]);
  assert.ok(call.messages.length >= 1);
});


test('operator generation respects zero graph requests before dispatching long-argument compaction', async()=>{
 let generated=0,captured=0;
 const driver=Object.assign(async()=>{generated++;return {text:'summary',completion_tokens:1};},{neuralese:true});
 const runtime=createNatlangRuntime({calls:false,model:driver,limits:{maxModelRequests:0},
 neuralese:{store:new MemoryNeuraleseStore(),dialect:'native-state/test',compaction:'nl',capture:async()=>{captured++;throw Error('no capture expected');}}});
 const fn=loadVirtualNatlang({'run.nl':'---\nargs: {value: string}\nreturns: number\n---\nCount the lines of value.\n'},'run.nl');
 await assert.rejects(runtime.run(()=>fn('line of data\n'.repeat(20000))),/budget/);
 assert.equal(generated,0);assert.equal(captured,0);
});

test('operator generation uses the shared token reservation and never captures an overrun', async()=>{
 for(const overrun of [false,true]){
 const requests=[];let captured=0;
 const driver=Object.assign(async request=>{requests.push(request);return {text:'summary',completion_tokens:overrun?4:3};},{neuralese:true});
 const runtime=createNatlangRuntime({calls:false,model:driver,limits:{maxModelRequests:2,maxModelTokens:3},
 neuralese:{store:new MemoryNeuraleseStore(),dialect:'native-state/test',compaction:'nl',capture:async()=>{captured++;throw Error('fixture does not archive states');}}});
 const fn=loadVirtualNatlang({'run.nl':'---\nargs: {value: string}\nreturns: number\n---\nCount the lines of value.\n'},'run.nl');
 await assert.rejects(runtime.run(()=>fn('line of data\n'.repeat(20000))),/budget/);
 assert.equal(requests.length,1);assert.equal(requests[0].max_tokens,3);
 assert.match(requests[0].invocation_id,/#operator-1$/);assert.equal(captured,overrun?0:1);
 }
});

test('learned compaction cuts and compacts values over its threshold whatever the context budget', async () => {
  const { compactionPageChars, OPERATOR_COMPACT_CHARS } = await import('../dist/native/operator-frames.js');
  const { operatorSite } = await import('../dist/native/operator-site.js');
  assert.equal(OPERATOR_COMPACT_CHARS, 8000);
  assert.equal(compactionPageChars(131072), 57344);
  assert.equal(compactionPageChars(131072, OPERATOR_COMPACT_CHARS), 8000);
  const output = 'line of data\n'.repeat(800);
  const site = operatorSite({ operator: 'compact-value', value: output, purpose: 'count lines', tool: 'bash', context_tokens: 131072 });
  assert.ok(site.compacted && site.text.includes('<<cut off:'));
  assert.deepEqual(site.messages, viewSiteMessages(output, 'Read this bash result for: count lines'));
  assert.ok(site.state.before.endsWith(OPERATOR_FRAMES['compact-value'].open));
  const small = operatorSite({ operator: 'compact-value', value: 'line\n'.repeat(1000), purpose: 'p', context_tokens: 131072 });
  assert.equal(small.compacted, false);
});

test('an operator call imports its own soft prompt state where view\'s body stood, and its capture keeps the import', async () => {
  const { systemPromptBank, SYSTEM_PROMPT_TYPE } = await import('../dist/native/system-prompts.js');
  const { neuraleseRef } = await import('../dist/native/neuralese.js');
  const SOFT = 'nzs1_' + 'b'.repeat(52), VIEW = 'nzs1_' + 'c'.repeat(52);
  const captures = [], operatorCalls = [];
  const capture = async call => {
    captures.push({ prompt: call.prompt, promptSegments: call.promptSegments, reply: call.reply });
    return { id: ID, dialect: 'native-state/test', type: 'Neuralese<string>', length: 3, width: 8,
      producer: { native_source_capture: { segments: [], payload_start: 5, payload_stop: 8 } } };
  };
  const driver = Object.assign(async request => {
    if (request.invocation_id?.includes('#operator-')) {
      operatorCalls.push(request);
      return { text: 'summary', completion_tokens: 2, raw_response: { x_natlang_rl_capture: { prompt: [
        { kind: 'tokens', token_ids: [1, 61] }, { kind: 'native_state', id: SOFT }, { kind: 'tokens', token_ids: [60, 4] }],
        generated_inputs: [{ token_id: 9, input_role: 'ordinary' }] } } };
    }
    return { calls: [['return_result', { status: 'success', value: 7 }]], completion_tokens: 1 };
  }, { neuralese: true });
  const bank = systemPromptBank({ 'view@compact-params': neuraleseRef(SYSTEM_PROMPT_TYPE, SOFT),
    'view@compact-value': neuraleseRef(SYSTEM_PROMPT_TYPE, VIEW) });
  const runtime = createNatlangRuntime({ calls: false, model: driver, neuralese: { store: new MemoryNeuraleseStore(),
    dialect: 'native-state/test', capture, compaction: 'nl', systemPrompts: bank } });
  const fn = loadVirtualNatlang({ 'run.nl': '---\nargs: {value: string}\nreturns: number\n---\nCount the lines of value.\n' }, 'run.nl');
  await runtime.run(() => fn('line of data\n'.repeat(20000))).catch(() => undefined);
  assert.equal(operatorCalls.length, 1);
  const system = JSON.stringify(operatorCalls[0].messages[0]);
  assert.ok(system.includes(SOFT) && !system.includes(VIEW) && !system.includes(VIEW_PROMPT.slice(0, 40)),
    'compact-params imports its own state, not compact-value\'s, in place of the text');
  assert.deepEqual(captures, [{ prompt: undefined, reply: [9], promptSegments: [{ kind: 'tokens', token_ids: [1, 61] },
    { kind: 'native_state', id: SOFT }, { kind: 'tokens', token_ids: [60, 4] }] }]);
});

test('scoped prompt-bank entries soften only their own scope; prompt states from a build are checked against the piece text', async () => {
  const { scopedBank, softenText, systemPromptBank, SYSTEM_PROMPT_TYPE } = await import('../dist/native/system-prompts.js');
  const { promptStateIds } = await import('../dist/model/prompt-states.js');
  const { neuraleseRef } = await import('../dist/native/neuralese.js');
  const { hexDigest } = await import('../dist/native/hash.js');
  const A = 'nzs1_' + 'd'.repeat(52);
  const bank = systemPromptBank({ 'view@compact-value': neuraleseRef(SYSTEM_PROMPT_TYPE, A) });
  assert.equal(softenText(VIEW_PROMPT, bank), VIEW_PROMPT, 'outside its scope the text stays');
  assert.notEqual(softenText(VIEW_PROMPT, scopedBank(bank, 'compact-value')), VIEW_PROMPT);
  assert.equal(softenText(VIEW_PROMPT, scopedBank(bank, 'compact-params')), VIEW_PROMPT);
  const item = { binding: 'prompt:view@compact-value', kind: 'instructions', target: null, state: A, textSha256: hexDigest(VIEW_PROMPT) };
  assert.deepEqual(promptStateIds([item, { ...item, binding: 'lambda:x' }]), { 'view@compact-value': A });
  assert.throws(() => promptStateIds([{ ...item, textSha256: hexDigest('old text') }]), /system-prompt-stale/);
  assert.throws(() => promptStateIds([{ ...item, binding: 'prompt:nope' }]), /system-prompt-unknown-piece/);
});
