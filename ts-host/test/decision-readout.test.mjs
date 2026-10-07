import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { finiteValues, softmax } from '../dist/native/decision.js';
import { parseType, TypeEnv } from '../dist/native/types.js';
import { promptLogprobDecider } from '../dist/model/chat-completion.js';

test('finite result types list their values; open types are not finite', () => {
  const env = new TypeEnv({ Verdict: parseType('"knight" | "knave"') });
  assert.deepEqual(finiteValues(parseType('Verdict | null'), env), ['knight', 'knave', null]);
  assert.deepEqual(finiteValues(parseType('boolean'), env), [true, false]);
  assert.deepEqual(finiteValues(parseType('1 | 2 | 3 | 2'), env), [1, 2, 3]);
  assert.equal(finiteValues(parseType('string'), env), null);
  assert.equal(finiteValues(parseType('"a" | number'), env), null);
  assert.deepEqual(parseType('\n  | /** a */ { op: "a" }\n  | { op: "b" }'), parseType('{ op: "a" } | { op: "b" }'), 'a leading | as in TypeScript');
  assert.deepEqual(softmax([Math.log(1), Math.log(3)]).map(p => +p.toFixed(6)), [0.25, 0.75]);
});

test('the prompt-logprob decider scores where the options differ, through the end of message', async () => {
  const bodies = [];
  const row = (id, logprob) => ({ [id]: { logprob, rank: 1 } });
  const transport = async body => {
    bodies.push(body);
    const reply = body.messages.at(-1).content;
    // Shared prompt tokens get different (noisy) scores per request: they must not count.
    const shared = [null, row('1', reply === '"yes"' ? -0.5 : -0.9), row('2', -0.1)];
    const own = reply === '"yes"' ? [row('10', -0.2), row('99', -0.1)] : [row('11', -2), row('12', -1), row('99', -0.1)];
    return { prompt_logprobs: [...shared, ...own] };
  };
  const decide = promptLogprobDecider(transport, { request: { max_tokens: 50, chat_template_kwargs: { enable_thinking: false } } });
  const scores = await decide({ messages: [{ role: 'user', content: 'q' }], options: ['"yes"', '"no"'] });
  assert.deepEqual(scores.tokens, [2, 3]);
  assert.deepEqual(scores.log_probs.map(v => +v.toFixed(6)), [-0.3, -3.1]);
  assert.equal(bodies[0].add_generation_prompt, false);
  assert.equal(bodies[0].max_tokens, 1);
  assert.deepEqual(bodies[0].chat_template_kwargs, { enable_thinking: false });
  await assert.rejects(promptLogprobDecider(async () => ({ choices: [] }))({ messages: [], options: ['1', '2'] }),
    /^Error: decision-unsupported/);
});

const VERDICT = '---\nargs: { statement: string }\nreturns: "knight" | "knave"\nreadout: decision\n---\nWho said it?\n';

test('a readout: decision function answers by scoring its values once, without the tool loop', async () => {
  const traces = [], requests = [];
  const driver = Object.assign(async () => { throw new Error('the tool loop must not run'); }, {
    decide: async request => { requests.push(request); return { log_probs: [Math.log(0.2), Math.log(0.8)], tokens: [2, 2] }; } });
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, trace: trace => traces.push(trace) });
  const fn = loadVirtualNatlang({ 'verdict.nl': VERDICT }, 'verdict.nl');
  assert.equal(await runtime.run(() => fn('I always lie.')), 'knave');
  assert.deepEqual(requests[0].options, ['"knight"', '"knave"']);
  assert.match(String(requests[0].messages.at(-1).content), /exactly one of these JSON values/);
  const event = traces.flatMap(trace => trace.events).find(item => item.kind === 'decision_readout' || item.type === 'decision_readout' || item.event === 'decision_readout');
  assert.ok(event, 'the readout is traced');
  assert.deepEqual(event.probabilities.map(p => +p.toFixed(6)), [0.2, 0.8]);
});

test('without a scorer the call falls back to the tool loop; finite-returns applies the readout to every finite call', async () => {
  const plain = async () => ({ calls: [['return_result', { status: 'success', value: 'knight' }]] });
  const fn = loadVirtualNatlang({ 'verdict.nl': VERDICT }, 'verdict.nl');
  assert.equal(await createNatlangRuntime({ model: plain, seed: { mode: 'backend' } }).run(() => fn('x')), 'knight');
  const open = loadVirtualNatlang({ 'yes.nl': '---\nargs: { q: string }\nreturns: boolean\n---\nAnswer q.\n' }, 'yes.nl');
  const driver = Object.assign(plain, { decide: async () => ({ log_probs: [-0.1, -3] }) });
  assert.equal(await createNatlangRuntime({ model: { driver, decisionReadout: 'finite-returns' }, seed: { mode: 'backend' } })
    .run(() => open('q')), true);
  assert.throws(() => loadVirtualNatlang({ 'bad.nl': '---\nargs: {}\nreturns: string\nreadout: decision\n---\nx\n' }, 'bad.nl'),
    /readout: decision needs a finite returns type/);
});

test('a readout: template function forces its first reply to return_result, writing a Neuralese result', async () => {
  const requests = [];
  const block = 'nz1_' + 'a'.repeat(52);
  const driver = Object.assign(async request => {
    requests.push(request);
    return { calls: [['return_result', { status: 'success', value: [{ type: 'neuralese', id: block }] }]] };
  }, { neuralese: true });
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'note.nl': '---\nargs: { text: string }\nreturns: Neuralese<string>\nreadout: template\n---\nNote it.\n' }, 'note.nl');
  const result = await runtime.run(() => fn('the meeting moved'));
  assert.equal(result.$neuralese.id, block);
  assert.equal(requests.length, 1);
  assert.deepEqual(requests[0].template, { call: 'return_result', arguments: { status: 'success' }, value_type: 'string', value: 'write' });
  // A plain result is decoded after the forced call opening; a backend without Neuralese gets no template.
  const plain = loadVirtualNatlang({ 'say.nl': '---\nargs: { text: string }\nreturns: string\nreadout: template\n---\nSay it.\n' }, 'say.nl');
  requests.length = 0;
  const decoded = Object.assign(async request => { requests.push(request); return { calls: [['return_result', { status: 'success', value: 'ok' }]] }; }, { neuralese: true });
  assert.equal(await createNatlangRuntime({ model: { driver: decoded }, seed: { mode: 'backend' } }).run(() => plain('x')), 'ok');
  assert.equal(requests[0].template.value, 'decode');
  requests.length = 0;
  const crisp = async request => { requests.push(request); return { calls: [['return_result', { status: 'success', value: 'ok' }]] }; };
  assert.equal(await createNatlangRuntime({ model: { driver: crisp }, seed: { mode: 'backend' } }).run(() => plain('x')), 'ok');
  assert.equal(requests[0].template, undefined);
});

test('guidance is sent to natlang servers only, when configured', async () => {
  const fn = loadVirtualNatlang({ 'say.nl': '---\nargs: { text: string }\nreturns: string\n---\nSay it.\n' }, 'say.nl');
  const requests = [];
  const reply = async request => { requests.push(request); return { calls: [['return_result', { status: 'success', value: 'ok' }]] }; };
  const ours = Object.assign(reply, { neuralese: true });
  await createNatlangRuntime({ model: { driver: ours, guidance: { repeat: 3 } }, seed: { mode: 'backend' } }).run(() => fn('x'));
  assert.deepEqual(requests[0].guidance, { repeat: 3 });
  requests.length = 0;
  await createNatlangRuntime({ model: { driver: async request => { requests.push(request); return { calls: [['return_result', { status: 'success', value: 'ok' }]] }; },
    guidance: true }, seed: { mode: 'backend' } }).run(() => fn('x'));
  assert.equal(requests[0].guidance, undefined);
});

test('decide(fn, ...args) in eval gives a function the distribution of a decision it calls', async () => {
  const { scriptedModel } = await import('./support/natlang.mjs');
  const files = {
    'gate.nl': '---\nargs: { statement: string }\nreturns: string\n---\nRefuse statement when verdict gives knave a probability of at least 0.7.\n',
    'gate/verdict.nl': VERDICT,
  };
  const model = scriptedModel(opening => opening.includes('Refuse statement') ?
    'const d = await decide(verdict, statement); const knave = d.probabilities.find(p => p.value === "knave").probability; ' +
    'return `${d.value} ${knave.toFixed(2)} ${d.scored} ${knave >= 0.7 ? "refuse" : "accept"}`;' : null);
  const driver = Object.assign(model.driver, { decide: async () => ({ log_probs: [Math.log(0.25), Math.log(0.75)], tokens: [2, 2] }) });
  const gate = loadVirtualNatlang(files, 'gate.nl');
  assert.equal(await createNatlangRuntime({ model: { driver } }).run(() => gate('I always lie.')), 'knave 0.75 true refuse');
  // A driver that cannot score: the sampled answer, with probability 1.
  const plain = scriptedModel(opening => opening.includes('Refuse statement') ?
    'const d = await decide(verdict, statement); return `${d.value} ${d.confidence} ${d.scored}`;' : 'return "knight";');
  assert.equal(await createNatlangRuntime({ model: plain.driver }).run(() => loadVirtualNatlang(files, 'gate.nl')('x')), 'knight 1 false');
});
