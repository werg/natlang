import assert from 'node:assert/strict';
import { test } from 'node:test';
import { wrapDriver, remoteDriver } from '../dist/runtime/driver.js';
import { UsageGateway, BudgetExhausted } from '../dist/evaluation/usage.js';
import { promptLogprobDecider } from '../dist/model/chat-completion.js';

const turn = Object.assign(async () => ({ calls: [], prompt_tokens: 10, completion_tokens: 2 }), {
  contextWindow: () => Promise.resolve(4096), guidance: true,
  decide: Object.assign(async request => ({ log_probs: request.options.map(() => -1), prompt_tokens: 7 }), { scoreMany: async () => [] }) });
const request = { messages: [], options: ['"a"', '"b"'] };
const limits = overrides => ({ maxRollouts: 4, maxProposals: 3, maxModelCalls: 8, ...overrides });

test('a wrapped driver keeps every capability of the driver it wraps', async () => {
  const seen = [];
  const wrapped = wrapDriver(turn, { turn: async (...args) => { seen.push('turn'); return turn(...args); },
    decide: inner => async (item, signal) => { seen.push('decide'); return inner(item, signal); } });
  assert.equal(wrapped.guidance, true);
  assert.equal(await wrapped.contextWindow(), 4096);
  await wrapped({ messages: [] });
  assert.deepEqual((await wrapped.decide(request)).log_probs, [-1, -1]);
  assert.deepEqual(seen, ['turn', 'decide']);
  const plain = wrapDriver(turn);
  assert.equal(plain.decide, turn.decide, 'without a decide wrapper the scorer is the wrapped driver\'s own');
  const unscored = wrapDriver(Object.assign(async () => ({ calls: [] }), { guidance: true }), { decide: inner => inner });
  assert.equal(unscored.decide, undefined, 'a wrapper never gives a driver a scorer it does not have');
});

test('a worker thread scores replies exactly where the host driver does', async () => {
  const calls = [];
  const send = async call => { calls.push(call.kind); return { log_probs: [0, 0] }; };
  assert.equal(remoteDriver(send, false).decide, undefined);
  const driver = remoteDriver(send, true);
  await driver({ messages: [] });
  await driver.decide(request);
  assert.deepEqual(calls, ['turn', 'decision']);
});

test('the usage gateway meters decision readouts like turns and keeps the driver\'s capabilities', async () => {
  const gateway = new UsageGateway(limits({ maxInputTokens: 100, maxOutputTokens: 100 }));
  const driver = gateway.driver(turn, 'executor');
  assert.equal(driver.guidance, true);
  await driver({ messages: [] });
  await driver.decide(request);
  assert.equal(gateway.ledger.usage.modelCalls, 2);
  assert.equal(gateway.ledger.usage.inputTokens, 17);
  assert.equal(gateway.ledger.usage.outputTokens, 2, 'a decision generates nothing');
  assert.equal(gateway.ledger.roles.executor.modelCalls, 2);
  // A scorer that reports no prompt size leaves the input unknown, as a turn without usage does.
  const silent = gateway.driver(Object.assign(async () => ({ calls: [] }), { decide: async () => ({ log_probs: [0, 0] }) }), 'judge');
  await silent.decide(request);
  assert.equal(gateway.ledger.usage.inputTokens, null);
  assert.equal(gateway.ledger.roles.judge.modelCalls, 1);
});

test('decision readouts count against the model-call limit', async () => {
  const gateway = new UsageGateway(limits({ maxModelCalls: 1 }));
  const driver = gateway.driver(turn, 'executor');
  await driver.decide(request);
  await assert.rejects(driver.decide(request), error => error instanceof BudgetExhausted && error.dimension === 'modelCalls');
});

test('a backend that declines to score is no request: the ledger is not charged or made unknown', async () => {
  const gateway = new UsageGateway(limits({ maxInputTokens: 100 }));
  const declining = Object.assign(async () => ({ calls: [] }), { decide: async () => { throw new Error('decision-unsupported: no scores here'); } });
  await assert.rejects(gateway.driver(declining, 'executor').decide(request), /^Error: decision-unsupported/);
  assert.equal(gateway.ledger.usage.modelCalls, 0);
  assert.equal(gateway.ledger.roles.executor.modelCalls, 0);
  assert.equal(gateway.ledger.unknownRequests, 0);
  assert.equal(gateway.ledger.usage.inputTokens, 0);
  const failing = Object.assign(async () => ({ calls: [] }), { decide: async () => { throw new Error('server down'); } });
  await assert.rejects(gateway.driver(failing, 'executor').decide(request), /server down/);
  assert.equal(gateway.ledger.unknownRequests, 1, 'any other failure may have been billed');
});

test('a scorer reports the prompt tokens its readout evaluated, where the server says', async () => {
  const row = (id, logprob) => ({ [id]: { logprob, rank: 1 } });
  const transport = usage => async () => ({ prompt_logprobs: [null, row('1', -0.1), row('2', -0.2)], ...(usage ? { usage: { prompt_tokens: 40 } } : {}) });
  const request = { messages: [{ role: 'user', content: 'q' }], options: ['"a"', '"b"'] };
  assert.equal((await promptLogprobDecider(transport(true))(request)).prompt_tokens, 80, 'each option evaluates the prompt');
  assert.equal('prompt_tokens' in await promptLogprobDecider(transport(false))(request), false, 'unknown when the server does not say');
  const { postExplicitBatch } = await import('../dist/model/scoring.js');
  const { registerLocalEndpoint } = await import('../dist/model/chat-completion.js');
  const unregister = registerLocalEndpoint('http://score.local', async () =>
    new Response(JSON.stringify({ results: [{ log_probs: [-1, -2], tokens: [2, 2], prompt_tokens: 12 }] }), { status: 200 }));
  try {
    const [settled] = await postExplicitBatch({ url: 'http://score.local/v1/natlang/score' }, [request]);
    assert.equal(settled.value.prompt_tokens, 12);
  } finally { unregister(); }
});
