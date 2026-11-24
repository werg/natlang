// Structured `readout: decision` (semantic-FP B10, WS2 R3): finite records and unions of records, scored jointly up to
// DECISION_JOINT_OPTIONS values and field by field above that, on the same client for every engine.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, decide, loadVirtualNatlang } from '../dist/index.js';
import { DECISION_JOINT_OPTIONS, armForm, decisionDomain, domainMatch, domainValues, scoredJointly, stepCandidates } from '../dist/native/decision.js';
import { decisionFieldsPrompt, framedDecision } from '../dist/decision-frame.js';
import { batchScoreItem, promptLogprobRequest, scoresFromBatchResult } from '../dist/model/decision-scoring.js';
import { explicitBatchScoreMany } from '../dist/model/scoring.js';
import { parseType, TypeEnv } from '../dist/native/types.js';

const env = new TypeEnv({});
const domainOf = text => decisionDomain(parseType(text), env);

test('records whose fields are all finite, and unions of them, are finite; optional or open fields are not', () => {
  const triage = domainOf('{ label: "spam" | "ham"; urgent: boolean }');
  assert.equal(triage.size, 4);
  assert.deepEqual([...domainValues(triage)].map(item => item.text),
    ['{"label":"spam","urgent":true}', '{"label":"spam","urgent":false}', '{"label":"ham","urgent":true}', '{"label":"ham","urgent":false}']);
  assert.deepEqual([...domainValues(triage)][1].value, { label: 'spam', urgent: false });
  const tagged = domainOf('"none" | { kind: "refund"; amount: "small" | "large" } | { kind: "deny"; reason: "late" | "used" | "other" }');
  assert.equal(tagged.size, 6);
  assert.deepEqual(tagged.arms.map(armForm), ['"none"', '{"kind":"refund","amount":<"small" | "large">}', '{"kind":"deny","reason":<"late" | "used" | "other">}']);
  assert.equal(domainOf('{ label: "a" | "b"; note: string }'), null);
  assert.equal(domainOf('{ label?: "a" | "b" }'), null);
  assert.equal(domainOf('"a" | "b"').size, 2, 'a flat type is the same domain as before');
  const many = domainOf(Array.from({ length: 40 }, (_, i) => `"v${i}"`).join(' | '));
  assert.equal(many.size, 40); assert.ok(scoredJointly(many), 'a flat union is scored jointly at any size, as before');
  assert.ok(!scoredJointly(domainOf('{ a: 1 | 2 | 3 | 4 | 5 | 6; b: 1 | 2 | 3 | 4 | 5 | 6 }')));
  assert.deepEqual([...domainValues(domainOf('"a" | "b"'))].map(item => item.text), ['"a"', '"b"']);
  const nested = domainOf('{ who: { role: "admin" | "user" }; ok: boolean }');
  assert.deepEqual([...domainValues(nested)][0], { text: '{"who":{"role":"admin"},"ok":true}', value: { who: { role: 'admin' }, ok: true } });
  assert.deepEqual(domainMatch(tagged, { amount: 'large', kind: 'refund' }), { text: '{"kind":"refund","amount":"large"}', value: { kind: 'refund', amount: 'large' } });
  assert.equal(domainMatch(tagged, { kind: 'refund', amount: 'huge' }), undefined);
  assert.equal(domainMatch(tagged, { kind: 'refund', amount: 'large', extra: 1 }), undefined);
});

test('field-by-field candidates end where the next key name starts, so they are token prefixes of the whole reply', () => {
  const big = domainOf('{ label: "spam" | "ham" | "other"; urgency: "low" | "mid" | "high"; route: "billing" | "tech" | "sales" | "abuse" }');
  assert.equal(big.size, 36);
  assert.ok(big.size > DECISION_JOINT_OPTIONS);
  const first = stepCandidates(big, big.arms.map((_, arm) => ({ arm, chosen: [] })));
  assert.deepEqual(first.map(item => item.text), ['{"label":"spam","', '{"label":"ham","', '{"label":"other","']);
  const second = stepCandidates(big, first[1].paths);
  assert.deepEqual(second.map(item => item.text), ['{"label":"ham","urgency":"low","', '{"label":"ham","urgency":"mid","', '{"label":"ham","urgency":"high","']);
  const last = stepCandidates(big, second[2].paths);
  assert.equal(last[0].text, '{"label":"ham","urgency":"high","route":"billing"}');
  assert.ok(last.every(item => item.complete));
  for (const item of [...first, ...second]) assert.ok(last[0].text.startsWith(item.text.slice(0, 10)));
  // A union's first step chooses between arms; a value arm is complete at once.
  const tagged = domainOf('"none" | { kind: "refund"; amount: 1 | 2 } | { kind: "deny"; reason: "late" | "used" }');
  const arms = stepCandidates(tagged, tagged.arms.map((_, arm) => ({ arm, chosen: [] })));
  assert.deepEqual(arms.map(item => [item.text, item.complete]),
    [['"none"', true], ['{"kind":"refund","', false], ['{"kind":"deny","', false]]);
  assert.equal(decisionFieldsPrompt(big.arms.map(armForm)),
    'Answer this call now, without tools. Reply with exactly one JSON value of this form and nothing else: ' +
    '{"label":<"spam" | "ham" | "other">,"urgency":<"low" | "mid" | "high">,"route":<"billing" | "tech" | "sales" | "abuse">}.');
});

const ROUTE = [
  '---', 'args: { message: string }', 'returns: "{ label: \\"spam\\" | \\"ham\\"; urgent: boolean }"', 'readout: instant', '---',
  'Classify the message and say whether it is urgent.', ''].join('\n');
const BIG = [
  '---', 'args: { message: string }',
  'types:', '  Ticket: "{ label: \\"spam\\" | \\"ham\\" | \\"other\\"; urgency: \\"low\\" | \\"mid\\" | \\"high\\"; route: \\"billing\\" | \\"tech\\" | \\"sales\\" | \\"abuse\\" }"',
  'returns: Ticket', 'readout: instant', '---', 'Triage the message.', ''].join('\n');

test('a record decision within the joint limit scores every record once and reports per-field marginals', async () => {
  const requests = [];
  const driver = Object.assign(async () => { throw new Error('the tool loop must not run'); }, {
    decide: async request => { requests.push(request); return { log_probs: [0.1, 0.4, 0.2, 0.3].map(Math.log) }; } });
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'route.nl': ROUTE }, 'route.nl');
  const decision = await runtime.run(() => decide(fn, 'WIN A PRIZE'));
  assert.deepEqual(decision.value, { label: 'spam', urgent: false });
  assert.equal(requests.length, 1); assert.equal(requests[0].open_reply, undefined);
  assert.deepEqual(requests[0].options, ['{"label":"spam","urgent":true}', '{"label":"spam","urgent":false}', '{"label":"ham","urgent":true}', '{"label":"ham","urgent":false}']);
  assert.match(requests[0].messages.at(-1).content, /exactly one of these JSON values and nothing else: \{"label":"spam","urgent":true\}, /);
  assert.equal(+decision.confidence.toFixed(6), 0.4);
  const marginal = Object.fromEntries(Object.entries(decision.fields).map(([key, items]) => [key, items.map(item => [item.value, +item.probability.toFixed(6)])]));
  assert.deepEqual(marginal, { label: [['spam', 0.5], ['ham', 0.5]], urgent: [[true, 0.3], [false, 0.7]] });
});

test('a record decision above the joint limit is scored field by field on open replies', async () => {
  const requests = [];
  const pick = { label: '"ham"', urgency: '"high"', route: '"tech"' };
  const driver = Object.assign(async () => { throw new Error('the tool loop must not run'); }, {
    decide: async request => {
      requests.push(request);
      // the option whose newest field has the wanted value scores highest
      return { log_probs: request.options.map(option => Object.values(pick).some(value => option.endsWith(`:${value},"`) || option.endsWith(`:${value}}`)) ? -0.2 : -2) };
    } });
  const traces = [];
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, trace: trace => traces.push(trace) });
  const fn = loadVirtualNatlang({ 'ticket.nl': BIG }, 'ticket.nl');
  const decision = await runtime.run(() => decide(fn, 'my invoice app crashes'));
  assert.deepEqual(decision.value, { label: 'ham', urgency: 'high', route: 'tech' });
  assert.equal(requests.length, 3);
  assert.ok(requests.every(request => request.open_reply === true));
  assert.ok(requests.every(request => request.messages.at(-1).content === requests[0].messages.at(-1).content), 'one prompt for every step');
  assert.match(requests[0].messages.at(-1).content, /of this form and nothing else: \{"label":<"spam" \| "ham" \| "other">,/);
  assert.deepEqual(requests[2].options, ['{"label":"ham","urgency":"high","route":"billing"}', '{"label":"ham","urgency":"high","route":"tech"}',
    '{"label":"ham","urgency":"high","route":"sales"}', '{"label":"ham","urgency":"high","route":"abuse"}']);
  assert.deepEqual(Object.keys(decision.fields), ['label', 'urgency', 'route']);
  const p = (n) => Math.exp(-0.2) / (Math.exp(-0.2) + (n - 1) * Math.exp(-2));
  assert.ok(Math.abs(decision.confidence - p(3) * p(3) * p(4)) < 1e-9, 'confidence is the chosen path\'s probability');
  assert.deepEqual(decision.probabilities.map(item => item.value), [{ label: 'ham', urgency: 'high', route: 'tech' }]);
  const event = traces.flatMap(trace => trace.events).find(item => item.kind === 'decision_readout' && item.phase === 'scored');
  assert.equal(event.mode, 'fields'); assert.equal(event.steps.length, 3);
});

test('without open-reply scoring a field-by-field decision generates the record; joint decisions keep scoring', async () => {
  let turns = 0;
  const driver = Object.assign(async request => { turns++; assert.deepEqual(request.tools, []);
    return { text: '{"route":"sales","label":"other","urgency":"low"}', calls: [] }; }, {
    decide: async request => { if (request.open_reply) throw new Error('decision-open-unsupported: old server'); return { log_probs: [-1, -2, -3, -4] }; } });
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' } });
  const big = loadVirtualNatlang({ 'ticket.nl': BIG }, 'ticket.nl');
  assert.deepEqual(await runtime.run(() => big('x')), { label: 'other', urgency: 'low', route: 'sales' });
  assert.equal(turns, 1);
  const small = loadVirtualNatlang({ 'route.nl': ROUTE }, 'route.nl');
  assert.deepEqual(await runtime.run(() => small('x')), { label: 'spam', urgent: true });
  assert.equal(turns, 1, 'the joint readout still scored');
});

test('every engine route carries open_reply and an engine that ignores it is refused, not trusted', async () => {
  assert.equal(batchScoreItem({ messages: [], options: ['{"a":"x","'], open_reply: true }).open_reply, true);
  assert.equal('open_reply' in batchScoreItem({ messages: [], options: ['"x"'] }), false);
  // both tokenizations of every option are summed by default; one option has nothing to split against
  assert.equal(batchScoreItem({ messages: [], options: ['"spam"', '"billing"'] }).split_tokenization, true);
  assert.equal('split_tokenization' in batchScoreItem({ messages: [], options: ['"x"'] }), false);
  // a reasoned readout carries its reasoning's sampled ids when the generation recorded them, never without the reasoning
  assert.deepEqual(batchScoreItem({ messages: [], options: ['"a"', '"b"'], reasoning: 'Hello', reasoning_ids: [5, 6, 7] }).reasoning_ids, [5, 6, 7]);
  assert.equal('reasoning_ids' in batchScoreItem({ messages: [], options: ['"a"', '"b"'], reasoning_ids: [5] }), false);
  process.env.NATLANG_DECISION_REASONING_IDS = 'rendered';
  try { assert.equal('reasoning_ids' in batchScoreItem({ messages: [], options: ['"a"', '"b"'], reasoning: 'Hello', reasoning_ids: [5] }), false); }
  finally { delete process.env.NATLANG_DECISION_REASONING_IDS; }
  process.env.NATLANG_DECISION_TOKENIZATIONS = 'canonical';
  try { assert.equal('split_tokenization' in batchScoreItem({ messages: [], options: ['"a"', '"b"'] }), false); }
  finally { delete process.env.NATLANG_DECISION_TOKENIZATIONS; }
  assert.equal(promptLogprobRequest([], '{"a":', {}, true).continue_final_message, true);
  assert.equal(promptLogprobRequest([], '"a"').continue_final_message, false);
  assert.throws(() => scoresFromBatchResult({ log_probs: [-1, -2] }, { options: ['a', 'b'], open_reply: true }), /^Error: decision-open-unsupported/);
  assert.deepEqual(scoresFromBatchResult({ log_probs: [-1, -2], open_reply: true }, { options: ['a', 'b'], open_reply: true }).log_probs, [-1, -2]);
  // the stock-model frame wraps open replies without closing them
  const framed = framedDecision({ messages: [{ role: 'user', content: decisionFieldsPrompt(['{"a":<1 | 2>,"b":<true | false>}']) }],
    options: ['{"a":1,"', '{"a":2,"'], open_reply: true }, 'result-object');
  assert.deepEqual(framed.options, ['{"result": {"a":1,"', '{"result": {"a":2,"']);
  assert.match(framed.messages[0].content, /^Answer this call now, without tools\. Reply with exactly one JSON object \{"result": VALUE\} and nothing else, where VALUE is a JSON value of this form and nothing else: \{"a":<1 \| 2>/);
});

test('a discovered score endpoint falls back on any client error; a natlang server is asked first', async () => {
  const calls = [];
  const fallback = async items => { calls.push('fallback'); return items.map(() => ({ status: 'fulfilled', value: { log_probs: [-1, -2] } })); };
  const original = globalThis.fetch;
  try {
    globalThis.fetch = async () => { calls.push('score'); return new Response('{"error":"unknown route"}', { status: 400 }); };
    const many = explicitBatchScoreMany({ url: 'http://x/v1/natlang/score', minItems: 1, discovered: true }, fallback);
    const [first] = await many([{ messages: [], options: ['"a"', '"b"'] }]);
    assert.equal(first.status, 'fulfilled');
    await many([{ messages: [], options: ['"a"', '"b"'] }]);
    assert.deepEqual(calls, ['score', 'fallback', 'fallback'], 'asked once, then remembered as missing');
  } finally { globalThis.fetch = original; }
});

test('the loader accepts readout: decision on a finite record and names records in its refusal', () => {
  assert.ok(loadVirtualNatlang({ 'route.nl': ROUTE }, 'route.nl'));
  assert.throws(() => loadVirtualNatlang({ 'x.nl': '---\nargs: {}\nreturns: "{ note: string }"\nreadout: decision\n---\nx\n' }, 'x.nl'),
    /a record whose fields are all finite/);
});
