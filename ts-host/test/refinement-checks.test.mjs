import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, RefinementCallError, RefinementError, MemoryVerdictCache } from '../dist/index.js';
import { refine, assume } from '../dist/runtime/surface.js';
import { refinements as crispTable } from '../dist/native/types.js';
import { RefinementChecker, collectObligations, decisionJudge, parseRefinementSettings, verdictKey } from '../dist/native/refinement.js';
import { TypeEnv, parseType } from '../dist/native/types.js';
import { modelTurnsSoFar } from '../dist/native/agent.js';

const POLITE = 'a reply that is polite';
const REPLY = `---\nargs: { complaint: string }\nreturns: 'Is<string, "${POLITE}">'\n---\nAnswer the complaint.\n`;
const SUBJECTS = `---\nargs: { topic: string }\nreturns: 'Is<string, "${POLITE}">[]'\n---\nWrite replies.\n`;

/**
 * A model whose tool-loop replies come from `answers` (one return_result value per model turn) and whose judge scores a
 * value by `truth(value, predicate)`. The scorer is told apart from the tool loop by its true/false options.
 */
function stubModel({ answers, truth, probabilityOf }) {
  const turns = [], judged = [];
  let answer = 0;
  const driver = Object.assign(async ({ messages }) => {
    turns.push(messages.at(-1));
    const value = answers[Math.min(answer++, answers.length - 1)];
    return { calls: [['return_result', { status: 'success', value }]] };
  }, {
    decide: async request => {
      assert.deepEqual(request.options, ['true', 'false']);
      const prompt = String(request.messages.at(-1).content);
      const value = /<<<value\n([^]*?)\nvalue>>>/.exec(prompt)[1];
      judged.push(value);
      const p = probabilityOf ? probabilityOf(value) : truth(value) ? 0.97 : 0.04;
      return { log_probs: [Math.log(p), Math.log(1 - p)] };
    },
  });
  return { driver, turns, judged };
}

function run(model, files, settings = {}, call, extra = {}) {
  const traces = [];
  const runtime = createNatlangRuntime({ model: { driver: model.driver, ...extra.model }, seed: { mode: 'backend' }, calls: false,
    trace: trace => traces.push(trace), ...extra.runtime, refinements: { ...settings, ...extra.runtime?.refinements } });
  const fn = loadVirtualNatlang(files, Object.keys(files)[0]);
  return { runtime, fn, traces, result: () => runtime.run(() => call(fn)) };
}
const events = (traces, kind) => traces.flatMap(trace => trace.events).filter(event => event.kind === kind);

test('a refined return that the judge accepts is returned, with a traced check', async () => {
  const model = stubModel({ answers: ['Thank you for telling us.'], truth: () => true });
  const t = run(model, { 'reply.nl': REPLY }, {}, fn => fn('late parcel'));
  assert.equal(await t.result(), 'Thank you for telling us.');
  const [check] = events(t.traces, 'refinement_check');
  assert.equal(check.outcome, 'pass');
  assert.equal(check.predicate, POLITE);
  assert.equal(check.phase, 'return');
  assert.ok(check.probability > 0.9);
  assert.deepEqual(model.judged.length, 1, 'the net re-check after the agent hit the cache');
});

test('a failing refined return is sent back as a tool error and the executor repairs it', async () => {
  const model = stubModel({ answers: ['This is your own fault.', 'I am sorry about the delay.'], truth: value => !/fault/.test(value) });
  const t = run(model, { 'reply.nl': REPLY }, {}, fn => fn('late parcel'));
  assert.equal(await t.result(), 'I am sorry about the delay.');
  const feedback = String(model.turns.at(-1).content);
  assert.match(feedback, /^rejected\nrefinement-unsatisfied: Revise the value at return so that it is "a reply that is polite"/);
  const outcomes = events(t.traces, 'refinement_check').map(event => event.outcome);
  assert.deepEqual(outcomes.slice(0, 2), ['fail', 'pass']);
});

test('a refined return that stays unsatisfied fails the call with refinement-unsatisfied after the repair budget', async () => {
  const model = stubModel({ answers: ['your fault'], truth: () => false });
  const t = run(model, { 'reply.nl': REPLY }, { repairs: 2 }, fn => fn('late parcel'));
  await assert.rejects(t.result(), error => {
    assert.ok(error instanceof RefinementCallError, String(error));
    assert.equal(error.code, 'refinement-unsatisfied');
    assert.match(error.message, /refinement-unsatisfied: Revise the value at return so that it is "a reply that is polite"/);
    return true;
  });
  assert.equal(model.turns.length, 3, 'the first answer plus two repairs');
  assert.equal(model.judged.length, 1, 'identical rejected values are judged once');
});

test('a verdict inside the uncertainty band follows the policy', async () => {
  const band = { low: 0.3, high: 0.7 };
  const maybe = () => stubModel({ answers: ['Fine, I suppose.'], probabilityOf: () => 0.5 });
  let t = run(maybe(), { 'reply.nl': REPLY }, { band, policy: 'accept' }, fn => fn('x'));
  assert.equal(await t.result(), 'Fine, I suppose.');
  t = run(maybe(), { 'reply.nl': REPLY }, { band, policy: 'reject', repairs: 0 }, fn => fn('x'));
  await assert.rejects(t.result(), error => error.code === 'refinement-undecided' && /inside its uncertainty band/.test(error.message));
  const strong = stubModel({ answers: ['Fine, I suppose.'], probabilityOf: () => 0.5 });
  const teacher = stubModel({ answers: [], probabilityOf: () => 0.9 });
  t = run(strong, { 'reply.nl': REPLY }, { band, policy: 'escalate', escalate: 'teacher' }, fn => fn('x'),
    { runtime: { models: { teacher: { driver: teacher.driver, id: 'teacher' } } } });
  assert.equal(await t.result(), 'Fine, I suppose.');
  assert.equal(teacher.judged.length, 1);
  assert.equal(events(t.traces, 'refinement_check')[0].source, 'escalation');
});

test('the second check of the same value is a cache hit and skips the scorer', async () => {
  const model = stubModel({ answers: ['Thank you.'], truth: () => true });
  const cache = new MemoryVerdictCache();
  const t = run(model, { 'reply.nl': REPLY }, {}, async fn => [await fn('a'), await fn('b')], { runtime: { refinements: { cache } } });
  assert.deepEqual(await t.result(), ['Thank you.', 'Thank you.']);
  assert.equal(model.judged.length, 1);
  assert.equal(cache.size, 1);
  assert.ok(events(t.traces, 'refinement_check').some(event => event.source === 'cache'));
});

test('a list of refined values is checked element by element in one batch', async () => {
  const model = stubModel({ answers: [['Hello.', 'Rude!', 'Welcome.']], truth: value => value !== 'Rude!' });
  const t = run(model, { 'replies.nl': SUBJECTS }, { repairs: 0 }, fn => fn('greeting'));
  await assert.rejects(t.result(), error => error.code === 'refinement-unsatisfied' && /at return\/1 /.test(error.message));
  assert.deepEqual([...model.judged].sort(), ['Hello.', 'Rude!', 'Welcome.']);
  const paths = events(t.traces, 'refinement_check').map(event => event.path).sort();
  assert.deepEqual(paths, ['return/0', 'return/1', 'return/2']);
});

test('an argument into a refined parameter is checked before the callee starts', async () => {
  const files = { 'send.nl': `---\nargs: { body: 'Is<string, "${POLITE}">' }\nreturns: string\n---\nSend body.\n` };
  const model = stubModel({ answers: ['sent'], truth: value => !/fault/.test(value) });
  let t = run(model, files, {}, fn => fn('Thanks for waiting.'));
  assert.equal(await t.result(), 'sent');
  const rude = stubModel({ answers: ['sent'], truth: value => !/fault/.test(value) });
  t = run(rude, files, {}, fn => fn('your fault'));
  await assert.rejects(t.result(), error => error instanceof RefinementError && error.code === 'refinement-unsatisfied' && /value at body /.test(error.message));
  assert.equal(rude.turns.length, 0, 'the callee never ran');
});

test('refine() checks and returns the value; assume() records it unchecked', async () => {
  const model = stubModel({ answers: ['x'], truth: value => !/fault/.test(value) });
  const runtime = createNatlangRuntime({ model: { driver: model.driver }, calls: false });
  await runtime.run(async () => {
    assert.equal(await refine('Kind words.', POLITE), 'Kind words.');
    await assert.rejects(refine('your fault', POLITE), error => error.code === 'refinement-unsatisfied');
    await assert.rejects(refine('anything', '  '), error => error.code === 'refinement-predicate-invalid');
    assert.equal(assume('unchecked', POLITE), 'unchecked');
  });
  assert.equal(model.judged.length, 2, 'assume never scores');
});

test('crisp checkers decide when they return a boolean and defer when they return undefined; shadow records disagreements', async () => {
  const SHORT = 'one line of at most 10 characters';
  const files = { 'subject.nl': `---\nargs: { topic: string }\nreturns: 'Is<string, "${SHORT}">'\n---\nWrite a subject.\n` };
  const crisp = { [SHORT]: value => typeof value === 'string' ? value.length <= 10 : undefined };
  const model = stubModel({ answers: ['way too long a subject', 'short'], truth: () => true });
  let t = run(model, files, {}, fn => fn('t'), { runtime: { refinements: { crisp } } });
  assert.equal(await t.result(), 'short');
  assert.equal(model.judged.length, 0, 'the crisp checker decided both answers');
  assert.deepEqual(events(t.traces, 'refinement_check').map(event => event.source).slice(0, 2), ['crisp', 'crisp']);

  const shadow = stubModel({ answers: ['way too long a subject'], truth: () => true });
  t = run(shadow, files, { mode: 'shadow' }, fn => fn('t'), { runtime: { refinements: { crisp } } });
  assert.equal(await t.result(), 'way too long a subject', 'in shadow mode the judge decides');
  const [disagreement] = events(t.traces, 'refinement_shadow');
  assert.equal(disagreement.agree, false);
  assert.equal(disagreement.crisp, false);
  assert.equal(disagreement.nl, true);

  const deferred = stubModel({ answers: ['ok'], truth: () => true });
  t = run(deferred, files, {}, fn => fn('t'), { runtime: { refinements: { crisp: { [SHORT]: () => undefined } } } });
  assert.equal(await t.result(), 'ok');
  assert.ok(deferred.judged.length >= 1, 'undefined defers to the judge');

  crispTable[SHORT] = value => value.length <= 10;
  try {
    const builtin = stubModel({ answers: ['short'], truth: () => false });
    t = run(builtin, files, {}, fn => fn('t'));
    assert.equal(await t.result(), 'short');
    assert.equal(builtin.judged.length, 0);
  } finally { delete crispTable[SHORT]; }
});

test('without a scoring driver the judge runs as an ordinary boolean call and is traced as judge: call', async () => {
  const asked = [];
  const driver = async ({ messages }) => {
    const text = messages.map(message => String(message.content)).join('\n');
    if (/Decide whether value satisfies predicate/.test(text)) {
      asked.push(text);
      return { calls: [['return_result', { status: 'success', value: !/rude/.test(text) }]] };
    }
    const rude = !messages.some(message => message.role === 'tool' && /^rejected/.test(String(message.content)));
    return { calls: [['return_result', { status: 'success', value: rude ? 'rude reply' : 'Hello.' }]] };
  };
  const t = run({ driver }, { 'reply.nl': REPLY }, { repairs: 2 }, fn => fn('x'));
  assert.equal(await t.result(), 'Hello.');
  const checks = events(t.traces, 'refinement_check');
  assert.ok(checks.length >= 2 && checks.every(check => check.judge === 'call'), JSON.stringify(checks.map(check => check.judge)));
  assert.deepEqual(checks.slice(0, 2).map(check => check.outcome), ['fail', 'pass']);
  assert.equal(asked.length, 2);
});

test('service results declared refined are checked like an nl return', async () => {
  const judged = [];
  const driver = Object.assign(async ({ messages }) => {
    if (modelTurnsSoFar(messages) > 0) return { calls: [['return_result', { status: 'success', value: String(messages.at(-1).content).slice(0, 400) }]] };
    return { calls: [['eval', { code: 'let bad = "no error"; try { await mail.draft(); } catch (error) { bad = error.message; } return bad + " | " + await mail.fine();', finish: true }]] };
  }, { decide: async request => {
    const value = /<<<value\n([^]*?)\nvalue>>>/.exec(String(request.messages.at(-1).content))[1];
    judged.push(value);
    const p = value === 'bad' ? 0.02 : 0.98;
    return { log_probs: [Math.log(p), Math.log(1 - p)] };
  } });
  const files = { 'ask.nl': '---\nargs: { x: string }\nreturns: string\n---\nUse the mail service.\n' };
  const runtime = createNatlangRuntime({ model: { driver }, calls: false, seed: { mode: 'backend' },
    services: { mail: { draft: async () => 'bad', fine: async () => 'good' } },
    refinements: { services: { 'mail.draft': `Is<string, "${POLITE}">`, 'mail.fine': `Is<string, "${POLITE}">` } } });
  const fn = loadVirtualNatlang(files, 'ask.nl');
  const answer = await runtime.run(() => fn('x'));
  assert.match(answer, /^refinement-unsatisfied: Revise the value at mail\.draft so that it is "a reply that is polite".* \| good$/);
  assert.deepEqual(judged.sort(), ['bad', 'good']);
});

test('settings are validated; obligations are found in unions and aliases', () => {
  assert.deepEqual(parseRefinementSettings({ threshold: 0.6, band: { low: 0.4, high: 0.6 }, policy: 'escalate', escalate: 'teacher',
    predicates: { 'a   b': { mode: 'shadow' } } }).predicates, { 'a b': { mode: 'shadow' } });
  for (const bad of [{ threshold: 2 }, { band: { low: 0.8, high: 0.2 } }, { policy: 'maybe' }, { mode: 'x' }, { surprise: 1 }, { repairs: -1 }])
    assert.throws(() => parseRefinementSettings(bad), TypeError);
  const env = new TypeEnv({ Note: parseType('Is<string, "brief">') });
  const union = parseType('Note | { n: number }');
  assert.deepEqual(collectObligations('hi', union, env).map(item => [item.path, item.predicate]), [['return', 'brief']]);
  assert.deepEqual(collectObligations({ n: 1 }, union, env), []);
  assert.deepEqual(collectObligations(3, parseType('Is<string, "x">'), env), [], 'a structurally wrong value is not judged');
});

test('the judge scores true against false and renders the value as data', async () => {
  const seen = [];
  const judge = decisionJudge(async request => { seen.push(request); return { log_probs: [Math.log(0.75), Math.log(0.25)] }; }, { id: 'j' });
  assert.equal(+(await judge.probability({ a: 1 }, 'a record')).toFixed(6), 0.75);
  assert.deepEqual(seen[0].options, ['true', 'false']);
  assert.match(String(seen[0].messages[1].content), /<<<value\n\{"a":1\}\nvalue>>>/);
  const checker = new RefinementChecker({ judge });
  assert.deepEqual(await checker.check([{ path: 'p', predicate: 'a record', value: { a: 1 } }], { phase: 'refine' }), []);
});
