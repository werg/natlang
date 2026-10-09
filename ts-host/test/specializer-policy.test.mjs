import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, CallStore, specializerOutput } from '../dist/index.js';
import { targets, specializeOne } from '../../applications/dist/specializer/main.js';
import { crispWorth, sanitizeWorth, worthPolicy, declinePolicy, crispDeclineText, madeProgress, stateKey, guidanceProblems, guidanceFeedback,
  checkedGuidance, DEFAULTS, GUIDANCE_LIMIT } from '../../applications/dist/specializer/policy.js';
import { scriptedModel } from './support/natlang.mjs';

const freshStore = () => CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-specpol-')));
const done = store => { const root = store.root; store.close(); rmSync(root, { recursive: true, force: true }); };

const candidate = (over = {}) => ({ name: 'f', agent_calls: 100, tokens: 5000, new_calls_since_compilation: 0, decline: null, has_compilation: false, ...over });

test('the crisp target rule: volume, tokens, standing decline, new calls since a compilation', () => {
  assert.deepEqual(crispWorth(candidate(), 20), { look: true, reason: 'enough unspecialized volume' });
  assert.equal(crispWorth(candidate({ agent_calls: 19 }), 20).look, false);
  assert.equal(crispWorth(candidate({ tokens: 0 }), 20).look, false);
  assert.equal(crispWorth(candidate({ decline: { calls_at_decline: 60, reason: 'semantic' } }), 20).look, false);
  assert.equal(crispWorth(candidate({ decline: { calls_at_decline: 50, reason: 'semantic' } }), 20).look, true);
  assert.equal(crispWorth(candidate({ has_compilation: true, new_calls_since_compilation: 5 }), 20).look, false);
  assert.equal(crispWorth(candidate({ has_compilation: true, new_calls_since_compilation: 20 }), 20).look, true);
});

test('the natural-language target answer never lowers the crisp bound or turns malformed into a look', () => {
  assert.equal(sanitizeWorth({ look: true, reason: 'x' }, candidate({ agent_calls: 3 }), 20).look, false);
  assert.equal(sanitizeWorth({ look: 'yes' }, candidate(), 20).look, false);
  assert.deepEqual(sanitizeWorth({ look: true }, candidate(), 20), { look: true, reason: '(no reason given)' });
});

test('worthPolicy: crisp never asks the model, nl serves the model, shadow serves crisp and records disagreement', async () => {
  const asked = [];
  const judge = value => (c, min) => { asked.push(c.name); return value; };
  assert.equal((await worthPolicy('crisp', judge({ look: false, reason: 'x' }))(candidate(), 20)).look, true);
  assert.deepEqual(asked, []);
  assert.equal((await worthPolicy('nl', judge({ look: false, reason: 'x' }))(candidate(), 20)).look, false);
  assert.equal((await worthPolicy('shadow', judge({ look: false, reason: 'x' }))(candidate(), 20)).look, true);
  assert.deepEqual(asked, ['f', 'f']);
});

test('declinePolicy: crisp joins group lines; nl stores the reading; shadow stores crisp', async () => {
  const groups = [{ id: 'g1', label: 'orders.refund', calls: 12, outcome: 'skipped', reason: 'semantic', why: 'needs the meaning' }];
  assert.equal(crispDeclineText(groups), 'no case was accepted: g1 (orders.refund, 12 calls): semantic: needs the meaning');
  const reading = () => 'The main obstacle is meaning. Anchor the condition on the whole input.';
  assert.equal(await declinePolicy('crisp', reading)(groups), crispDeclineText(groups));
  assert.match(await declinePolicy('nl', reading)(groups), /^no case was accepted: The main obstacle/);
  assert.equal(await declinePolicy('shadow', reading)(groups), crispDeclineText(groups));
  assert.equal(await declinePolicy('nl', () => '  ')(groups), crispDeclineText(groups));
});

test('the repair loop continues while any group changed and stops at a fixed point', () => {
  const a = stateKey({}), b = stateKey({ part: 'x' }), c = stateKey({ report: 'r' }), d = stateKey({ done: 'accepted', part: 'x' });
  assert.equal(madeProgress([a, a], [b, a]), true);
  assert.equal(madeProgress([b, a], [c, a]), true);
  assert.equal(madeProgress([c, a], [c, a]), false);
  assert.equal(madeProgress([b], [d]), true);
  assert.equal(DEFAULTS.rounds, 3);
});

test('guidance checks: length, emptiness and sentences written as prohibitions', async () => {
  assert.deepEqual(guidanceProblems('Call orders.refund with the number.'), { tooLong: false, empty: false, negative: [] });
  assert.equal(guidanceProblems('x'.repeat(GUIDANCE_LIMIT + 1)).tooLong, true);
  assert.equal(guidanceProblems('').empty, true);
  assert.deepEqual(guidanceProblems('Call orders.refund. Never call lookup first.\nDo not guess.').negative, ['Never call lookup first.', 'Do not guess.']);
  assert.match(guidanceFeedback('Never call lookup first.'), /Rewrite these sentences as actions/);
  const seen = [];
  const repaired = await checkedGuidance(async feedback => { seen.push(feedback); return feedback ? 'Call lookup after refund, in that order.' : 'Never call lookup before refund, ever.'; });
  assert.equal(repaired, 'Call lookup after refund, in that order.\n');
  assert.match(seen[1], /Never call lookup before refund/);
  assert.equal(await checkedGuidance(async () => 'Do not call lookup at all, ever.'), undefined);
});

const HANDLE = '---\nargs: { request: string }\nreturns: string\n---\nHandle a support request: refund an order, or say its status.\n';
const orders = () => ({ async refund() { return { ok: true }; }, lookup(id) { return { status: `shipped ${id}` }; } });
function executor() {
  return Object.assign(async request => {
    const tools = request.messages.filter(message => message.role === 'tool').map(message => String(message.content));
    const opening = String(request.messages.find(message => message.role === 'user')?.content ?? '');
    if (opening.includes('Two executions')) return { prompt_tokens: 10, calls: [['return_result', { status: 'success', value: 'equal' }]] };
    const seen = tools.map(text => /(refund|status) (\d+)/.exec(text)).find(Boolean);
    if (tools.length <= 1) return { prompt_tokens: 10, calls: [['eval', { code: 'return request' }]] };
    if (tools.length === 2) return { prompt_tokens: 10, calls: [['eval', { code: seen[1] === 'refund' ? "const id = request.split(' ')[1];\nawait orders.refund(id);\nreturn 'refunded ' + id" :
      "return orders.lookup(request.split(' ')[1]).status" }]] };
    return { prompt_tokens: 10, calls: [['return_result', { status: 'success', value: seen[1] === 'refund' ? `refunded ${seen[2]}` : `shipped ${seen[2]}` }]] };
  }, { model: 'test-model' });
}
async function recordCalls(store, count) {
  const handle = loadVirtualNatlang({ 'handle.nl': HANDLE }, 'handle.nl');
  for (let index = 0; index < count; index++)
    await createNatlangRuntime({ model: executor(), calls: store, services: { orders: orders() } }).run(() => handle(`${index % 2 ? 'refund' : 'status'} ${100 + index}`));
}
const options = (over = {}) => ({ loop: false, interval: 600, rounds: 3, jobs: 50, includeSelf: false, dryRun: false, jobsOnly: false, maxBusy: 2, idleWait: 0,
  maxGroups: 8, guidance: true, ...over });

test('targets follow targetPolicy: crisp rule by default, worthLooking.nl under nl, crisp bound kept', async () => {
  const store = freshStore();
  try {
    await recordCalls(store, 30);
    assert.equal((await targets(store, options(), null)).length, 1);
    assert.equal((await targets(store, options({ minCalls: 100 }), null)).length, 0);
    const asked = [];
    const model = scriptedModel(opening => { asked.push(opening); return 'return { look: false, reason: "the scripted policy says wait" }'; });
    const runtime = createNatlangRuntime({ model: model.driver, calls: false });
    store.writeSettings({ targetPolicy: 'nl' });
    assert.equal((await targets(store, options(), null, runtime)).length, 0);
    assert.equal(asked.length, 1);
    store.writeSettings({ targetPolicy: 'shadow' });
    assert.equal((await targets(store, options(), null, runtime)).length, 1, 'shadow serves the crisp answer');
    assert.equal(asked.length, 2);
    const yes = createNatlangRuntime({ model: scriptedModel(() => 'return { look: true, reason: "plenty" }').driver, calls: false });
    store.writeSettings({ targetPolicy: 'nl' });
    assert.equal((await targets(store, options({ minCalls: 100 }), null, yes)).length, 0, 'nl cannot lower minCalls');
  } finally { done(store); }
});

// Writers for the end-to-end pass: chooseCondition measures a condition per group, writeBody writes its case.ts,
// writeGuidance returns guidance (a prohibition first, to exercise the one repair).
const REFUND = "import { orders } from 'natlang:services';\\nexport const when = (args: { request: string }) => /^refund \\\\d+$/.test(args.request);\\n" +
  "export const run = async (args: { request: string }) => { const id = args.request.split(' ')[1]; await orders.refund(id); return 'refunded ' + id; };\\n";
const STATUS = "import { orders } from 'natlang:services';\\nexport const when = (args: { request: string }) => /^status \\\\d+$/.test(args.request);\\n" +
  "export const run = async (args: { request: string }) => orders.lookup(args.request.split(' ')[1]).status;\\n";
function writers(log, stubborn = false) {
  return scriptedModel(opening => {
    if (opening.includes('You choose the condition')) { log.push('choose');
      return "const text = await folder.file('group.md').readText();\n" +
        "const condition = text.includes('orders.refund') ? '/^refund \\\\d+$/.test(args.request)' : '/^status \\\\d+$/.test(args.request)';\n" +
        "const m = await group.measure(condition);\nreturn { kind: 'condition', condition, admits: m.ofGroup };"; }
    if (opening.includes('You write case.ts')) { log.push('body');
      return `await folder.file('case.ts').writeText(condition.includes('refund') ? "${REFUND}" : "${STATUS}");\nreturn { kind: 'case' };`; }
    if (opening.includes('Write short guidance')) { log.push(`guidance:${/feedback[^\n]*\n?/.test(opening)}`);
      return `return feedback && !${stubborn} ? 'Recognise a request that starts with refund and call orders.refund with its number.\\nRecognise a request that starts with status and return the status that orders.lookup gives.\\nFinish with a short string.' : 'Never call lookup for a refund.'`; }
    return null;
  });
}
const combined = (script) => Object.assign(async request => {
  const text = request.messages.map(message => String(message.content ?? '')).join('\n');
  return /You choose the condition|You write case\.ts|Write short guidance/.test(text) ? script.driver(request) : executor()(request);
}, { model: 'test-model' });

test('specializeOne: chooseCondition then writeBody per group, and the compilation carries instructions.md (tier 2)', async () => {
  const store = freshStore();
  try {
    await recordCalls(store, 30);
    const key = store.hot()[0].definition_key;
    const log = [];
    const context = { model: combined(writers(log)), executorIdentity: undefined, package: undefined, args: [], io: { output: process.stdout, error: process.stderr } };
    const summary = await specializeOne(context, store, key, options());
    assert.match(summary, /compilation .* with 2 case\(s\), in shadow/, summary);
    assert.equal(log.filter(item => item === 'choose').length, 2);
    assert.equal(log.filter(item => item === 'body').length, 2);
    const row = store.currentCompilation(key);
    assert.match(row.files['instructions.md'], /^Recognise a request that starts with refund/);
    assert.doesNotMatch(row.files['instructions.md'], /Never/);
    assert.ok(row.files['cases.ts'].includes('orders.refund'));
    const output = specializerOutput(store, key, row.interface_hash);
    assert.match(output.guidance, /orders\.lookup/, 'the tier ladder reads the guidance');
  } finally { done(store); }
});

test('specializeOne without guidance, and with guidance that never passes, stores no instructions.md', async () => {
  for (const settings of [{ guidance: false }, { guidance: true, stubborn: true }]) {
    const store = freshStore();
    try {
      await recordCalls(store, 30);
      const key = store.hot()[0].definition_key;
      const model = combined(writers([], !!settings.stubborn));
      const context = { model, executorIdentity: undefined, package: undefined, args: [], io: { output: process.stdout, error: process.stderr } };
      const summary = await specializeOne(context, store, key, options({ guidance: settings.guidance }));
      assert.match(summary, /compilation/, summary);
      assert.equal(store.currentCompilation(key).files['instructions.md'], undefined);
    } finally { done(store); }
  }
});

test('a declined pass stores the decline text of declinePolicy', async () => {
  for (const [policy, expected] of [['crisp', /^no case was accepted: g\d \(.*\): semantic: needs meaning/], ['nl', /^no case was accepted: The main obstacle is meaning\./]]) {
    const store = freshStore();
    try {
      await recordCalls(store, 30);
      store.writeSettings({ declinePolicy: policy });
      const key = store.hot()[0].definition_key;
      const script = scriptedModel(opening => {
        if (opening.includes('You choose the condition')) return "return { kind: 'skip', reason: 'semantic', why: 'needs meaning' }";
        if (opening.includes('Write the reason a function stays')) return "return 'The main obstacle is meaning. Anchor the condition on the whole input.'";
        return null;
      });
      const model = Object.assign(async request => {
        const text = request.messages.map(message => String(message.content ?? '')).join('\n');
        return /You choose the condition|Write the reason a function stays/.test(text) ? script.driver(request) : executor()(request);
      }, { model: 'test-model' });
      const context = { model, executorIdentity: undefined, package: undefined, args: [], io: { output: process.stdout, error: process.stderr } };
      const summary = await specializeOne(context, store, key, options());
      assert.match(summary, /declined \(semantic\)/, summary);
      assert.match(store.declineFor(key).why, expected);
    } finally { done(store); }
  }
});
