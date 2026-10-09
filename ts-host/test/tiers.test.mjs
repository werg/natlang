import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, CallStore, TierEngine, caseHashes } from '../dist/index.js';
import { tiersOf, caseTiersOf } from '../dist/cli/calls.js';

const freshStore = () => CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-tiers-')));
const done = store => { const root = store.root; store.close(); rmSync(root, { recursive: true, force: true }); };
const SHOUT = '---\nargs: { text: string }\nreturns: string\n---\nReturn the text in upper case.\n';
const shout = () => loadVirtualNatlang({ 'shout.nl': SHOUT }, 'shout.nl');

/** A scripted model profile: answers `value`, counts its calls, declares a model name and token use. */
function scripted(name, value) {
  const model = Object.assign(async () => { model.asked++; return { prompt_tokens: 10, calls: [['return_result', { status: 'success', value: model.value }]] }; }, { model: name });
  model.asked = 0; model.value = value;
  return model;
}
const crisp = (overrides = {}) => ({ id: 'upper', when: args => typeof args.text === 'string' && args.text.length < 10,
  run: args => args.text.toUpperCase(), pure: true, ...overrides });
const setup = (functions, store, extra = {}) => {
  const teacher = scripted('teacher', 'TEACHER'), student = scripted('student', 'STUDENT');
  const engine = new TierEngine({ functions: { shout: { tier0: { model: 'teacher' }, tier1: { model: 'student' }, ...functions } }, random: () => 0 });
  const traces = [];
  const runtime = createNatlangRuntime({ model: teacher, models: { teacher, student }, calls: store, tiers: engine, ...extra });
  const handle = shout();
  const call = text => runtime.run(() => handle(text), { trace: trace => traces.push(trace) });
  const events = kind => traces.flatMap(trace => trace.events).filter(event => event.kind === kind);
  return { teacher, student, engine, call, traces, events };
};
const rows = store => store.tierRows().map(row => row.value);

test('the highest admissible tier serves, and the call is recorded as a crisp case', async () => {
  const store = freshStore();
  try {
    const t = setup({ crisp: { implementations: [crisp()], start: 'active' } }, store);
    assert.equal(await t.call('hello'), 'HELLO');
    assert.equal(t.teacher.asked + t.student.asked, 0);
    assert.deepEqual(rows(store).map(row => `${row.tier}:${row.event}`), ['tier3:upper:served']);
    const [served] = t.events('tier_served');
    assert.equal(served.tier, 'tier3:upper');
    const call = store.call(store.calls({ limit: 1 })[0].call_id);
    assert.equal(call.executor.kind, 'crisp');
    assert.equal(call.executor.case_hash, 'tier:upper');
  } finally { done(store); }
});

test('a guard miss goes to the student; a rejected student output goes to the teacher', async () => {
  const store = freshStore();
  try {
    const t = setup({ crisp: { implementations: [crisp()], start: 'active' } }, store);
    assert.equal(await t.call('a very long input text'), 'STUDENT');
    const deopt = t.events('tier_deopt');
    assert.equal(deopt.length, 1);
    assert.equal(deopt[0].tier, 'tier3:upper');
    assert.equal(deopt[0].reason_kind, 'guard');
    const v = setup({ tier1: { model: 'student', verify: (args, out) => out === 'TEACHER' || 'student output rejected' } }, store);
    assert.equal(await v.call('x'), 'TEACHER');
    assert.equal(v.student.asked, 1);
    const kinds = rows(store).filter(row => row.event === 'deopt' && row.tier === 'tier1');
    assert.equal(kinds.length, 1);
    assert.equal(kinds[0].kind, 'verify');
    assert.equal(kinds[0].own, true);
    assert.equal(kinds[0].to, 'tier0');
  } finally { done(store); }
});

test('an output check that fails in the crisp tier deoptimizes before serving', async () => {
  const store = freshStore();
  try {
    const t = setup({ crisp: { implementations: [crisp({ verify: () => 'output is not trusted' })], start: 'active' } }, store);
    assert.equal(await t.call('abc'), 'STUDENT');
    const deopt = t.events('tier_deopt');
    assert.equal(deopt[0].tier, 'tier3:upper');
    assert.equal(deopt[0].reason, 'output is not trusted');
    assert.equal(t.events('tier_served')[0].tier, 'tier1');
  } finally { done(store); }
});

test('repeated failures demote a tier by the shared rule, and the demotion survives a restart of the engine', async () => {
  const store = freshStore();
  try {
    let runs = 0;
    const broken = crisp({ run: () => { runs++; return 'WRONG'; }, verify: () => 'wrong' });
    const t = setup({ crisp: { implementations: [broken], start: 'active' } }, store);
    for (let index = 0; index < 14; index++) assert.equal(await t.call('abc'), 'STUDENT');
    assert.equal(runs, 10, 'after callMinimum hand-offs over the bound the tier is not tried again');
    assert.equal(t.events('tier_demoted').length, 1);
    assert.equal(t.events('tier_demoted')[0].evidence.by, 'crisp');
    assert.equal(t.engine.ledger.state('shout', 'tier3:upper', 'active'), 'demoted');
    const again = setup({ crisp: { implementations: [broken], start: 'active' } }, store);
    assert.equal(await again.call('abc'), 'STUDENT');
    assert.equal(runs, 10, 'a new engine on the same store starts demoted');
  } finally { done(store); }
});

test('guard misses and infrastructure failures are not failures of a tier', async () => {
  const store = freshStore();
  try {
    const t = setup({ crisp: { implementations: [crisp()], start: 'active' } }, store);
    for (let index = 0; index < 15; index++) assert.equal(await t.call('a very long input text'), 'STUDENT');
    assert.equal(t.engine.ledger.state('shout', 'tier3:upper', 'active'), 'active');
    assert.equal(t.engine.ledger.stats('shout', 'tier3:upper').guard_misses, 15);
    const flaky = crisp({ run: () => { throw new Error('connect ETIMEDOUT'); } });
    const u = setup({ crisp: { implementations: [flaky], start: 'active' } }, store);
    for (let index = 0; index < 15; index++) assert.equal(await u.call('abc'), 'STUDENT');
    assert.equal(u.engine.ledger.state('shout', 'tier3:upper', 'active'), 'active');
    assert.equal(u.engine.ledger.stats('shout', 'tier3:upper').infrastructure > 0, true);
  } finally { done(store); }
});

test('shadow evidence promotes a tier that agrees, and not one that does not, by the store settings', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionComparisons: 3, promotionLiveComparisons: 0 });
    const t = setup({ crisp: { implementations: [crisp({ run: () => 'STUDENT' })], start: 'shadow' }, shadowRate: 1 }, store);
    for (let index = 0; index < 3; index++) assert.equal(await t.call('abc'), 'STUDENT');
    assert.equal(t.student.asked, 3, 'in shadow the student serves');
    assert.equal(t.events('tier_promoted').length, 1);
    assert.equal(t.events('tier_promoted')[0].tier, 'tier3:upper');
    assert.equal(await t.call('abc'), 'STUDENT');
    assert.equal(t.student.asked, 3, 'promoted: the crisp tier serves');
    assert.equal(t.events('tier_served').at(-1).tier, 'tier3:upper');
  } finally { done(store); }
  const other = freshStore();
  try {
    other.writeSettings({ promotionComparisons: 3, promotionLiveComparisons: 0 });
    const t = setup({ crisp: { implementations: [crisp()], start: 'shadow' }, shadowRate: 1 }, other);
    for (let index = 0; index < 5; index++) await t.call('abc');
    assert.equal(t.events('tier_promoted').length, 0);
    assert.equal(t.student.asked, 5);
  } finally { done(other); }
});

test('without tier configuration for a function, dispatch is untouched', async () => {
  const store = freshStore();
  try {
    const teacher = scripted('teacher', 'TEACHER');
    const engine = new TierEngine({ functions: { other: { crisp: { implementations: [crisp()], start: 'active' } } } });
    const runtime = createNatlangRuntime({ model: teacher, calls: store, tiers: engine });
    assert.equal(await runtime.run(() => shout()('abc')), 'TEACHER');
    assert.equal(store.tierRows().length, 0);
    assert.equal(store.call(store.calls({ limit: 1 })[0].call_id).executor.kind, 'agent');
  } finally { done(store); }
});

const SHOUT_CASES = `export const cases = [
  { when: (args: { text: string }) => args.text.length < 10, run: (args: { text: string }) => args.text.toUpperCase() },
];
`;
/** Store SHOUT_CASES as the current compilation of shout.nl, with extra files and the given case tier. */
async function compiled(store, tier, files = {}) {
  const teacher = scripted('seed', 'SEED');
  await createNatlangRuntime({ model: teacher, calls: store, specialization: 'off' }).run(() => shout()('seed'));
  const seed = store.call(store.calls()[0].call_id);
  const hashes = caseHashes(SHOUT_CASES);
  store.saveCompilation({ definitionKey: seed.definition.key, definitionId: seed.definition.id, definitionName: 'shout', definitionSource: seed.definition.source,
    interfaceHash: seed.definition.interface, programRoot: null, files: { 'cases.ts': SHOUT_CASES, ...files }, caseHashes: hashes,
    links: [{ caseHash: hashes[0], callId: seed.call_id, role: 'training' }] });
  hashes.forEach(hash => store.setTier(hash, tier));
  return hashes;
}

test('the ladder has no crisp tier without a compilation, and so leaves no deopt noise', async () => {
  const store = freshStore();
  try {
    const t = setup({}, store);
    assert.equal(await t.call('abc'), 'STUDENT');
    assert.deepEqual(store.calls({ limit: 10 }).map(call => call.outcome), ['done']);
    assert.deepEqual(rows(store).map(row => `${row.tier}:${row.event}`), ['tier1:served']);
  } finally { done(store); }
});

test('the specializer compilation becomes tier 3 by itself, and the stored case keeps its own state', async () => {
  const store = freshStore();
  try {
    const [hash] = await compiled(store, 'active');
    const t = setup({}, store);
    assert.equal(await t.call('abc'), 'ABC');
    assert.equal(t.teacher.asked + t.student.asked, 0);
    assert.equal(t.events('tier_served')[0].tier, 'tier3');
    assert.equal(store.caseStats(hash).served, 1);
    assert.equal(await t.call('a very long input text'), 'STUDENT', 'a guard miss falls to the student');
    const row = tiersOf(store.tierRows(), 'shout', caseTiersOf(store, 'shout')).find(item => item.tier === 'tier3');
    assert.equal(row.calls, 1, 'the served count is the store crisp calls');
    assert.equal(row.state, 'active');
  } finally { done(store); }
});

test('a shadow case is not a tier yet', async () => {
  const store = freshStore();
  try {
    await compiled(store, 'shadow');
    const t = setup({}, store);
    assert.equal(await t.call('abc'), 'STUDENT');
    assert.deepEqual(rows(store).map(row => `${row.tier}:${row.event}`), ['tier1:served']);
  } finally { done(store); }
});

test('recorded specialized guidance becomes tier 2 on the student, after the crisp tier', async () => {
  const store = freshStore();
  try {
    await compiled(store, 'active', { 'instructions.md': 'Shout the text.' });
    const t = setup({ specialized: { start: 'active' } }, store);
    assert.equal(await t.call('a very long input text'), 'STUDENT');
    assert.equal(t.events('tier_served')[0].tier, 'tier2');
    assert.equal(t.events('tier_deopt')[0].tier, 'tier3');
    const off = setup({ specialized: { enabled: false } }, store);
    await off.call('a very long input text');
    assert.equal(off.events('tier_served')[0].tier, 'tier1');
  } finally { done(store); }
});

test('tiersOf summarizes calls, deopt rate and cost per tier', () => {
  const event = (tier, name, extra = {}) => ({ value: { fn: 'f', tier, event: name, ...extra }, definition_name: 'f', tokens_in: 10, tokens_out: 5, wall_ms: 20 });
  const table = tiersOf([event('tier1', 'served'), event('tier1', 'served'), event('tier1', 'deopt', { own: true }), event('tier0', 'served')]);
  const student = table.find(row => row.tier === 'tier1');
  assert.equal(student.calls, 2);
  assert.equal(student.deopt_rate, 0.333);
  assert.equal(student.tokens_per_call, 15);
  assert.equal(student.wasted_tokens, 15);
});
