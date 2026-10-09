import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, CallStore, TierEngine } from '../dist/index.js';
import { tiersOf } from '../dist/cli/calls.js';

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
    const t = setup({ tier3: { implementation: crisp(), start: 'active' } }, store);
    assert.equal(await t.call('hello'), 'HELLO');
    assert.equal(t.teacher.asked + t.student.asked, 0);
    assert.deepEqual(rows(store).map(row => `${row.tier}:${row.event}`), ['tier3:served']);
    const [served] = t.events('tier_served');
    assert.equal(served.tier, 'tier3');
    const call = store.call(store.calls({ limit: 1 })[0].call_id);
    assert.equal(call.executor.kind, 'crisp');
    assert.equal(call.executor.case_hash, 'tier:upper');
  } finally { done(store); }
});

test('a guard miss goes to the student; a rejected student output goes to the teacher', async () => {
  const store = freshStore();
  try {
    const t = setup({ tier3: { implementation: crisp(), start: 'active' } }, store);
    assert.equal(await t.call('a very long input text'), 'STUDENT');
    const deopt = t.events('tier_deopt');
    assert.equal(deopt.length, 1);
    assert.equal(deopt[0].tier, 'tier3');
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
    const t = setup({ tier3: { implementation: crisp({ verify: () => 'output is not trusted' }), start: 'active' } }, store);
    assert.equal(await t.call('abc'), 'STUDENT');
    const deopt = t.events('tier_deopt');
    assert.equal(deopt[0].tier, 'tier3');
    assert.equal(deopt[0].reason, 'output is not trusted');
    assert.equal(t.events('tier_served')[0].tier, 'tier1');
  } finally { done(store); }
});

test('repeated failures demote a tier, and the demotion survives a restart of the engine', async () => {
  const store = freshStore();
  try {
    let runs = 0;
    const broken = crisp({ run: () => { runs++; return 'WRONG'; }, verify: () => 'wrong' });
    const t = setup({ tier3: { implementation: broken, start: 'active' }, demoteAfter: 3 }, store);
    for (let index = 0; index < 5; index++) assert.equal(await t.call('abc'), 'STUDENT');
    assert.equal(runs, 3, 'after three failures the tier is not tried again');
    assert.equal(t.events('tier_demoted').length, 1);
    assert.equal(t.engine.ledger.state('shout', 'tier3', 'active'), 'demoted');
    const again = setup({ tier3: { implementation: broken, start: 'active' }, demoteAfter: 3 }, store);
    assert.equal(await again.call('abc'), 'STUDENT');
    assert.equal(runs, 3, 'a new engine on the same store starts demoted');
  } finally { done(store); }
});

test('shadow evidence promotes a tier that agrees, and not one that does not', async () => {
  const store = freshStore();
  try {
    const t = setup({ tier3: { implementation: crisp({ run: () => 'STUDENT' }), start: 'shadow' }, shadowRate: 1, promoteAfter: 3 }, store);
    for (let index = 0; index < 3; index++) assert.equal(await t.call('abc'), 'STUDENT');
    assert.equal(t.student.asked, 3, 'in shadow the student serves');
    assert.equal(t.events('tier_promoted').length, 1);
    assert.equal(t.events('tier_promoted')[0].tier, 'tier3');
    assert.equal(await t.call('abc'), 'STUDENT');
    assert.equal(t.student.asked, 3, 'promoted: the crisp tier serves');
    assert.equal(t.events('tier_served').at(-1).tier, 'tier3');
  } finally { done(store); }
  const other = freshStore();
  try {
    const t = setup({ tier3: { implementation: crisp(), start: 'shadow' }, shadowRate: 1, promoteAfter: 3 }, other);
    for (let index = 0; index < 5; index++) await t.call('abc');
    assert.equal(t.events('tier_promoted').length, 0);
    assert.equal(t.student.asked, 5);
  } finally { done(other); }
});

test('without tier configuration for a function, dispatch is untouched', async () => {
  const store = freshStore();
  try {
    const teacher = scripted('teacher', 'TEACHER');
    const engine = new TierEngine({ functions: { other: { tier3: { implementation: crisp(), start: 'active' } } } });
    const runtime = createNatlangRuntime({ model: teacher, calls: store, tiers: engine });
    assert.equal(await runtime.run(() => shout()('abc')), 'TEACHER');
    assert.equal(store.tierRows().length, 0);
    assert.equal(store.call(store.calls({ limit: 1 })[0].call_id).executor.kind, 'agent');
  } finally { done(store); }
});

test('the compiled tier declines without a compilation and leaves no half-recorded call behind', async () => {
  const store = freshStore();
  try {
    const t = setup({ tier2: {} }, store);
    assert.equal(await t.call('abc'), 'STUDENT');
    assert.deepEqual(store.calls({ limit: 10 }).map(call => call.outcome), ['done']);
    assert.deepEqual(rows(store).map(row => `${row.tier}:${row.event}`), ['tier2:deopt', 'tier1:served']);
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
