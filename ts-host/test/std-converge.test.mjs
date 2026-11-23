// @natlang/std convergence combinators (semanticRepair, fixedPoint, semanticMerge) and the shipped judgments they use,
// with a scripted model: the loop mechanics, the stop rules, the traces, and examples/release's notes and merge.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createNatlangRuntime, loadVirtualNatlang, std as fromRoot } from '../dist/index.js';
import * as std from '../dist/std/index.js';
import { STD_MODULE_DTS } from '../dist/compiler/intrinsics.js';
import { STD_SOURCES } from '../dist/std/sources.generated.js';
import { buildExample, localModel } from './support/std.mjs';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const { semanticRepair, fixedPoint, semanticMerge } = std;

test('the library is one module: @natlang/node/std, the std namespace of @natlang/node, and the virtual declarations agree', () => {
  assert.equal(fromRoot.semanticRepair, std.semanticRepair, 'one instance');
  const exported = Object.keys(std).sort();
  const declared = [...STD_MODULE_DTS.matchAll(/export declare (?:function|const) (\w+)/g)].map(match => match[1]).sort();
  assert.deepEqual(declared, exported, 'intrinsics STD_MODULE_DTS declares exactly what src/std exports');
  // The shipped judgments are the files in ts-host/std, byte for byte, and load under @natlang/std/.
  for (const [name, source] of Object.entries(STD_SOURCES)) assert.equal(readFileSync(join(repo, 'ts-host/std', name), 'utf8'), source);
  assert.deepEqual(Object.keys(STD_SOURCES).sort(), ['beats.nl', 'combine.nl', 'compatible.nl', 'compileRule.nl', 'intentOf.nl', 'meets.nl', 'relates.nl', 'settled.nl', 'shouldEscalate.nl']);
  for (const name of ['intentOf', 'compatible', 'combine', 'settled', 'beats', 'meets', 'relates', 'compileRuleFn', 'shouldEscalate']) {
    assert.equal(typeof std[name], 'function');
    assert.equal(typeof std[name].in, 'function', `${name} is a natlang function (.in, .on)`);
  }
});

test('semanticRepair: the exact check feeds the fix until it passes; the trace lists each stage', async () => {
  const traces = [];
  const runtime = createNatlangRuntime({ trace: trace => traces.push(trace) });
  const check = words => words.length > 3 ? [`${words.length} words; at most 3.`] : [];
  const seen = [];
  const fix = async (words, problems) => { seen.push(problems); return words.slice(1); };
  const result = await runtime.run(() => semanticRepair(['a', 'b', 'c', 'd', 'e'], { check, fix, attempts: 5 }));
  assert.deepEqual(result.value, ['c', 'd', 'e']);
  assert.equal(result.ok, true); assert.equal(result.attempts, 2); assert.deepEqual(result.problems, []);
  assert.deepEqual(seen, [['5 words; at most 3.'], ['4 words; at most 3.']]);
  assert.deepEqual(result.history.map(entry => entry.problems.length), [1, 1, 0]);
  const span = traces.find(trace => trace.definitionId === 'std:semanticRepair');
  assert.ok(span, 'one std:semanticRepair trace');
  assert.equal(span.outcome, 'done'); assert.match(span.detail, /passed after 2 fixes/);
  assert.deepEqual(span.events.filter(event => event.kind === 'std_stage').map(event => event.stage), ['check', 'fix', 'fix']);
  assert.ok(traces.some(trace => trace.definitionId.startsWith('iterateOn:')), 'the loop is an iterateOn');
});

test('semanticRepair answers at its bound with the closest value, stops on a repeated value, and needs a bound', async () => {
  const runtime = createNatlangRuntime({});
  const check = value => value % 2 ? ['odd'] : value > 10 ? ['too big', 'still too big'] : [];
  // Never passes: 1 (1 problem) -> 13 (2) -> 15 (1) -> 17 (1); the latest of the fewest is 17.
  const climbing = await runtime.run(() => semanticRepair(1, { check, fix: value => value < 13 ? 13 : value + 2, attempts: 3 }));
  assert.equal(climbing.ok, false); assert.equal(climbing.value, 17); assert.equal(climbing.attempts, 3);
  // A fix that returns an earlier value ends the repair.
  const stuck = await runtime.run(() => semanticRepair(3, { check, fix: () => 3, attempts: 50 }));
  assert.equal(stuck.ok, false); assert.equal(stuck.attempts, 1);
  // Zero attempts only checks.
  assert.equal((await runtime.run(() => semanticRepair(4, { check, fix: () => 0, attempts: 0 }))).ok, true);
  await assert.rejects(runtime.run(() => semanticRepair(1, { check, fix: value => value })), /^TypeError: iteration-unbounded: semanticRepair needs attempts/);
  await assert.rejects(runtime.run(() => semanticRepair(1, { check: () => 'odd', fix: value => value, attempts: 1 })), /std-check-result: check returns the list of problems/);
});

test('semanticRepair with an inline nl fix: the problems reach the model as the fix argument', async () => {
  const fix = loadVirtualNatlang({ 'fix.nl': '---\nargs: { text: string, problems: "string[]" }\nreturns: string\n---\nRewrite text so that none of problems applies.\n' }, 'fix.nl');
  const local = localModel(opening => opening.includes('none of problems') ? 'return text.toUpperCase()' : null, () => [0.5, 0.5]);
  const result = await createNatlangRuntime({ model: local.driver }).run(() =>
    semanticRepair('quiet', { check: value => value === value.toUpperCase() ? [] : ['The text must be upper case.'], fix, attempts: 2 }));
  assert.deepEqual([result.ok, result.value, result.attempts], [true, 'QUIET', 1]);
  assert.match(local.openings[0], /The text must be upper case\./);
});

test('fixedPoint: the shipped settled judgment ends the loop without a step bound; an exact settled test needs maxSteps', async () => {
  // settled.nl: the first revision still changed the text, the second only reworded it.
  // Keyed on the judged values, not on call order: a reasoned readout scores its options once per reasoning sample.
  const local = localModel(() => null, (options, prompt) => /draft two\./.test(prompt) ? [0.9, 0.1] : [0.1, 0.9]);
  const traces = [];
  const runtime = createNatlangRuntime({ model: local.driver, trace: trace => traces.push(trace) });
  const steps = ['draft', 'draft two', 'draft two.'];
  let at = 0;
  const result = await runtime.run(() => fixedPoint(async () => steps[Math.min(++at, steps.length - 1)], steps[0]));
  assert.equal(result.value, 'draft two.'); assert.equal(result.settled, true); assert.equal(result.steps, 2);
  const judged = local.decisions.filter(decision => /says the same as previous/.test(decision.prompt));
  assert.ok(judged.length >= 1, 'settled.nl was scored');
  assert.ok(traces.some(trace => trace.definitionId === 'std:fixedPoint' && trace.outcome === 'done'));
  // A step that changes nothing has reached its fixed point without a judgment.
  assert.deepEqual(await runtime.run(() => fixedPoint(value => value, 'same')), { value: 'same', steps: 1, settled: true });
  // Exact settledness: equality after rounding, within a bound.
  const halving = await runtime.run(() => fixedPoint(value => Math.round(value / 2), 40, { settled: (a, b) => a === b, maxSteps: 20 }));
  assert.deepEqual(halving, { value: 1, steps: 7, settled: true }, '40 20 10 5 3 2 1 1');
  const cut = await runtime.run(() => fixedPoint(value => value + 1, 0, { settled: () => false, maxSteps: 3 }));
  assert.deepEqual(cut, { value: 3, steps: 3, settled: false });
  await assert.rejects(runtime.run(() => fixedPoint(value => value + 1, 0, { settled: () => false })), /iteration-unbounded: fixedPoint with an exact settled test needs maxSteps/);
});

const mergeModel = ({ compatible = 0.9 } = {}) => localModel(opening => {
  if (opening.includes('Say what the edit is for')) return 'return "The edit " + (typeof after === "string" && after.includes("search") ? "adds the faster search" : "announces the removed endpoint") + "."';
  if (opening.includes('Write the value that makes')) return 'return ours.after + " " + theirs.after.split(". ").slice(1).join(". ")';
  return null;
}, (options, prompt) => /one value can carry out both purposes/.test(prompt) ? [compatible, 1 - compatible] : [0.5, 0.5]);

test('semanticMerge: one-sided edits are taken, both-sided ones judged by intent, combined and checked', async () => {
  const base = { title: 'Sync 2.4', highlights: 'Uploads resume.', notes: { a: 1, b: 2 }, removed: 'x' };
  const ours = { title: 'Sync 2.4', highlights: 'Uploads resume, and search is faster.', notes: { a: 1, b: 3 }, removed: 'x' };
  const theirs = { title: 'Sync 3.0', highlights: 'Uploads resume. The v1 endpoint is removed.', notes: { a: 5, b: 2 } };
  const local = mergeModel();
  const traces = [];
  const merge = await createNatlangRuntime({ model: local.driver, trace: trace => traces.push(trace) }).run(() => semanticMerge(base, ours, theirs));
  assert.deepEqual(merge.value, { title: 'Sync 3.0', highlights: 'Uploads resume, and search is faster. The v1 endpoint is removed.',
    notes: { a: 5, b: 3 } }, 'disjoint edits merge exactly; the deleted key stays deleted');
  assert.deepEqual(merge.merged.map(item => item.path), [['highlights']]);
  assert.deepEqual(merge.merged[0].intents, { ours: 'The edit adds the faster search.', theirs: 'The edit announces the removed endpoint.' });
  assert.ok(Math.abs(merge.merged[0].compatibility - 0.9) < 1e-9);
  assert.deepEqual(merge.unresolved, []);
  const span = traces.find(trace => trace.definitionId === 'std:semanticMerge');
  const stages = span.events.filter(event => event.kind === 'std_stage');
  assert.deepEqual(stages.map(event => event.stage), ['walk', 'resolve']);
  assert.ok(stages[1].call_ids.length >= 3, 'intentOf twice, then combine, under the resolve stage');
});

test('semanticMerge lists what it cannot merge: incompatible purposes, and a combined value that fails its check', async () => {
  const base = { version: '2.4.0', highlights: 'Uploads resume.' };
  const ours = { version: '2.5.0', highlights: 'Uploads resume, and search is faster.' };
  const theirs = { version: '3.0.0', highlights: 'Uploads resume. The v1 endpoint is removed.' };
  const incompatible = await createNatlangRuntime({ model: mergeModel({ compatible: 0.2 }).driver }).run(() => semanticMerge(base, ours, theirs, { keep: 'theirs' }));
  assert.deepEqual(incompatible.value, theirs, 'keep: theirs where nothing merged');
  assert.deepEqual(incompatible.unresolved.map(item => [item.path.join('.'), item.reason]), [['highlights', 'incompatible'], ['version', 'incompatible']]);
  const checked = await createNatlangRuntime({ model: mergeModel().driver }).run(() => semanticMerge(base, ours, theirs, {
    check: (value, path) => path[0] === 'version' && !/^\d+\.\d+\.\d+$/.test(value) ? [`${value} is not a version.`] : [] }));
  assert.equal(checked.value.version, '2.4.0', 'an unresolved field keeps the base by default');
  const version = checked.unresolved.find(item => item.path[0] === 'version');
  assert.equal(version.reason, 'check'); assert.match(version.problems[0], /is not a version/);
  // A string base merges as one unit.
  const unit = await createNatlangRuntime({ model: mergeModel().driver }).run(() =>
    semanticMerge('Uploads resume.', 'Uploads resume, and search is faster.', 'Uploads resume. The v1 endpoint is removed.'));
  assert.equal(unit.value, 'Uploads resume, and search is faster. The v1 endpoint is removed.');
});

test('values pass through by reference: an opaque state object reaches the fix and comes back unchanged', async () => {
  const state = Object.freeze({ kind: 'opaque', handle: Symbol('soft') });
  let received;
  const result = await createNatlangRuntime({}).run(() => semanticRepair(state, { check: value => value === state && received ? [] : ['again'],
    fix: value => { received = value; return value; }, identity: () => String(Math.random()), attempts: 2 }));
  assert.equal(received, state); assert.equal(result.value, state);
});

const changes = JSON.parse(readFileSync(join(repo, 'examples/release/changes.json'), 'utf8'));

test('examples/release notes: the exact rules repair the draft; the summary revises to a fixed point', async () => {
  const load = await buildExample('release');
  const { draftNotes, summarize, notesProblems } = await load('notes.js');
  assert.deepEqual(notesProblems(['#412 Uploads resume.'], changes.slice(0, 1)), []);
  assert.deepEqual(notesProblems(['#412: Uploads resume.'], changes.slice(0, 1)), [], 'the id followed by a colon (what LFM3 writes) starts the line');
  assert.equal(notesProblems(['Uploads resume.', '#999 x'], changes.slice(0, 1)).length, 3);
  const local = localModel(opening => {
    // The first draft forgets #421; the fix adds it.
    if (opening.includes('Write one line of release notes')) return 'return ["#412 Uploads resume after a dropped connection.", "#418 The /v1/sync endpoint is removed; use /v2/sync."]';
    if (opening.includes('none of problems applies')) return 'return [...notes, "#421 Rebuilding the search index is three times faster."]';
    if (opening.includes('Write the paragraph')) return 'return "Uploads resume and search is faster."';
    if (opening.includes('Revise summary')) return 'return "Uploads resume, search is faster, and /v1/sync is gone: use /v2/sync."';
    return null;
  }, (options, prompt) => /says the same as previous/.test(prompt) ? [0.9, 0.1] : [0.5, 0.5]);
  const runtime = createNatlangRuntime({ model: local.driver });
  const drafted = await runtime.run(() => draftNotes(changes));
  assert.deepEqual(drafted.problems, []);
  assert.equal(drafted.notes.length, 3);
  assert.ok(local.openings.some(opening => opening.includes('No line starts with #421')), 'the fix saw the exact problem');
  const summary = await runtime.run(() => summarize(drafted.notes));
  assert.match(summary, /v2\/sync/);
});

test('examples/release merge: disjoint fields merge exactly; the version conflict is checked', async () => {
  const load = await buildExample('release');
  const { mergeRelease } = await load('merge.js');
  const { base, ours, theirs } = JSON.parse(readFileSync(join(repo, 'examples/release/release.json'), 'utf8'));
  const local = localModel(opening => {
    if (opening.includes('Say what the edit is for')) return 'return "The edit records " + JSON.stringify(after) + "."';
    if (opening.includes('Write the value that makes')) return 'return typeof base === "string" && /^\\d/.test(base) ? "3.0.0" : ours.after + " The legacy /v1/sync endpoint is removed."';
    return null;
  }, (options, prompt) => /one value can carry out both purposes/.test(prompt) ? [0.9, 0.1] : [0.5, 0.5]);
  const merge = await createNatlangRuntime({ model: local.driver }).run(() => mergeRelease(base, ours, theirs));
  assert.equal(merge.value.upgrade, theirs.upgrade, 'only theirs changed the upgrade note');
  assert.equal(merge.value.version, '3.0.0');
  assert.match(merge.value.highlights, /search index.*\/v1\/sync/);
  assert.deepEqual(merge.merged.map(item => item.path[0]).sort(), ['highlights', 'title', 'version'].filter(key => base[key] !== ours[key] && base[key] !== theirs[key] && ours[key] !== theirs[key]));
  assert.deepEqual(merge.unresolved, []);
});
