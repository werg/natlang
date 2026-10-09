import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime } from '../dist/index.js';
import { RepositoryMigration, migrate } from '../../applications/dist/migration/index.js';
import flow from '../../applications/dist/migration/migrate.nl.js';
const settledCrisp = flow.settledCrisp;
import { scriptedModel } from './support/natlang.mjs';

const source = {
  'lib.mjs': 'export function sum(a, b) { return a + b; }\n',
  'caller.mjs': "import { sum } from './lib.mjs';\nexport const total = sum(2, 3);\n",
  'test.mjs': "import { strict as assert } from 'node:assert';\nimport { total } from './caller.mjs';\nassert.equal(total, 5);\n",
};
// A repository whose scenario already fails at the base: it expects the new name.
const pending = { ...source, 'test.mjs': "import { add } from './lib.mjs';\nimport { strict as assert } from 'node:assert';\nassert.equal(add(2, 3), 5);\n" };
const CHECKS = [{ id: 'scenario', argv: [process.execPath, '--test', 'test.mjs'] }];

// What an interpreter of each stage might write: the algorithm its instructions spell out, over the same data.
const STAGES = [
  ['Migrate the repository as request says', 'driver'], ['Restate request for the later stages', 'understand'],
  ['Find the sites of intent at revision', 'survey'], ['Locate the site of hit at revision', 'locate'],
  ['Classify the use of intent.old', 'classify'], ['Plan the migration of intent over classified', 'plan'],
  ['Edit the site of classified at revision', 'edit'], ['Write the patches for classified.site', 'patch'],
  ['Check patches, written for classified.site', 'exactjudge'], ['Find the causes of the failure', 'triage'],
  ['Repair the candidate at snapshot.revision', 'repair'], ['Run one round on state', 'round'],
  ['Decide whether the repair loop stops', 'settled'], ['Write the closing words of the migration', 'summarize'],
];

const DEFAULTS = {
  driver: `const intent = await understand(request, snapshot, checks, seeds);
const sites = await survey(intent, snapshot.revision);
const usages = await Promise.all(sites.map(site => classify(intent, site)));
const classified = sites.map((site, i) => ({ site, usage: usages[i] }));
const planned = await plan(intent, classified);
const edits = await Promise.all(classified.filter(c => planned.edits.includes(c.site.id)).map(c => edit(intent, c, snapshot.revision)));
for (const l of planned.leave) edits.push({ site: l.site, usage: classified.find(c => c.site.id === l.site).usage, patches: [], status: 'left', note: l.reason });
const patches = planned.edits.flatMap(id => edits.find(e => e.site === id).patches);
let state = { snapshot, validation: null, rejected: '', findings: [], remaining: attempts, revisions: [snapshot.revision] };
if (!patches.length) state = { ...state, rejected: 'no site produced a patch: ' + edits.filter(e => e.status === 'failed').map(e => e.note).join('; ') };
else {
  try {
    const candidate = repository.apply(snapshot.revision, patches);
    const validation = await repository.validate(candidate.revision);
    state = { ...state, snapshot: candidate, validation, revisions: [snapshot.revision, candidate.revision] };
  } catch (error) { state = { ...state, rejected: error.message }; }
}
const stop = repository.implementation('settled') === 'crisp' ? settledCrisp : settled;
let final;
try { final = await round.iterateOn(state, intent).withMeasure(s => s.remaining).until(stop); }
catch (error) { if (error.name === 'IterationLimitError') final = error.lastState; else throw error; }
const validation = final.validation ?? await repository.validate(final.snapshot.revision);
const report = repository.report(final.snapshot.revision, validation);
const closing = await summarize(intent, planned, edits, final, report);
return { report, intent, plan: planned, edits, findings: final.findings, rounds: attempts - final.remaining, summary: closing.summary, next: closing.next };`,
  understand: `return { summary: 'rename sum to add', old: 'sum', new: 'add', queries: [...new Set([...seeds, 'sum'])], invariants: ['total stays 5'] };`,
  survey: `const seen = new Set(); const hits = [];
for (const q of intent.queries) for (const h of repository.search(q, revision).hits) { const k = h.path + ':' + h.offset; if (!seen.has(k)) { seen.add(k); hits.push(h); } }
hits.sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : a.line - b.line);
const firsts = hits.filter((h, i) => i === 0 || hits[i - 1].path !== h.path || hits[i - 1].line !== h.line);
const sites = await Promise.all(firsts.map(h => locate(intent, h, revision)));
const merged = [];
for (const s of sites) { const p = merged.at(-1);
  if (p && p.path === s.path && s.from <= p.to) { p.to = Math.max(p.to, s.to); p.text = repository.lines(p.path, p.from, p.to, revision); p.hits += s.hits; } else merged.push({ ...s }); }
return merged.map((s, i) => ({ ...s, id: 's' + (i + 1) }));`,
  locate: `return { id: '', path: hit.path, from: hit.line, to: hit.line, text: repository.lines(hit.path, hit.line, hit.line, revision), hits: 1 };`,
  classify: `const t = site.text;
const pattern = /^\\s*import /.test(t) ? 'import' : /function\\s+sum\\b/.test(t) ? 'declaration' : /sum\\(/.test(t) ? 'call' : /assert/.test(t) ? 'test' : 'unrelated';
return { pattern, action: pattern === 'unrelated' && !evidence ? 'leave' : 'edit', reason: pattern + ' at ' + site.path + ':' + site.from };`,
  plan: `const rank = { declaration: 0, export: 1, import: 2, call: 5, test: 6 };
const edits = classified.filter(c => c.usage.action === 'edit').sort((a, b) => (rank[a.usage.pattern] ?? 9) - (rank[b.usage.pattern] ?? 9) || (a.site.path < b.site.path ? -1 : 1)).map(c => c.site.id);
return { edits, leave: classified.filter(c => c.usage.action !== 'edit').map(c => ({ site: c.site.id, reason: c.usage.reason })), risks: [] };`,
  edit: `if (classified.usage.action === 'leave') return { site: classified.site.id, usage: classified.usage, patches: [], status: 'left', note: classified.usage.reason };
let patches = await patch(intent, classified, revision, problem);
let verdict = await exact(patches, classified, revision);
if (!verdict.exact) { patches = await patch(intent, classified, revision, verdict.problem); verdict = await exact(patches, classified, revision); }
return verdict.exact ? { site: classified.site.id, usage: classified.usage, patches, status: 'patched', note: classified.usage.reason }
  : { site: classified.site.id, usage: classified.usage, patches: [], status: 'failed', note: verdict.problem };`,
  patch: `const text = classified.site.text, path = classified.site.path, patches = [];
for (const m of [...text.matchAll(/\\bsum\\b/g)]) {
  let a = m.index, b = m.index + 3;
  for (let i = 0; i < text.length && repository.count(path, text.slice(a, b), revision) !== 1; i++) { a = Math.max(0, a - 1); b = Math.min(text.length, b + 1); }
  patches.push({ path, old: text.slice(a, b), new: text.slice(a, m.index) + 'add' + text.slice(m.index + 3, b) });
}
return patches;`,
  exactjudge: `if (!patches.length) return { exact: false, problem: 'no patch for the site' };
for (const p of patches) {
  if (p.path !== classified.site.path || !p.old || p.old === p.new) return { exact: false, problem: 'bad patch in ' + p.path };
  const found = repository.count(p.path, p.old, revision);
  if (found !== 1) return { exact: false, problem: 'old text occurs ' + found + ' times in ' + p.path };
}
return { exact: true, problem: '' };`,
  triage: `const out = [];
if (rejected) out.push({ check: 'apply', kind: 'wrong-edit', path: '', line: 0, evidence: rejected, repairable: true });
for (const c of validation?.checks ?? []) if (c.status === 'failed') {
  const m = [...c.output.matchAll(/([\\w-]+\\.mjs):(\\d+)/g)].find(x => snapshot.files.some(f => f.path === x[1] && x[1] !== 'test.mjs'));
  out.push({ check: c.id, kind: 'missed-site', path: m ? m[1] : '', line: m ? Number(m[2]) : 0, evidence: c.output.slice(-300), repairable: true });
}
return out;`,
  repair: `const lists = await Promise.all(findings.map(async finding => {
  const hit = finding.path ? { path: finding.path, offset: 0, line: Math.max(1, finding.line), excerpt: '' } : repository.search(intent.old, snapshot.revision).hits[0];
  if (!hit) return [];
  const site = { ...(await locate(intent, hit, snapshot.revision)), id: 'r' };
  const usage = { ...(await classify(intent, site, finding.evidence)), action: 'edit' };
  return (await edit(intent, { site, usage }, snapshot.revision, finding.evidence)).patches;
}));
const seen = new Set(), out = [];
for (const p of lists.flat()) { const k = p.path + '|' + p.old; if (!seen.has(k)) { seen.add(k); out.push(p); } }
return out;`,
  round: `const findings = await triage(intent, state.snapshot, state.validation, state.rejected);
const repairable = findings.filter(f => f.repairable);
if (!repairable.length) return { ...state, findings, remaining: state.remaining - 1 };
const patches = await repair(intent, state.snapshot, repairable);
if (!patches.length) return { ...state, findings, rejected: 'the repair wrote no patch for the findings', remaining: state.remaining - 1 };
try {
  const candidate = repository.apply(state.snapshot.revision, patches);
  const validation = await repository.validate(candidate.revision);
  return { snapshot: candidate, validation, rejected: '', findings, remaining: state.remaining - 1, revisions: [...state.revisions, candidate.revision] };
} catch (error) { return { ...state, findings, rejected: error.message, remaining: state.remaining - 1 }; }`,
  settled: `return state.validation?.status === 'passed' || state.remaining === 0 || (state.findings.length > 0 && state.findings.every(f => !f.repairable))
  || new Set(state.revisions).size < state.revisions.length;`,
  summarize: `return { summary: report.status + ': ' + report.changed.length + ' files changed', next: state.findings.map(f => f.evidence.slice(0, 40)) };`,
};

async function run({ overrides = {}, policy, attempts = 3, files = Object.keys(source), contents = source, seeds = [] } = {}) {
  const root = await mkdtemp(join(tmpdir(), 'natlang-repo-'));
  for (const [name, text] of Object.entries(contents)) await writeFile(join(root, name), text);
  const repository = new RepositoryMigration(root, { files, checks: CHECKS, policy });
  await repository.open();
  const seen = [], openings = [];
  const model = scriptedModel(opening => {
    const stage = STAGES.find(([phrase]) => opening.includes(phrase))?.[1];
    if (!stage) return null;
    seen.push(stage); openings.push([stage, opening]);
    const script = overrides[stage] ?? DEFAULTS[stage];
    return typeof script === 'function' ? script(opening) : script;
  });
  const runtime = createNatlangRuntime({ model: model.driver });
  const result = await migrate(runtime, repository, 'Rename sum to add while preserving the calculation', { attempts, seeds, checks: ['scenario'] });
  return { result, root, repository, seen, openings };
}
const finish = ({ root }) => rm(root, { recursive: true, force: true });

test('natlang plans, edits site by site, checks and repairs a migration without editing the original repository', async () => {
  // The classifier wrongly leaves the callers alone; the checks fail and the repair, from their output, finds the sites.
  const wrongly = `const t = site.text;
if (!evidence && site.path === 'caller.mjs') return { pattern: 'unrelated', action: 'leave', reason: 'looks like a plain constant' };
return { pattern: /function/.test(t) ? 'declaration' : 'call', action: 'edit', reason: 'uses sum at ' + site.path + ':' + site.from };`;
  const run1 = await run({ overrides: { classify: wrongly } });
  try {
    const { result, root, seen } = run1;
    assert.equal(result.report.status, 'reviewable', JSON.stringify(result.report.checks));
    assert.equal(result.rounds, 2, 'one round per missed site: the import, then the call');
    assert.deepEqual(result.report.changed.map(c => c.path).sort(), ['caller.mjs', 'lib.mjs']);
    assert.equal(result.intent.old, 'sum');
    assert.equal(result.plan.leave.length, 2, 'both caller sites were left in the plan');
    assert.deepEqual(result.edits.map(e => e.status).sort(), ['left', 'left', 'patched']);
    assert.match(result.summary, /reviewable/);
    // Every unit of the first proposal ran once per site, then the loop: triage, repair (locate, classify, edit) per round.
    assert.equal(seen.filter(s => s === 'locate').length, 3 + 2, 'three sites, then one located per repair round');
    assert.equal(seen.filter(s => s === 'triage').length, 2);
    assert.equal(seen.filter(s => s === 'settled').length, 3, 'asked on the first candidate and after each round');
    assert.equal(await readFile(join(root, 'caller.mjs'), 'utf8'), source['caller.mjs']);
    assert.equal(await readFile(join(root, 'lib.mjs'), 'utf8'), source['lib.mjs']);
  } finally { await finish(run1); }
});

test('a complete first candidate passes its checks and the loop never runs a round', async () => {
  const r = await run();
  try {
    assert.equal(r.result.report.status, 'reviewable');
    assert.equal(r.result.rounds, 0);
    assert.equal(r.result.report.changed.length, 2);
    assert.deepEqual(r.result.plan.edits.length, 3);
    assert.ok(!r.seen.includes('round') && !r.seen.includes('triage') && !r.seen.includes('repair'));
    assert.equal(r.seen.filter(s => s === 'settled').length, 1, 'the stopping judgment is checked on the initial state');
    assert.equal(r.seen.filter(s => s === 'patch').length, 3);
  } finally { await finish(r); }
});

test('a patch that is not exact goes back to the writer once with the problem', async () => {
  // The first answer for caller.mjs line 2 is ambiguous; the writer is told how many times the old text occurs.
  const patch = opening => opening.includes('occurs 2 times') ? DEFAULTS.patch
    : `return [{ path: classified.site.path, old: 'sum', new: 'add' }];`;
  for (const policy of [{ exact: 'natural-language' }, { exact: 'crisp' }]) {
    const r = await run({ overrides: { patch }, policy });
    try {
      assert.equal(r.result.report.status, 'reviewable', JSON.stringify(policy));
      const retried = r.openings.filter(([stage, opening]) => stage === 'patch' && opening.includes('occurs 2 times'));
      assert.ok(retried.length >= 1, 'the retry opening carries the exactness problem');
      assert.equal(r.seen.includes('exactjudge'), policy.exact === 'natural-language');
    } finally { await finish(r); }
  }
  // A writer that never gets it right leaves the site failed, with the problem as the note, and nothing is applied.
  const stubborn = await run({ overrides: { patch: `return [{ path: classified.site.path, old: 'sum', new: 'add' }];` }, policy: { exact: 'crisp' }, attempts: 1 });
  try {
    assert.ok(stubborn.result.edits.some(e => e.status === 'failed' && /occurs 2 times/.test(e.note)));
    assert.equal(await readFile(join(stubborn.root, 'lib.mjs'), 'utf8'), source['lib.mjs']);
  } finally { await finish(stubborn); }
});

test('a spent budget keeps the last candidate honestly, however the stopping judgment answers', async () => {
  const nothing = { patch: `return [];` };
  const spent = await run({ overrides: nothing, attempts: 2, contents: pending });
  try {
    assert.equal(spent.result.report.status, 'checks-failed');
    assert.equal(spent.result.rounds, 2);
    assert.deepEqual(spent.result.report.changed, []);
    assert.equal(spent.seen.filter(s => s === 'round').length, 2);
    assert.ok(spent.result.next.length > 0, 'the open findings are in the closing words');
  } finally { await finish(spent); }
  // A judgment that never says stop is ended by the measure.
  const never = await run({ overrides: { ...nothing, settled: `return false;` }, attempts: 2, contents: pending });
  try {
    assert.equal(never.result.report.status, 'checks-failed');
    assert.equal(never.result.rounds, 2);
  } finally { await finish(never); }
});

test('findings that cannot be repaired end the loop at once', async () => {
  const environment = `return [{ check: 'scenario', kind: 'environment', path: '', line: 0, evidence: 'node is not installed', repairable: false }];`;
  const r = await run({ overrides: { patch: `return [];`, triage: environment }, attempts: 5, contents: pending });
  try {
    assert.equal(r.result.rounds, 1);
    assert.equal(r.result.findings[0].kind, 'environment');
    assert.equal(r.result.report.status, 'checks-failed');
  } finally { await finish(r); }
});

test('the stopping point is pluggable: crisp or natural-language, the same four rules', async () => {
  const crisp = await run({ policy: { settled: 'crisp' } });
  try {
    assert.equal(crisp.result.report.status, 'reviewable');
    assert.ok(!crisp.seen.includes('settled'));
  } finally { await finish(crisp); }
  const state = (over = {}) => ({ snapshot: { revision: 'a', files: [] }, validation: null, rejected: '', findings: [], remaining: 2, revisions: ['a'], ...over });
  assert.equal(settledCrisp(state()), false);
  assert.equal(settledCrisp(state({ validation: { revision: 'b', status: 'passed', checks: [] } })), true);
  assert.equal(settledCrisp(state({ remaining: 0 })), true);
  assert.equal(settledCrisp(state({ findings: [{ repairable: false }] })), true);
  assert.equal(settledCrisp(state({ findings: [{ repairable: false }, { repairable: true }] })), false);
  assert.equal(settledCrisp(state({ revisions: ['a', 'b', 'a'] })), true);
});

test('failed checks, stale patch context and exact lines remain visible for repair', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-repo-'));
  for (const [name, text] of Object.entries(source)) await writeFile(join(root, name), text);
  const repository = new RepositoryMigration(root, { files: Object.keys(source), checks: CHECKS });
  try {
    const base = (await repository.open()).revision;
    assert.equal(repository.lines('caller.mjs', 2, 2), 'export const total = sum(2, 3);');
    assert.equal(repository.count('caller.mjs', 'sum'), 2);
    assert.throws(() => repository.lines('caller.mjs', 0, 1), /invalid line range/);
    assert.equal(repository.implementation('exact'), 'natural-language');
    assert.throws(() => repository.implementation('speed'), /unknown policy point/);
    const broken = repository.apply(base, [{ path: 'lib.mjs', old: 'function sum(', new: 'function add(' }]);
    const validation = await repository.validate(broken.revision);
    assert.equal(validation.status, 'failed', JSON.stringify(validation));
    assert.equal(repository.report(broken.revision, validation).status, 'checks-failed');
    const repaired = repository.apply(broken.revision, [
      { path: 'caller.mjs', old: '{ sum }', new: '{ add }' },
      { path: 'caller.mjs', old: 'sum(2, 3)', new: 'add(2, 3)' },
    ]);
    assert.equal((await repository.validate(repaired.revision)).status, 'passed');
    assert.throws(() => repository.apply(broken.revision, [{ path: 'lib.mjs', old: 'function sum(', new: 'function add(' }]), /context missing: lib.mjs/);
    assert.throws(() => repository.apply(base, [{ path: 'caller.mjs', old: 'sum', new: 'add' }]), /context ambiguous: caller.mjs.*neighboring text/);
    assert.throws(() => repository.apply(base, [{ path: 'other.mjs', old: 'a', new: 'b' }]), /not a file of the manifest/);
  } finally { await rm(root, { recursive: true, force: true }); }
});

test('long checks stream output and retain a bounded diagnostic tail', async () => {
  const root = await mkdtemp(join(tmpdir(), 'natlang-repo-'));
  await writeFile(join(root, 'lib.mjs'), source['lib.mjs']);
  const repository = new RepositoryMigration(root, { files: ['lib.mjs'], checks: [
    { id: 'verbose', argv: [process.execPath, '-e', "process.stdout.write('x'.repeat(2_000_000))"] },
  ] });
  try {
    const snapshot = await repository.open();
    const checked = await repository.validate(snapshot.revision);
    assert.equal(checked.status, 'passed');
    assert.equal(checked.checks[0].output_bytes, 2_000_000);
    assert.equal(checked.checks[0].truncated, true);
    assert.equal(checked.checks[0].output.length, 4000);
  } finally { await rm(root, { recursive: true, force: true }); }
});
