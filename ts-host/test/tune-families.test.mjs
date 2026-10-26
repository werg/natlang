import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, existsSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { activeNativeAdapters } from '../dist/native/model-recording.js';
import { prepareData, tuneProgram } from '../dist/tune/pipeline.js';
import { currentRelease, readActive } from '../dist/adaptation/weights/release.js';
import { crispProposals, familyRegistry, proposeFamilies, registerTunedProgram, runFamilies } from '../dist/tune/families.js';
import { scriptedModel } from './support/natlang.mjs';

const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
const respond = opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()';

// A stand-in for the student's Python entry: `-m <module>` selects the build, discovery or family fit.
const STUB = `import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
const args = process.argv.slice(2), module = args[args.indexOf('-m') + 1];
const arg = name => args[args.indexOf(name) + 1];
const plan = JSON.parse(readFileSync(arg('--plan'), 'utf8')), out = arg('--out');
mkdirSync(out, { recursive: true });
if (module === 'natlang_neuralese.tune.build') {
  const rows = readFileSync(plan.rows, 'utf8').split('\\n').filter(Boolean).map(line => JSON.parse(line));
  if (rows.some(row => row.partition !== 'train')) throw new Error('non-training row reached the trainer');
  const tag = plan.manifest.families.length ? '+f' : '';
  const artifacts = plan.manifest.tree.filter(node => node.level === 'lambda').map(node => ({ node: node.id, artifact: 'nnt1:' + node.id + tag, rank: 2, parameterization: 'native-subspace', requires: [] }));
  writeFileSync(out + '/result.json', JSON.stringify({ artifacts, families: plan.manifest.families, memberships: plan.manifest.memberships,
    receipt: { rows: rows.length, imported: plan.artifacts ?? [] } }));
} else if (args.includes('discover')) {
  const members = [];
  for (const program of plan.programs) for (const binding of JSON.parse(readFileSync(program.release + '/natlang.adapters.json', 'utf8')).bindings)
    members.push([program.id, binding.id]);
  members.sort();
  writeFileSync(out + '/discovery.json', JSON.stringify({ schema: 'natlang.tune-family-discovery/v1', members,
    similarities: [{ a: members[0], b: members.at(-1), cosine: 0.9 }], clusters: [members], tags: plan.tags }));
} else {
  const rows = readFileSync(plan.rows, 'utf8').split('\\n').filter(Boolean).map(line => JSON.parse(line));
  writeFileSync(out + '/rows-seen.json', JSON.stringify(rows.map(row => [row.binding, row.partition])));
  writeFileSync(out + '/result.json', JSON.stringify({ schema: 'natlang.tune-family-fit/v1', family: plan.family, version: 'v1',
    artifact: 'nnt1:family-' + plan.family, amplitudeHex: '0x1.0000000000000p+0', members: plan.members }));
}`;

function project(name, stub) {
  const root = mkdtempSync(join(tmpdir(), `natlang-family-${name}-`));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase.'));
  writeFileSync(join(root, 'package.json'), JSON.stringify({ name, type: 'module' }));
  const examples = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'].map(value =>
    ({ function: 'outer.nl', input: { value: value + name }, expected: (value + name).toUpperCase() }));
  writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
  const built = host.buildProject({ project: root, emit: false, write: false });
  assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
  const settings = { student: 'lfm', students: {}, teacher: 'teacher', backend: 'managed-local', teacherRouting: { minSuccess: .5 },
    placement: { maxSuccessLoss: .02, costWeights: { rank: 1, bytes: 1e-6, stack: 4, latencyMs: .01 } } };
  const student = { name: 'lfm', base, checkpoint: '/unused', trainer: [process.execPath, stub, '-m', 'natlang_neuralese.tune.build'],
    moduleMap: 'lfm2', baseGgufSha256: 'f'.repeat(64) };
  const seen = [];
  const serve = async ({ manifest }) => {
    const model = scriptedModel(respond);
    const driver = async (request, ...rest) => { seen.push(activeNativeAdapters().map(item => item.id)); return model.driver(request, ...rest); };
    return { driver, loaded: new Set([...manifest.artifacts, ...manifest.families].map(item => item.artifact)), close: async () => {} };
  };
  const teacher = scriptedModel(respond);
  return { root, seen, options: { root, program: built.manifest.adaptation, settings, student, teacher: teacher.driver, teacherId: 'teacher:test', serve } };
}

test('families: discovery across programs, a fit on training rows only, member refits with gates, activation', async () => {
  const scratch = mkdtempSync(join(tmpdir(), 'natlang-family-store-'));
  const stub = join(scratch, 'stub.mjs');
  writeFileSync(stub, STUB);
  const a = project('alpha-prog', stub), b = project('beta-prog', stub), c = project('held-prog', stub);
  try {
    for (const item of [a, b, c]) assert.equal((await tuneProgram(item.options)).activated, true);
    const heldActive = readActive(c.root).release;
    const held = [{ options: c.options, data: await prepareData(c.options), release: currentRelease(c.root).manifest, releaseDir: currentRelease(c.root).directory }];
    const before = [a, b].map(item => readActive(item.root).release);
    const programs = [];
    for (const item of [a, b]) {
      const active = currentRelease(item.root);
      programs.push({ options: item.options, data: await prepareData(item.options), release: active.manifest, releaseDir: active.directory });
    }
    const store = join(scratch, 'store');
    const report = await runFamilies(programs, { store, held });
    assert.equal(report.published.length, 1);
    // The held program never entered discovery or the fit; it is only measured zero-shot, and stays as it was.
    assert.equal(new Set(report.discovery.members.map(([namespace]) => namespace)).size, 2, 'only the two member programs were discovered');
    assert.equal(report.discovery.members.length, 4);
    const transfer = report.transfer.filter(item => item.kind === 'held');
    assert.equal(transfer.length, 1);
    assert.equal(transfer[0].program, c.options.program.id);
    assert.ok(transfer[0].development.roots > 0);
    assert.ok(transfer[0].development.interval.length === 2);
    assert.equal(readActive(c.root).release, heldActive, 'a transfer measurement never activates');
    const composed = JSON.parse(readFileSync(join(c.root, '.natlang/tune/releases', transfer[0].release, 'natlang.adapters.json'), 'utf8'));
    assert.equal(composed.memberships.length, composed.bindings.length, 'the family is applied to every binding at gate 1');
    assert.ok(composed.memberships.every(entry => entry.gate === 1));
    assert.ok(c.seen.some(stack => stack[0] === report.published[0].artifact), 'the held program ran with the family first');
    assert.equal(report.transfer.filter(item => item.kind === 'non-member').length, 0, 'both named programs are members');
    const [family] = report.published;
    assert.equal(family.members.length, 4, 'two bindings in each of two programs');
    assert.ok(family.members.every(member => /^p[0-9a-f]{12}\//.test(member)), 'members are namespaced by program');
    // The family fit saw only training rows, namespaced.
    const jobs = readdirSync(join(store, 'jobs'));
    const seenRows = JSON.parse(readFileSync(join(store, 'jobs', jobs[0], 'rows-seen.json'), 'utf8'));
    assert.ok(seenRows.length > 0 && seenRows.every(([binding, partition]) => partition === 'train' && binding.includes('/')));
    assert.ok(existsSync(join(store, 'versions.json')));
    for (const [index, item] of [a, b].entries()) {
      const outcome = report.programs.find(entry => entry.program === item.options.program.id);
      assert.equal(outcome.activated, true, outcome.reason);
      assert.equal(outcome.previous, before[index]);
      const active = currentRelease(item.root).manifest;
      assert.notEqual(active.digest, before[index], 'the refit is the active release');
      assert.deepEqual(active.families.map(entry => entry.artifact), [family.artifact]);
      assert.equal(active.memberships.length, 2);
      assert.ok(active.memberships.every(entry => entry.gate === 1 && entry.family === family.family));
      assert.equal(readActive(item.root).previous, before[index], 'the previous release stays for rollback');
      // Calls run their family first, then their lexical stack.
      assert.ok(item.seen.some(stack => stack[0] === family.artifact && stack.length > 1));
    }
    // A rerun reuses the published version and the fits.
    const again = await runFamilies(programs.map(program => ({ ...program })), { store });
    assert.equal(again.published[0].version, family.version);
    assert.equal(readdirSync(join(store, 'jobs')).length, jobs.length);
  } finally { for (const root of [a.root, b.root, c.root, scratch]) rmSync(root, { recursive: true, force: true }); }
});

test('the family registry counts newly tuned releases since the last run', () => {
  const store = mkdtempSync(join(tmpdir(), 'natlang-family-registry-'));
  try {
    assert.equal(registerTunedProgram(store, '/p/a', 'r1', 2), false);
    assert.equal(registerTunedProgram(store, '/p/a', 'r1', 2), false, 'the same release counts once');
    assert.equal(registerTunedProgram(store, '/p/b', 'r2', 2), true);
    assert.equal(registerTunedProgram(store, '/p/c', 'r3', undefined), false, 'no schedule configured');
    assert.deepEqual(Object.keys(familyRegistry(store).programs).sort(), ['/p/a', '/p/b', '/p/c']);
  } finally { rmSync(store, { recursive: true, force: true }); }
});

test('proposals: tags first, clusters span programs, a natural-language answer may only name discovered members', async () => {
  const discovery = { schema: 'natlang.tune-family-discovery/v1', members: [['p1', 'x'], ['p1', 'y'], ['p2', 'x'], ['p3', 'z']],
    similarities: [], clusters: [[['p1', 'x'], ['p2', 'x']], [['p1', 'y'], ['p1', 'z']]], tags: { shouting: [['p1', 'y'], ['p3', 'z']] } };
  const crisp = crispProposals(discovery);
  assert.deepEqual(crisp.map(item => item.source), ['tag', 'cluster']);
  assert.equal(crisp[0].name, 'shouting');
  assert.deepEqual(crisp[1].members, [['p1', 'x'], ['p2', 'x']]);
  const picked = await proposeFamilies(discovery, 'nl', () => ({ families: [{ name: 'echo', members: ['p1/x', 'p2/x'] }] }));
  assert.deepEqual(picked, [{ name: 'echo', source: 'nl', members: [['p1', 'x'], ['p2', 'x']] }]);
  const invented = await proposeFamilies(discovery, 'nl', () => ({ families: [{ name: 'echo', members: ['p1/x', 'p9/q'] }] }));
  assert.deepEqual(invented, crisp, 'an unknown member falls back to the crisp proposals');
});
