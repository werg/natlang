import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { tuneProgram } from '../dist/tune/pipeline.js';
import { scaffoldManifest } from '../dist/tune/collect.js';
import { programSites } from '../dist/tune/sites.js';
import { incrementalHistoryArtifacts, incrementalSelector, readCounter, runIncrementalBuilds } from '../dist/tune/incremental.js';
import { currentRelease } from '../dist/adaptation/weights/release.js';
import { scriptedModel } from './support/natlang.mjs';

const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
const respond = opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()';

function project() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-incremental-'));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase.'));
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  const examples = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'].map(value => ({ function: 'outer.nl', input: { value }, expected: value.toUpperCase() }));
  writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
  // A trainer stand-in honoring the plan's pins and warm starts: one artifact per node with a residual at its level,
  // pinned nodes kept exactly, trained nodes new versions; every plan is appended to plans.jsonl.
  writeFileSync(join(root, 'trainer.mjs'), `import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { encodeNativeTensorArtifact, nativeTensorArtifactContentId } from ${JSON.stringify(new URL('../dist/native/neuralese-store.js', import.meta.url).href)};
const args = process.argv.slice(2), planPath = args[args.indexOf('--plan') + 1], out = args[args.indexOf('--out') + 1];
const plan = JSON.parse(readFileSync(planPath, 'utf8'));
appendFileSync(${JSON.stringify(join(root, 'plans.jsonl'))}, JSON.stringify(plan) + '\\n');
const rows = readFileSync(plan.rows, 'utf8').split('\\n').filter(Boolean).map(line => JSON.parse(line));
const ranks = plan.recipe.ranks ?? { tree: 8, folder: 4, lambda: 2 };
const used = new Set(plan.manifest.bindings.flatMap(binding => binding.ancestry));
const version = createHash('sha256').update(JSON.stringify([planPath, rows.length])).digest('hex').slice(0, 8);
const artifacts = plan.manifest.tree.filter(node => used.has(node.id) && ranks[node.level] > 0).map(node => {
  if (plan.pinned?.[node.id]) return { node: node.id, artifact: plan.pinned[node.id], rank: ranks[node.level], parameterization: 'native-subspace', requires: [] };
  const metadata = { schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'parameter-set', native_identity: { ...plan.manifest.base,
    tokenizer_fingerprint: /^[0-9a-f]{64}$/.test(plan.manifest.base.tokenizer_fingerprint) ? plan.manifest.base.tokenizer_fingerprint : 'd'.repeat(64) },
    source_artifact_id: null, provenance: { fixture: true, node: node.id, version }, tensors: [{ name: 'weight', dtype: 'F32', shape: [1] }] };
  const tensors = { weight: { dtype: 'F32', shape: [1], data: Uint8Array.from([0, 0, 128, 63]) } };
  const artifact = nativeTensorArtifactContentId(metadata, tensors);
  mkdirSync(out + '/artifacts', { recursive: true });
  writeFileSync(out + '/artifacts/' + artifact + '.nnt', encodeNativeTensorArtifact({ metadata, tensors }));
  return { node: node.id, artifact, rank: ranks[node.level], parameterization: 'native-subspace', requires: [] };
});
const optimizer_states = Object.fromEntries(artifacts.filter(item => !plan.pinned?.[item.node]).map(item => [item.node, 'opt:' + item.artifact]));
writeFileSync(out + '/result.json', JSON.stringify({ artifacts, receipt: { rows: rows.length, optimizer_states,
  bindings: [...new Set(rows.map(row => row.binding))].sort() } }));`);
  return root;
}
const programOf = root => {
  const built = host.buildProject({ project: root, emit: false, write: false });
  assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
  return built.manifest.adaptation;
};
const optionsFor = (root, settings) => ({ root, program: programOf(root), settings, teacher: scriptedModel(respond).driver, teacherId: 'teacher:test',
  student: { name: 'lfm', base, checkpoint: '/unused', trainer: [process.execPath, join(root, 'trainer.mjs')], moduleMap: 'lfm2', baseGgufSha256: 'f'.repeat(64) },
  serve: async ({ manifest }) => ({ driver: scriptedModel(respond).driver, loaded: new Set(manifest.artifacts.map(item => item.artifact)), close: async () => {} }) });
const plans = root => readFileSync(join(root, 'plans.jsonl'), 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
/** What the CLI does for `--incremental` (cli/tune.ts): scaffold the edited program, then run the next builds. */
function incremental(root, settings, full = false) {
  const program = programOf(root);
  const current = scaffoldManifest(program, base, { sites: programSites(root, program).sites });
  return runIncrementalBuilds(root, current, program.buildHash, { full, settings: settings.incremental, recipe: { seed: 0, ...settings.recipe } },
    step => { const options = optionsFor(root, settings); return tuneProgram(options, incrementalSelector(options, step)); });
}
const nodeOf = (manifest, suffix) => manifest.tree.find(node => node.level === 'lambda' && node.path.endsWith(suffix)).id;

test('a leaf edit with lambda-only residuals builds one candidate: only the leaf trains, against pinned released nodes', async () => {
  const root = project();
  const settings = { student: 'lfm', students: {}, teacher: 'teacher', backend: 'managed-local', recipe: { ranks: { tree: 0, folder: 0, lambda: 2 } } };
  try {
    const first = await tuneProgram(optionsFor(root, settings));
    assert.equal(first.activated, true, first.reason);
    assert.equal(await incremental(root, settings), null, 'nothing changed');
    const before = currentRelease(root).manifest;
    writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase letters.'));
    const reports = await incremental(root, settings);
    assert.deepEqual(reports.map(report => [report.build, report.activated]), [['candidate', true]]);
    const plan = plans(root).at(-1), inner = nodeOf(before, 'inner.nl'), outer = nodeOf(before, 'outer.nl');
    const history = incrementalHistoryArtifacts(root, { manifest: before, directory: join(root, '.natlang', 'tune', 'releases', before.digest) });
    assert.ok(plan.artifacts.includes(history[0]), 'warm optimizer history comes from the verified fit output');
    const outerArtifact = before.artifacts.find(item => item.node === outer).artifact;
    assert.deepEqual(plan.pinned, { [outer]: outerArtifact });
    assert.deepEqual(Object.keys(plan.warm), [inner]);
    assert.equal(plan.warm[inner].artifact, before.artifacts.find(item => item.node === inner).artifact);
    assert.equal(plan.warm[inner].optimizer, 'opt:' + plan.warm[inner].artifact, 'moments offered from the released fit');
    const innerBinding = before.bindings.find(entry => entry.ancestry.at(-1) === inner).id;
    const rows = readFileSync(plan.rows, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
    assert.ok(rows.length && rows.every(row => row.binding === innerBinding), 'only rows whose stack trains');
    const after = currentRelease(root).manifest;
    assert.equal(after.receipts.build, 'candidate');
    assert.equal(after.receipts.pinnedFrom, before.digest);
    assert.deepEqual(after.receipts.stalePins, {}, 'no residual-bearing shared parent was dirty');
    assert.equal(after.artifacts.find(item => item.node === outer).artifact, outerArtifact, 'the pinned node is the exact released artifact');
    assert.equal(readCounter(root).sinceFull, 1);
    const resultPath = join(history[0], '..', 'result.json');
    writeFileSync(resultPath, '{}');
    assert.throws(() => incrementalHistoryArtifacts(root, { manifest: before, directory: root }), /receipt mismatch/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('a dirty shared ancestor: candidate with stale pins, then the shared refit seeded by it; the counter forces a full refit', async () => {
  const root = project();
  const settings = { student: 'lfm', students: {}, teacher: 'teacher', backend: 'managed-local', recipe: {}, incremental: { fullRefitEvery: 2 } };
  try {
    assert.equal((await tuneProgram(optionsFor(root, settings))).activated, true);
    const before = currentRelease(root).manifest;
    const inner = nodeOf(before, 'inner.nl'), outer = nodeOf(before, 'outer.nl');
    const released = Object.fromEntries(before.artifacts.map(item => [item.node, item.artifact]));
    writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase letters.'));
    const reports = await incremental(root, settings);
    assert.deepEqual(reports.map(report => [report.build, report.activated]), [['candidate', true], ['shared-refit', true]]);
    const [candidate, refit] = plans(root).slice(-2);
    // The candidate trains the edited leaf; the shared ancestors it sits under stay pinned at their older versions.
    assert.deepEqual(Object.keys(candidate.warm), [inner]);
    assert.deepEqual(Object.keys(candidate.pinned).sort(), ['folder:outer', outer, 'tree:'].sort());
    // The refit trains the dirty ancestors and everything fit against them, the leaf seeded by the candidate.
    assert.deepEqual(refit.pinned ?? {}, {});
    assert.deepEqual(Object.keys(refit.warm).sort(), ['folder:outer', inner, outer, 'tree:'].sort());
    assert.notEqual(refit.warm[inner].artifact, released[inner], 'seeded by the candidate, not the older release');
    assert.equal(refit.warm['tree:'].artifact, released['tree:']);
    const active = currentRelease(root).manifest;
    assert.equal(active.receipts.build, 'shared-refit');
    assert.equal(active.receipts.pinnedFrom, before.digest, 'planned against the release the candidate pinned');
    assert.ok(active.receipts.completes, 'the refit names the candidate it completes');
    assert.deepEqual(active.receipts.stalePins, {});
    assert.equal(readCounter(root).sinceFull, 1, 'a candidate and its shared refit are one incremental build');
    // The next edit reaches fullRefitEvery (2): a full joint refit, no pins, counter reset.
    writeFileSync(join(root, 'outer.nl'), named('Pass value to inner, unchanged.'));
    const next = await incremental(root, settings);
    assert.deepEqual(next.map(report => report.build), ['full']);
    assert.equal(plans(root).at(-1).pinned, undefined);
    assert.equal(readCounter(root).sinceFull, 0);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('an interrupted candidate with stale pins is completed by the next incremental run, not reported fresh', async () => {
  const root = project();
  const settings = { student: 'lfm', students: {}, teacher: 'teacher', backend: 'managed-local', recipe: {} };
  try {
    assert.equal((await tuneProgram(optionsFor(root, settings))).activated, true);
    const before = currentRelease(root).manifest;
    writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase letters.'));
    // Only the candidate step runs (as if the shared refit had failed).
    const program = programOf(root);
    const current = scaffoldManifest(program, base, { sites: programSites(root, program).sites });
    let steps = 0;
    await runIncrementalBuilds(root, current, program.buildHash, { recipe: { seed: 0 } }, async step => {
      if (steps++) return { activated: false, build: step.kind };
      const options = optionsFor(root, settings);
      return tuneProgram(options, incrementalSelector(options, step));
    });
    assert.equal(currentRelease(root).manifest.receipts.build, 'candidate');
    assert.ok(Object.keys(currentRelease(root).manifest.receipts.stalePins).includes('tree:'));
    const reports = await incremental(root, settings);
    assert.deepEqual(reports.map(report => report.build), ['shared-refit']);
    assert.equal(currentRelease(root).manifest.receipts.pinnedFrom, before.digest);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
