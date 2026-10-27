import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { activeNativeAdapters } from '../dist/native/model-recording.js';
import { tuneProgram, prepareData, evaluateRelease, saveArm } from '../dist/tune/pipeline.js';
import { runPlacement, deepestFirst } from '../dist/tune/placement.js';
import { runSweep } from '../dist/tune/sweep.js';
import { pairedComparison } from '../dist/tune/stats.js';
import { loadOverview, formatOverview } from '../dist/tune/report.js';
import { protectedExamples, readRootExamples } from '../dist/tune/collect.js';
import { scriptedModel } from './support/natlang.mjs';

const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 'd'.repeat(64), embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
const words = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta', 'iota', 'kappa', 'lambda', 'mu', 'nu', 'xi', 'omicron', 'pi'];

function project() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-tune-place-'));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase.'));
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  const examples = words.map(value => ({ function: 'outer.nl', input: { value }, expected: value.toUpperCase() }));
  examples.push({ function: 'outer.nl', input: { value: 'held' }, expected: 'HELD', protected: true });
  writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
  // A trainer stand-in honouring the recipe's per-level ranks (0: no adapter) and removed (scale 0) nodes; it counts
  // its invocations so resumption can be checked.
  writeFileSync(join(root, 'trainer.mjs'), `import { readFileSync, writeFileSync, appendFileSync } from 'node:fs';
import { writeNativeTuneTensor } from ${JSON.stringify(new URL('./fixtures/native-tune.mjs', import.meta.url).href)};
const args = process.argv.slice(2), plan = JSON.parse(readFileSync(args[args.indexOf('--plan') + 1], 'utf8')), out = args[args.indexOf('--out') + 1];
appendFileSync(${JSON.stringify(join(root, 'trainer-runs.log'))}, 'run\\n');
const rows = readFileSync(plan.rows, 'utf8').split('\\n').filter(Boolean).map(line => JSON.parse(line));
if (rows.some(row => row.partition !== 'train')) throw new Error('non-training row reached the trainer');
const ranks = { tree: 8, folder: 4, lambda: 2, ...(plan.recipe.ranks ?? {}) };
const artifacts = plan.manifest.tree.filter(node => ranks[node.level] > 0 && (plan.manifest.scales[node.id] ?? 1) !== 0)
  .map(node => ({ node: node.id, artifact: writeNativeTuneTensor(out, plan.manifest.base, [node.id, ranks]), rank: ranks[node.level], parameterization: 'native-subspace', requires: [] }));
writeFileSync(out + '/result.json', JSON.stringify({ artifacts, receipt: { rows: rows.length } }));`);
  return root;
}
const programOf = root => {
  const built = host.buildProject({ project: root, emit: false, write: false });
  assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
  return built.manifest.adaptation;
};
const settings = { student: 'lfm', students: {}, teacher: 'teacher', backend: 'managed-local' };
const student = root => ({ name: 'lfm', base, checkpoint: '/unused', trainer: [process.execPath, join(root, 'trainer.mjs')], moduleMap: 'lfm2' });
/** The stand-in student answers correctly only when its call runs a lambda-level adapter: tree and folder residuals
 * add cost but no measured quality, so placement should remove them and keep the lambdas. */
function model(manifest) {
  let active = [];
  const lambdaArtifacts = new Set(manifest.artifacts.filter(item => item.node.startsWith('lambda:')).map(item => item.artifact));
  const scripted = scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)'
    : active.some(id => lambdaArtifacts.has(id)) ? 'return value.toUpperCase()' : 'return value');
  return async (request, ...rest) => { active = activeNativeAdapters().map(item => item.id); return scripted.driver(request, ...rest); };
}
const teacher = scriptedModel(opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()').driver;
const serve = async ({ manifest }) => ({ driver: model(manifest), loaded: new Set(manifest.artifacts.map(item => item.artifact)), close: async () => {} });
const optionsFor = root => ({ root, program: programOf(root), settings: { ...settings, placement: { maxSuccessLoss: 0, costWeights: { rank: 1, bytes: 0, stack: 1, latencyMs: 0 } } },
  student: student(root), teacher, teacherId: 'teacher:scripted', serve });

test('cluster-paired comparisons resample components, are reproducible and flag few clusters', () => {
  const arm = (name, outcome) => ({ schema: 'natlang.tune-arm/v1', arm: name, release: null, partition: 'development',
    roots: [0, 1, 2, 3, 4, 5].map(index => ({ id: 'r' + index, component: 'c' + (index % 3), oracle: true, correct: outcome(index), ok: true, stacks: [] })) });
  const a = arm('untuned', () => false), b = arm('tuned', index => index % 3 !== 0);
  const first = pairedComparison(a, b), second = pairedComparison(a, b);
  assert.deepEqual(first, second);
  assert.equal(first.clusters, 3); assert.equal(first.pairs, 6);
  assert.ok(Math.abs(first.effect - 4 / 6) < 1e-12);
  assert.ok(first.ci[0] <= first.effect && first.effect <= first.ci[1]);
  assert.equal(first.underpowered, true);
});

test('placement refits deepest-first removals on development roots and keeps the cheapest stack within the envelope', async () => {
  const root = project();
  try {
    const options = optionsFor(root);
    const report = await tuneProgram(options);
    assert.equal(report.activated, true, report.reason);
    const data = await prepareData(options);
    const reference = host.currentManifest(root);
    assert.deepEqual(deepestFirst(reference, new Set()).slice(0, 1), [reference.tree.find(node => node.path === 'outer/inner.nl').id]);
    const placed = await runPlacement(options, data, reference);
    // Only the inner lambda's residual carries measured quality in the stand-in; everything else is removable cost.
    const innerLambda = reference.tree.find(node => node.path === 'outer/inner.nl').id;
    assert.deepEqual([...placed.selected.removed].sort(), reference.artifacts.map(item => item.node).filter(node => node !== innerLambda).sort());
    assert.ok(placed.selected.cost.total < placed.frontier[0].cost.total);
    assert.ok(placed.frontier.some(item => item.removed.some(node => node.startsWith('lambda:')) && !item.withinEnvelope),
      'removing a lambda adapter lost measured success and stayed out of the envelope');
    assert.ok(placed.frontier.every(item => item.kind === 'unpruned' || item.comparison.pairs > 0));
    // Subtree promotion proposals are evaluated as candidates of their own.
    const promoted = await runPlacement(options, data, reference, { treeRoots: [['outer']], propose: () => [] });
    assert.equal(promoted.frontier.at(-1).kind, 'tree-roots');
    assert.ok(existsSync(join(root, '.natlang/tune/placement', reference.digest + '.json')));
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('sweeps journal structural arms by independent roots, resume, and select on development data', async () => {
  const root = project();
  try {
    const options = optionsFor(root);
    const data = await prepareData(options);
    const spec = { name: 'capacity', recipes: [{ name: 'default', recipe: {} }], sizes: [2, 'full', 1000],
      arms: ['hierarchy', 'flat-lambda', 'program-single', 'full-finetune'] };
    const first = await runSweep(options, data, spec);
    const status = (arm, size) => first.cells.find(cell => cell.arm === arm && cell.size === size).status;
    assert.equal(status('hierarchy', 2), 'done');
    assert.equal(status('hierarchy', 1000), 'skipped');
    assert.equal(status('full-finetune', 'full'), 'unsupported');
    assert.equal(first.cells.find(cell => cell.arm === 'program-single' && cell.size === 'full').development.success, 0,
      'one program adapter without lambda adapters is measured, not assumed');
    assert.equal(first.selected.arm, 'flat-lambda', 'equal success, lower serving cost');
    assert.equal(first.cells.find(cell => cell.arm === 'hierarchy' && cell.size === 2).roots, 2);
    const runs = readFileSync(join(root, 'trainer-runs.log'), 'utf8').split('\n').filter(Boolean).length;
    const again = await runSweep(options, data, spec);
    assert.equal(readFileSync(join(root, 'trainer-runs.log'), 'utf8').split('\n').filter(Boolean).length, runs, 'resumed without retraining');
    assert.equal(again.cells.length, first.cells.length);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('evaluation arms pair by root and the report shows strata, comparisons and author actions', async () => {
  const root = project();
  try {
    const options = optionsFor(root);
    assert.equal((await tuneProgram(options)).activated, true);
    const data = await prepareData(options);
    const development = data.examples.filter(example => example.partition === 'development');
    saveArm(root, await evaluateRelease(options, data, null, development, 'untuned', 'development'));
    saveArm(root, await evaluateRelease(options, data, host.currentManifest(root), development, 'hierarchy', 'development'));
    const overview = loadOverview(root, 'development');
    const hierarchy = overview.arms.find(arm => arm.arm === 'hierarchy'), untuned = overview.arms.find(arm => arm.arm === 'untuned');
    assert.equal(untuned.success, 0);
    assert.equal(hierarchy.success, 1);
    assert.equal(hierarchy.versusUntuned.pairs, development.length);
    assert.equal(hierarchy.degradedCalls, 0);
    const text = formatOverview(overview);
    assert.match(text, /hierarchy \| \d+ \| 100\.0%/);
    assert.match(text, /manual author actions: 0/);
    const held = protectedExamples(readRootExamples(root));
    assert.equal(held.length, 1);
    const heldArm = await evaluateRelease(options, data, host.currentManifest(root), held, 'hierarchy', 'protected');
    assert.equal(heldArm.roots[0].partition, 'protected');
  } finally { rmSync(root, { recursive: true, force: true }); }
});
