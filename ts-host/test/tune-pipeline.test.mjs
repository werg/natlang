import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, existsSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as host from '../dist/index.js';
import { activeNativeAdapters } from '../dist/native/model-recording.js';
import { tuneProgram } from '../dist/tune/pipeline.js';
import { splitExamples } from '../dist/tune/collect.js';
import { openAICompatibleModelTurn } from '../dist/model/openai-compatible.js';
import { scriptedModel } from './support/natlang.mjs';

const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
const respond = opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()';

function project() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-tune-'));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase.'));
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  const examples = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta'].map(value => ({ function: 'outer.nl', input: { value }, expected: value.toUpperCase() }));
  examples.push({ function: 'outer.nl', input: { value: 'held' }, expected: 'HELD', protected: true });
  writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
  // A trainer stand-in: one rank-2 artifact per lambda node, recorded exactly as the real trainer reports them.
  writeFileSync(join(root, 'trainer.mjs'), `import { readFileSync, writeFileSync } from 'node:fs';
const args = process.argv.slice(2), plan = JSON.parse(readFileSync(args[args.indexOf('--plan') + 1], 'utf8')), out = args[args.indexOf('--out') + 1];
const rows = readFileSync(plan.rows, 'utf8').split('\\n').filter(Boolean).map(line => JSON.parse(line));
if (rows.some(row => row.partition !== 'train')) throw new Error('non-training row reached the trainer');
const artifacts = plan.manifest.tree.filter(node => node.level === 'lambda').map(node => ({ node: node.id, artifact: 'nnt1:' + node.id, rank: 2, parameterization: 'native-subspace', requires: [] }));
writeFileSync(out + '/result.json', JSON.stringify({ artifacts, receipt: { rows: rows.length, bindings: [...new Set(rows.map(row => row.binding))].length } }));
console.log('progress: trained');`);
  return root;
}
const program = root => {
  const built = host.buildProject({ project: root, emit: false, write: false });
  assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
  return built.manifest.adaptation;
};
const settings = { student: 'lfm', students: {}, teacher: 'teacher', backend: 'managed-local' };
const student = root => ({ name: 'lfm', base, checkpoint: '/unused', trainer: [process.execPath, join(root, 'trainer.mjs')], moduleMap: 'lfm2', baseGgufSha256: 'f'.repeat(64) });
/** A backend stand-in that serves the release and records which adapters each request carried. */
function serve(seen) {
  return async ({ manifest }) => {
    const model = scriptedModel(respond);
    const driver = async (request, ...rest) => { seen.push(activeNativeAdapters().map(item => item.id)); return model.driver(request, ...rest); };
    return { driver, loaded: new Set(manifest.artifacts.map(item => item.artifact)), close: async () => {} };
  };
}

test('splits happen per component before expansion and protected roots are excluded', () => {
  const examples = [{ function: 'f.nl', input: 1, group: 'w1' }, { function: 'f.nl', input: 2, group: 'w1' },
    { function: 'f.nl', input: 3, group: 'w2' }, { function: 'f.nl', input: 4, protected: true }];
  const split = splitExamples(examples);
  assert.equal(split.length, 3);
  assert.equal(split[0].partition, split[1].partition, 'one world stays on one side');
  assert.ok(split.some(item => item.partition === 'development'));
  assert.deepEqual([...splitExamples(examples)], [...split], 'deterministic');
  // Adding examples never moves an assigned component; the receipt carries the assignment forward.
  const more = [...examples, ...Array.from({ length: 20 }, (_, index) => ({ function: 'f.nl', input: 100 + index }))];
  const extended = splitExamples(more, split.receipt);
  for (const item of split) assert.equal(extended.find(other => other.id === item.id).partition, item.partition);
  assert.ok(new Set(extended.map(item => item.partition)).size >= 2);
});

test('natlang tune collects per-binding trajectories, trains, validates through the engine and activates', async () => {
  const root = project();
  try {
    const seen = [], teacherCalls = [];
    const teacher = scriptedModel(opening => { teacherCalls.push(opening); return respond(opening); });
    const options = { root, program: program(root), settings, student: student(root), teacher: teacher.driver, teacherId: 'teacher:test', serve: serve(seen) };
    const report = await tuneProgram(options);
    assert.equal(report.activated, true, report.reason);
    assert.equal(report.data.protected, 1);
    assert.ok(!teacherCalls.some(opening => opening.includes('held')), 'the teacher never saw a protected root');
    const rootsDir = join(root, '.natlang/tune/data/roots');
    const turns = readdirSync(rootsDir).flatMap(file => JSON.parse(readFileSync(join(rootsDir, file), 'utf8')).turns);
    assert.equal(new Set(turns.map(turn => turn.binding)).size, 2, 'outer and inner turns carry their own bindings');
    assert.ok(turns.every(turn => turn.evidence.enclosingSuccess && turn.evidence.rootOracle === true));
    assert.ok(seen.some(adapters => adapters.length === 1), 'validation calls ran with their own lambda stack');
    assert.equal(report.validation.tuned.oracle, report.validation.tuned.roots);
    const active = host.readActive(root);
    assert.equal(active.release, report.release);
    assert.equal(host.tuneStatus(root, { program: program(root) }).status, 'ready');
    // Ordinary execution now discovers the release with no further author action.
    const weights = host.activeWeights(root, { program: program(root) });
    assert.equal(weights()[0].release, report.release);
    // A rerun reuses the collected data and the training receipt.
    const again = await tuneProgram({ ...options, teacher: () => { throw new Error('teacher must not be asked again'); } });
    assert.equal(again.activated, true, again.reason);
    assert.deepEqual(again.timings.filter(item => item.reused).map(item => item.stage), ['data', 'train']);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('a failed validation keeps the previous release active and says why', async () => {
  const root = project();
  try {
    const options = { root, program: program(root), settings, student: student(root), teacher: scriptedModel(respond).driver, teacherId: 'teacher:test', serve: serve([]) };
    const first = await tuneProgram(options);
    assert.equal(first.activated, true, first.reason);
    writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase, trimmed.'));
    const failing = await tuneProgram({ ...options, program: program(root),
      serve: async () => ({ driver: scriptedModel(respond).driver, loaded: new Set(), close: async () => {} }) });
    assert.equal(failing.activated, false);
    assert.match(failing.reason, /weights-unavailable|tuned stack/);
    assert.equal(host.readActive(root).release, first.release);
    assert.equal(existsSync(join(root, '.natlang/tune/job.lock')), false);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('OpenAI-compatible requests carry the calling invocation\'s tuned fields, turns and coalesced decisions alike', async () => {
  const bodies = [];
  const server = createServer((request, response) => {
    let text = ''; request.on('data', chunk => { text += chunk; });
    request.on('end', () => {
      const body = JSON.parse(text); bodies.push(body);
      response.setHeader('content-type', 'application/json');
      if (body.prompt_logprobs === 0) response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: '' } }], prompt_logprobs: [{ 1: { logprob: -1 } }] }));
      else response.end(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    let tag = 'a';
    const driver = openAICompatibleModelTurn({ endpoint: `http://127.0.0.1:${server.address().port}/v1`, model: 'm', tunedRequest: () => ({ lora: [{ id: 0, scale: tag === 'a' ? 1 : 0 }] }) });
    await driver({ messages: [{ role: 'user', content: 'hi' }], tools: [], seed: null, max_tokens: 4 });
    const first = driver.decide({ messages: [{ role: 'user', content: 'pick' }], options: ['"x"'] });
    tag = 'b';
    const second = driver.decide({ messages: [{ role: 'user', content: 'pick' }], options: ['"y"'] });
    await Promise.all([first, second]);
    assert.deepEqual(bodies[0].lora, [{ id: 0, scale: 1 }]);
    const decisions = bodies.filter(body => body.prompt_logprobs === 0);
    assert.deepEqual(decisions.map(body => body.lora[0].scale).sort(), [0, 1], 'each coalesced decision kept its own stack');
  } finally { server.close(); }
});

function readdirOne(path) { return readdirSync(path)[0]; }

test('an edit invalidates exactly its dependency closure and recollects only affected roots', async () => {
  const { invalidation } = await import('../dist/tune/keys.js');
  const { scaffoldManifest } = await import('../dist/tune/collect.js');
  const root = project();
  try {
    // A second root function in its own folder, untouched by the edit below.
    mkdirSync(join(root, 'other'));
    writeFileSync(join(root, 'other', 'echo.nl'), named('Return value in uppercase too.'));
    const examples = readFileSync(join(root, 'tune.examples.jsonl'), 'utf8').trim().split('\n');
    writeFileSync(join(root, 'tune.examples.jsonl'), [...examples, ...['one', 'two', 'three', 'four'].map(value =>
      JSON.stringify({ function: 'other/echo.nl', input: { value }, expected: value.toUpperCase() }))].join('\n') + '\n');
    const asked = [];
    const teacher = scriptedModel(opening => { asked.push(opening); return opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()'; });
    const options = { root, program: program(root), settings, student: student(root), teacher: teacher.driver, teacherId: 'teacher:test', serve: serve([]) };
    const first = await tuneProgram(options);
    assert.equal(first.activated, true, first.reason);
    const before = host.currentManifest(root);
    writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase, please.'));
    const edited = program(root);
    const change = invalidation(before, scaffoldManifest(edited, base));
    const inner = before.bindings.find(entry => entry.component.endsWith('outer/inner.nl::instructions'));
    assert.deepEqual(change.changed, [inner.id]);
    assert.ok(change.dirty.includes('folder:outer') && change.dirty.includes('tree:') && change.dirty.includes(inner.ancestry.at(-1)));
    assert.ok(change.dependents.some(node => node.includes('other')), 'residuals below the dirty root depend on its new version');
    asked.length = 0;
    const second = await tuneProgram({ ...options, program: edited });
    assert.equal(second.activated, true, second.reason);
    assert.ok(asked.length > 0, 'roots that reached the edited lambda were recollected');
    assert.ok(!asked.some(opening => opening.includes('uppercase too')), 'roots that only reached unchanged lambdas reused their data');
  } finally { rmSync(root, { recursive: true, force: true }); }
});
