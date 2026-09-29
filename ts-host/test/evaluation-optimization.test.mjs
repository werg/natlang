import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { loadEvaluationSuite, evaluate, preflightEvaluation, validateCases, validateMetrics } from '../dist/evaluation/index.js';
import { optimize, resumeOptimization, revalidateAdaptation } from '../dist/optimization/index.js';
import { candidateArtifact } from '../dist/evaluation/runner.js';
import { scriptedModel } from './support/natlang.mjs';

function fixtureSuite({ disposeFailure = false } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-adapt-suite-'));
  const evaluationURL = new URL('../dist/evaluation/index.js', import.meta.url).href;
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  writeFileSync(join(root, 'classify.nl'), '---\nargs: { value: string }\nreturns: string\n---\nReturn value.\n');
  writeFileSync(join(root, 'main.ts'), 'import classify from "./classify.nl";\nlet calls = 0;\nexport async function run(value: string): Promise<string> { calls++; if (calls !== 1) throw new Error("state leaked"); return classify(value); }\n');
  writeFileSync(join(root, 'suite.mjs'), `import { defineEvaluationSuite } from ${JSON.stringify(evaluationURL)};
export default defineEvaluationSuite({ id: 'test', program: { root: '.', id: 'fixture', entry: 'main.ts' },
 components: 'lambda-only', executorIdentity: { id: 'scripted', configuration: {} },
 cases: [{ id: 'train', group: 'train-family', split: 'train', input: 'urgent help', expected: 'urgent' },
 { id: 'valid', group: 'valid-family', split: 'validation', input: 'ordinary task', expected: 'ordinary' },
 { id: 'test', group: 'test-family', split: 'test', input: 'held-out text', expected: 'held-out' }],
 budget: { maxRollouts: 30, maxProposals: 2, maxModelCalls: 80 },
 fixture: { async create(testCase, context) { const app = await context.loadFreshProgram(); return {
 execute: () => context.runtime.run(() => app.run(testCase.input)),
 dispose: () => { ${disposeFailure ? 'throw new Error("dispose failed");' : ''} }
 }; } },
 score(testCase, observation) { return { quality: observation.result === testCase.expected ? 1 : 0, gates: { typed: typeof observation.result === 'string' }, feedback: observation.result === testCase.expected ? 'Correct' : 'Only the first word is needed' }; }
});`);
  return { root, path: join(root, 'suite.mjs') };
}

test('suite split leakage and malformed independent metrics are rejected', () => {
  assert.throws(() => validateCases([{ id: 'a', group: 'same', split: 'train' }, { id: 'b', group: 'same', split: 'test' }]), /leakage/);
  assert.throws(() => validateCases([{ id: 'a', group: 'a', split: 'train' }, { id: 'a', group: 'b', split: 'train' }]), /duplicate/);
  assert.throws(() => validateMetrics({ quality: NaN }), /quality/);
  assert.throws(() => validateMetrics({ quality: 1, gates: { safe: 'yes' } }), /boolean/);
  assert.throws(() => validateMetrics({ quality: 1, metrics: { latency: Infinity } }), /finite/);
});

test('evaluation imports application state afresh, scores real runtime execution, and observes coverage/unknown tokens', async () => {
  const fixture = fixtureSuite();
  const suite = await loadEvaluationSuite(fixture.path);
  const model = scriptedModel(() => 'return value.split(" ")[0]');
  const first = await evaluate(suite, { split: 'validation', driver: model.driver });
  const second = await evaluate(suite, { split: 'validation', driver: model.driver, replicate: 1 });
  assert.equal(first.quality, 1); assert.equal(second.quality, 1);
  assert.equal(first.results[0].traces.length, 1);
  assert.ok(suite.components.every(key => first.results[0].coverage.includes(key)));
  assert.ok(first.results[0].coverage.includes('fixture::program.guidance'));
  assert.equal(first.results[0].usage.modelCalls, 2);
  assert.equal(first.results[0].usage.inputTokens, null);
  assert.notEqual(first.results[0].id, second.results[0].id);
  assert.ok(model.openings.every(opening => !opening.includes('Only the first word is needed')), 'scoring feedback is not in executor source');
});

test('fixture disposal and model transport failures remain infrastructure failures', async () => {
  const bad = await loadEvaluationSuite(fixtureSuite({ disposeFailure: true }).path);
  await assert.rejects(() => evaluate(bad, { driver: scriptedModel(() => 'return value').driver }), /dispose failed/);
  const good = await loadEvaluationSuite(fixtureSuite().path);
  await assert.rejects(() => evaluate(good, { driver: () => { throw new Error('provider outage'); } }), error => {
    assert.match(error.message, /provider outage/);
    assert.equal(error.name, 'EvaluationInfrastructureError');
    assert.equal(error.caseId, 'valid');
    assert.equal(error.ledger.usage.modelCalls, 1);
    assert.equal(error.ledger.unknownRequests, 1);
    assert.equal(error.ledger.usage.inputTokens, null);
    return true;
  });
});

test('preflight creates and disposes every fixture; semantic failures reach independent scoring', async () => {
  const prepared = await loadEvaluationSuite(fixtureSuite().path);
  assert.deepEqual(await preflightEvaluation(prepared), { cases: 3, suiteHash: prepared.suiteHash });
  const batch = await evaluate(prepared, { driver: () => ({ calls: [['return_result', { status: 'blocked', reason: 'Missing input' }]] }) });
  assert.equal(batch.results[0].status, 'scored');
  assert.equal(batch.results[0].outcome.kind, 'threw');
  assert.equal(batch.results[0].outcome.error.outcome, 'quiesced');
  assert.match(batch.results[0].outcome.error.message, /Missing input/);
  assert.equal(batch.quality, 0);
  assert.equal(batch.results[0].observation.result, undefined);
});

test('full observations and judge calls cross the worker boundary without live handles', async () => {
  const fixture = fixtureSuite();
  let source = readFileSync(fixture.path, 'utf8');
  source = source.replace('execute: () => context.runtime.run(() => app.run(testCase.input)),',
    `execute: async () => { context.observations.put('effect', { payload: 'x'.repeat(10000) });
      await context.judge({ messages: [], tools: [], seed: 1, max_tokens: 2 });
      return () => 'live function'; }, observe: () => ({ result: 'ordinary' }),`);
  writeFileSync(fixture.path, source);
  const prepared = await loadEvaluationSuite(fixture.path);
  const gateway = new (await import('../dist/evaluation/usage.js')).UsageGateway(prepared.suite.budget);
  const batch = await evaluate(prepared, { gateway, driver: () => { throw new Error('unexpected executor'); },
    judge: () => ({ text: 'accepted', prompt_tokens: 3, completion_tokens: 1 }) });
  assert.equal(batch.quality, 1);
  assert.equal(batch.results[0].observations.effect.payload.length, 10000);
  assert.equal(batch.results[0].outcome.value, undefined);
  assert.equal(gateway.ledger.roles.judge.modelCalls, 1);
  assert.equal(gateway.ledger.roles.executor, undefined);
});

test('worker timeout aborts coordinator requests and retains unknown incurred usage', async () => {
  const prepared = await loadEvaluationSuite(fixtureSuite().path);
  const gateway = new (await import('../dist/evaluation/usage.js')).UsageGateway(prepared.suite.budget);
  await assert.rejects(evaluate(prepared, { gateway, driver: () => new Promise(() => {}), timeoutMs: 2000 }), /timed out/);
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(gateway.ledger.unknownRequests, 1);
  assert.equal(gateway.snapshot().reserved.modelCalls, 0);
});

test('GEPA and reflection search evaluate real runtime, retain baseline, persist and resume without new model calls', async () => {
  for (const strategy of ['gepa', 'reflection']) {
    const fixture = fixtureSuite(); const suite = await loadEvaluationSuite(fixture.path);
    const model = scriptedModel(opening => opening.includes('first word') ? 'return value.split(" ")[0]' : 'return value');
    const key = suite.components[0];
    const reflection = () => ({ text: JSON.stringify({ [key]: 'Return the first word of value.' }), prompt_tokens: 5, completion_tokens: 8 });
    const options = { executor: model.driver, reflection, strategy, seed: 11, out: join(fixture.root, 'run') };
    const result = await optimize(suite, options);
    assert.equal(result.report.baseline.quality, 0);
    assert.equal(result.report.selectedValidation.quality, 1);
    assert.equal(result.report.lockedTest.quality, 1);
    assert.equal(result.artifact.provenance.strategy, strategy);
    const before = model.openings.length;
    const resumed = await resumeOptimization(suite, result.directory, options);
    assert.equal(resumed.artifact.digest, result.artifact.digest);
    assert.equal(model.openings.length, before);
    assert.equal(JSON.parse(readFileSync(join(result.directory, 'checkpoint.json'), 'utf8')).incumbent, result.state.incumbent);
    await assert.rejects(() => resumeOptimization(suite, result.directory, { ...options, seed: 12 }), /fingerprint mismatch/);
  }
});

test('orderly interruption after a committed decision resumes the same scripted incumbent and PRNG state', async () => {
  const fixture = fixtureSuite(), suite = await loadEvaluationSuite(fixture.path), key = suite.components[0];
  const makeModel = () => scriptedModel(opening => opening.includes('first word') ? 'return value.split(" ")[0]' : 'return value');
  const reflection = () => ({ text: JSON.stringify({ [key]: 'Return the first word of value.' }), prompt_tokens: 1, completion_tokens: 1 });
  const whole = await optimize(suite, { executor: makeModel().driver, reflection, seed: 44, out: join(fixture.root, 'whole') });
  const abort = new AbortController();
  const interrupted = await optimize(suite, { executor: makeModel().driver, reflection, seed: 44, out: join(fixture.root, 'interrupted'), signal: abort.signal,
    progress: event => { if (event.type === 'accepted') abort.abort(new Error('test interrupt at boundary')); } });
  assert.equal(interrupted.state.stopReason, 'cancelled');
  assert.equal(interrupted.state.selected, false);
  const resumed = await resumeOptimization(suite, interrupted.directory, { executor: makeModel().driver, reflection, seed: 44 });
  assert.deepEqual(resumed.artifact.components, whole.artifact.components);
  assert.deepEqual(resumed.state.selector, whole.state.selector);
  assert.equal(resumed.state.rng, whole.state.rng);
  assert.equal(resumed.state.incumbent, whole.state.incumbent);
  assert.equal(resumed.state.ledger.usage.modelCalls, whole.state.ledger.usage.modelCalls);
});

test('revalidation evaluates a source-drifted artifact against paired baseline and records fresh provenance', async () => {
  const fixture = fixtureSuite();
  const previous = await loadEvaluationSuite(fixture.path);
  const component = previous.program.components.find(item => item.key === previous.components[0]);
  const value = { kind: 'lambda.instructions', template: { segments: ['Return the first word of value.'], slotIds: [] } };
  const old = candidateArtifact(previous, { [component.key]: value }, 'previous-run');
  writeFileSync(join(fixture.root, 'classify.nl'), '---\nargs: { value: string }\nreturns: string\n---\nReturn the full input unchanged.\n');
  const current = await loadEvaluationSuite(fixture.path);
  assert.notEqual(previous.program.buildHash, current.program.buildHash);
  assert.equal(component.contractHash, current.program.components.find(item => item.key === component.key).contractHash);
  const model = scriptedModel(opening => /first word|first space/.test(opening) ? 'return value.split(" ")[0]' : 'return value');
  const promoted = await revalidateAdaptation(old, current, model.driver);
  assert.equal(promoted.provenance.promotion, 'revalidated');
  assert.equal(promoted.provenance.evidence.previous, old.digest);
  assert.equal(promoted.provenance.evidence.quality, 1);
  assert.notEqual(promoted.provenance.evidence.baseline, promoted.provenance.evidence.validation);
  assert.ok(promoted.provenance.evidence.ledger.usage.modelCalls >= 4, 'baseline and candidate requests share one recorded ledger');
  assert.ok(promoted.provenance.evidence.ledger.roles.executor.modelCalls >= 4);
  assert.equal(promoted.provenance.suiteHash, current.suiteHash);
  assert.notEqual(promoted.digest, old.digest);
});

test('revalidation rejects a paired candidate that fails a required regression gate', async () => {
  const fixture = fixtureSuite();
  const previous = await loadEvaluationSuite(fixture.path);
  const component = previous.program.components.find(item => item.key === previous.components[0]);
  const value = { kind: 'lambda.instructions', template: { segments: ['Return the first word of value.'], slotIds: [] } };
  const old = candidateArtifact(previous, { [component.key]: value }, 'previous-run');
  const drifted = fixtureSuite();
  writeFileSync(join(drifted.root, 'classify.nl'), '---\nargs: { value: string }\nreturns: string\n---\nReturn the full input unchanged.\n');
  let source = readFileSync(drifted.path, 'utf8');
  source = source.replace("expected: 'ordinary'", "expected: 'ordinary task'")
    .replace("gates: { typed: typeof observation.result === 'string' }", "gates: { correct: observation.result === testCase.expected }")
    .replace("budget: { maxRollouts", "requiredGates: ['correct'],\n budget: { maxRollouts");
  assert.match(source, /requiredGates: \['correct'\]/);
  writeFileSync(drifted.path, source);
  const current = await loadEvaluationSuite(drifted.path);
  const model = scriptedModel(opening => opening.includes('first word') ? 'return value.split(" ")[0]' : 'return value');
  const baseline = await evaluate(current, { split: 'validation', driver: model.driver });
  const candidate = await evaluate(current, { split: 'validation', candidate: { [component.key]: value }, driver: model.driver });
  assert.equal(baseline.gatesPassed, true);
  assert.equal(candidate.gatesPassed, false);
  await assert.rejects(() => revalidateAdaptation(old, current, model.driver), /revalidation failed required regression gates/);
});

test('bounded search history retains a complete hash-indexed event log', async () => {
  const fixture = fixtureSuite(), suite = await loadEvaluationSuite(fixture.path), key = suite.components[0];
  const model = scriptedModel(opening => /first word|first space/.test(opening) ? 'return value.split(" ")[0]' : 'return value');
  let proposals = 0;
  const reflection = () => ({ text: JSON.stringify({ [key]: ++proposals === 1 ? 'Return the first word of value.' : 'Return text before the first space.' }), prompt_tokens: 1, completion_tokens: 1 });
  const result = await optimize(suite, { executor: model.driver, reflection, seed: 19, out: join(fixture.root, 'bounded'),
    maxHistory: 1, finalTest: false });
  assert.ok(result.state.history.length <= 1);
  assert.ok(result.state.historyTruncated > 0);
  const hashes = JSON.parse(readFileSync(join(result.directory, 'events.json'), 'utf8'));
  assert.ok(hashes.length > result.state.history.length);
  for (const hash of hashes) assert.ok(readFileSync(join(result.directory, 'blobs', `${hash}.json`), 'utf8').length > 0);
});

test('locked test gate failure keeps the frozen selection and prevents artifact promotion', async () => {
  const fixture = fixtureSuite();
  let source = readFileSync(fixture.path, 'utf8');
  source = source.replace("components: 'lambda-only',", "components: 'lambda-only', requiredGates: ['release'],")
    .replace("gates: { typed: typeof observation.result === 'string' }", "gates: { release: testCase.split !== 'test' }");
  writeFileSync(fixture.path, source);
  const suite = await loadEvaluationSuite(fixture.path);
  const model = scriptedModel(() => 'return value.split(" ")[0]');
  const directory = join(fixture.root, 'heldout-failure');
  const options = { executor: model.driver, reflection: () => { throw new Error('no proposal allowed'); }, out: directory,
    budget: { ...suite.suite.budget, maxProposals: 0 } };
  await assert.rejects(() => optimize(suite, options), /frozen finalist failed locked test gates/);
  const report = JSON.parse(readFileSync(join(directory, 'report.json'), 'utf8'));
  assert.equal(report.selectedValidation.gatesPassed, true);
  assert.equal(report.lockedTest.gatesPassed, false);
  assert.equal(report.promotion, 'incumbent');
  assert.equal((await import('node:fs')).existsSync(join(directory, 'selected.json')), false);
  const checkpoint = JSON.parse(readFileSync(join(directory, 'checkpoint.json'), 'utf8'));
  assert.equal(checkpoint.selected, true, 'the validation finalist remains frozen');
  const requests = model.openings.length;
  await assert.rejects(() => resumeOptimization(suite, directory, options), /frozen finalist failed locked test gates/);
  assert.equal(model.openings.length, requests, 'test results are reused without selecting another candidate');
});

test('an exhausted holdout allowance returns an incumbent artifact without finalized promotion', async () => {
  const fixture = fixtureSuite(), suite = await loadEvaluationSuite(fixture.path);
  const model = scriptedModel(() => 'return value.split(" ")[0]');
  const result = await optimize(suite, { executor: model.driver, reflection: () => { throw new Error('unused'); },
    out: join(fixture.root, 'incomplete-holdout'), budget: { maxRollouts: 10, maxProposals: 0, maxModelCalls: 5 } });
  assert.equal(result.report.holdoutComplete, false);
  assert.equal(result.artifact.provenance.promotion, 'incumbent');
  assert.equal(result.report.lockedTest, null);
  assert.equal((await import('node:fs')).existsSync(join(result.directory, 'selected.json')), false);
  assert.equal((await import('node:fs')).existsSync(join(result.directory, 'incumbent.json')), true);
  assert.ok(result.state.history.some(event => event.type === 'holdout-incomplete'));
  assert.equal(result.state.ledger.usage.modelCalls, 5);
});

test('revalidation of changed guidance requires validation coverage of authored helpers', async () => {
  const fixture = fixtureSuite();
  (await import('node:fs')).mkdirSync(join(fixture.root, 'classify'));
  writeFileSync(join(fixture.root, 'classify/unused.nl'), '---\nargs: {}\nreturns: string\n---\nReturn "unused".\n');
  writeFileSync(fixture.path, readFileSync(fixture.path, 'utf8').replace("components: 'lambda-only'", "components: 'guidance-only'"));
  const previous = await loadEvaluationSuite(fixture.path);
  const key = previous.components[0];
  const artifact = candidateArtifact(previous, { [key]: { kind: 'program.guidance', text: 'Keep the application objective.' } });
  writeFileSync(join(fixture.root, 'classify.nl'), '---\nargs: { value: string }\nreturns: string\n---\nReturn value unchanged.\n');
  const current = await loadEvaluationSuite(fixture.path);
  const model = scriptedModel(() => 'return value.split(" ")[0]');
  await assert.rejects(() => revalidateAdaptation(artifact, current, model.driver), /whole-program guidance coverage/);
});
