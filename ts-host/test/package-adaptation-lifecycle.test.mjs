import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, symlinkSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { basename, join, resolve } from 'node:path';
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { test } from 'node:test';
import { scriptedModel } from './support/natlang.mjs';

const root = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const nodePackage = join(root, 'npm-packages/node');
const cliBin = join(root, 'npm-packages/cli/bin/natlang.mjs');
const requireNode = createRequire(join(nodePackage, 'package.json'));
const staged = existsSync(join(nodePackage, 'dist/adaptation/node.js'));
const importNode = specifier => import(pathToFileURL(requireNode.resolve(specifier)));
const identity = { id: 'package-fixture', configuration: { revision: 1 } };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;

function project() {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-package-adaptation-'));
  writeFileSync(join(directory, 'echo.nl'), named('Return value.'));
  writeFileSync(join(directory, 'main.ts'), "import echo from './echo.nl'; export default echo;\n");
  return directory;
}

test('staged package compiler, binding, concurrent baseline and adapted calls, and inspect CLI work without inference', { skip: !staged }, async () => {
  const directory = project();
  try {
    const host = await importNode('@natlang/node');
    const adaptation = await importNode('@natlang/node/adaptation');
    const compiled = host.compileVirtualProject({ files: {
      'echo.nl': named('Return value.'),
      'main.ts': "import echo from './echo.nl'; export default echo;\n",
    } }, host, { target: 'node' });
    assert.equal(compiled.ok, true, host.formatDiagnostics(compiled.diagnostics));
    const program = compiled.manifest.adaptation;
    const component = program.components.find(value => value.origin === 'named');
    assert.ok(component);
    const artifact = { schema: 'natlang.adaptation/v1', digest: '',
      program: { id: program.id, buildHash: program.buildHash, protocol: program.protocol, guidanceScope: program.guidanceScope },
      executor: identity, policy: { codeEdits: 'deny', settings: {} },
      components: [{ key: component.key, baselineHash: component.baselineHash, contractHash: component.contractHash,
        value: { kind: 'lambda.instructions', template: { segments: ['Return value in uppercase.'], slotIds: [] } } }],
      provenance: { runId: 'installed-package-test', strategy: 'test', engine: 'natlang/test', suiteHash: adaptation.fingerprint({ suite: 'fixture' }),
        seed: 1, promotion: 'selected', evidence: {} } };
    artifact.digest = adaptation.artifactDigest(artifact);
    const binding = adaptation.bindAdaptation(artifact, program, identity);
    const callable = compiled.require('main.ts').default;
    const model = scriptedModel(opening => opening.includes('uppercase') ? 'return value.toUpperCase()' : 'return value');
    const runtime = host.createNatlangRuntime({ program, executorIdentity: identity, adaptation: binding, model: model.driver });
    assert.deepEqual(await Promise.all([
      runtime.run(() => callable('hello world')),
      runtime.run(() => callable('hello world'), { adaptation: null }),
    ]), ['HELLO WORLD', 'hello world']);

    writeFileSync(join(directory, 'echo.nl'), named('Return value.'));
    const env = { ...process.env, NATLANG_SERVER: 'http://127.0.0.1:1', NATLANG_MODEL: 'no-network-fixture' };
    const inspect = spawnSync(process.execPath, [cliBin, 'adapt', 'inspect', directory, '--json'], { cwd: directory, env, encoding: 'utf8', timeout: 30000 });
    assert.equal(inspect.status, 0, inspect.stderr);
    assert.equal(JSON.parse(inspect.stdout).components.some(value => value.origin === 'named'), true);
    const invalid = spawnSync(process.execPath, [cliBin, 'adapt', 'inspect', directory, '--unknown-option'],
      { cwd: directory, env, encoding: 'utf8', timeout: 30000 });
    assert.equal(invalid.status, 2);
    assert.match(invalid.stderr, /unknown option: --unknown-option/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

function cli(args, cwd, env) {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(process.execPath, [cliBin, ...args], { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    const timeout = setTimeout(() => child.kill('SIGKILL'), 45000);
    timeout.unref();
    child.stdout.setEncoding('utf8').on('data', chunk => { stdout += chunk; });
    child.stderr.setEncoding('utf8').on('data', chunk => { stderr += chunk; });
    child.once('error', error => { clearTimeout(timeout); reject(error); });
    child.once('close', (status, signal) => { clearTimeout(timeout); resolvePromise({ status, signal, stdout, stderr }); });
  });
}

test('installed CLI eval, reflection optimize/resume, artifact inspect/export and revalidation use a local scripted model',
  { skip: !staged }, async () => {
    const directory = project();
    const config = join(directory, '.natlang-config');
    mkdirSync(config, { recursive: true });
    writeFileSync(join(directory, 'package.json'), '{"type":"module"}');
    mkdirSync(join(directory, 'node_modules', '@natlang'), { recursive: true });
    symlinkSync(nodePackage, join(directory, 'node_modules', '@natlang', 'node'), 'dir');
    const seen = [];
    const fake = createServer(async (request, response) => {
      const chunks = [];
      for await (const chunk of request) chunks.push(chunk);
      const body = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      seen.push(body);
      // Reuse the runtime test oracle at the OpenAI-compatible boundary: its eval calls become function calls,
      // and its terminal "done" reply becomes a normal assistant message.
      const turn = await scriptedModel(opening => opening.includes('uppercase') ? 'return value.toUpperCase()' : 'return value')
        .driver({ messages: body.messages, tools: body.tools ?? [] });
      const message = turn.calls?.length ? { role: 'assistant', tool_calls: turn.calls.map(([name, args], index) => ({
        id: `scripted_${seen.length}_${index}`, type: 'function', function: { name, arguments: JSON.stringify(args) },
      })) } : { role: 'assistant', content: turn.text ?? '' };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ id: `scripted_${seen.length}`, object: 'chat.completion', choices: [{ index: 0, message,
        finish_reason: turn.calls?.length ? 'tool_calls' : 'stop' }], usage: { prompt_tokens: 11, completion_tokens: 3 } }));
    });
    fake.requestTimeout = 20000;
    fake.headersTimeout = 10000;
    await new Promise((resolvePromise, reject) => { fake.once('error', reject); fake.listen(0, '127.0.0.1', resolvePromise); });
    try {
      const endpoint = `http://127.0.0.1:${fake.address().port}`;
      const configPath = join(config, 'config.json');
      writeFileSync(configPath, JSON.stringify({ defaultProfile: 'fixture', profiles: { fixture: { endpoint, model: 'scripted-fixture' } } }));
      const env = { ...process.env, NATLANG_CONFIG_HOME: config, NATLANG_API_KEY: 'local-test-key' };
      const suitePath = join(directory, 'suite.mjs');
      const candidateSuitePath = join(directory, 'candidate-suite.mjs');
      const evaluationURL = pathToFileURL(join(nodePackage, 'dist/evaluation/index.js')).href;
      const programId = basename(directory);
      const modelApi = await importNode('@natlang/node/model');
      const executorIdentity = modelApi.executorIdentityForChoice(modelApi.resolveModelChoice({ endpoint, model: 'scripted-fixture' }));
      const suiteSource = expected => `import { defineEvaluationSuite } from ${JSON.stringify(evaluationURL)};
export default defineEvaluationSuite({ id: 'installed-cli', program: { root: '.', id: ${JSON.stringify(programId)}, entry: 'main.ts' },
  components: 'lambda-only', models: { executor: 'fixture', reflection: 'fixture' },
  executorIdentity: ${JSON.stringify(executorIdentity)},
  cases: [
    { id: 'train', group: 'train', split: 'train', input: 'urgent help', expected: 'urgent help' },
    { id: 'valid', group: 'valid', split: 'validation', input: 'ordinary task', expected: ${JSON.stringify(expected)} },
    { id: 'test', group: 'heldout', split: 'test', input: 'held out', expected: 'held out' }
  ],
  budget: { maxRollouts: 20, maxProposals: 0, maxModelCalls: 100 }, requiredGates: ['correct'],
  fixture: { async create(testCase, context) { const app = await context.loadFreshProgram(); return {
    execute: () => context.runtime.run(() => app.default(testCase.input))
  }; } },
  score(testCase, observation) { return { quality: observation.result === testCase.expected ? 1 : 0,
    gates: { correct: observation.result === testCase.expected } }; }
});`;
      writeFileSync(suitePath, suiteSource('ordinary task'));
      writeFileSync(candidateSuitePath, suiteSource('ORDINARY TASK'));
      const evaluation = await importNode('@natlang/node/evaluation');
      const adaptation = await importNode('@natlang/node/adaptation');
      const prepared = await evaluation.loadEvaluationSuite(candidateSuitePath);
      const component = prepared.program.components.find(value => value.kind === 'lambda.instructions');
      const adapted = (await import(pathToFileURL(join(nodePackage, 'dist/evaluation/runner.js'))))
        .candidateArtifact(prepared, { [component.key]: { kind: 'lambda.instructions',
          template: { segments: ['Return value in uppercase.'], slotIds: [] } } }, 'installed-cli-adaptation');
      const artifactPath = join(directory, 'uppercase.json');
      writeFileSync(artifactPath, JSON.stringify(adapted));

      const baseline = await cli(['eval', suitePath, '--split', 'validation', '--baseline', '--json'], directory, env);
      assert.equal(baseline.status, 0, baseline.stderr);
      assert.equal(JSON.parse(baseline.stdout).quality, 1);
      const optimized = await cli(['optimize', suitePath, '--strategy', 'reflection', '--seed', '5', '--out', 'run'], directory, env);
      assert.equal(optimized.status, 0, optimized.stderr);
      const run = JSON.parse(optimized.stdout);
      assert.equal(run.report.baseline.quality, 1);
      assert.equal(JSON.parse(readFileSync(join(directory, 'run', 'checkpoint.json'), 'utf8')).selected, true);
      const beforeResume = seen.length;
      const resumed = await cli(['optimize', 'resume', join(directory, 'run'), '--suite', suitePath, '--strategy', 'reflection', '--seed', '5'], directory, env);
      assert.equal(resumed.status, 0, resumed.stderr);
      assert.equal(seen.length, beforeResume, 'selected runs resume without fresh inference');

      const rollback = await cli(['eval', suitePath, '--split', 'validation', '--artifact', artifactPath, '--json'], directory, env);
      assert.equal(rollback.status, 3, rollback.stderr);
      assert.equal(JSON.parse(rollback.stdout).quality, 0, 'the explicit baseline remains available when an artifact regresses');
      const candidateEval = await cli(['eval', candidateSuitePath, '--split', 'validation', '--artifact', artifactPath, '--json'], directory, env);
      assert.equal(candidateEval.status, 0, candidateEval.stderr);
      assert.equal(JSON.parse(candidateEval.stdout).quality, 1);
      const inspected = await cli(['adapt', 'inspect', artifactPath, '--project', directory, '--json'], directory, env);
      assert.equal(inspected.status, 0, inspected.stderr);
      assert.equal(JSON.parse(inspected.stdout).compatible, true);
      const exported = await cli(['adapt', 'export-source', artifactPath, '--project', directory, '--out', 'adaptation.patch'], directory, env);
      assert.equal(exported.status, 0, exported.stderr);
      assert.match(readFileSync(join(directory, 'adaptation.patch'), 'utf8'), /Return value in uppercase/);
      const revalidated = await cli(['adapt', 'revalidate', artifactPath, '--suite', candidateSuitePath, '--out', 'revalidated.json'], directory, env);
      assert.equal(revalidated.status, 0, revalidated.stderr);
      const newArtifact = JSON.parse(readFileSync(join(directory, 'revalidated.json'), 'utf8'));
      assert.equal(newArtifact.provenance.promotion, 'revalidated');
      assert.equal(newArtifact.provenance.evidence.previous, adapted.digest);
      assert.ok(newArtifact.provenance.evidence.ledger.usage.modelCalls > 0);
      assert.ok(seen.length > beforeResume);
      assert.equal(adaptation.validateAdaptation(newArtifact).digest, newArtifact.digest);
    } finally {
      const closing = new Promise(resolvePromise => fake.close(resolvePromise));
      fake.closeAllConnections?.();
      let closeTimer;
      try { await Promise.race([closing, new Promise((_, reject) => { closeTimer = setTimeout(() => reject(new Error('fake model server did not close')), 5000); })]); }
      finally { clearTimeout(closeTimer); }
      rmSync(directory, { recursive: true, force: true });
    }
  });
