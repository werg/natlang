import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as host from '../dist/index.js';
import { collect, scaffoldManifest } from '../dist/tune/collect.js';
import { currentSplit, tuneProgram } from '../dist/tune/pipeline.js';
import { programSites } from '../dist/tune/sites.js';
import { exportCases, importCases } from '../dist/tune/pool.js';
import { activeNativeAdapters } from '../dist/native/model-recording.js';
import { scriptedModel } from './support/natlang.mjs';

const COLLECTOR = fileURLToPath(new URL('../dist/teacher/cli.js', import.meta.url));
const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
const respond = opening => opening.includes('Pass value') ? 'return await inner(value)' : 'return value.toUpperCase()';

function project() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-pool-'));
  mkdirSync(join(root, 'outer'));
  writeFileSync(join(root, 'outer.nl'), named('Pass value to inner.'));
  writeFileSync(join(root, 'outer', 'inner.nl'), named('Return value in uppercase.'));
  writeFileSync(join(root, 'package.json'), '{"type":"module"}');
  const examples = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta', 'theta'].map(value =>
    ({ function: 'outer.nl', input: { value }, expected: value.toUpperCase(), group: `g-${value}` }));
  examples.push({ function: 'outer.nl', input: { value: 'iota' }, expected: 'WRONG', group: 'g-iota' });
  examples.push({ function: 'outer.nl', input: { value: 'held' }, expected: 'HELD', protected: true });
  writeFileSync(join(root, 'tune.examples.jsonl'), examples.map(example => JSON.stringify(example)).join('\n') + '\n');
  writeFileSync(join(root, 'trainer.mjs'), `import { readFileSync, writeFileSync } from 'node:fs';
const args = process.argv.slice(2), plan = JSON.parse(readFileSync(args[args.indexOf('--plan') + 1], 'utf8')), out = args[args.indexOf('--out') + 1];
const rows = readFileSync(plan.rows, 'utf8').split('\\n').filter(Boolean).map(line => JSON.parse(line));
const artifacts = plan.manifest.tree.filter(node => node.level === 'lambda').map(node => ({ node: node.id, artifact: 'nnt1:' + node.id, rank: 2, parameterization: 'native-subspace', requires: [] }));
writeFileSync(out + '/result.json', JSON.stringify({ artifacts, receipt: { rows: rows.length, teachers: [...new Set(rows.map(row => row.teacher))] } }));`);
  return root;
}

/** A chat-completions stand-in answering exactly as `scriptedModel` does: first own turn evals, then replies done. */
function scriptedServer() {
  const server = createServer((request, response) => {
    let body = ''; request.on('data', chunk => body += chunk); request.on('end', () => {
      response.setHeader('content-type', 'application/json');
      if (request.url.endsWith('/props')) return response.end(JSON.stringify({ default_generation_settings: { n_ctx: 32768 } }));
      if (request.url.endsWith('/models')) return response.end(JSON.stringify({ data: [{ id: 'scripted-luna' }] }));
      const { messages } = JSON.parse(body);
      const own = messages.filter(message => message.role === 'assistant' && !(message.tool_calls ?? []).some(call => String(call.id).startsWith('scope_')));
      const message = own.length === 0
        ? { role: 'assistant', content: '', tool_calls: [{ id: `c${messages.length}`, type: 'function', function: { name: 'eval', arguments: JSON.stringify({ code: respond(String(messages[1].content)) }) } }] }
        : { role: 'assistant', content: 'done' };
      response.end(JSON.stringify({ id: 'x', object: 'chat.completion', model: 'scripted-luna',
        choices: [{ index: 0, message, finish_reason: message.tool_calls ? 'tool_calls' : 'stop' }], usage: { prompt_tokens: 10, completion_tokens: 3 } }));
    });
  });
  return new Promise(resolveServer => server.listen(0, '127.0.0.1', () => resolveServer(server)));
}

function runCollector(args) {
  return new Promise((resolveRun, reject) => {
    const child = spawn(process.execPath, [COLLECTOR, ...args], { stdio: ['ignore', 'pipe', 'pipe'] });
    let log = ''; child.stdout.on('data', chunk => log += chunk); child.stderr.on('data', chunk => log += chunk);
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolveRun(log) : reject(new Error(`collector exited ${code}\n${log.slice(-3000)}`)));
  });
}

function contextOf(root) {
  const built = host.buildProject({ project: root, emit: false, write: false });
  assert.equal(built.ok, true, host.formatDiagnostics(built.diagnostics));
  const program = built.manifest.adaptation, examples = currentSplit(root);
  return { root, program, examples, splitReceipt: examples.receipt, scaffold: scaffoldManifest(program, base, { sites: programSites(root, program).sites }) };
}

test('pool cases: export, the real collector, import, and a tune that reuses the pool data without its own teacher', async () => {
  const root = project(), work = mkdtempSync(join(tmpdir(), 'natlang-pool-work-'));
  const server = await scriptedServer();
  try {
    const context = contextOf(root);
    const exported = exportCases(context, join(work, 'cases'));
    const rows = readFileSync(join(work, 'cases', 'source.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
    const collectable = context.examples.filter(example => example.partition !== 'confirmation');
    assert.equal(exported.cases, collectable.length);
    assert.ok(rows.every(row => row.version === 'natlang.program/2' && row.source_groups.length === 1 && ['train', 'development'].includes(row.split)));
    assert.ok(!rows.some(row => JSON.stringify(row).includes('held')), 'protected roots never leave the program');
    const { cases } = JSON.parse(readFileSync(join(work, 'cases', 'cases.json'), 'utf8'));
    assert.deepEqual(cases.map(item => item.index), rows.map((_, index) => index));
    assert.throws(() => exportCases(context, join(work, 'cases')), /immutable/);

    // The pool's per-case path: the same collector entry the reviewed queue runner invokes.
    const port = server.address().port;
    await runCollector([join(work, 'cases', 'source.jsonl'), join(work, 'jobs'), join(work, 'results', 'case.results.jsonl'),
      '--model-id', 'scripted-luna', '--root-seed', '7', '--all', '--workers', '2', '--server', `http://127.0.0.1:${port}`, '--kv-tokens', '0']);

    const summary = importCases(context, join(work, 'results'));
    assert.equal(summary.rows, rows.length);
    assert.equal(summary.imported, rows.length, JSON.stringify(summary.rejected));
    assert.equal(summary.unboundTurns, 0, 'every turn maps to its own invocation binding');
    assert.equal(summary.teachers.length, 1);
    assert.match(summary.teachers[0], /^collector:openai-compatible:scripted-luna:/);
    const records = readdirSync(join(root, '.natlang/tune/data/roots')).map(file => JSON.parse(readFileSync(join(root, '.natlang/tune/data/roots', file), 'utf8')));
    const iota = context.examples.find(example => example.input.value === 'iota');
    if (iota.partition !== 'confirmation') {
      const failed = records.find(record => record.root === iota.id);
      assert.equal(failed.outcome.oracle, false, 'a wrong root is kept as a failure');
      assert.ok(failed.turns.every(turn => turn.evidence.enclosingSuccess === false), 'never a positive target');
    }

    // Equivalence with the tuner's own collection for the same root and the same scripted answers.
    const alpha = context.examples.find(example => example.input.value === 'alpha');
    const own = await collect({ root, program: context.program, scaffold: context.scaffold, examples: [alpha],
      teacher: scriptedModel(respond).driver, teacherId: 'own' });
    const pooled = records.find(record => record.root === alpha.id).turns;
    assert.equal(pooled.length, own.turns.length);
    // Tool-call IDs are minted by each transport, and a reply with calls carries '' or no text depending on the driver.
    const ids = messages => { const seen = new Map(), id = value => seen.get(value) ?? (seen.set(value, `id${seen.size}`), seen.get(value));
      return JSON.parse(JSON.stringify(messages, (key, value) => (key === 'id' || key === 'tool_call_id') && typeof value === 'string' && !value.startsWith('scope_') ? id(value) : value)); };
    const shape = turns => turns.map(turn => ({ binding: turn.binding, turn: turn.turn, messages: ids(turn.messages), tools: turn.tools,
      calls: turn.reply.calls ?? null, text: turn.reply.text || null })).sort((a, b) => (a.binding + a.turn).localeCompare(b.binding + b.turn));
    assert.deepEqual(shape(pooled), shape(own.turns), 'the imported rows are what the serving runtime renders');

    // A changed build rejects the rows.
    const stale = { ...context, program: { ...context.program, buildHash: 'other' } };
    assert.equal(importCases(stale, join(work, 'results')).rejected.length, rows.length);

    // The ordinary tune reuses the imported data; the configured teacher is never called.
    let teacherCalls = 0;
    const report = await tuneProgram({ root, program: context.program, settings: { student: 'lfm', students: {}, teacher: 'teacher', backend: 'managed-local' },
      student: { name: 'lfm', base, checkpoint: '/unused', trainer: [process.execPath, join(root, 'trainer.mjs')], moduleMap: 'lfm2', baseGgufSha256: 'f'.repeat(64) },
      teacher: async () => { teacherCalls++; throw new Error('the configured teacher must not run'); }, teacherId: 'teacher:configured',
      serve: async ({ manifest }) => {
        const model = scriptedModel(respond);
        return { driver: async (request, ...rest) => { activeNativeAdapters(); return model.driver(request, ...rest); },
          loaded: new Set(manifest.artifacts.map(item => item.artifact)), close: async () => {} };
      } });
    assert.equal(teacherCalls, 0);
    assert.equal(report.activated, true, report.reason);
    assert.ok(report.timings.find(item => item.stage === 'data').reused, 'collection was skipped');
    const job = readdirSync(join(root, '.natlang/tune/jobs'))[0];
    assert.deepEqual(JSON.parse(readFileSync(join(root, '.natlang/tune/jobs', job, 'out/result.json'), 'utf8')).receipt.teachers, summary.teachers,
      'training rows carry the pool actor identity');
  } finally {
    server.close();
    rmSync(root, { recursive: true, force: true }); rmSync(work, { recursive: true, force: true });
  }
});
