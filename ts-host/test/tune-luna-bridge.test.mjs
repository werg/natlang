import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, mkdirSync, rmSync, writeFileSync, readFileSync, readdirSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as host from '../dist/index.js';
import { collect, scaffoldManifest } from '../dist/tune/collect.js';
import { currentSplit, tuneProgram, validRootRecord } from '../dist/tune/pipeline.js';
import { sharedSourceHash } from '../dist/tune/keys.js';
import { programSites } from '../dist/tune/sites.js';
import { exportCases, importCases, honestTuneBlockedRoot } from '../dist/tune/pool.js';
import { activeNativeAdapters } from '../dist/native/model-recording.js';
import { canonicalJson } from '../dist/package/archive.js';
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

    const snapshotDir = join(root, '.natlang/tune/pool-import-custody-snapshots');
    const custody = fixtureCustody(work, join(work, 'results'), join(work, 'cases', 'source.jsonl'), snapshotDir);
    writeFileSync(join(work, 'results', 'case-late.results.jsonl'), readFileSync(join(work, 'results', 'case.results.jsonl')));
    assert.throws(() => importCases(context, join(work, 'results')), /reviewed dispatcher custody/);
    const summary = importCases(context, join(work, 'results'), custody);
    assert.equal(summary.rows, rows.length);
    assert.equal(summary.imported, rows.length, JSON.stringify(summary.rejected));
    assert.equal(summary.unboundTurns, 0, 'every turn maps to its own invocation binding');
    assert.equal(summary.teachers.length, 1);
    assert.match(summary.teachers[0], /^collector:openai-compatible:scripted-luna:/);
    const records = readdirSync(join(root, '.natlang/tune/data/roots')).map(file => JSON.parse(readFileSync(join(root, '.natlang/tune/data/roots', file), 'utf8')));
    const custodyRecord = records.find(record => record.imported?.result_row_sha256 === custody.selected_rows[0].result_row_sha256);
    assert.ok(custodyRecord, 'the exact imported result row has a durable tune-root record');
    assert.equal(custodyRecord.imported.campaign.plan.sha256, custody.plan.sha256);
    assert.equal(custodyRecord.imported.campaign.runtime_receipt.sha256, custody.runtime.receipt.sha256);
    assert.ok(existsSync(custodyRecord.imported.campaign.claim_ledger.snapshot.path));
    const retainedLedgerSnapshot = custodyRecord.imported.campaign.claim_ledger.snapshot.sha256;
    writeFileSync(custody.claim_ledger.observed.path,
      readFileSync(custody.claim_ledger.observed.path).toString() + '{"event":"later claim"}\n');
    writeFileSync(custody.launch_record.observed.path, '{"status":"running","later":true}\n');
    const laterCustody = fixtureCustody(work, join(work, 'results'), join(work, 'cases', 'source.jsonl'), snapshotDir);
    assert.equal(importCases(context, join(work, 'results'), laterCustody).imported, rows.length,
      'the same exact closed results are idempotent as campaign supervision files advance');
    const afterRetry = readdirSync(join(root, '.natlang/tune/data/roots')).map(file => JSON.parse(readFileSync(join(root, '.natlang/tune/data/roots', file), 'utf8')))
      .find(record => record.imported?.result_row_sha256 === custody.selected_rows[0].result_row_sha256);
    assert.equal(afterRetry.imported.campaign.claim_ledger.snapshot.sha256, retainedLedgerSnapshot,
      're-import preserves the first immutable terminal evidence');
    const detachedPath = join(work, 'merged-results.jsonl');
    writeFileSync(detachedPath, readFileSync(join(work, 'results', 'case.results.jsonl')));
    const detachedCustody = { ...laterCustody,
      input: { kind: 'file', ...binding(detachedPath) },
      selected_rows: laterCustody.selected_rows.map((row, index) => ({ ...row,
        input: { path: resolve(detachedPath), line: index + 1 } })) };
    assert.equal(importCases(context, detachedPath, detachedCustody).imported, rows.length,
      'detached merged imports map each exact input line to the original accounted campaign output');
    const queuePath = custody.selected_rows[0].queue.path, queueBytes = readFileSync(queuePath);
    writeFileSync(queuePath, Buffer.concat([queueBytes, Buffer.from('tampered\n')]));
    assert.throws(() => importCases(context, join(work, 'results'), custody), /case queue changed/);
    writeFileSync(queuePath, queueBytes);
    const outputPath = join(work, 'results', 'case.results.jsonl'), outputBytes = readFileSync(outputPath);
    writeFileSync(outputPath, Buffer.concat([outputBytes, Buffer.from(' ')]));
    assert.throws(() => importCases(context, join(work, 'results'), custody), /selected closed-case result file changed/);
    writeFileSync(outputPath, outputBytes);
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
    assert.equal(importCases(stale, join(work, 'results'), custody).rejected.length, rows.length);

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
    const retained = validRootRecord(root, context.examples[0], context.scaffold, sharedSourceHash(context.program), 'teacher:configured');
    assert.ok(retained?.imported, 'the exact imported row remains reusable while its snapshot is intact');
    const ledgerSnapshot = retained.imported.campaign.claim_ledger.snapshot.path;
    writeFileSync(ledgerSnapshot, 'tampered');
    assert.equal(validRootRecord(root, context.examples[0], context.scaffold, sharedSourceHash(context.program), 'teacher:configured'), undefined,
      'a missing or changed immutable terminal snapshot invalidates imported custody');
  } finally {
    server.close();
    rmSync(root, { recursive: true, force: true }); rmSync(work, { recursive: true, force: true });
  }
});

function hash(bytes) { return createHash('sha256').update(bytes).digest('hex'); }
function binding(path) { const bytes = readFileSync(path); return { path: resolve(path), bytes: bytes.length, sha256: hash(bytes) }; }
function campaignSnapshot(livePath, label, snapshotDir) {
  const bytes = readFileSync(livePath), sha256 = hash(bytes), snapshotPath = join(snapshotDir, `${label}-${sha256}.snapshot`);
  mkdirSync(snapshotDir, { recursive: true });
  writeFileSync(snapshotPath, bytes);
  return { observed: { path: resolve(livePath), bytes: bytes.length, sha256,
    stat: { bytes: bytes.length }, prefix: { bytes: bytes.length, sha256 } },
    snapshot: { path: resolve(snapshotPath), bytes: bytes.length, sha256 } };
}
function fixtureCustody(work, inputPath, sourcePath, snapshotDir) {
  const output = join(inputPath, 'case.results.jsonl'), outputBytes = readFileSync(output);
  const manifestPath = `${output}.manifest.json`, resultRows = outputBytes.toString().split('\n').filter(Boolean).map(line => JSON.parse(line));
  const sourceText = readFileSync(sourcePath, 'utf8'), sourceRows = sourceText.split('\n').filter(Boolean).map(line => JSON.parse(line));
  const sourceRawRows = sourceText.split('\n').filter(Boolean).map(line => Buffer.from(line + '\n'));
  const planPath = join(work, 'fixture-plan.json'), ledgerPath = join(work, 'fixture-ledger.jsonl');
  const launchPath = join(work, 'fixture-launch.json'), runtimePath = join(work, 'fixture-runtime.json');
  for (const [path, body] of [[planPath, '{}'], [ledgerPath, '{}\n'], [launchPath, '{}'], [runtimePath, '{}']])
    if (!existsSync(path)) writeFileSync(path, body);
  const rows = resultRows.map((row, index) => {
    const queue = join(work, `fixture-queue-${index}.jsonl`); writeFileSync(queue, JSON.stringify({ index }) + '\n');
    return { case_index: index, input: { path: resolve(output), line: index + 1 },
      result_row_sha256: hash(Buffer.from(canonicalJson(row))), result_line_sha256: hash(Buffer.from(outputBytes.toString().split('\n')[index] + '\n')), result_file: binding(output),
      result_manifest: binding(manifestPath), queue: binding(queue),
      source_row_sha256: hash(sourceRawRows[index]),
      provenance_sha256: hash(Buffer.from(canonicalJson(row.provenance))), execution_identity: row.execution_identity,
      actor: { provider: row.provenance.provider, model: row.provenance.model,
        dispatch_plan_sha256: 'a'.repeat(64), runtime_receipt_sha256: hash(readFileSync(runtimePath)), source_inventory_sha256: null } };
  });
  return { schema: 'natlang.tune-pool-import-custody/2', input: { kind: 'directory', path: resolve(inputPath),
    files: [binding(output)] }, plan: binding(planPath), source: binding(sourcePath), claim_ledger: campaignSnapshot(ledgerPath, 'claim-ledger', snapshotDir),
    launch_record: campaignSnapshot(launchPath, 'launch', snapshotDir), dispatcher_identity_sha256: 'b'.repeat(64),
    reviewed_artifacts: [],
    runtime: { root: resolve(work), receipt_sha256: hash(readFileSync(runtimePath)), receipt: binding(runtimePath) },
    source_inventory: null, selected_rows: rows };
}

test('reviewed tune import accepts only an exact honest blocked root, never a blocked child or failed stop', () => {
  const expected = { status: 'blocked', reason: 'required evidence is unavailable' };
  const root = { invocation_id: 'root', parent_invocation_id: null, completion_status: 'quiesced', completion_source: 'execution_graph' };
  const row = { outcome: { status: 'quiesced', accepted: true, oracle: { accepted: true, level: 'exact' },
      checks: Object.fromEntries(['seeded_failure_requirement', 'expected_status', 'honest_stop', 'effects', 'files', 'world',
        'file_return_consistency', 'answer'].map(key => [key, true])), invocation_ledger: [root],
      action_ledger: [{ call_id: 'root', name: 'return_result', outcome: 'blocked', arguments: expected, seq: 3 }] },
    trajectory: [{ invocation_id: 'root', assistant: { calls: [{ source_tool: 'return_result', arguments: expected }] } }] };
  assert.equal(honestTuneBlockedRoot(row, expected), true);
  const failedRoot = structuredClone(row); failedRoot.outcome.checks.answer = false;
  assert.equal(honestTuneBlockedRoot(failedRoot, expected), false);
  const blockedChild = structuredClone(row);
  blockedChild.outcome.invocation_ledger.push({ invocation_id: 'child', parent_invocation_id: 'root', completion_status: 'done', completion_source: 'execution_graph' });
  blockedChild.trajectory.push({ invocation_id: 'child', assistant: { calls: [{ source_tool: 'return_result', arguments: { status: 'blocked', reason: 'child is blocked' } }] } });
  assert.equal(honestTuneBlockedRoot(blockedChild, expected), false);
  const childFailure = structuredClone(row);
  childFailure.outcome.invocation_ledger.push({ invocation_id: 'child', parent_invocation_id: 'root', completion_status: 'done', completion_source: 'execution_graph' });
  childFailure.trajectory.push({ invocation_id: 'child', assistant: { calls: [{ source_tool: 'return_result', arguments: { status: 'failed', reason: 'child failed' } }] } });
  assert.equal(honestTuneBlockedRoot(childFailure, expected), false, 'a successful root cannot excuse a failed child');
  assert.equal(honestTuneBlockedRoot(row, { status: 'success', value: 'x' }), false);
});
