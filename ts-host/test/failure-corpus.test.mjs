import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { interpreter } from './support/natlang.mjs';
import { NativeSession } from '../dist/native/runtime.js';
import { TypeEnv } from '../dist/native/types.js';
import { buildPending, MISSING } from '../dist/native/values.js';
import { defaultSystemPrompt, expectedProvenance, nativeJobRunner, validateFocusedRecord } from '../dist/teacher/collector.js';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';
import { failureCases, failureProgramRecords } from '../scripts/failure-corpus/cases.mjs';
import { freezeFailureCorpus } from '../scripts/failure-corpus/freeze.mjs';
import { teacherSeeds } from '../scripts/code-corpus/teacher-seeds.mjs';

test('failure-repair corpus has distinct, valid, semantically annotated records', () => {
  const records = failureProgramRecords();
  assert.equal(records.length, failureCases.length);
  assert.ok(records.length >= 10);
  assert.equal(new Set(records.map(record => record.id)).size, records.length);
  assert.ok(new Set(failureCases.map(item => item.family)).size >= 8);
  for (const record of records) {
    validateFocusedRecord(record);
    assert.ok(record.failure_insight.length >= 50);
    assert.equal(record.semantics.failure_seed.kind, 'runtime');
  }
});

test('frozen failure corpus matches source and refuses changed output', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'failure-freeze-')), output = join(dir, 'cases.jsonl');
  await freezeFailureCorpus(output);
  await freezeFailureCorpus(output);
  const rows = (await readFile(output, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(rows, failureProgramRecords());
  await writeFile(output, '{"changed":true}\n');
  await assert.rejects(freezeFailureCorpus(output), /refusing to overwrite/);
});

test('teacher seed assembly puts failure cases inside a bounded collection range', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'failure-seeds-'));
  const synthetic = join(dir, 'synthetic.jsonl'), failures = join(dir, 'failures.jsonl');
  const output = join(dir, 'teacher.jsonl');
  await writeFile(synthetic, '');
  await freezeFailureCorpus(failures);
  const manifest = await teacherSeeds(synthetic, output, 20, 42, failures);
  const rows = (await readFile(output, 'utf8')).trim().split('\n').map(JSON.parse);
  assert.deepEqual(rows.slice(0, failureCases.length).map(row => row.id), failureProgramRecords().map(row => row.id));
  assert.equal(manifest.programs, rows.length);
  await teacherSeeds(synthetic, output, 20, 42, failures);
});

for (const item of failureCases) test(`${item.id}: failed eval exposes debug and repair reaches the oracle`, async () => {
  const lam = buildPending({ $lambda: { type: item.type, instructions: item.instructions, args: item.inputs } });
  const runtime = interpreter();
  {
    const session = new NativeSession(runtime, lam, new TypeEnv());
    const failed = await session.applyAsync('eval', { code: item.failed });
    assert.equal(failed.kind, 'error', failed.text);
    assert.equal(lam.return, MISSING);
    assert.equal(session.failureDebug.kind, 'runtime');
    assert.equal(session.failureDebug.code, item.failed);
    assert.ok(runtime.trace.events.some(event => event.kind === 'scope_failure' && event.failure_kind === 'runtime'));
    const repaired = await session.applyAsync('eval', { code: item.repaired });
    assert.equal(repaired.kind, 'ok', repaired.text);
    assert.deepEqual(lam.return, item.expected);
    assert.equal(session.failureDebug, undefined);
  }
});

test('teacher collector seeds a real failure and admits only the repair continuation', async () => {
  const caseItem = failureCases[0], record = failureProgramRecords()[0];
  // The repair eval returns the value; the empty reply after it says done.
  const replies = [['eval', { code: caseItem.repaired }]];
  let requests = 0;
  const server = createServer((request, response) => {
    request.resume(); request.on('end', () => {
      const [name, args] = replies[requests++] ?? [];
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message: name ? { role: 'assistant', content: '',
        tool_calls: [{ id: `repair_${requests}`, type: 'function', function: { name, arguments: JSON.stringify(args) } }] } :
        { role: 'assistant', content: '' } }], usage: { prompt_tokens: 10, completion_tokens: 2 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try {
    const dir = await mkdtemp(join(tmpdir(), 'failure-teacher-'));
    const options = { endpoint: `http://127.0.0.1:${server.address().port}`, modelId: 'fixture', rootSeed: 3,
      systemPrompt: defaultSystemPrompt, segmentTurns: 8, segmentMessages: 20, toolSurfaceSha256: 'fixture',
      jobs: join(dir, 'jobs'), output: join(dir, 'out.jsonl'), workers: 1 };
    const item = { index: 0, record };
    const row = await nativeJobRunner(options)(item, expectedProvenance(record, options));
    assert.equal(row.outcome.accepted, true);
    assert.equal(row.outcome.value, caseItem.expected);
    assert.equal(requests, 2, 'the seeded failure must not consume a model request');
    assert.equal(row.trajectory.length, 3);
    assert.match(row.trajectory[1].context.at(-1).content, /Nothing else from this eval was kept/);
    const materialized = materializeNativeRows([row]);
    assert.equal(materialized.turns[0].training_admission.approved, false);
    assert.equal(materialized.turns[1].training_admission.approved, true);
  } finally { await new Promise(resolve => server.close(resolve)); }
});
