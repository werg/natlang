import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtemp, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { buildHardStateQueue, selectHardState } from '../scripts/build-hard-state-queue.mjs';
import { preferencePairs } from '../scripts/build-preference-pairs.mjs';
import { collectBatch, defaultSystemPrompt, defaultToolSurfaceHash, nativeJobRunner } from '../dist/teacher/collector.js';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';

const record = { version: 'natlang.program/2', id: 'student-hard-case', kind: 'lambda_source',
  source_groups: ['student-hard-case'], split: 'train', semantics: {
    root: 'one.nl', files: { 'one.nl': '---\nargs: {}\nreturns: number\n---\nReturn one.\n' },
    inputs: {}, expected: 1, operation: 'exact' } };
function modelServer(responses) {
  let count = 0;
  const requests = [];
  const server = createServer((request, response) => {
    let body = '';
    request.setEncoding('utf8');
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      requests.push(JSON.parse(body));
      const calls = responses[Math.min(count++, responses.length - 1)];
      const message = { role: 'assistant', content: '', ...(calls ? { tool_calls: calls.map(([name, args], index) => ({
        id: `call-${count}-${index}`, type: 'function', function: { name, arguments: JSON.stringify(args) } })) } : {}) };
      response.writeHead(200, { 'content-type': 'application/json' });
      response.end(JSON.stringify({ choices: [{ message }], usage: { prompt_tokens: 10, completion_tokens: 4 } }));
    });
  });
  return { server, count: () => count, requests };
}
const config = (dir, endpoint, role, surface) => ({ jobs: join(dir, `${role}-jobs`),
  output: join(dir, `${role}.jsonl`), workers: 1, modelId: role, rootSeed: 7,
  systemPrompt: defaultSystemPrompt, contextTokens: 16384,
  toolSurfaceSha256: surface, endpoint, collectionRole: role, maxTurns: 2 });

test('student failure is replayed exactly, teacher repairs it, prefix is not positive SFT', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'student-handoff-'));
  const student = modelServer([[['eval', { code: 'throw new Error("wrong branch")' }]], null]);
  const teacher = modelServer([[['return_result', { status: 'success', value: 1 }]]]);
  await Promise.all([new Promise(resolve => student.server.listen(0, '127.0.0.1', resolve)),
    new Promise(resolve => teacher.server.listen(0, '127.0.0.1', resolve))]);
  try {
    const surface = await defaultToolSurfaceHash();
    const studentConfig = config(dir, `http://127.0.0.1:${student.server.address().port}`, 'student', surface);
    const item = { index: 0, record };
    await collectBatch([item], studentConfig, nativeJobRunner(studentConfig));
    const studentRow = JSON.parse((await readFile(studentConfig.output, 'utf8')).trim());
    assert.equal(studentRow.outcome.accepted, false);
    assert.equal(studentRow.trajectory.length, 2);
    assert.equal(studentRow.outcome.scope_failures.length, 1);
    const external = structuredClone(studentRow);
    external.outcome.host_events.push({ event: { operation: 'network.fetch', replayable: false } });
    assert.equal(selectHardState(external).reason, 'nonreplayable_host_effect');
    const queue = join(dir, 'queue.jsonl');
    const manifest = await buildHardStateQueue(studentConfig.output, queue);
    assert.equal(manifest.count, 1);
    const handoff = JSON.parse((await readFile(queue, 'utf8')).trim());
    assert.equal(handoff.handoff_at, 1);
    assert.equal(handoff.failure.kind, 'scope_eval');
    const teacherConfig = { ...config(dir, `http://127.0.0.1:${teacher.server.address().port}`, 'teacher', surface),
      handoffs: new Map([[record.id, handoff]]) };
    await collectBatch([item], teacherConfig, nativeJobRunner(teacherConfig));
    const repaired = JSON.parse((await readFile(teacherConfig.output, 'utf8')).trim());
    assert.equal(repaired.outcome.accepted, true);
    assert.equal(repaired.handoff.handoff_at, 1);
    assert.equal(student.count(), 2);
    assert.equal(teacher.count(), 1, 'student prefix must replay without another student or teacher decode');
    assert.match(teacher.requests[0].messages.at(-1).content, /wrong branch/);
    const { turns } = materializeNativeRows([repaired]);
    assert.equal(turns.length, 2);
    assert.equal(turns[0].training_admission.approved, false);
    assert.match(turns[0].training_admission.reason, /student replay prefix/);
    assert.equal(turns[1].training_admission.approved, true);
    const preferences = preferencePairs([studentRow], [repaired], turns);
    assert.equal(preferences.pairs.length, 1);
    assert.equal(preferences.pairs[0].evidence.kind, 'premature_reply');
    assert.equal(preferences.pairs[0].request_sha256, handoff.target_request_sha256);
    const divergent = structuredClone(repaired);
    divergent.trajectory[1].request_sha256 = 'different';
    assert.equal(preferencePairs([studentRow], [divergent], turns).pairs.length, 0);
    assert.equal((await buildHardStateQueue(studentConfig.output, queue)).count, 1);
  } finally {
    await Promise.all([new Promise(resolve => student.server.close(resolve)),
      new Promise(resolve => teacher.server.close(resolve))]);
  }
});
