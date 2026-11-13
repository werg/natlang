// natlangRuntime(): a TypeScript host gets the runtime `natlang run` would configure, from the same profiles.
import assert from 'node:assert/strict';
import http from 'node:http';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Writable } from 'node:stream';
import { test } from 'node:test';
import { loadNatlang, natlangRuntime } from '../dist/index.js';

async function chatServer(reply) {
  const requests = [];
  const server = http.createServer((request, response) => {
    let body = '';
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      if (!request.url.endsWith('/chat/completions')) { response.statusCode = 404; response.end('{}'); return; }
      requests.push(JSON.parse(body));
      const answer = reply(requests.length);
      if (answer === null) { response.statusCode = 500; response.end(JSON.stringify({ error: { message: 'down' } })); return; }
      response.setHeader('content-type', 'application/json');
      response.end(JSON.stringify({ choices: [{ message: { content: answer, tool_calls: [] }, finish_reason: 'stop' }], usage: { completion_tokens: 1 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return { endpoint: `http://127.0.0.1:${server.address().port}`, requests, close: () => new Promise(resolve => server.close(resolve)) };
}

const collect = () => { const lines = []; return { lines, stream: new Writable({ write(chunk, _e, done) { lines.push(String(chunk)); done(); } }) }; };

test('an unknown escalation profile cannot silently resolve to the default backend', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-configured-'));
  try {
    writeFileSync(join(directory, 'config.json'), JSON.stringify({ defaultProfile: 'student', profiles: {
      student: { endpoint: 'http://127.0.0.1:1', model: 'student', escalate: 'missing' } } }));
    const environment = { ...process.env, NATLANG_CONFIG_HOME: directory, NATLANG_CALL_STORE: 'off' };
    await assert.rejects(natlangRuntime({ environment, log: collect().stream }), /escalation profile "missing" is not explicitly configured/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('profile escalation cycles fail before a model request', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-configured-'));
  try {
    writeFileSync(join(directory, 'config.json'), JSON.stringify({ defaultProfile: 'student', profiles: {
      student: { endpoint: 'http://127.0.0.1:1', model: 'student', escalate: 'expert' },
      expert: { endpoint: 'http://127.0.0.1:2', model: 'expert', escalate: 'student' } } }));
    const environment = { ...process.env, NATLANG_CONFIG_HOME: directory, NATLANG_CALL_STORE: 'off' };
    await assert.rejects(natlangRuntime({ environment, log: collect().stream }), /escalation profile cycle/);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test('a host runs functions on the configured profile, with escalation, and close() ends every session', async () => {
  const student = await chatServer(() => null), teacher = await chatServer(() => '42');
  const directory = mkdtempSync(join(tmpdir(), 'natlang-configured-'));
  try {
    writeFileSync(join(directory, 'answer.nl'), '---\nargs:\n  question: string\nreturns: number\n---\nAnswer the question with a number.\n');
    writeFileSync(join(directory, 'config.json'), JSON.stringify({ defaultProfile: 'student', profiles: {
      student: { endpoint: student.endpoint, model: 'student-model', escalate: 'teacher' },
      teacher: { endpoint: teacher.endpoint, model: 'teacher-model' } } }));
    const environment = { ...process.env, NATLANG_CONFIG_HOME: directory, NATLANG_CALL_STORE: 'off' };
    const log = collect();
    const natlang = await natlangRuntime({ programRoot: directory, environment, log: log.stream });
    assert.equal(natlang.selection.name, 'student');
    assert.equal(natlang.executorIdentity.id, 'external:student-model');
    const answer = loadNatlang(join(directory, 'answer.nl'));
    assert.equal(await natlang.runtime.run(() => answer('six times seven')), 42);
    assert.ok(teacher.requests.length >= 1);
    assert.match(log.lines.join(''), /answer handed off from .*student-model to .*teacher-model/);
    await natlang.close();
    await natlang.close();   // idempotent

    // escalate: false runs the student alone; an explicit profile selects it.
    const direct = await natlangRuntime({ profile: 'teacher', environment, log: collect().stream, escalate: false });
    assert.equal(direct.selection.name, 'teacher');
    assert.equal(await direct.runtime.run(() => answer('six times seven')), 42);
    await direct.close();
  } finally { rmSync(directory, { recursive: true, force: true }); await student.close(); await teacher.close(); }
});

test('a root call the student fails and the teacher runs adds its input to the program examples once', async () => {
  const student = await chatServer(() => null), teacher = await chatServer(() => '42');
  const directory = mkdtempSync(join(tmpdir(), 'natlang-configured-'));
  try {
    writeFileSync(join(directory, 'answer.nl'), '---\nargs:\n  question: string\nreturns: number\n---\nAnswer the question with a number.\n');
    writeFileSync(join(directory, 'config.json'), JSON.stringify({ defaultProfile: 'student', profiles: {
      student: { endpoint: student.endpoint, model: 'student-model', escalate: 'teacher' },
      teacher: { endpoint: teacher.endpoint, model: 'teacher-model' } } }));
    const environment = { ...process.env, NATLANG_CONFIG_HOME: directory, NATLANG_CALL_STORE: join(directory, 'calls') };
    const answer = loadNatlang(join(directory, 'answer.nl'));
    for (let round = 0; round < 2; round++) {
      const log = collect();
      const natlang = await natlangRuntime({ programRoot: directory, environment, log: log.stream });
      assert.equal(await natlang.runtime.run(() => answer('six times seven')), 42);
      await natlang.close();
      const lines = JSON.parse(readFileSync(join(directory, 'examples.json'), 'utf8'));
      assert.equal(lines.length, 1, 'the same input is listed once');
      assert.equal(lines[0].function, 'answer.nl');
      assert.deepEqual(lines[0].inputs, { question: 'six times seven' });
      assert.ok(!('expected' in lines[0]), 'a teacher answer is not an oracle');
      assert.match(lines[0].group, /^escalations-\d{8}-[0-9a-f]{8}$/);
      if (round === 0) assert.match(log.lines.join(''), /answer: its inputs join .*examples\.json/);
    }
    const off = await natlangRuntime({ programRoot: directory, environment: { ...environment, NATLANG_ESCALATION_EXAMPLES: 'off' }, log: collect().stream });
    await off.runtime.run(() => answer('two plus two'));
    await off.close();
    assert.equal(JSON.parse(readFileSync(join(directory, 'examples.json'), 'utf8')).length, 1);
  } finally { rmSync(directory, { recursive: true, force: true }); await student.close(); await teacher.close(); }
});

test('guided generation is on for natlang backends unless the profile says otherwise', async () => {
  const { runtimeModel } = await import('../dist/configured-runtime.js');
  const ours = Object.assign(async () => ({}), { guidance: true }), other = async () => ({});
  assert.equal((await runtimeModel({ kind: 'managed-local', model: 'm' }, ours)).model.guidance, true);
  assert.equal((await runtimeModel({ kind: 'managed-local', model: 'm', runtime: { guidance: false } }, ours)).model.guidance, false);
  assert.deepEqual((await runtimeModel({ kind: 'managed-local', model: 'm', runtime: { guidance: { repeat: 3 } } }, ours)).model.guidance, { repeat: 3 });
  assert.equal('guidance' in (await runtimeModel({ kind: 'external', endpoint: 'http://x', model: 'm', apiKeyEnv: 'K' }, other)).model, false);
});

test('a managed stock model reads its decisions as {"result": VALUE}; a model natlang trained, an endpoint, or a profile that says so keeps the bare value', async () => {
  const { resolveModelChoice } = await import('../dist/model/config.js');
  assert.equal(resolveModelChoice({}).decisionReply, 'result-object');
  assert.equal(resolveModelChoice({ local: { stock: 'lfm3-1.7b-q4_k_m' } }).decisionReply, 'result-object');
  assert.equal(resolveModelChoice({ decisionReply: 'value' }).decisionReply, 'value');
  assert.equal('decisionReply' in resolveModelChoice({ local: { modelPath: '/models/natlang.gguf' } }), false);
  assert.equal('decisionReply' in resolveModelChoice({ endpoint: 'http://x', model: 'm' }), false);
  assert.equal(resolveModelChoice({ endpoint: 'http://x', model: 'm', decisionReply: 'result-object' }).decisionReply, 'result-object');
  assert.throws(() => resolveModelChoice({ decisionReply: 'object' }), /decisionReply must be value or result-object/);
});
