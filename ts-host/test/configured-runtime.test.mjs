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
    assert.match(log.lines.join(''), /answer failed on student .*answered by teacher/);
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
      const lines = readFileSync(join(directory, 'tune.examples.jsonl'), 'utf8').trim().split('\n').map(line => JSON.parse(line));
      assert.equal(lines.length, 1, 'the same input is listed once');
      assert.equal(lines[0].function, 'answer.nl');
      assert.deepEqual(lines[0].input, { question: 'six times seven' });
      assert.ok(!('expected' in lines[0]), 'a teacher answer is not an oracle');
      assert.match(lines[0].group, /^escalations-\d{8}-[0-9a-f]{8}$/);
      if (round === 0) assert.match(log.lines.join(''), /answer: its input joins .*tune\.examples\.jsonl/);
    }
    const off = await natlangRuntime({ programRoot: directory, environment: { ...environment, NATLANG_ESCALATION_EXAMPLES: 'off' }, log: collect().stream });
    await off.runtime.run(() => answer('two plus two'));
    await off.close();
    assert.equal(readFileSync(join(directory, 'tune.examples.jsonl'), 'utf8').trim().split('\n').length, 1);
  } finally { rmSync(directory, { recursive: true, force: true }); await student.close(); await teacher.close(); }
});
