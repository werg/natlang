// A profile's `escalate` teacher answers a call its student fails (calls/tiers.ts tiers 1 and 0), announced on stderr.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import http from 'node:http';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

const bin = new URL('../bin/natlang.mjs', import.meta.url).pathname;

async function chatServer(reply) {
  const requests = [];
  const server = http.createServer((request, response) => {
    let body = '';
    request.on('data', chunk => { body += chunk; });
    request.on('end', () => {
      if (request.url.endsWith('/v1/models')) { response.end(JSON.stringify({ data: [] })); return; }
      requests.push(body);
      const answer = reply(requests.length);
      if (answer === null) { response.statusCode = 500; response.end(JSON.stringify({ error: { message: 'student is down' } })); return; }
      response.setHeader('content-type', 'application/json');
      response.end(JSON.stringify({ choices: [{ message: { content: answer, tool_calls: [] }, finish_reason: 'stop' }], usage: { completion_tokens: 1 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return { endpoint: `http://127.0.0.1:${server.address().port}`, requests, close: () => new Promise(resolve => server.close(resolve)) };
}

function run(args, env, cwd) {
  return new Promise(resolve => {
    const child = spawn(process.execPath, [bin, ...args], { env, cwd, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    child.stdout.on('data', chunk => { out += chunk; }); child.stderr.on('data', chunk => { err += chunk; });
    child.on('close', code => resolve({ code, out, err }));
  });
}

test('a call the student fails is answered by the escalation profile, and --escalate none turns it off', async () => {
  const student = await chatServer(() => null), teacher = await chatServer(() => '42');
  const directory = mkdtempSync(join(tmpdir(), 'natlang-escalation-'));
  try {
    writeFileSync(join(directory, 'answer.nl'), '---\nargs:\n  question: string\nreturns: number\n---\nAnswer the question with a number.\n');
    writeFileSync(join(directory, 'config.json'), JSON.stringify({ defaultProfile: 'student', profiles: {
      student: { endpoint: student.endpoint, model: 'student-model', escalate: 'teacher' },
      teacher: { endpoint: teacher.endpoint, model: 'teacher-model' } } }));
    const env = { ...process.env, NATLANG_CONFIG_HOME: directory, NATLANG_CALL_STORE: 'off', NATLANG_HASH_CACHE: join(directory, 'hashes.jsonl') };
    writeFileSync(join(directory, 'inputs.json'), JSON.stringify({ question: 'six times seven' }));
    const escalated = await run(['call', 'answer.nl', '--inputs', 'inputs.json'], env, directory);
    assert.equal(escalated.code, 0, escalated.err);
    assert.equal(escalated.out.trim(), '42');
    assert.ok(student.requests.length >= 1 && teacher.requests.length >= 1);
    assert.match(escalated.err, /answer failed on student \(.*\); answered by teacher/);
    const before = teacher.requests.length;
    const direct = await run(['call', 'answer.nl', '--inputs', 'inputs.json', '--escalate', 'none'], env, directory);
    assert.notEqual(direct.code, 0);
    assert.equal(teacher.requests.length, before, 'no escalation when turned off');
  } finally { rmSync(directory, { recursive: true, force: true }); await student.close(); await teacher.close(); }
});
