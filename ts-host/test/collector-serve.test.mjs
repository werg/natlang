import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { mkdtempSync, existsSync } from 'node:fs';
import { connect } from 'node:net';
import { join } from 'node:path';
import { test } from 'node:test';

const cli = new URL('../dist/teacher/cli.js', import.meta.url).pathname;
const delay = ms => new Promise(done => setTimeout(done, ms));

function request(socket, argv, { abandon = false } = {}) {
  return new Promise((resolve, reject) => {
    const connection = connect(socket);
    let text = '';
    connection.on('error', error => (abandon ? resolve(null) : reject(error)));
    connection.on('connect', () => {
      if (argv === null) { connection.destroy(); resolve(null); return; }   // a probe: connect and go away
      connection.write(JSON.stringify({ argv }) + '\n');
      if (abandon) { connection.destroy(); resolve(null); }
    });
    connection.on('data', chunk => { text += chunk; });
    connection.on('end', () => resolve(text.trim().split('\n').filter(Boolean).map(line => JSON.parse(line))));
  });
}

test('a shared collector survives peers that leave and keeps its capacity', async () => {
  const socket = join(mkdtempSync('/tmp/nlserve-'), 's.sock');
  const server = spawn(process.execPath, [cli, '--serve', socket, '--capacity', '1', '--idle-seconds', '30'],
    { stdio: ['ignore', 'pipe', 'pipe'] });
  let stderr = '';
  server.stderr.on('data', chunk => { stderr += chunk; });
  try {
    for (let i = 0; i < 100 && !existsSync(socket); i++) await delay(100);
    assert.ok(existsSync(socket), 'server listening');
    // Probes (no request) hold no capacity.
    for (let i = 0; i < 5; i++) await request(socket, null);
    await delay(100);
    assert.deepEqual((await request(socket, [1]))?.at(-1), { exit: 1, error: 'argv must be a list of strings' });
    // Peers that send a request and leave before the reply: every write to them fails with EPIPE.
    await Promise.all(Array.from({ length: 50 }, () => request(socket, [1], { abandon: true })));
    await delay(300);
    assert.equal(server.exitCode, null, `server still running: ${stderr}`);
    assert.deepEqual((await request(socket, [1]))?.at(-1), { exit: 1, error: 'argv must be a list of strings' },
      'capacity is free again');
  } finally {
    server.kill();
  }
});
