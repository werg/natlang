// Run with Node 24: node --test test/app.test.mjs   (pytest wrapper: tests/ops_dashboard/test_app.py)
import assert from 'node:assert/strict';
import { test } from 'node:test';

test('the WebSocket close guard turns a client-invalid close code into a normal close', async () => {
  const calls = [];
  const real = globalThis.WebSocket;
  globalThis.WebSocket = class { close(code, reason) { calls.push([code, reason]); } };
  try {
    await import('../helpers/websocket-close-guard.mjs');
    const socket = new globalThis.WebSocket();
    for (const code of [1011, 1008, 1006, 999, 5000]) socket.close(code, 'why');
    for (const code of [1000, 3000, 4003, 4999, undefined]) socket.close(code, 'ok');
    assert.deepEqual(calls.slice(0, 5).map(c => c[0]), [1000, 1000, 1000, 1000, 1000]);
    assert.deepEqual(calls.slice(5).map(c => c[0]), [1000, 3000, 4003, 4999, undefined]);
    assert.ok(calls.every(c => typeof c[1] === 'string'));
  } finally { globalThis.WebSocket = real; }
});

test('the participant and its setup modules load against the installed Tendriller packages', async () => {
  const peer = await import('../ops-peer.mjs');
  assert.equal(typeof peer.runOpsPeer, 'function');
  const session = await import('../session.mjs');
  assert.equal(typeof session.openSession, 'function');
  const { definition } = await import('../domain.mjs');
  assert.equal(definition.id, 'natlang-ops');
});
