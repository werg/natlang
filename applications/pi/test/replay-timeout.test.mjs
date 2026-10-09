/**
 * Replay stops before a bash step whose recorded observation is OpenHands' notice that the command is still running
 * (its soft timeout, or the call's own timeout): pi's bash waits for the command, so pi would never show that notice or
 * the interaction it offers (send keys, execute_bash's timeout parameter).
 * Run: node --test applications/pi/test/replay-timeout.test.mjs
 */
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { openHandsTimeout, replay } from '../bench/replay.ts';

const offer = `You may wait longer to see additional output by sending empty command '', send other commands to interact with the current process, send keys ("C-c", "C-z", "C-d") to interrupt/kill the previous command before sending your new command, or use the timeout parameter in execute_bash for future commands.`;
const soft = `Help on function f:\n-- more --\n[The command has no new output after 30 seconds. ${offer}]`;
const hard = `running tests...\n[The command timed out after 300.0 seconds. ${offer}]`;

test('OpenHands timeout notices are recognized in their exact form', () => {
  assert.equal(openHandsTimeout(soft), 'bash soft timeout (OpenHands-only interaction)');
  assert.equal(openHandsTimeout(hard), 'bash timeout (OpenHands-only interaction)');
  assert.equal(openHandsTimeout('ok\n[The command completed with exit code 0.]'), null);
  assert.equal(openHandsTimeout('[The command completed with exit code 130. CTRL+C was sent.]'), null);
  assert.equal(openHandsTimeout('the command has no new output after 30 seconds'), null);
});

test('replay diverges at a soft-timeout bash observation and keeps the verified prefix', async () => {
  const root = mkdtempSync(join(tmpdir(), 'pi-replay-timeout-'));
  try {
    const call = (id, name, args) => ({ role: 'assistant', content: [{ type: 'toolCall', id, name, arguments: args }] });
    const result = (id, name, text) => ({ role: 'toolResult', toolCallId: id, toolName: name, content: [{ type: 'text', text }] });
    const prepared = { id: 't', repo: 'test/repo', cwd: '/workspace/project', base_commit: 'x', messages: [
      { role: 'user', content: 'Fix the bug.' },
      call('c1', 'bash', { command: 'echo hi' }), result('c1', 'bash', 'hi\n[The command completed with exit code 0.]'),
      call('c2', 'bash', { command: 'python -c "help(f)"' }), result('c2', 'bash', soft),
      call('c3', 'execute_bash', { command: 'C-c', is_input: 'true' }), result('c3', 'execute_bash', '[The command completed with exit code 130. CTRL+C was sent.]'),
    ] };
    const { messages, report } = await replay(prepared, root);
    assert.deepEqual(report.diverged, { message: 4, call: 'c2', reason: 'bash soft timeout (OpenHands-only interaction)' });
    assert.equal(report.replayed, 1);
    assert.equal(report.recordedBash, 1);
    // The step before is in pi's form; the diverging observation is left as recorded (records cut before its turn).
    assert.deepEqual(messages[2].content, [{ type: 'text', text: 'hi' }]);
    assert.equal(messages[4].content[0].text, soft);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
