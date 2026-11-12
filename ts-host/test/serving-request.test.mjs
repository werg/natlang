import assert from 'node:assert/strict';
import { test } from 'node:test';
import { execFileSync } from 'node:child_process';
import { createNatlangRuntime } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';
import { FUNCTION_TOOLS_PROMPT, GENERATION_GUIDANCE, SYSTEM_PROMPT, TOOLS_PROMPT } from '../dist/native/prompt.js';
import { pieceBank, servingRequest } from '../dist/teacher/serving-request.js';

/** The first request the runtime sends for `files` (entry `entry`). */
async function liveRequest(files, entry) {
  const requests = [];
  const runtime = createNatlangRuntime({ model: { driver: async request => { requests.push(structuredClone(request));
    return { calls: [['return_result', { status: 'success', value: 2 }]] }; } } });
  await runtime.run(() => loadVirtualNatlang(files, entry)());
  return requests[0];
}

const ownCode = { 'solve.nl': '---\nargs: {}\nreturns: number\n---\nReturn double(1).',
  'solve/double.nl': '---\nargs: { x: number }\nreturns: number\n---\nDouble x.' };

test('a call recorded with its notes in the system prompt is rebuilt into exactly what the runtime sends', async () => {
  const live = await liveRequest(ownCode, 'solve.nl');
  const opening = String(live.messages[1].content);
  const notesAt = opening.indexOf(FUNCTION_TOOLS_PROMPT.trim());
  assert.ok(notesAt > 0, 'the runtime puts the function-tools paragraph in the call opening');
  // The earlier layout: the call's paragraphs and guidance in the system prompt, the opening without them, and a
  // return_result typed by the call.
  const recorded = [{ role: 'system', content: `${TOOLS_PROMPT}${FUNCTION_TOOLS_PROMPT}\n\nKeep answers short.` },
    { role: 'user', content: opening.slice(0, notesAt).trimEnd() }, ...live.messages.slice(2)];
  const { messages, tools, report } = servingRequest({ messages: recorded, tools: [] }, { entry: 'runtime', bank: pieceBank() });
  assert.equal(report.mode, 'serving-runtime', JSON.stringify(report));
  assert.deepEqual(report.call_surface, { depthLimited: false, ownCode: true, folder: null, fileTools: 'all' });
  assert.deepEqual(tools, live.tools);
  assert.equal(messages[0].content, live.messages[0].content);
  assert.equal(messages[1].content, `${opening.slice(0, notesAt).trimEnd()}\n\nKeep answers short.\n\n${opening.slice(notesAt)}`);
  assert.deepEqual(messages.slice(2), live.messages.slice(2));
  // Rebuilding a request of the current layout changes nothing.
  assert.equal(servingRequest({ messages: live.messages, tools: live.tools }, { entry: 'runtime', bank: pieceBank() }).report.reason,
    'notes already in the call opening');
});

test('both entry points serve the one system prompt, and other records stay as recorded', () => {
  const recorded = [{ role: 'system', content: TOOLS_PROMPT }, { role: 'user', content: 'You are inside this call: f(): number\n\nInstructions:\nReturn 1.' }];
  for (const entry of ['collector', 'runtime'])
    assert.equal(servingRequest({ messages: recorded }, { entry, bank: pieceBank() }).messages[0].content, `${TOOLS_PROMPT}${GENERATION_GUIDANCE}`);
  const { messages } = servingRequest({ messages: recorded }, { entry: 'collector', bank: pieceBank() });
  assert.equal(messages[1].content, recorded[1].content);
  assert.equal(servingRequest({ messages: [{ role: 'system', content: 'Judge it.' }] }, { entry: 'runtime', bank: pieceBank() }).report.reason,
    'not a tool-loop call');
  const lacking = [...recorded, { role: 'assistant', tool_calls: [{ function: { name: 'write_file', arguments: '{}' } }] }];
  assert.deepEqual(servingRequest({ messages: lacking }, { entry: 'runtime', bank: pieceBank() }).report.tools, ['write_file']);
});

test('the Python bridge answers identity, serving and render requests', () => {
  const lines = [{ op: 'identity' }, { op: 'serving', entry: 'runtime',
    messages: [{ role: 'system', content: TOOLS_PROMPT }, { role: 'user', content: 'You are inside this call: f(): number\n\nInstructions:\nReturn 1.' }] },
  { op: 'render', text: `${TOOLS_PROMPT}\n\nKeep answers short.`, mode: 'program-header' }, { op: 'nothing' }];
  const out = execFileSync(process.execPath, [new URL('../scripts/serving-request.mjs', import.meta.url).pathname],
    { input: lines.map(line => JSON.stringify(line)).join('\n') + '\n', encoding: 'utf8' }).trim().split('\n').map(line => JSON.parse(line));
  assert.match(out[0].pieces_sha256, /^[0-9a-f]{64}$/);
  assert.equal(out[1].messages[0].content, SYSTEM_PROMPT);
  assert.match(out[2].text, /Keep answers short\.$/);
  assert.doesNotMatch(out[2].text, /return_result/);
  assert.match(out[3].error, /unknown op/);
});
