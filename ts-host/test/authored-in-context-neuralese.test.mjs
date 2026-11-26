// Authored trajectories read Neuralese as the serving engines do (model/in-context-neuralese.ts): learned compaction's
// operator call is a call of `view` answered by its entry's summary, captured in its own context and spliced where its
// consumer reads it (a native_state part); a Neuralese<string> result is its producing call's text, read in place by its
// consumer. No sentinel reaches a request, and the runtime's own operator and typed-write paths run.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { replayReference, OPERATOR_CALL } from '../dist/teacher/curriculum.js';
import { createInContextNeuralese } from '../dist/model/in-context-neuralese.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { hasNeuraleseSentinel } from '../dist/native/neuralese.js';

const log = Array.from({ length: 400 }, (_, i) => `2026-11-2${i % 9} 10:${String(i % 60).padStart(2, '0')} svc-${i % 7} ok heartbeat ${i}\n`).join('');
const SUMMARY = 'A heartbeat log of 400 lines from svc-0 to svc-6; every line reports ok.';
const GIST = 'Every service reported ok heartbeats; nothing failed.';
const BRIEFING = 'All seven services stayed healthy. No failures. Nothing to hand over.';
const record = { version: 'natlang.program/2', id: 'authored-neuralese', kind: 'lambda_source', source: 'fixture', split: 'train',
  source_ids: ['authored-neuralese'], source_groups: ['authored-neuralese'], license: 'test', family: 'fixture',
  semantics: { root: 'brief.nl', inputs: { log }, expected: BRIEFING, oracle: 'exact', files: {
    'brief.nl': '---\nargs:\n  log: string\nreturns: string\n---\nGist the log with gistLog, then write the briefing with writeBriefing.\n',
    'brief/gistLog.nl': '---\nargs:\n  log: string\nreturns: Neuralese<string>\n---\nWrite the gist of the log: which services failed and when.\n',
    'brief/writeBriefing.nl': '---\nargs:\n  gist: Neuralese<string>\nreturns: string\n---\nWrite a three-sentence briefing for the on-call engineer from the gist.\n' } },
  curriculum: { reference: { root: [['eval', { code: 'const g = await gistLog(log);\nreturn await writeBriefing(g);', finish: true }]],
    children: [
      { match: [`You are inside this call: ${OPERATOR_CALL}(`], value: SUMMARY },
      { match: ['You are inside this call: gistLog('], value: GIST },
      { match: ['You are inside this call: writeBriefing('], value: BRIEFING }] } } };

test('learned compaction and Neuralese values replay in their serving form', async () => {
  const turns = [];
  const neuralese = createInContextNeuralese({ compaction: 'nl' });
  const { run } = await replayReference(record, TOOLS_PROMPT, () => '', {
    neuralese,
    onTurn: (request, response) => turns.push({ request, response }) });
  assert.equal(run.outcome.accepted, true, JSON.stringify(run.outcome.detail ?? run.outcome.status).slice(0, 800));
  assert.equal(run.outcome.value, BRIEFING);
  for (const { request } of turns) assert.ok(!hasNeuraleseSentinel(JSON.stringify(request.messages)), 'no state sentinel reaches a request');
  const operators = turns.filter(({ request }) => /#operator-\d+$/.test(request.invocation_id ?? ''));
  assert.ok(operators.length >= 1, 'the long argument gets an operator call');
  for (const { request, response } of operators) {
    assert.deepEqual(request.tools, []);
    assert.equal(response.text, SUMMARY);
  }
  const shown = name => turns.find(({ request }) => String(request.messages[1]?.content ?? '').startsWith(`You are inside this call: ${name}(`));
  // the operator's summary is a capture, which engines splice: the consumer's listing holds it as a native_state part
  // where the value stands, and the transport names the operator call that produced it
  const listing = shown('gistLog').request.messages.find(m => Array.isArray(m.content));
  assert.ok(listing, 'the consumer reads the captured state as a content part');
  const part = listing.content.find(p => p.type === 'native_state');
  assert.deepEqual(Object.keys(part).sort(), ['id', 'type']);
  assert.ok(listing.content[0].text.endsWith('log: string = ') && listing.content.at(-1).text.startsWith('  // view of the value'));
  assert.ok(!JSON.stringify(shown('gistLog').request.messages).includes(SUMMARY));
  const capture = neuralese.captures().find(c => c.id === part.id);
  assert.equal(capture.text, SUMMARY);
  assert.match(capture.invocation_id, /#operator-\d+$/);
  assert.equal(capture.turn, 0);
  assert.ok(JSON.stringify(shown('writeBriefing').request.messages).includes(`gist: Neuralese<string> = ${GIST}`),
    'a Neuralese<string> argument (a span) reads as its text in place');
});

test('without a Neuralese transport the replay keeps the crisp cut and cannot write a Neuralese<string> result', async () => {
  const turns = [];
  await assert.rejects(replayReference(record, TOOLS_PROMPT, () => '', { onTurn: request => turns.push(request) }),
    /plain-text Neuralese<string> cannot be written/);
  assert.ok(turns.length && !turns.some(request => /#operator-/.test(request.invocation_id ?? '')));
});
