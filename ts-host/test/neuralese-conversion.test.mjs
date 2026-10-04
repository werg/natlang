import assert from 'node:assert/strict';
import { test } from 'node:test';
import { convertTrajectory } from '../dist/compiler/neuralese-conversion.js';
import { COMPACTION_NOTICE, GENERATION_GUIDANCE, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN, TOOLS_PROMPT } from '../dist/native/prompt.js';
import { programGuidance } from '../dist/adaptation/prompts.js';

const note = 'Checked lines 1-3; line 2 is a fee with no PO line. Left: prior invoices.';
const compact = (id, text) => ({ role: 'assistant', content: '', tool_calls: [{ id, type: 'function',
  function: { name: 'compact_history', arguments: JSON.stringify({ note: text }) } }] });
const record = () => ({ id: 'r1', messages: [
  { role: 'system', content: TOOLS_PROMPT + GENERATION_GUIDANCE + programGuidance('Prefer the contract over emails.') },
  { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nIs line 2 a fee?\n\nIn eval you can use state.' },
  { role: 'user', content: HANDOVER_NOTE_OPEN + note + HANDOVER_NOTE_CLOSE },
  compact('c1', note),
  { role: 'tool', tool_call_id: 'c1', content: 'Compacted.' },
  { role: 'assistant', content: '', tool_calls: [{ id: 'e1', type: 'function', function: { name: 'eval',
    arguments: JSON.stringify({ code: 'const v = await nl<boolean>`Is it a fee?`(state);' }) } }] },
  { role: 'tool', tool_call_id: 'e1', content: 'console:\ntrue' + COMPACTION_NOTICE },
], target: compact('c2', 'Line 2 is a fee. Return true.') });

test('prompts, guidance and handover notes become Neuralese; the rest is counted', () => {
  const { record: out, pieces } = convertTrajectory(record());
  const system = out.messages[0].content;
  assert.deepEqual(system.filter(p => p.type === 'soft').map(p => p.name).slice(0, 2), ['prompt:interpreter', 'prompt:generation-guidance']);
  assert.ok(system.some(p => p.type === 'soft' && p.name.startsWith('guidance@')), 'program guidance is its own soft parameter');
  assert.ok(!JSON.stringify(system).includes('You are running one call'));

  const pinned = out.messages[2].content;
  assert.deepEqual(pinned.map(p => p.type), ['soft', 'read', 'soft']);
  const write = JSON.parse(out.messages[3].tool_calls[0].function.arguments).note.$write;
  assert.equal(write.name, pinned[1].name, 'the pinned note reads the block the compaction call writes');
  assert.equal(write.source, note);
  assert.equal(write.type, 'Neuralese<HandoverNote>');
  assert.ok(JSON.parse(out.target.tool_calls[0].function.arguments).note.$write, 'a compaction target is a write');

  assert.ok(out.messages[6].content.some(p => p.type === 'soft' && p.name === 'prompt:compaction-notice'));
  assert.equal(typeof out.messages[1].content, 'string', 'instructions wait for their curriculum step');
  const { sites } = out.neuralese_conversion;
  assert.equal(sites['handover-write'].converted, 2);
  assert.equal(sites['handover-read'].converted, 1);
  assert.equal(sites['nl-literal'].exact['later-curriculum-step'], 1);
  assert.equal(sites['tool-output'].exact['later-curriculum-step'], 2);
  assert.equal(sites.instructions.exact['later-curriculum-step'], 1);
  assert.ok(pieces.some(p => p.name === 'prompt:interpreter' && p.text === TOOLS_PROMPT), 'pieces carry their initial text');
});

test('instructions become soft bodies when asked; unregistered system text is a versioned piece', () => {
  const input = record();
  input.messages[0].content = 'An older runtime prompt.';
  const { record: out, pieces } = convertTrajectory(input, { convert: ['instructions'] });
  assert.match(out.messages[0].content[0].name, /^prompt:system@[0-9a-f]{12}$/);
  const body = out.messages[1].content.find(p => p.type === 'soft');
  assert.match(body.name, /^instructions@/);
  assert.equal(pieces.find(p => p.name === body.name).text, 'Is line 2 a fee?');
  assert.equal(out.neuralese_conversion.sites.instructions.converted, 1);
});

test('tool outputs: copied exact values keep an output exact; model-only outputs are encoded when asked', () => {
  const call = (id, code) => ({ role: 'assistant', content: '', tool_calls: [{ id, type: 'function', function: { name: 'eval', arguments: JSON.stringify({ code }) } }] });
  const input = { id: 'r2', messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: f(x: string): boolean\n\nInstructions:\nDecide.\n\nIn eval you can use x.' },
    call('e1', 'console.log(x)'),
    { role: 'tool', tool_call_id: 'e1', content: 'console:\ninvoice INV-20931 total 1734.50 for "Acme Industrial"' },
    call('e2', 'const id = "INV-20931";'),
    { role: 'tool', tool_call_id: 'e2', content: 'console:\nThe vendor seems reliable and the tone is polite.' },
  ], target: { role: 'assistant', content: 'true' } };
  const counted = convertTrajectory(input).record.neuralese_conversion.sites['tool-output'];
  assert.deepEqual(counted, { converted: 0, exact: { 'copied-exact-values': 1, 'later-curriculum-step': 1 } });
  const { record: out } = convertTrajectory(input, { convert: ['tool-outputs'] });
  assert.equal(typeof out.messages[3].content, 'string', 'an output whose ID is copied stays exact');
  const [part] = out.messages[5].content;
  assert.equal(part.type, 'encode');
  assert.match(part.source, /tone is polite/);
  assert.equal(out.neuralese_conversion.sites['tool-output'].converted, 1);
});
