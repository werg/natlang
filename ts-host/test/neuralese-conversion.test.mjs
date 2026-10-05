import assert from 'node:assert/strict';
import { test } from 'node:test';
import { childCallIds, childFunctionNames, childReturn, convertTrajectory, instructionsDigest, printedResults } from '../dist/compiler/neuralese-conversion.js';
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
  assert.equal(typeof out.messages[1].content, 'string', 'single-use instructions stay text');
  const { sites } = out.neuralese_conversion;
  assert.equal(sites['handover-write'].converted, 2);
  assert.equal(sites['handover-read'].converted, 1);
  assert.equal(sites['nl-literal'].exact['later-curriculum-step'], 1);
  assert.equal(sites['tool-output'].exact['single-use'], 1);
  assert.equal(sites['child-result'].exact['producer-missing'], 1, 'without the run\'s child returns a printed result stays exact');
  assert.equal(sites.instructions.exact['single-use'], 1);
  assert.ok(pieces.some(p => p.name === 'prompt:interpreter' && p.text === TOOLS_PROMPT), 'pieces carry their initial text');
});

test('instructions used by several calls become one shared soft parameter; unregistered system text is versioned', () => {
  const input = record();
  input.messages[0].content = 'An older runtime prompt.';
  const digest = instructionsDigest('Is line 2 a fee?');
  const { record: out, pieces } = convertTrajectory(input, { instructionCalls: new Map([[digest, 3]]), instructionsShare: 0 });
  assert.match(out.messages[0].content[0].name, /^prompt:system@[0-9a-f]{12}$/);
  const body = out.messages[1].content.find(p => p.type === 'soft');
  assert.equal(body.name, `instructions@${digest}`);
  assert.equal(pieces.find(p => p.name === body.name).text, 'Is line 2 a fee?');
  assert.equal(out.neuralese_conversion.sites['instructions-reused'].converted, 1);
  const single = convertTrajectory(record(), { instructionCalls: new Map([[digest, 1]]), instructionsShare: 1 }).record;
  assert.equal(single.neuralese_conversion.sites['instructions-coverage'].converted, 1, 'a share of single-use instructions converts for coverage');
});

test('a cut-off value in the root call\'s opening listing becomes a digest site of the full value', () => {
  const full = { task: 'Review the packet.', lines: Array.from({ length: 50 }, (_, i) => ({ id: `L${i}`, amount: i * 10 })) };
  const input = { id: 'r3', task: { program_ir: { semantics: { root: 'judge.nl', inputs: { state: full } } } }, messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nDecide.\n\nIn eval you can use state.' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'scope_0', type: 'function', function: { name: 'eval', arguments: '{"code":"const state = read_inputs().state;"}' } }] },
    { role: 'tool', tool_call_id: 'scope_0', content: 'state: unknown = { task: "Review the packet.", <<cut off: 1 of 2 fields not shown; state holds all of it>> }\nDeclared state for the rest of this call.' },
  ], target: { role: 'assistant', content: 'true' } };
  const { record: out } = convertTrajectory(input);
  const parts = out.messages[3].content;
  const digest = parts.find(p => p.type === 'digest');
  assert.deepEqual(JSON.parse(digest.source), full);
  assert.match(digest.preview, /cut off/);
  assert.equal(parts[0].text, 'state: unknown = ');
  assert.equal(out.neuralese_conversion.sites.digest.converted, 1);
  const nested = convertTrajectory({ ...input, task: undefined }).record;
  assert.equal(nested.neuralese_conversion.sites.digest.exact['full-value-unavailable'], 1);
});

test('a child call\'s returned value that its caller prints becomes a write in the child and a read in the caller', () => {
  const summary = 'Line 2 bills 7.5 hours of on-site work at the contract rate.';
  const run = { source_ref: { trajectory_id: 'run-1' }, task: { program_ir: { semantics: { root: 'judge.nl' } } } };
  const child = { ...run, id: 'child', messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: summarise(line: string): string\n\nInstructions:\nSummarise the line.\n\nIn eval you can use line.' },
  ], target: { role: 'assistant', content: '', tool_calls: [{ id: 'r1', type: 'function', function: { name: 'return_result',
    arguments: JSON.stringify({ status: 'success', value: summary }) } }] } };
  const caller = { ...run, id: 'caller', messages: [
    { role: 'system', content: TOOLS_PROMPT },
    { role: 'user', content: 'You are inside this call: judge(state: unknown): boolean\n\nInstructions:\nIs line 2 a fee?\n\nIn eval you can use state.' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'e1', type: 'function', function: { name: 'eval',
      arguments: JSON.stringify({ code: 'const s = await nl`Summarise the line.`(state.lines[1]); console.log(s)' }) } }] },
    { role: 'tool', tool_call_id: 'e1', content: `console:\n${summary}\nStored local s.` },
  ], target: { role: 'assistant', content: 'true' } };
  assert.equal(childReturn(child), summary);
  assert.equal(childReturn({ ...child, messages: [child.messages[0], caller.messages[1]] }), undefined, 'the root call returns no child result');
  assert.deepEqual([...childCallIds(caller.messages)], ['e1']);
  const read = new Set(printedResults(caller.messages[3].content, [summary, 'true']));
  const childResults = new Map([['run-1', { returned: [summary, 'true'], read }]]);

  const written = convertTrajectory(child, { childResults }).record;
  const write = JSON.parse(written.target.tool_calls[0].function.arguments);
  assert.equal(write.status, 'success');
  assert.equal(write.value.$write.source, summary);
  assert.equal(write.value.$write.type, 'Neuralese<string>');

  const reading = convertTrajectory(caller, { childResults }).record;
  const parts = reading.messages[3].content;
  assert.deepEqual(parts.map(p => p.type), ['text', 'read', 'text']);
  assert.equal(parts[1].name, write.value.$write.name, 'the caller reads the block the child writes');
  assert.equal(parts.map(p => p.type === 'read' ? p.source : p.text).join(''), caller.messages[3].content);
  assert.equal(reading.neuralese_conversion.sites['child-result'].converted, 1);

  const short = convertTrajectory(caller, { childResults: new Map([['run-1', { returned: ['true'], read: new Set() }]]) }).record;
  assert.equal(short.neuralese_conversion.sites['child-result'].exact['crisp-value'], 1, 'a short value is its exact form');
});


test('named file calls are recognized and equal returns in independent runs stay separate', () => {
  const value = 'A private observation supports the contractual claim.';
  const make = run => ({ source_ref: { trajectory_id: run }, task: { program_ir: { semantics: {
    root: 'folder/review.nl', files: { 'folder/review.nl': '', 'folder/review/assess.nl': '' } } } },
    messages: [{ role: 'user', content: 'You are inside this call: review(): string' },
      { role: 'assistant', tool_calls: [{ id: 'e1', function: { name: 'eval', arguments: JSON.stringify({ code: 'await assess()' }) } }] },
      { role: 'tool', tool_call_id: 'e1', content: value }] });
  const a = make('run-a'), b = make('run-b');
  assert.deepEqual([...childCallIds(a.messages, childFunctionNames(a))], ['e1']);
  const childResults = new Map(['run-a', 'run-b'].map(id => [id, { returned: [value], read: new Set([value]) }]));
  const read = row => convertTrajectory(row, { childResults }).record.messages.at(-1).content.find(p => p.type === 'read');
  assert.equal(read(a).source, value);
  assert.notEqual(read(a).name, read(b).name, 'equal text in different executions is not one producer');
  assert.equal(childReturn({ ...a, target: { role: 'assistant', tool_calls: [{ function: {
    name: 'return_result', arguments: JSON.stringify({ status: 'success', value }) } }] } }), undefined,
    'a root inside a folder is still a root');
});
