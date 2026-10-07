import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeToolAgent, collapseHistory } from '../dist/native/agent.js';
import { session as open } from './support/natlang.mjs';

const automaticNote = message => message.role === 'user' && /^This conversation reached its context limit/.test(message.content);

test('compaction keeps the opening, the note, and the latest exchange; everything between leaves the conversation', () => {
  const messages = [{ role: 'system', content: 'system' }, { role: 'user', content: 'task' }, { role: 'user', content: 'note' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', tool_call_id: 'a', content: 'one' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', tool_call_id: 'b', content: 'two' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', tool_call_id: 'c', content: 'three' }];
  assert.equal(collapseHistory(messages, 3), 4);
  assert.deepEqual(messages.map(message => message.content), ['system', 'task', 'note', '', 'three']);
  assert.equal(collapseHistory(messages, 3), 0, 'the latest exchange itself is never removed');
});

test('a long call compacts old outputs instead of rolling over, and the model can read them back from transcript', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  const requests = [];
  let turn = 0, recovered;
  // Each eval prints about 2,000 characters; the server reports 1 token per 4 characters of request.
  const driver = request => {
    requests.push(structuredClone(request.messages));
    const prompt = Math.round((JSON.stringify(request.messages).length + JSON.stringify(request.tools).length) / 4);
    turn++;
    if (turn < 12) return { calls: [['eval', { code: `console.log('y'.repeat(2000) + ' mark${turn}'); ${turn}` }]], prompt_tokens: prompt };
    if (turn === 12) return { calls: [['eval', { code: 'transcript.search(/ mark1\\b/, { in: "output" }).map(match => match.turn)' }]], prompt_tokens: prompt };
    recovered = requests.at(-1).at(-1).content;
    return { calls: [['return_result', { status: 'success', value: turn }]], prompt_tokens: prompt };
  };
  await new NativeToolAgent(driver, { contextTokens: 4096, maxTurns: 20 }).run(session);
  assert.equal(session.lam.return, 13);
  const last = requests.at(-1);
  assert.match(recovered, /\[\s*1\s*\]/, 'an eval found the first output in transcript after it left the conversation');
  assert.equal(last.filter(automaticNote).length, 1, 'the automatic note says where the earlier turns are');
  assert.ok(!JSON.stringify(last).includes('mark1\''), 'the first turns left the conversation');
  assert.ok(requests.every(request => Math.round(JSON.stringify(request).length / 4) < 4096), 'every prompt stays within the budget');
});

test('without a budget nothing is compacted, and a parameter named transcript keeps its meaning', async () => {
  const { session } = open({ type: '(transcript: string) => string', instructions: 'Echo.', args: { transcript: 'mine' } });
  let turn = 0, last;
  const driver = request => { last = request.messages; turn++;
    return turn < 8 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${turn}` }]], prompt_tokens: 100000 } :
      { calls: [['eval', { code: 'return transcript' }]] }; };
  await new NativeToolAgent(driver, { contextTokens: null, maxTurns: 20 }).run(session);
  assert.equal(last.some(automaticNote), false);
  assert.ok(last.length > 14, 'every turn is still in the conversation');
  assert.equal(session.lam.return, 'mine');
});

const toolNames = request => request.tools.map(tool => tool.function.name);
// The managed local default window. The opening (system prompt and task) takes about 3,400 tokens, so a 4,096 window
// would leave no room for a turn's reply (a quarter of the window) and nothing for compaction to work with.
const CONTEXT = 8192;
const promptOf = request => Math.round((JSON.stringify(request.messages).length + JSON.stringify(request.tools).length) / 4);

test('near the budget the model is asked to compact: only compact_history is offered, and its note is pinned', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  const requests = [];
  let turn = 0, notes = 0;
  const driver = request => {
    requests.push(structuredClone(request));
    turn++;
    if (toolNames(request).length === 1 && toolNames(request)[0] === 'compact_history')
      return { calls: [['compact_history', { note: `Counting; reached ${turn}; stop at 16. Note ${++notes}.` }]], prompt_tokens: promptOf(request) };
    return turn < 16 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${turn}` }]], prompt_tokens: promptOf(request) } :
      { calls: [['return_result', { status: 'success', value: turn }]], prompt_tokens: promptOf(request) };
  };
  await new NativeToolAgent(driver, { contextTokens: CONTEXT, maxTurns: 30 }).run(session);
  assert.equal(session.completed, true);
  const forced = requests.filter(request => toolNames(request).join() === 'compact_history');
  assert.ok(forced.length >= 2, 'the model was asked to compact, more than once in a long call');
  assert.ok(forced.every(request => request.tool_choice === 'required'), 'the compaction turn requires the tool call');
  assert.ok(forced.every(request => /near its context limit/.test(request.messages.at(-1).content)), 'the compaction turn says why');

  assert.ok(requests.filter(request => toolNames(request).length > 1).every(request => request.tool_choice === undefined));
  const kinds = requests.map(request => toolNames(request).join() === 'compact_history');
  assert.equal(kinds.some((forcedTurn, index) => forcedTurn && kinds[index + 1]), false, 'compaction never repeats back to back');
  const last = requests.at(-1);
  const pinned = last.messages.filter(message => message.role === 'user' && /^Your note from compacting/.test(message.content));
  assert.equal(pinned.length, 1, 'only the latest note is kept');
  assert.match(pinned[0].content, new RegExp(`Note ${notes}\\.`));
  assert.match(pinned[0].content, /Continue from where this note leaves off\. Look into the history only when something specific matters/);
  assert.ok(last.messages.length < requests.length, 'turns before the last compaction left the conversation');
  assert.ok(requests.every(request => promptOf(request) < CONTEXT), 'no request exceeded the window');
  assert.ok(session.transcript.some(entry => entry.tool === 'compact_history'), 'the compaction is part of the transcript');
});

test('the model may compact on its own, a note over the limit is rejected, and a text reply to the compaction turn is not a result', async () => {
  const { session } = open({ type: '() => string', instructions: 'Say hello.' });
  let turn = 0, sawRejection = false;
  const driver = request => {
    turn++;
    if (turn === 1) return { calls: [['eval', { code: "console.log('z'.repeat(3000)); 1" }]] };
    if (turn === 2) return { calls: [['compact_history', { note: 'x'.repeat(2001) }]] };
    if (turn === 3) { sawRejection = /1 to 2000 characters/.test(request.messages.at(-1).content);
      return { calls: [['compact_history', { note: 'Printed a long line; next say hello.' }]] }; }
    return { calls: [['return_result', { status: 'success', value: 'hello' }]] };
  };
  await new NativeToolAgent(driver, { maxTurns: 10 }).run(session);
  assert.equal(sawRejection, true);
  assert.equal(session.lam.return, 'hello');
  const forcedText = open({ type: '() => string', instructions: 'Say hello.' });
  let calls = 0;
  const texting = request => {
    calls++;
    if (toolNames(request).join() === 'compact_history') return { text: 'I would rather not.', prompt_tokens: promptOf(request) };
    return calls < 10 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${calls}` }]], prompt_tokens: promptOf(request) } :
      { calls: [['return_result', { status: 'success', value: 'hello' }]], prompt_tokens: promptOf(request) };
  };
  await new NativeToolAgent(texting, { contextTokens: 4096, maxTurns: 30 }).run(forcedText.session);
  assert.equal(forcedText.lam.return, 'hello', 'the text reply to the compaction turn did not become the result');
});

test('transcript is searched, not read through: search finds lines, entry gives one call, printing shows a summary', async () => {
  const { session } = open({ type: '() => number', instructions: 'Look around.' });
  await session.applyAsync('eval', { code: 'console.log("alpha\\nthe locker is locked\\nomega"); 1' });
  await session.applyAsync('eval', { code: 'const key = "brass"; console.log("found a brass key"); 2' });
  const found = await session.applyAsync('eval', { code: 'transcript.search("LOCKER").map(match => [match.entry, match.in, match.line])' });
  assert.match(found.text, /\[0, "code", .*\], \[0, "output", "the locker is locked"\]\]/, 'matches in code and output, line by line');
  const regex = await session.applyAsync('eval', { code: 'transcript.search(/brass/, { in: "code" }).length' });
  assert.match(regex.text, /^1\b/);
  const one = await session.applyAsync('eval', { code: 'transcript.entry(-1).code' });
  assert.match(one.text, /transcript\.search\(\/brass\//, 'entry(-1) is the latest earlier call');
  const printed = await session.applyAsync('eval', { code: 'console.log(transcript); String(transcript)' });
  assert.match(printed.text, /transcript: 5 earlier calls; use transcript\.search\(query\) or transcript\.entry\(n\)/);
  assert.equal(/locker is locked/.test(printed.text), false, 'printing the transcript does not dump it');
  await session.applyAsync('eval', { code: 'const rows = [{ id: 1 }, { id: 2 }]; return undefined.boom' });
  const record = await session.applyAsync('eval', { code: 'const first = transcript.entry(1); [first.status, first.value, first.console, transcript.entry(-1).status]' });
  assert.match(record.text, /\["ok", 2, "found a brass key", "error"\]/, 'entries carry status, the returned value as data, and console output');
  const failures = await session.applyAsync('eval', { code: 'transcript.search("", { status: "error" }).map(match => match.entry)' });
  assert.match(failures.text, /^\[6(, 6)*\]/, 'search can be restricted to failed calls');
  const indexed = await session.applyAsync('eval', { code: 'transcript[0]' });
  assert.equal(/locker/.test(indexed.text), false, 'there is no array access');
});

test('transcript keeps the reasoning of the turn that made each call, and search finds it', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count.' });
  let turn = 0, found;
  const driver = request => {
    turn++;
    if (turn === 1) return { reasoning: 'The rows need a total; the hidden column holds the discount.', calls: [
      ['eval', { code: 'const a = 1; a' }], ['eval', { code: 'const b = 2; b' }]] };
    if (turn === 2) return { reasoning: 'Look back at why I added b.', calls: [['eval', { code: 'transcript.search("discount", { in: "reasoning" }).map(m => [m.entry, m.turn])' }]] };
    found = request.messages.at(-1).content;
    return { calls: [['return_result', { status: 'success', value: 3 }]] };
  };
  await new NativeToolAgent(driver, { maxTurns: 5 }).run(session);
  assert.equal(session.transcript[0].reasoning, 'The rows need a total; the hidden column holds the discount.');
  assert.equal(session.transcript[1].reasoning, undefined, 'a turn\'s reasoning is kept once, with its first call');
  assert.equal(session.transcript[2].reasoning, 'Look back at why I added b.');
  assert.match(found, /\[\[0, 1\]\]/);
});


test('a request the server refuses as too long teaches the agent its true size, and the turn is taken again, compacted', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  let turn = 0, refused = 0;
  const sizes = [];
  const driver = request => {
    const size = promptOf(request);
    if (size >= CONTEXT) {
      refused++;
      throw new Error(`model HTTP 400: {"error":{"code":400,"type":"exceed_context_size_error","n_prompt_tokens":${size},"n_ctx":${CONTEXT}}}`);
    }
    sizes.push(size); turn++;
    // The server's reported size is half the truth, so the agent's own estimate runs low until it is corrected.
    const reported = Math.round(size / 2);
    if (toolNames(request).join() === 'compact_history')
      return { calls: [['compact_history', { note: `Counting; reached ${turn}.` }]], prompt_tokens: reported };
    return turn < 14 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${turn}` }]], prompt_tokens: reported } :
      { calls: [['return_result', { status: 'success', value: turn }]], prompt_tokens: reported };
  };
  await new NativeToolAgent(driver, { contextTokens: CONTEXT, maxTurns: 40 }).run(session);
  assert.equal(session.completed, true);
  assert.ok(refused >= 1, 'the server refused at least one request');
  assert.ok(sizes.every(size => size < CONTEXT));
});

test('a turn is limited to a quarter of the window, and long reasoning leaves the conversation with its turn', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  const requests = [];
  let turn = 0;
  const driver = request => {
    requests.push(structuredClone(request)); turn++;
    const long = 'Thinking hard. '.repeat(400);
    if (request.tools.map(tool => tool.function.name).join() === 'compact_history')
      return { calls: [['compact_history', { note: `Counting; reached ${turn}.` }]], prompt_tokens: promptOf(request) };
    return turn < 12 ? { reasoning: long, calls: [['eval', { code: `${turn}` }]], prompt_tokens: promptOf(request) } :
      { calls: [['return_result', { status: 'success', value: turn }]], prompt_tokens: promptOf(request) };
  };
  await new NativeToolAgent(driver, { contextTokens: CONTEXT, maxTurns: 30 }).run(session);
  assert.equal(session.completed, true);
  assert.ok(requests.every(request => request.max_tokens === CONTEXT / 4), 'each turn may use at most a quarter of the window');
  assert.ok(requests.every(request => promptOf(request) < CONTEXT));
  assert.ok(session.transcript.every(entry => entry.tool !== 'eval' || /Thinking hard/.test(entry.reasoning ?? '')),
    'the reasoning that left the conversation is still in transcript');
});
