import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { codemodeDriver, runAgent, systemPrompt } from '../../applications/dist/pi/agent.js';
import { runTool, truncate, MAX_LINES } from '../../applications/dist/pi/tools.js';
import { scriptedModel } from './support/natlang.mjs';

test('pi tools keep pi semantics: unique non-overlapping edits, offsets, tail-truncated bash', async () => {
  const cwd = mkdtempSync(join(tmpdir(), 'pi-tools-'));
  writeFileSync(join(cwd, 'a.txt'), 'one\ntwo\nthree\ntwo again\n');
  assert.match((await runTool('edit', { path: 'a.txt', edits: [{ oldText: 'two', newText: '2' }] }, cwd)).text, /not unique/);
  assert.match((await runTool('edit', { path: 'a.txt', edits: [{ oldText: 'four', newText: '4' }] }, cwd)).text, /Could not find/);
  assert.match((await runTool('edit', { path: 'a.txt', edits: [{ oldText: 'one\ntwo', newText: 'x' }, { oldText: 'two\nthree', newText: 'y' }] }, cwd)).text, /overlap/);
  const edited = await runTool('edit', { path: 'a.txt', edits: [{ oldText: 'three', newText: '3' }, { oldText: 'one', newText: '1' }] }, cwd);
  assert.ok(edited.ok);
  assert.equal(readFileSync(join(cwd, 'a.txt'), 'utf8'), '1\ntwo\n3\ntwo again\n');
  assert.equal((await runTool('read', { path: 'a.txt', offset: 2, limit: 2 }, cwd)).text, 'two\n3');
  const long = await runTool('bash', { command: `seq 1 ${MAX_LINES + 500}; exit 3` }, cwd);
  assert.equal(long.ok, false);
  assert.match(long.text, /\[Output truncated; full output: .*\]\n\[Exit code 3\]$/);
  assert.ok(long.text.startsWith('501\n'));
  assert.equal(truncate('a\nb', 'head').truncated, false);
});

test('runtime.decide reports the scored distribution, or the sampled value when the driver cannot score', async () => {
  const { default: progress } = await import('../../applications/dist/pi/system1/progress.nl.js');
  const plain = scriptedModel(() => 'return "stuck";');
  const unscored = await createNatlangRuntime({ model: plain.driver }).decide(progress, 'fix the build', ['ran make: failed']);
  assert.deepEqual(unscored, { value: 'stuck', probabilities: [{ value: 'stuck', probability: 1 }], confidence: 1, scored: false });
  const scored = Object.assign(async () => { throw new Error('scored decisions need no generation'); },
    { decide: async ({ options }) => ({ log_probs: options.map(option => Math.log(option === '"repeating"' ? 0.7 : 0.15)) }) });
  const decision = await createNatlangRuntime({ model: scored }).decide(progress, 'fix the build', ['ran make: failed', 'ran make: failed']);
  assert.equal(decision.value, 'repeating');
  assert.equal(decision.scored, true);
  assert.ok(Math.abs(decision.confidence - 0.7) < 1e-9);
  assert.deepEqual(decision.probabilities.map(item => item.value).sort(), ['progressing', 'repeating', 'stuck']);
});

test('the agent loop runs pi tools while System One gates, digests, reviews, steers, checks and runs codemode', async () => {
  const cwd = mkdtempSync(join(tmpdir(), 'pi-agent-'));
  writeFileSync(join(cwd, 'a.txt'), 'foo is here\nand foo again\n');
  // The small model: generative calls are scripted, decisions are scored from what the messages show.
  const small = scriptedModel(opening => {
    if (opening.includes('Keep what the agent needs to act')) return 'return "numbers 1 to 200, nothing failed";';
    if (opening.includes('Decide whether text mentions bar')) return 'return input.includes("bar");';
    return null;
  });
  const decided = [];
  const decide = async ({ messages, options }) => {
    const shown = JSON.stringify(messages);
    const table = shown.includes('wants to run command') ? (shown.includes('DANGER') ? { destructive: 0.9 } : { safe: 0.95 })
      : shown.includes('intent is what it said') ? { incomplete: 0.8 }
      : shown.includes('recent lists a coding agent') ? { repeating: 0.7 }
      : shown.includes('says it is finished with task') ? (shown.includes('All done.') ? { unfinished: 0.9 } : { done: 0.95 }) : {};
    const [winner, p] = Object.entries(table)[0] ?? ['', 1 / options.length];
    decided.push(winner);
    return { log_probs: options.map(option => Math.log(option === JSON.stringify(winner) ? p : (1 - p) / (options.length - 1))) };
  };
  const scripts = {};
  const runtime = createNatlangRuntime({ model: codemodeDriver(Object.assign(small.driver, { decide }), scripts) });

  const turns = [
    { text: 'Clean up first.', calls: [['bash', { command: 'echo DANGER && touch ran.txt' }]] },
    { calls: [['bash', { command: 'seq 1 200' }]] },
    { text: 'I will rename foo to bar in a.txt.', calls: [['edit', { path: 'a.txt', edits: [{ oldText: 'foo is here', newText: 'bar is here' }] }]] },
    { calls: [['codemode', { script: 'const out = await shell.run("cat a.txt");\nconst yes = await nl<boolean>`Decide whether text mentions bar.`(out.output);\nconsole.log(`mentions bar: ${yes}`);' }]] },
    { text: 'All done.' },
    { text: 'Finished: the first foo is renamed.' },
  ];
  const seen = [];
  const big = async request => { seen.push(request); return turns[seen.length - 1]; };
  const session = join(cwd, 'session.jsonl');
  const result = await runAgent({ task: 'Rename foo to bar in a.txt.', cwd, big, runtime, scripts, session,
    systemOne: { progressEvery: 4 } });

  assert.deepEqual(result.interventions.map(item => `${item.kind}:${item.action}${item.error ? `: ${item.error}` : ''}`), ['scout:too few files to scout', 'risk:refused',
    'risk:allowed', 'digest:digested 200 lines', 'review:flagged', 'risk:allowed', 'progress:steered', 'done:sent back']);
  assert.equal(result.stopped, 'answered');
  assert.equal(result.answer, 'Finished: the first foo is renamed.');
  assert.equal(existsSync(join(cwd, 'ran.txt')), false, 'the destructive command never ran');
  const results = seen.at(-1).messages.filter(message => message.role === 'tool').map(message => message.content);
  assert.match(results[0], /Refused by the safety check/);
  assert.match(results[1], /^numbers 1 to 200, nothing failed\n\[A small model condensed 200 lines/);
  assert.match(results[2], /Edited a\.txt.*\n\[Edit review: this change looks incomplete/);
  assert.match(results[3], /mentions bar: true/);
  assert.match(results[3], /\[Progress check: your recent actions look repetitive/);
  assert.match(seen.at(-1).messages.at(-1).content, /A quick check judged this unfinished/);
  assert.equal(readFileSync(join(cwd, 'a.txt'), 'utf8'), 'bar is here\nand foo again\n');
  const entries = readFileSync(session, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.equal(entries[0].type, 'session');
  assert.equal(entries.at(-1).type, 'result');
  assert.ok(entries.filter(entry => entry.type === 'system_one').length === 8);
  assert.ok(seen[0].tools.some(tool => tool.function.name === 'codemode'));
  assert.match(seen[0].messages[0].content, /^You are an expert coding assistant operating inside pi/);
  assert.match(systemPrompt(cwd, { codemode: false, systemOne: false }), /<cwd>\n/);
});
