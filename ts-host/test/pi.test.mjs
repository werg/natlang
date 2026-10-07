import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
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
  const { default: { progress } } = await import('../../applications/dist/pi/pi.nl.js');
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

test('route hands routine turns to the small model and keeps doubtful ones with the big model', async () => {
  const cwd = mkdtempSync(join(tmpdir(), 'pi-route-'));
  const decide = async ({ messages, options }) => {
    const shown = JSON.stringify(messages);
    const p = shown.includes('routine enough') || shown.includes('rerunning a command after a fix') ? (shown.includes('FIXED') ? 0.9 : 0.6) : 0.9;
    return { log_probs: options.map((option, index) => Math.log(index === 0 ? p : (1 - p) / (options.length - 1))) };
  };
  const smallTurns = [];
  const small = Object.assign(async request => {
    if (request.tools.some(tool => tool.function?.name === 'bash')) { smallTurns.push(request); return { text: 'Tests pass now.' }; }
    throw new Error('no generative System One calls in this test');
  }, { decide });
  const runtime = createNatlangRuntime({ model: codemodeDriver(small, {}) });
  const bigTurns = [{ calls: [['bash', { command: 'echo tests fail' }]] }, { calls: [['bash', { command: 'echo FIXED' }]] }];
  let bigCalls = 0;
  const big = async () => bigTurns[bigCalls++];
  const result = await runAgent({ task: 'Make the tests pass.', cwd, big, small, runtime, scripts: {},
    systemOne: { route: true, done: false, progress: false, scout: false } });
  assert.equal(result.answer, 'Tests pass now.');
  assert.equal(bigCalls, 2, 'the first turn and the doubtful second stay with the big model');
  assert.equal(result.smallTurns, 1);
  assert.deepEqual(result.interventions.filter(item => item.kind === 'route').map(item => item.action), ['big model', 'small model']);
});

test('pi skills: SKILL.md directories are listed for the model to read on demand; hidden ones are not', async () => {
  const { discoverSkills } = await import('../../applications/dist/pi/agent.js');
  const cwd = mkdtempSync(join(tmpdir(), 'pi-skills-')), home = mkdtempSync(join(tmpdir(), 'pi-home-'));
  const skill = (dir, text) => { mkdirSync(dir, { recursive: true }); writeFileSync(join(dir, 'SKILL.md'), text); };
  skill(join(cwd, '.pi/skills/release'), '---\nname: release\ndescription: "Cut a release: bump, tag & publish"\n---\nSteps.');
  skill(join(cwd, '.pi/skills/internal'), '---\ndescription: Internal only\ndisable-model-invocation: true\n---\n');
  skill(join(home, '.pi/agent/skills/group/review'), '---\ndescription: Review a pull request\n---\n');
  const skills = discoverSkills(cwd, [], home);
  assert.deepEqual(skills.map(item => [item.name, item.description]), [['review', 'Review a pull request'], ['release', 'Cut a release: bump, tag & publish']]);
  const prompt = systemPrompt(cwd, { codemode: false, systemOne: false, skills });
  assert.match(prompt, /<skills>\nThe following skills provide specialized instructions[^]*<name>release<\/name>\n    <description>Cut a release: bump, tag &amp; publish<\/description>\n    <location>[^<]*\.pi\/skills\/release\/SKILL\.md<\/location>[^]*<\/skills>\n\n<cwd>/);
  assert.doesNotMatch(prompt, /Internal only/);
});

test('pure pi: pi.nl runs on the big model and calls its tools, which run on the small model with their judgments', async () => {
  const { runPure } = await import('../../applications/dist/pi/pure.js');
  const cwd = mkdtempSync(join(tmpdir(), 'pi-pure-'));
  writeFileSync(join(cwd, 'a.txt'), 'foo is here\nand foo again\n');
  // The big model runs pi.nl only: one eval that works through the tools.
  const big = scriptedModel(opening => opening.includes('operating inside pi') ? `
    const known = await context(task, cwd, skillDirs);
    const refused = await bash('echo DANGER && touch ran.txt', 'clean up');
    const listed = await bash('seq 1 200', 'see the numbers');
    const edited = await edit('a.txt', [{ oldText: 'foo is here', newText: 'bar is here' }], 'rename every foo to bar in a.txt');
    const shown = await read('a.txt', 1, 1);
    const check = await decide(done, task, 'Renamed foo.', edited);
    return [known.start.length, refused, listed, edited, shown, check.value].join('\\n---\\n');` : null);
  // The small model carries out the tools as their instructions say, as a model would in eval.
  const small = scriptedModel(opening => {
    if (opening.includes('what pi puts in its system prompt')) return 'return { instructions: [], skills: [], start: [] };';
    if (opening.includes("as pi's bash tool does")) return `
      const d = await decide(risk, command, purpose);
      const D = d.probabilities.find(p => p.value === 'destructive').probability;
      if (D >= 0.7) return '[Refused by the safety check: this command looks destructive (p(destructive)=' + D.toFixed(2) + ').]';
      const run = await shell.run(command, timeout);
      const lines = run.output.trimEnd().split('\\n');
      return lines.length > 80 ? (await digest(run.output, purpose)) + '\\n[A small model condensed ' + lines.length + ' lines of output]' : run.output;`;
    if (opening.includes('Keep what the agent needs to act')) return 'return "numbers 1 to 200";';
    if (opening.includes("as pi's edit tool does")) return `
      const before = await files.read(path);
      let after = before;
      for (const e of edits) after = after.replace(e.oldText, e.newText);
      await files.write(path, after);
      const d = await decide(review, intent, '-' + edits[0].oldText + '\\n+' + edits[0].newText);
      return 'Edited ' + path + ': ' + edits.length + ' replacement' + (d.value !== 'as-intended' && d.confidence >= 0.6 ? '\\n[Edit review: this change looks ' + d.value + ' (p=' + d.confidence.toFixed(2) + ')]' : '');`;
    if (opening.includes('Read the file at path')) return 'return (await files.read(path)).split("\\n").slice(offset - 1, offset - 1 + limit).join("\\n");';
    return null;
  });
  const scores = { risk: shown => shown.includes('DANGER') ? ['destructive', 0.9] : ['safe', 0.95],
    review: () => ['incomplete', 0.8], done: () => ['done', 0.95] };
  const decide = async ({ messages, options }) => {
    const shown = JSON.stringify(messages);
    const [winner, p] = shown.includes('wants to run command') ? scores.risk(shown) : shown.includes('intent is what it said') ? scores.review()
      : shown.includes('says it is finished with task') ? scores.done() : ['', 1 / options.length];
    return { log_probs: options.map(option => Math.log(option === JSON.stringify(winner) ? p : (1 - p) / (options.length - 1))) };
  };
  const session = join(cwd, '.pi', 'session.jsonl');
  const result = await runPure({ task: 'Rename foo to bar in a.txt.', cwd, base: { model: Object.assign(small.driver, { decide }) },
    big: Object.assign(big.driver, { decide }), session });

  const [start, refused, listed, edited, shown, verdict] = result.answer.split('\n---\n');
  assert.equal(start, '0');
  assert.match(refused, /^\[Refused by the safety check: this command looks destructive \(p\(destructive\)=0\.90\)/);
  assert.equal(existsSync(join(cwd, 'ran.txt')), false, 'the destructive command never ran');
  assert.equal(listed, 'numbers 1 to 200\n[A small model condensed 200 lines of output]');
  assert.equal(edited, 'Edited a.txt: 1 replacement\n[Edit review: this change looks incomplete (p=0.80)]');
  assert.equal(shown, 'bar is here');
  assert.equal(verdict, 'done');
  assert.equal(big.openings.length, 1, 'the big model ran pi.nl and nothing else');
  assert.equal(small.openings.length, 6, 'the small model ran context, three bash/edit/read tools and the digest');
  assert.equal(result.toolCalls, 4);
  assert.deepEqual(result.interventions.map(item => `${item.kind}:${item.value}`).sort(),
    ['digest:undefined', 'done:done', 'review:incomplete', 'risk:destructive', 'risk:safe']);
  const entries = readFileSync(session, 'utf8').trim().split('\n').map(line => JSON.parse(line));
  assert.equal(entries[0].type, 'session');
  assert.equal(entries.at(-1).name, 'pi', 'the agent call finishes last');
});
