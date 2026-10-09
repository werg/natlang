import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, openFolder } from '../dist/index.js';
import { BuildWorkspace, buildGoal, buildServices } from '../../applications/dist/build/index.js';
import buildDriver from '../../applications/dist/build/build.nl.js';
import { scriptedModel } from './support/natlang.mjs';

const node = process.execPath;

// What an interpreter of each stage might write, cut down: the exact algorithm each stage's instructions spell out.
const STAGES = [
  ['Build goal from tasks with the stages', 'driver'], ['Read tasks, the declared tasks of a build', 'declare'],
  ['Compute the closure of goal over graph', 'closure'], ['Order needed, the closure of goal', 'order'],
  ['Run one round of the build in state', 'step'], ['Find the ready tasks exactly in eval', 'ready'],
  ['Choose one ID from ready', 'pick'], ['Decide whether task can be settled', 'judge'],
  ['Diagnose result, the outcome of task', 'diagnose'], ['Write the closing words of a build', 'summarize'],
];

const VALIDITY = `const r = evidence.recorded;
if (!r) return { valid: false, reason: 'no record of an earlier run' };
if (r.fingerprint !== evidence.fingerprint) return { valid: false, reason: 'declaration changed' };
for (const i of evidence.inputs) { const b = r.inputs.find(x => x.path === i.path); if (!b || b.sha256 !== i.sha256) return { valid: false, reason: 'input changed: ' + i.path }; }
for (const o of evidence.outputs) { const b = r.outputs.find(x => x.path === o.path);
  if (o.sha256 === null) return { valid: false, reason: 'output missing: ' + o.path };
  if (!b || b.sha256 !== o.sha256) return { valid: false, reason: 'output modified: ' + o.path }; }
return { valid: true, reason: 'up to date' };`;

const DEFAULTS = {
  driver: `const graph = await declare(tasks, goal);
const empty = { order: [], results: [], built: [], reused: [], judgments: [], problems: [] };
const close = async (state, fields) => { const closing = await summarize(state); return { goal, status: state.status, detail: state.detail,
  blocked: state.blocked, not_run: state.needed, ...empty, ...fields, summary: closing.summary, next: closing.next }; };
const blank = { goal, tasks, graph, needed: [], order: [], attempted: [], results: [], judgments: [], blocked: [], status: 'invalid', detail: '' };
if (graph.problems.length) {
  const detail = graph.problems.map(p => p.detail).join('; ');
  return close({ ...blank, detail }, { not_run: [], problems: graph.problems, detail });
}
const needed = await closure(graph, goal);
const plan = await order(graph, goal, needed);
if (plan.cycle.length) {
  const detail = 'tasks wait on each other: ' + plan.cycle.join(', ');
  return close({ ...blank, needed, blocked: needed, status: 'blocked', detail }, { plan, blocked: needed, detail });
}
const state = { ...blank, needed, status: 'running' };
const final = await step.iterateOn(state, files)
  .withMeasure(s => 2 * (s.needed.length - s.attempted.length) + (s.status === 'running' ? 1 : 0))
  .until(s => s.status !== 'running');
const closing = await summarize(final);
return { goal, status: final.status, detail: final.detail, order: final.order, results: final.results, blocked: final.blocked,
  built: final.judgments.filter(j => j.action === 'ran').map(j => j.task), reused: final.judgments.filter(j => j.action === 'reused').map(j => j.task),
  not_run: final.needed.filter(id => !final.attempted.includes(id)), judgments: final.judgments,
  ...(final.diagnosis ? { diagnosis: final.diagnosis } : {}), problems: [], plan, summary: closing.summary, next: closing.next };`,
  declare: `const problems = [];
const ids = tasks.map(t => t.id);
const uniq = list => [...new Set(list)].sort();
for (const id of uniq(ids)) {
  if (id === '') problems.push({ code: 'empty-id', tasks: [id], detail: 'give every task an id' });
  else if (ids.filter(x => x === id).length > 1) problems.push({ code: 'duplicate-id', tasks: [id], detail: 'task id ' + id + ' is used twice: rename one' });
}
if (!ids.includes(goal)) problems.push({ code: 'unknown-goal', tasks: [goal], detail: 'unknown goal: ' + goal });
const producerOf = {};
for (const t of tasks) for (const p of t.outputs) producerOf[p] = [...(producerOf[p] || []), t.id];
for (const p of Object.keys(producerOf)) if (producerOf[p].length > 1)
  problems.push({ code: 'duplicate-output', tasks: uniq(producerOf[p]), detail: 'tasks ' + producerOf[p].join(', ') + ' all write ' + p + ': give each output one task' });
const info = {};
for (const t of tasks) {
  if (!t.argv.length) problems.push({ code: 'no-command', tasks: [t.id], detail: 'task ' + t.id + ' has no command: give it an argv' });
  if (!t.outputs.length) problems.push({ code: 'no-output', tasks: [t.id], detail: 'task ' + t.id + ' declares no output: name the file it makes' });
  for (const n of t.needs) {
    if (n === t.id) problems.push({ code: 'self-dependency', tasks: [t.id], detail: 'task ' + t.id + ' needs itself: remove it from needs' });
    else if (!ids.includes(n)) problems.push({ code: 'unknown-need', tasks: [t.id, n], detail: 'task ' + t.id + ' needs ' + n + ', which is not declared' });
  }
  if (uniq(t.needs).length !== t.needs.length) problems.push({ code: 'duplicate-need', tasks: [t.id], detail: 'task ' + t.id + ' lists a need twice: list each once' });
  for (const p of t.inputs) if (t.outputs.includes(p)) problems.push({ code: 'self-read', tasks: [t.id], detail: 'task ' + t.id + ' reads its own output ' + p });
  const producers = uniq(t.inputs.flatMap(p => producerOf[p] || []).filter(x => x !== t.id));
  for (const x of producers) if (!t.needs.includes(x))
    problems.push({ code: 'undeclared-dependency', tasks: [t.id, x], detail: 'task ' + t.id + ' reads ' + t.inputs.find(p => (producerOf[p] || []).includes(x)) + ', which task ' + x + ' produces: add ' + x + ' to the needs of ' + t.id });
  if (!info[t.id]) info[t.id] = { t, producers };
}
const nodes = Object.keys(info).filter(id => id !== '').sort().map(id => {
  const { t, producers } = info[id];
  const needs = uniq(t.needs.filter(x => x !== id));
  const deps = uniq([...needs.filter(x => ids.includes(x)), ...producers]);
  return { id, needs, producers, deps, dependents: [], sources: uniq(t.inputs.filter(p => !(producerOf[p] || []).some(x => x !== id))) };
});
for (const n of nodes) n.dependents = nodes.filter(m => m.deps.includes(n.id)).map(m => m.id);
return { nodes, problems };`,
  closure: `const nodeOf = Object.fromEntries(graph.nodes.map(n => [n.id, n]));
let reached = [goal], frontier = [goal];
for (let i = 0; i < graph.nodes.length; i++) {
  const next = [...new Set(frontier.flatMap(id => nodeOf[id].deps))].filter(id => !reached.includes(id) && nodeOf[id]);
  if (!next.length) break;
  reached = [...reached, ...next]; frontier = next;
}
return reached.sort();`,
  order: `const nodeOf = Object.fromEntries(graph.nodes.map(n => [n.id, n]));
let placed = [], remaining = [...needed];
for (let i = 0; i < needed.length; i++) {
  const available = remaining.filter(id => nodeOf[id].deps.filter(d => needed.includes(d)).every(d => placed.includes(d))).sort();
  if (!available.length) break;
  placed = [...placed, ...available]; remaining = remaining.filter(id => !available.includes(id));
}
return { goal, needed, order: placed, cycle: remaining.sort() };`,
  step: `const ids = await ready(state.graph, state.needed, state.order);
if (!ids.length) return commit.settle(state, { kind: 'stop', task: '', why: 'no declared task is ready' }, null, null);
const candidates = state.tasks.filter(t => ids.includes(t.id));
const picked = await choose(candidates, state.goal, files);
const task = candidates.find(t => t.id === picked);
const admission = commit.admit(state, { kind: 'run', task: picked, why: '' });
if (!admission.ok) return commit.settle(state, { kind: 'reject', task: picked, why: admission.problem }, null, null);
const evidence = await build.inspect(task);
const verdict = await validity(task, evidence);
let decision = { kind: verdict.valid ? 'reuse' : 'run', task: picked, why: verdict.reason };
let result;
if (decision.kind === 'reuse') {
  result = await build.reuse(task);
  if (result.status !== 'ok') { decision = { kind: 'run', task: picked, why: result.detail }; result = await build.execute(task); }
} else result = await build.execute(task);
const diagnosis = result.status === 'ok' ? null : await diagnose(task, result, state.graph);
return commit.settle(state, decision, result, diagnosis);`,
  ready: `const nodeOf = Object.fromEntries(graph.nodes.map(n => [n.id, n]));
return needed.filter(id => !finished.includes(id) && nodeOf[id].deps.every(d => finished.includes(d))).sort();`,
  pick: `return ready.map(t => t.id).sort()[0];`,
  judge: VALIDITY,
  diagnose: `const d = result.detail;
const cause = result.status === 'unknown' ? 'interrupted' : d.startsWith('output already exists') ? 'output-conflict'
  : d.startsWith('declared input changed') ? 'input-mutated' : result.exit_code > 0 ? 'command-failed' : 'other';
const culprit = (d.match(/: (\\S+)$/) || [])[1] || '';
return { task: task.id, cause, culprit, summary: 'task ' + task.id + ' ' + cause, fix: 'see ' + culprit,
  retry: cause === 'interrupted' ? 'inspect-first' : cause === 'other' ? 'no' : 'after-fix' };`,
  summarize: `return { summary: 'build ' + state.status + ' for ' + state.goal, next: state.status === 'done' ? [] : [state.detail] };`,
};

function scripted(overrides = {}) {
  const seen = [];
  const model = scriptedModel(opening => {
    const stage = STAGES.find(([phrase]) => opening.includes(phrase))?.[1];
    if (!stage) return null;
    seen.push(stage);
    const script = overrides[stage] ?? DEFAULTS[stage];
    return typeof script === 'function' ? script(opening) : script;
  });
  return { model, seen };
}

/** One build in its own root. Returns the report, the workspace's events and the stages the model was asked. */
async function run(tasks, goal, { overrides, setup = () => {}, policy, folder = mkdtempSync(join(tmpdir(), 'natlang-build-')), keep = false } = {}) {
  mkdirSync(join(folder, 'out'), { recursive: true });
  if (!existsSync(join(folder, 'input.txt'))) writeFileSync(join(folder, 'input.txt'), 'hello');
  setup(folder);
  const workspace = await new BuildWorkspace(folder, { policy }).open();
  const { model, seen } = scripted(overrides);
  const runtime = createNatlangRuntime({ model: model.driver });
  const report = await buildGoal(runtime, workspace, goal, tasks, openFolder(folder).root());
  return { report, folder, events: workspace.drainEvents(), seen, model };
}
const finish = ({ folder }) => rmSync(folder, { recursive: true, force: true });

const chain = () => [
  { id: 'source', needs: [], description: 'prepare source',
    argv: [node, '-e', 'require("fs").copyFileSync("input.txt", "out/source.txt")'],
    inputs: ['input.txt'], outputs: ['out/source.txt'] },
  { id: 'goal', needs: ['source'], description: 'publish goal',
    argv: [node, '-e', 'const fs=require("fs"); fs.writeFileSync("out/goal.txt", fs.readFileSync("out/source.txt", "utf8").toUpperCase())'],
    inputs: ['out/source.txt'], outputs: ['out/goal.txt'] },
  { id: 'unrelated', needs: [], description: 'unrelated',
    argv: [node, '-e', 'require("fs").writeFileSync("out/unrelated.txt", "bad")'],
    inputs: [], outputs: ['out/unrelated.txt'] },
];

test('natlang builds the goal through the stages and real processes, and reports', async () => {
  const result = await run(chain(), 'goal');
  try {
    const { report, folder, events, seen } = result;
    assert.equal(report.status, 'done');
    assert.deepEqual(Array.from(report.order), ['source', 'goal']);
    assert.deepEqual(Array.from(report.built), ['source', 'goal']);
    assert.deepEqual(Array.from(report.not_run), []);
    assert.deepEqual(Array.from(report.plan.needed), ['goal', 'source'], 'the unrelated task is outside the goal\'s closure');
    assert.deepEqual(Array.from(report.plan.order), ['source', 'goal']);
    assert.equal(readFileSync(join(folder, 'out/goal.txt'), 'utf8'), 'HELLO');
    assert.ok(!existsSync(join(folder, 'out/unrelated.txt')));
    assert.ok(report.results.every(row => row.status === 'ok' && row.input_sha256.length === 64 && row.output_sha256.length === 64));
    assert.equal(events.filter(e => e.operation === 'build.execute').length, 2);
    assert.match(report.summary, /done/);
    // The stages ran in the architecture's order; the crisp ready filter did not ask the model.
    assert.deepEqual(seen.filter(s => s !== 'step'), ['driver', 'declare', 'closure', 'order', 'pick', 'judge', 'pick', 'judge', 'summarize']);
    assert.ok(!seen.includes('ready'));
  } finally { finish(result); }
});

test('a failed process stops the build with a diagnosis, and its dependent goal is not complete', async () => {
  const tasks = [
    { id: 'source', needs: [], description: 'prepare source', argv: [node, '-e', 'process.exit(7)'],
      inputs: ['input.txt'], outputs: ['out/source.txt'] },
    { id: 'goal', needs: ['source'], description: 'publish goal', argv: [node, '-e', 'process.exit(0)'],
      inputs: ['out/source.txt'], outputs: ['out/goal.txt'] },
  ];
  const result = await run(tasks, 'goal');
  try {
    const { report, seen } = result;
    assert.equal(report.status, 'failed');
    assert.deepEqual(Array.from(report.order), []);
    assert.equal(report.results[0].exit_code, 7);
    assert.equal(report.diagnosis.cause, 'command-failed');
    assert.equal(report.diagnosis.retry, 'after-fix');
    assert.deepEqual(Array.from(report.not_run), ['goal']);
    assert.ok(seen.includes('diagnose'));
    assert.deepEqual(Array.from(report.next), [report.detail]);
  } finally { finish(result); }
});

test('a cycle blocks the build before any process or round runs', async () => {
  const tasks = [
    { id: 'source', needs: ['goal'], description: 'source', argv: [node], inputs: [], outputs: ['out/source.txt'] },
    { id: 'goal', needs: ['source'], description: 'goal', argv: [node], inputs: [], outputs: ['out/goal.txt'] },
  ];
  const result = await run(tasks, 'goal');
  try {
    const { report, events, seen } = result;
    assert.equal(report.status, 'blocked');
    assert.deepEqual(Array.from(report.blocked), ['goal', 'source']);
    assert.deepEqual(Array.from(report.plan.cycle), ['goal', 'source']);
    assert.equal(events.filter(e => e.operation === 'build.execute').length, 0);
    assert.ok(!seen.includes('step') && !seen.includes('pick'));
  } finally { finish(result); }
});

test('faults in the declarations are reported with the fix, and nothing runs', async () => {
  const tasks = [
    { id: 'a', needs: [], description: 'makes x', argv: [node, '-e', '0'], inputs: ['input.txt'], outputs: ['out/x.txt'] },
    // b reads a's output without listing a in needs; both claim out/dup.txt.
    { id: 'b', needs: [], description: 'reads x', argv: [node, '-e', '0'], inputs: ['out/x.txt'], outputs: ['out/dup.txt'] },
    { id: 'c', needs: ['a', 'zzz'], description: 'dup', argv: [node, '-e', '0'], inputs: [], outputs: ['out/dup.txt'] },
  ];
  const result = await run(tasks, 'b');
  try {
    const { report, events, seen } = result;
    assert.equal(report.status, 'invalid');
    const codes = Array.from(report.problems).map(p => p.code).sort();
    assert.deepEqual(codes, ['duplicate-output', 'undeclared-dependency', 'unknown-need']);
    assert.match(report.detail, /add a to the needs of b/);
    assert.equal(events.length, 0);
    assert.ok(!seen.includes('closure'));
    const unknown = await run(chain(), 'nope');
    try {
      assert.equal(unknown.report.status, 'invalid');
      assert.match(unknown.report.detail, /unknown goal: nope/);
    } finally { finish(unknown); }
  } finally { finish(result); }
});

test('input mutation and preexisting outputs are rejected as verified builds, and diagnosed', async () => {
  const mutation = [{ id: 'source', needs: [], description: 'bad generator',
    argv: [node, '-e', 'const fs=require("fs"); fs.writeFileSync("input.txt", "changed"); fs.writeFileSync("out/source.txt", "result")'],
    inputs: ['input.txt'], outputs: ['out/source.txt'] }];
  const first = await run(mutation, 'source');
  try {
    assert.equal(first.report.status, 'failed');
    assert.match(first.report.detail, /declared input changed/);
    assert.equal(first.report.diagnosis.cause, 'input-mutated');
  } finally { finish(first); }

  const preexisting = [{ id: 'source', needs: [], description: 'bad declaration',
    argv: [node, '-e', 'process.exit(0)'], inputs: ['input.txt'], outputs: ['out/source.txt'] }];
  const second = await run(preexisting, 'source', { setup: folder => writeFileSync(join(folder, 'out/source.txt'), 'stale') });
  try {
    assert.equal(second.report.status, 'failed');
    assert.match(second.report.detail, /output already exists/);
    assert.equal(second.report.diagnosis.cause, 'output-conflict');
  } finally { finish(second); }
});

test('an interrupted process is an unknown outcome that is inspected before any retry', async () => {
  const tasks = [{ id: 'slow', needs: [], description: 'hangs', argv: [node, '-e', 'setTimeout(() => {}, 60000)'], inputs: [], outputs: ['out/slow.txt'] }];
  const folder = mkdtempSync(join(tmpdir(), 'natlang-build-'));
  mkdirSync(join(folder, 'out'));
  const workspace = await new BuildWorkspace(folder, { timeoutMs: 100 }).open();
  const { model } = scripted();
  try {
    const report = await createNatlangRuntime({ model: model.driver }).run(() => buildDriver('slow', tasks), buildServices(workspace));
    assert.equal(report.status, 'unknown');
    assert.equal(report.diagnosis.cause, 'interrupted');
    assert.equal(report.diagnosis.retry, 'inspect-first');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('a second build reuses valid outputs, and a changed input rebuilds exactly what it invalidates', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-build-'));
  try {
    const first = await run(chain(), 'goal', { folder });
    assert.deepEqual(Array.from(first.report.built), ['source', 'goal']);
    // Same declarations, same files: both tasks are settled from the ledger, nothing runs.
    const second = await run(chain(), 'goal', { folder });
    assert.equal(second.report.status, 'done');
    assert.deepEqual(Array.from(second.report.reused), ['source', 'goal']);
    assert.deepEqual(Array.from(second.report.built), []);
    assert.equal(second.events.filter(e => e.operation === 'build.execute').length, 0);
    assert.equal(second.events.filter(e => e.operation === 'build.reuse' && e.status === 'ok').length, 2);
    // A changed source input invalidates source, and its new output invalidates goal in turn.
    const third = await run(chain(), 'goal', { folder, setup: dir => writeFileSync(join(dir, 'input.txt'), 'goodbye') });
    assert.deepEqual(Array.from(third.report.built), ['source', 'goal']);
    assert.match(third.report.judgments[0].reason, /input changed: input.txt/);
    assert.match(third.report.judgments[1].reason, /input changed: out\/source.txt/);
    assert.equal(readFileSync(join(folder, 'out/goal.txt'), 'utf8'), 'GOODBYE');
    // A changed declaration invalidates the task even though its files match.
    const redefined = chain(); redefined[1].argv = [node, '-e', 'const fs=require("fs"); fs.writeFileSync("out/goal.txt", fs.readFileSync("out/source.txt", "utf8") + "!")'];
    const fourth = await run(redefined, 'goal', { folder });
    assert.deepEqual(Array.from(fourth.report.reused), ['source']);
    assert.deepEqual(Array.from(fourth.report.built), ['goal']);
    assert.match(fourth.report.judgments[1].reason, /declaration changed/);
    assert.equal(readFileSync(join(folder, 'out/goal.txt'), 'utf8'), 'goodbye!');
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('an output edited by hand is never overwritten, and a wrong validity verdict cannot reuse stale files', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-build-'));
  try {
    await run(chain(), 'goal', { folder });
    writeFileSync(join(folder, 'out/source.txt'), 'edited by hand');
    const edited = await run(chain(), 'goal', { folder });
    assert.equal(edited.report.status, 'failed');
    assert.match(edited.report.detail, /source: output already exists/);
    assert.equal(readFileSync(join(folder, 'out/source.txt'), 'utf8'), 'edited by hand');
    assert.equal(edited.report.diagnosis.cause, 'output-conflict');
  } finally { rmSync(folder, { recursive: true, force: true }); }

  const second = mkdtempSync(join(tmpdir(), 'natlang-build-'));
  try {
    await run(chain(), 'goal', { folder: second });
    // The judge wrongly says everything is valid; the workspace refuses to reuse source (its input changed) and the round runs it.
    const liar = await run(chain(), 'goal', { folder: second, overrides: { judge: `return { valid: true, reason: 'looks fine' };` },
      setup: dir => writeFileSync(join(dir, 'input.txt'), 'goodbye') });
    assert.equal(liar.report.status, 'done');
    // Both tasks were judged valid and both reuses were refused: goal's input is the output source just remade.
    assert.equal(liar.events.filter(e => e.operation === 'build.reuse' && e.status === 'refused').length, 2);
    assert.equal(readFileSync(join(second, 'out/goal.txt'), 'utf8'), 'GOODBYE');
    assert.match(liar.report.judgments[0].reason, /reuse refused: input changed: input.txt/);
    assert.match(liar.report.judgments[1].reason, /reuse refused: input changed: out\/source.txt/);
    assert.deepEqual(Array.from(liar.report.built), ['source', 'goal']);
  } finally { rmSync(second, { recursive: true, force: true }); }
});

test('each hot path runs its crisp or natural-language implementation, as the setting says', async () => {
  const defaults = await run(chain(), 'goal');
  try {
    assert.ok(defaults.seen.includes('pick') && defaults.seen.includes('judge') && !defaults.seen.includes('ready'));
  } finally { finish(defaults); }
  const crisp = await run(chain(), 'goal', { policy: { choose: 'crisp', validity: 'crisp' } });
  try {
    assert.equal(crisp.report.status, 'done');
    assert.ok(!crisp.seen.includes('pick') && !crisp.seen.includes('judge'));
  } finally { finish(crisp); }
  const language = await run(chain(), 'goal', { policy: { ready: 'natural-language', choose: 'natural-language', validity: 'natural-language' } });
  try {
    assert.equal(language.report.status, 'done');
    assert.deepEqual(Array.from(language.report.order), ['source', 'goal']);
    assert.equal(language.seen.filter(s => s === 'ready').length, 2);
  } finally { finish(language); }
  const workspace = await new BuildWorkspace(tmpdir()).open();
  assert.throws(() => workspace.implementation('speed'), /unknown policy point/);
});

test('the scheduling policy decides among ready tasks, and a wrong choice is refused with the ready tasks', async () => {
  const tasks = chain();
  tasks[1].needs = ['source', 'unrelated'];
  const wrong = await run(tasks, 'goal', { overrides: { pick: `return 'goal';` } });
  try {
    assert.equal(wrong.report.status, 'invalid');
    assert.match(wrong.report.detail, /chosen task is not ready: goal; choose the id of one of the ready tasks: source, unrelated/);
    assert.equal(wrong.events.filter(e => e.operation === 'build.execute').length, 0);
  } finally { finish(wrong); }
  // The scripted policy prefers the largest id; the default scripted one the smallest.
  const reverse = await run(tasks, 'goal', { overrides: { pick: `return ready.map(t => t.id).sort().reverse()[0];` } });
  try {
    assert.deepEqual(Array.from(reverse.report.order), ['unrelated', 'source', 'goal']);
  } finally { finish(reverse); }
  const forward = await run(tasks, 'goal');
  try {
    assert.deepEqual(Array.from(forward.report.order), ['source', 'unrelated', 'goal']);
  } finally { finish(forward); }
});

test('the commit and the ready filter are exact, pure and bounded', () => {
  const task = (id, needs = []) => ({ id, needs, description: id, argv: ['x'], inputs: [], outputs: [`${id}.out`] });
  const node = (id, deps) => ({ id, needs: deps, producers: [], deps, dependents: [], sources: [] });
  const graph = { nodes: [node('a', []), node('b', ['a']), node('c', ['a', 'b'])], problems: [] };
  const state = { goal: 'c', tasks: [task('a'), task('b', ['a']), task('c', ['a', 'b'])], graph, needed: ['a', 'b', 'c'], order: [], attempted: [],
    results: [], judgments: [], blocked: [], status: 'running', detail: '' };
  const { ready, commit } = buildDriver.step;
  assert.deepEqual(ready.crisp(graph, state.needed, []), ['a']);
  assert.deepEqual(ready.crisp(graph, state.needed, ['a']), ['b']);
  assert.deepEqual(ready.crisp(graph, ['a', 'b'], ['a', 'b']), []);
  assert.equal(commit.admit(state, { kind: 'run', task: 'b', why: '' }).ok, false);
  assert.match(commit.admit(state, { kind: 'reuse', task: 'c', why: '' }).problem, /ready tasks: a/);
  const ok = { id: 'a', status: 'ok', exit_code: 0, input_sha256: '', output_sha256: '', detail: '' };
  const next = commit.settle(state, { kind: 'run', task: 'a', why: 'no record of an earlier run' }, ok, null);
  assert.deepEqual([next.order, next.attempted, next.status], [['a'], ['a'], 'running']);
  assert.deepEqual(next.judgments, [{ task: 'a', action: 'ran', reason: 'no record of an earlier run' }]);
  assert.deepEqual(state.order, [], 'the committed state is a new value');
  assert.throws(() => commit.settle(next, { kind: 'run', task: 'a', why: '' }, ok, null), /already attempted/);
  const failed = commit.settle(next, { kind: 'run', task: 'b', why: '' }, { ...ok, id: 'b', status: 'failed', detail: 'exit 2: x' }, { task: 'b', cause: 'command-failed' });
  assert.deepEqual([failed.status, failed.detail, failed.order, failed.diagnosis.cause], ['failed', 'b: exit 2: x', ['a'], 'command-failed']);
  const stopped = commit.settle(next, { kind: 'stop', task: '', why: 'no declared task is ready' }, null, null);
  assert.deepEqual([stopped.status, stopped.blocked], ['blocked', ['b', 'c']]);
});

test('exact built-in cache reuses outputs, invalidates changed inputs, and rejects corrupt entries', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-build-cache-test-'));
  const cacheDir = join(root, 'cache');
  const task = { id: 'copy', needs: [], description: 'copy bytes', argv: ['@builtin', 'copy'],
    inputs: ['input.txt'], outputs: ['output.txt'] };
  const execute = async (name, value) => {
    const folder = join(root, name); mkdirSync(folder);
    writeFileSync(join(folder, 'input.txt'), value);
    const workspace = await new BuildWorkspace(folder, { cacheDir }).open();
    const result = await workspace.execute(task);
    return { result, events: workspace.drainEvents(), folder };
  };
  try {
    const first = await execute('first', 'hello');
    assert.equal(first.result.status, 'ok');
    assert.equal(first.result.detail, 'built-in copy');
    const key = first.events.find(e => e.operation === 'build.cache').key;
    const second = await execute('second', 'hello');
    assert.equal(second.result.detail, 'cache hit');
    assert.equal(readFileSync(join(second.folder, 'output.txt'), 'utf8'), 'hello');
    const changed = await execute('changed', 'goodbye');
    assert.equal(changed.result.detail, 'built-in copy');
    assert.notEqual(changed.result.output_sha256, first.result.output_sha256);
    writeFileSync(join(cacheDir, key, 'output.bin'), 'corrupt');
    const repaired = await execute('repaired', 'hello');
    assert.equal(repaired.result.status, 'ok');
    assert.equal(repaired.result.detail, 'built-in copy');
    assert.ok(repaired.events.some(e => e.operation === 'build.cache' && e.status === 'invalid'));
    assert.equal(readFileSync(join(repaired.folder, 'output.txt'), 'utf8'), 'hello');
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('tasks cannot reach the host cache or ledger, and paths stay inside the root', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-build-'));
  try {
    const workspace = await new BuildWorkspace(folder).open();
    const attempt = (inputs, outputs, argv = ['@builtin', 'copy']) => workspace.execute({ id: `t${inputs}${outputs}`, needs: [], description: '', argv, inputs, outputs });
    assert.match((await attempt(['.natlang-build-ledger/ledger.json'], ['o.txt'])).detail, /enters host cache/);
    assert.match((await attempt([], ['.natlang-build-cache/x'], [node, '-e', '0'])).detail, /enters host cache/);
    assert.match((await attempt(['../outside'], ['o.txt'])).detail, /escapes workspace/);
    assert.match((await attempt([], ['/tmp/abs.txt'], [node, '-e', '0'])).detail, /invalid workspace path/);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});
