import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, loadNatlang, loadCallables, iterateOn, EventLoop, NatlangContextError, NatlangRecursionError,
  NatlangCallError, IterationDivergedError, IterationLimitError, MemoryIterationStatistics, __natlang, Folder } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

function tree(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-runtime-'));
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}
const nlFile = (args, returns, body, extra = '') => `---\nargs:\n${Object.entries(args).map(([k, v]) => `  ${k}: ${JSON.stringify(v)}`).join('\n') || '  {}'}\nreturns: ${JSON.stringify(returns)}\n${extra}---\n${body}\n`;

const SUMMARIZE = {
  'types.ts': 'export type Summary = { text: string, short: boolean };\n',
  'summarize.nl': nlFile({ text: 'string' }, 'Summary', 'Summarize text. Use summarize.is_short and shorten.'),
  'summarize/is_short.ts': 'export default function is_short(text: string): boolean { return text.length < 20; }\n',
  'summarize/shorten.nl': nlFile({ text: 'string' }, 'string', 'Shorten text to its first word.'),
  'summarize/textTools.ts': 'export function normalize(text: string): string { return text.trim().toLowerCase(); }\n' +
    'export const rules = { strict: true };\n',
};

test('a named function exposes its callable folder as a typed hierarchy in host code and in the agent listing', async () => {
  const root = tree(SUMMARIZE);
  const summarize = loadNatlang(join(root, 'summarize.nl'));
  assert.equal(summarize.is_short('tiny'), true);
  assert.equal(summarize.textTools.normalize('  Loud '), 'loud');
  assert.equal(summarize.textTools.rules.strict, true);
  assert.equal(typeof summarize.shorten, 'function');
  assert.equal(typeof summarize.iterateOn, 'function');
  const model = scriptedModel(opening => opening.includes('Shorten text') ? 'return text.split(" ")[0]' :
    'const shortened = await shorten(text); return { text: textTools.normalize(shortened), short: is_short(shortened) }');
  const runtime = createNatlangRuntime({ model: model.driver });
  const value = await runtime.run(() => summarize('Hello brave new world'));
  assert.deepEqual(value, { text: 'hello', short: true });
  const listing = model.openings[0];
  assert.match(listing, /declare function is_short\(text: string\): boolean; {2}\/\/ TypeScript\n/);
  assert.match(listing, /declare function shorten\(text: string\): Promise<string>; {2}\/\/ natural language/);
  assert.match(listing, /declare namespace textTools \{\n {2}function normalize\(text: string\): string; {2}\/\/ TypeScript/);
  assert.match(listing, /type Summary = /);
  assert.equal(await runtime.run(() => summarize.shorten('Alpha beta')), 'Alpha');
});

test('a callable folder rejects reserved child names and open-ended loops at load time', () => {
  const reserved = tree({ 'f.nl': nlFile({}, 'string', 'Say hi.'), 'f/then.ts': 'export default function then(): string { return "x"; }\n' });
  assert.throws(() => loadNatlang(join(reserved, 'f.nl')), /collides with a built-in function property/);
  const looping = tree({ 'g.nl': nlFile({}, 'string', 'Say hi.'), 'g/spin.ts': 'export default function spin(): string { while (true) {} }\n' });
  assert.throws(() => loadNatlang(join(looping, 'g.nl')), /`while` loops are not allowed/);
});

test('natural-language calls need a task, run concurrently as siblings, and reject reentry in their own chain', async () => {
  const root = tree({ 'echo.nl': nlFile({ value: 'string' }, 'string', 'Return value.') });
  const echo = loadNatlang(join(root, 'echo.nl'));
  await assert.rejects(() => echo('x'), NatlangContextError);
  const model = scriptedModel(() => 'return value');
  const runtime = createNatlangRuntime({ model: model.driver });
  assert.deepEqual(await runtime.run(() => Promise.all(['a', 'b', 'c'].map(item => echo(item)))), ['a', 'b', 'c']);
  const guarded = await runtime.run(() => __natlang.guard('def:x', () => {
    try { __natlang.guard('def:x', () => 1); return 'no error'; } catch (error) { return error instanceof NatlangRecursionError; }
  }));
  assert.equal(guarded, true);
  const later = await runtime.run(async () => {
    const bound = runtime.bind(() => echo('bound'));
    return new Promise(resolve => setTimeout(() => resolve(bound()), 5));
  });
  assert.equal(later, 'bound');
});

test('recursion through a callback into an authored callable-folder function is rejected at entry', async () => {
  const root = tree({ 'visit.nl': nlFile({}, 'number', 'Walk.'),
    'visit/walk.ts': 'export default function walk(next: () => number): number { return next() + 1; }\n' });
  const visit = loadNatlang(join(root, 'visit.nl'));
  const runtime = createNatlangRuntime({ model: scriptedModel(() => '1').driver });
  const outcome = await runtime.run(() => {
    const again = () => visit.walk(() => 0);
    try { return visit.walk(again); } catch (error) { return error; }
  });
  assert.ok(outcome instanceof NatlangRecursionError, String(outcome));
  assert.equal(await runtime.run(() => visit.walk(() => 1)), 2, 'sequential reuse remains valid');
});

test('inline lambdas read live captures and write back mutable ones; a concurrent change is a conflict', async () => {
  const plan = (strings, returns, captures) => ({ sourceSpan: { file: 'app.ts', start: 0, end: 1, line: 1, column: 1 }, definitionId: `nl:${strings}`,
    strings: [strings], instructions: strings, parameters: [], returns: { text: returns, natlang: returns, aliases: {} },
    captures: captures.map(([name, type, mutable]) => ({ name, type: { text: type, natlang: type, aliases: {} }, mutable, source: 'local', mentionSpan: 0 })),
    inheritedCodebaseRevision: '' });
  let limit = 3, tally = 0;
  const model = scriptedModel(opening => opening.includes('Increment tally') ? 'tally = tally + limit; return null' :
    opening.includes('Double limit') ? 'return limit * 2' : null);
  const runtime = createNatlangRuntime({ model: model.driver });
  const read = __natlang.inline(plan('Double limit', 'number', [['limit', 'number', true]]), [], { limit: [() => limit, value => { limit = value; }] });
  limit = 5;
  assert.equal(await runtime.run(() => read()), 10, 'captures are read at invocation, not creation');
  const write = __natlang.inline(plan('Increment tally', 'null', [['tally', 'number', true], ['limit', 'number', false]]), [],
    { tally: [() => tally, value => { tally = value; }], limit: [() => limit] });
  await runtime.run(() => write());
  assert.equal(tally, 5);
  assert.match(model.openings.at(-1), /let tally: number = 0; \/\/ assignments are written back to the caller/);
  const racing = createNatlangRuntime({ model: scriptedModel(() => 'race.bump(); tally = tally + 1; return null').driver,
    services: { race: { bump() { tally = 100; } } } });
  await assert.rejects(() => racing.run(() => write()), NatlangCallError);
  assert.equal(tally, 100, 'a conflicting write-back leaves the other writer\'s value');
});

test('services are injected into eval and callable-folder modules, and every call is traced as an effect', async () => {
  const root = tree({ 'record.nl': nlFile({ entry: 'string' }, 'number', 'Store entry and return the count.'),
    'record/count.ts': "import { store } from 'natlang:services';\nexport default function count(): number { return store.size(); }\n" });
  const record = loadNatlang(join(root, 'record.nl'));
  const entries = [];
  const store = { add(value) { entries.push(value); return true; }, size() { return entries.length; } };
  const traces = [];
  const runtime = createNatlangRuntime({ model: scriptedModel(() => 'store.add(entry); return count()').driver,
    services: { store }, trace: trace => traces.push(trace) });
  assert.equal(await runtime.run(() => record('first')), 1);
  const effects = traces[0].events.filter(event => event.kind === 'effect' && event.phase === 'completed').map(event => event.capability);
  assert.deepEqual(effects, ['store.add']);
});

test('a callable folder may import siblings, including natural-language functions', async () => {
  const root = tree({ 'plan.nl': nlFile({ goal: 'string' }, 'string', 'Plan with steps.'),
    'plan/steps.ts': 'import expand from "./expand.nl";\nexport default async function steps(goal: string): Promise<string> { return (await expand(goal)).toUpperCase(); }\n',
    'plan/expand.nl': nlFile({ goal: 'string' }, 'string', 'Expand the goal.') });
  const plan = loadNatlang(join(root, 'plan.nl'));
  const runtime = createNatlangRuntime({ model: scriptedModel(opening => opening.includes('Expand the goal') ? 'return goal + " now"' :
    'return await steps(goal)').driver });
  assert.equal(await runtime.run(() => plan('ship')), 'SHIP NOW');
});

test('natlang.d folders load as callable trees for application code', async () => {
  const root = tree({ 'natlang.d/classify.nl': nlFile({ text: 'string' }, '"bug" | "feature"', 'Classify text.'),
    'natlang.d/labels.ts': 'export const all = ["bug", "feature"];\n' });
  const natlang = loadCallables(join(root, 'natlang.d'));
  assert.deepEqual(natlang.labels.all, ['bug', 'feature']);
  const runtime = createNatlangRuntime({ model: scriptedModel(() => 'return text.includes("crash") ? "bug" : "feature"').driver });
  assert.equal(await runtime.run(() => natlang.classify('it crashes')), 'bug');
  await assert.rejects(() => runtime.run(() => natlang.classify.call(null, 42)), /type-mismatch|expected string/);
});

test('iterateOn: method and free forms, zero-step success, streaming, limits, and step errors', async () => {
  const runtime = createNatlangRuntime();
  const step = async (state, by) => state + by;
  const done = state => state >= 10;
  assert.equal(await runtime.run(() => iterateOn(step, 1, 3).until(done)), 10);
  assert.equal(await runtime.run(() => iterateOn(step, 12, 3).until(done)), 12, 'initial-state success takes zero steps');
  const events = [];
  await runtime.run(async () => { for await (const event of iterateOn(step, 0, 4).streamUntil(done)) events.push(`${event.kind}:${event.state ?? ''}`); });
  assert.deepEqual(events, ['initial:0', 'step:4', 'step:8', 'step:12', 'done:12']);
  await assert.rejects(() => runtime.run(() => iterateOn(step, 0, 0).withLimit({ maxSteps: 5 }).until(done)), IterationLimitError);
  await assert.rejects(() => runtime.run(() => iterateOn(() => { throw new Error('bad step'); }, 0).until(done)),
    error => error.name === 'IterationStepError' && error.lastState === 0);
  const once = iterateOn(step, 0, 1);
  await runtime.run(() => once.until(done));
  await assert.rejects(() => runtime.run(() => once.until(done)), /only once/);
});

test('iterateOn reviews progress on fresh sites and stops only on a divergent verdict', async () => {
  const runtime = createNatlangRuntime({ progressJudge: async trajectory => ({ verdict: trajectory.repeats().length ? 'divergent' : 'continue',
    reason: `${trajectory.repeats().length} repeats` }) });
  const cycling = runtime.run(() => iterateOn(state => (state + 1) % 5, 0).until(() => false));
  await assert.rejects(cycling, error => error instanceof IterationDivergedError && /repeats/.test(error.reason));
  const reviews = [];
  const progressing = await runtime.run(() => iterateOn(state => state + 1, 0).onStep(event => { if (event.kind === 'review') reviews.push(event.review.verdict); })
    .until(state => state >= 25));
  assert.equal(progressing, 25);
  assert.ok(reviews.length >= 1 && reviews.every(verdict => verdict === 'continue'), 'a slow but improving run may continue');
});

test('iterateOn keeps per-site statistics when the site has an identity', async () => {
  const statistics = new MemoryIterationStatistics();
  const runtime = createNatlangRuntime({ statistics });
  for (let run = 0; run < 3; run++)
    await runtime.run(() => __natlang.site('app.ts#improve', iterateOn(state => state + 1, 0)).until(state => state >= 4));
  const [key, stats] = Object.entries(statistics.export())[0];
  assert.match(key, /^app\.ts#improve\|/);
  assert.equal(stats.successes, 3); assert.equal(stats.meanSteps, 4);
});

test('the event loop applies events serially, dedupes IDs, commits before publishing, and retries a failed view', async () => {
  const commits = [];
  let failView = false;
  const loop = new EventLoop({ initialState: { count: 0 },
    reduce: async (state, event) => ({ count: state.count + (event.by ?? 1) }),
    view: state => { if (failView) throw new Error('view failed'); return `count ${state.count}`; },
    onCommit: commit => { commits.push(commit.revision); } });
  assert.equal((await loop.start()).view, 'count 0');
  await Promise.all([loop.dispatch({ id: 'a', kind: 'add', by: 2 }), loop.dispatch({ id: 'b', kind: 'add' })]);
  assert.equal(await loop.dispatch({ id: 'a', kind: 'add', by: 2 }), null);
  assert.equal(loop.view, 'count 3'); assert.deepEqual(commits, [1, 2]);
  failView = true;
  await assert.rejects(() => loop.dispatch({ id: 'c', kind: 'add' }), /view failed/);
  assert.equal(loop.state.count, 4, 'the committed reduction stands');
  failView = false;
  assert.equal((await loop.refresh()).view, 'count 4');
});

test('directory reducers take a Folder and folder.apply installs their committed changes', async () => {
  const root = tree({ 'tidy.nl': nlFile({ note: 'string' }, 'string', 'Append note to log.txt and return "ok".', 'kind: directory-reducer\n') });
  const tidy = loadNatlang(join(root, 'tidy.nl'));
  const runtime = createNatlangRuntime({ model: scriptedModel(() =>
    'const log = folder.file("log.txt"); await log.writeText((await log.readText()) + note + "\\n"); return "ok"').driver });
  const folder = Folder.fromFiles({ 'log.txt': 'start\n' });
  assert.equal(await runtime.run(() => folder.apply(tidy, 'applied')), 'ok');
  assert.equal(await folder.readText('log.txt'), 'start\napplied\n');
  assert.equal(await runtime.run(() => tidy(folder.dir(''), 'discarded')), 'ok');
  assert.equal(await folder.readText('log.txt'), 'start\napplied\n');
  await assert.rejects(() => runtime.run(() => tidy('no folder')), /directory reducer/);
});
