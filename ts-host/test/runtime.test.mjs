import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, loadNatlang, loadCallables, iterateOn, EventLoop, NatlangContextError, NatlangRecursionError,
  NatlangCallError, IterationDivergedError, IterationLimitError, MemoryIterationStatistics, __natlang, Folder } from '../dist/index.js';
import { currentFrame } from '../dist/runtime/context.js';
import { invokeDefinition } from '../dist/runtime/kernel.js';
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

test('invokeDefinition releases supplied folder transactions on recursion and depth preflight errors', async () => {
  const child = { id: 'lease-child', name: 'lease-child', body: 'Return true.', params: [], returns: 'boolean',
    types: {}, codebase: {}, subtype: 'directory-reducer' };
  const makeFolder = () => Folder.fromFiles({ 'jobs/input.json': '{}' });

  const recursionRuntime = createNatlangRuntime({ model: scriptedModel(() => 'return true').driver });
  await recursionRuntime.run(async () => {
    const frame = currentFrame(); assert.ok(frame);
    const folder = makeFolder(), target = folder.dir('jobs'), transaction = await target.beginTransaction();
    await assert.rejects(invokeDefinition({ ...frame, chain: [...frame.chain, 'lease-child'] }, child, [target],
      { folder: { transaction, mode: 'apply' } }), NatlangRecursionError);
    assert.equal(transaction.open, false);
    const next = await target.beginTransaction(false); next.abort();
  });

  const depthRuntime = createNatlangRuntime({ model: scriptedModel(() => 'return true').driver, limits: { maxDepth: 0 } });
  await depthRuntime.run(async () => {
    const frame = currentFrame(); assert.ok(frame);
    const folder = makeFolder(), target = folder.dir('jobs'), transaction = await target.beginTransaction();
    await assert.rejects(invokeDefinition(frame, child, [target],
      { folder: { transaction, mode: 'apply' }, manifest: { delegate: true, path: 'jobs' } }),
      error => error instanceof NatlangCallError && /nested deeper/.test(error.message));
    assert.equal(transaction.open, false);
    const next = await target.beginTransaction(false); next.abort();
  });

  const validationRuntime = createNatlangRuntime({ model: scriptedModel(() => 'return true').driver });
  await validationRuntime.run(async () => {
    const frame = currentFrame(); assert.ok(frame);
    const folder = makeFolder(), target = folder.dir('jobs'), transaction = await target.beginTransaction();
    const invalidArgument = { ...child, id: 'typed-child', subtype: 'function',
      params: [{ name: 'count', type: 'number' }] };
    await assert.rejects(invokeDefinition(frame, invalidArgument, ['not a number'],
      { folder: { transaction, mode: 'apply' } }), /type-mismatch, expected number/);
    assert.equal(transaction.open, false);
    const next = await target.beginTransaction(false); next.abort();

    const malformedReturn = { ...child, id: 'bad-return-child',
      returns: 'The answer in the file, as plain text.' };
    await assert.rejects(invokeDefinition(frame, malformedReturn, [target]), /bad character/);
    const afterInternalAcquire = await target.beginTransaction(false); afterInternalAcquire.abort();
  });
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
  assert.equal(await runtime.run(() => iterateOn(step, 1, 3).withMeasure(n=>Math.max(0,10-n)).until(done)), 10);
  assert.equal(await runtime.run(() => iterateOn(step, 12, 3).until(done)), 12, 'initial-state success takes zero steps');
  const events = [];
  await runtime.run(async () => { for await (const event of iterateOn(step, 0, 4).withMeasure(n=>Math.max(0,10-n)).streamUntil(done)) events.push(`${event.kind}:${event.state ?? ''}`); });
  assert.deepEqual(events, ['initial:0', 'step:4', 'step:8', 'step:12', 'done:12']);
  await assert.rejects(() => runtime.run(() => iterateOn(step, 0, 0).withLimit({ maxSteps: 5 }).until(done)), IterationLimitError);
  await assert.rejects(() => runtime.run(() => iterateOn(() => { throw new Error('bad step'); }, 0).withLimit({maxSteps:1}).until(done)),
    error => error.name === 'IterationStepError' && error.lastState === 0);
  const once = iterateOn(step, 0, 1).withMeasure(n=>Math.max(0,10-n));
  await runtime.run(() => once.until(done));
  await assert.rejects(() => runtime.run(() => once.until(done)), /only once/);
});

test('iterateOn reviews progress on fresh sites and stops only on a divergent verdict', async () => {
  const runtime = createNatlangRuntime({ progressJudge: async trajectory => ({ verdict: trajectory.repeats().length ? 'divergent' : 'continue',
    reason: `${trajectory.repeats().length} repeats` }) });
  const cycling = runtime.run(() => iterateOn(state => (state + 1) % 5, 0).withLimit({maxSteps:50}).until(() => false));
  await assert.rejects(cycling, error => error instanceof IterationDivergedError && /repeats/.test(error.reason));
  const reviews = [];
  const progressing = await runtime.run(() => iterateOn(state => state + 1, 0).onStep(event => { if (event.kind === 'review') reviews.push(event.review.verdict); })
    .withLimit({maxSteps:25}).until(state => state >= 25));
  assert.equal(progressing, 25);
  assert.ok(reviews.length >= 1 && reviews.every(verdict => verdict === 'continue'), 'a slow but improving run may continue');
});

test('iterateOn keeps per-site statistics when the site has an identity', async () => {
  const statistics = new MemoryIterationStatistics();
  const runtime = createNatlangRuntime({ statistics });
  for (let run = 0; run < 3; run++)
    await runtime.run(() => __natlang.site('app.ts#improve', iterateOn(state => state + 1, 0)).withMeasure(n=>4-n).until(state => state >= 4));
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

test('bash natlang call and apply run children in the parent folder transaction', async () => {
  const root = tree({
    'outer.nl': nlFile({}, 'string', 'Use natlang commands on this folder.', 'kind: directory-reducer\n'),
    'outer/upper.nl': nlFile({ input: 'string' }, 'string', 'Uppercase input.'),
    'outer/tidy.nl': nlFile({}, 'string', 'Rewrite a.txt.', 'kind: directory-reducer\n'),
  });
  const outer = loadNatlang(join(root, 'outer.nl'));
  const runtime = createNatlangRuntime({ agent: async session => {
    if (session.lam.functionName === 'upper') {
      assert.equal((await session.applyAsync('eval', { code: 'return input.toUpperCase()' })).kind, 'ok');
    } else if (session.lam.functionName === 'tidy') {
      assert.equal((await session.applyAsync('eval', { code:
        'await folder.file("a.txt").writeText("changed"); return "done"' })).kind, 'ok');
    } else {
      const called = await session.applyAsync('bash', { command: "printf 'hi\\n' | natlang call upper --lines" });
      assert.equal(called.kind, 'ok', called.text);
      assert.equal(called.value.stdout, 'HI\n');
      const applied = await session.applyAsync('bash', { command: 'natlang apply tidy sub' });
      assert.equal(applied.kind, 'ok', applied.text);
      assert.equal(applied.value.stdout, 'done\n');
      session.lam.return = 'ok';
    }
  } });
  const folder = Folder.fromFiles({ 'sub/a.txt': 'old' });
  assert.equal(await runtime.run(() => folder.apply(outer)), 'ok');
  assert.equal(await folder.readText('sub/a.txt'), 'changed');
});

test('a runtime shows services by their declarations and limits scoped ones to their functions', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-scoped-'));
  mkdirSync(join(dir, 'answer'));
  const nlFile = text => `---\nargs: { question: string }\nreturns: string\n---\n${text}\n`;
  writeFileSync(join(dir, 'answer.nl'), nlFile('Answer question by asking lookup.'));
  writeFileSync(join(dir, 'answer', 'lookup.nl'), nlFile('Answer question from records.find(question).'));
  const seen = {};
  const agent = async session => {
    const name = session.lam.functionName;
    seen[name] = { used: (await session.applyAsync('eval', { code: 'const found = records.find("x"); found' })).text,
      read: session.apply('read_code', { name: 'records' }).text };
    if (name === 'answer') await session.applyAsync('eval', { code: 'const asked = await lookup("x"); asked' });
    session.lam.return = 'ok';
  };
  const runtime = createNatlangRuntime({ agent, services: { records: { find: question => `found ${question}` } },
    serviceDeclarations: { records: '/** Look a question up in the records. */\nexport function find(question: string): string;' },
    serviceScopes: { records: ['answer/lookup.nl'] } });
  await runtime.run(() => loadNatlang(join(dir, 'answer.nl'))('x'));
  assert.match(seen.answer.used, /records is not defined/, 'the caller cannot use a service scoped to its helper');
  assert.match(seen.lookup.used, /found x/);
  for (const name of ['answer', 'lookup'])
    assert.match(seen[name].read, /^declare namespace records \{\n {2}\/\*\* Look a question up in the records\. \*\/\n {2}export function find/);
});

test('service scope is exact unless descendant authority is explicitly requested',async()=>{
 const {loadVirtualNatlang}=await import('../dist/runtime/virtual-project.js');
 const step=loadVirtualNatlang({'step.nl':'---\nargs: {}\nreturns: boolean\n---\nAsk child.','step/child.nl':'---\nargs: {}\nreturns: boolean\n---\nInspect own services.'},'step.nl');
 for(const selector of ['step.nl','step.nl/**']){
  const seen={};const rt=createNatlangRuntime({services:{records:{read:()=>1}},serviceDeclarations:{records:'export function read():number;'},serviceScopes:{records:[selector]},agent:async session=>{
   seen[session.lam.functionName]=Object.keys(session.availableServices());
   const event=await session.applyAsync('eval',{code:session.lam.functionName==='step'?'await child(); return true;':'return true;'});assert.equal(event.kind,'ok',event.text);
  }});
  assert.equal(await rt.run(()=>step()),true);assert.deepEqual(seen.step,['records']);assert.deepEqual(seen.child,selector.endsWith('/**')?['records']:[]);
 }
});

test('app system instructions add to ordinary runtime guidance and retain normal tools',async()=>{
 const marker='APP_INSTRUCTIONS: prefer a brief answer.';
 const fn=loadNatlang(join(tree({'answer.nl':nlFile({},'number','Write seven.txt and return seven.','kind: directory-reducer\n')}),'answer.nl'));
 const requests=[];
 const runtime=createNatlangRuntime({systemPrompt:()=>marker,model:async request=>{
  requests.push(request);
  return {calls:[['write_file',{path:'seven.txt',content:'7'}],['eval',{code:'return 7;',finish:true}]]};
 }});
 const folder=Folder.fromFiles({});
 assert.equal(await runtime.run(()=>folder.apply(fn)),7);
 assert.equal(await folder.readText('seven.txt'),'7');
 const {TOOLS_PROMPT}=await import('../dist/native/prompt.js');
 const prompt=String(requests[0].messages[0].content);
 assert.ok(prompt.startsWith(TOOLS_PROMPT));
 assert.ok(prompt.includes(marker));
 const names=requests[0].tools.map(tool=>tool.function.name);
 for(const name of ['eval','write_file','edit_file'])assert.ok(names.includes(name),name+' missing');
});

test('inline nl created in eval has stable definition identities and seeds across fresh tasks',async()=>{
 const fn=loadNatlang(join(tree({'answer.nl':nlFile({value:'number'},'number','Delegate to an inline semantic call.')}),'answer.nl'));
 const seeds=[],definitions=[];
 for(let run=0;run<2;run++){
  const taskSeeds=[];const runtime=createNatlangRuntime({signal:AbortSignal.timeout(5000),seed:{mode:'derived',root:37},trace:trace=>definitions.push(trace.definitionId),model:async request=>{
   taskSeeds.push(request.seed);
   return {calls:[['eval',{code:String(request.messages[1].content).split('\n')[0].includes('answer(')?'const echo = nl<number>`Return value unchanged.`; return await echo(value);':'return value;',finish:true}]]};
  }});
  assert.equal(await runtime.run(()=>fn(9)),9);seeds.push(taskSeeds);
 }
 assert.deepEqual(seeds[0],seeds[1]);
 assert.equal(definitions[0],definitions[2]);
});

test('a child loaded directly retains its owning function and folder type scopes',async()=>{
 const {loadVirtualNatlang}=await import('../dist/runtime/virtual-project.js');
 const files={
  'types.ts':'export type Value = number;',
  'outer.nl':'---\nargs: {}\nreturns: number\ntypes: {Local: string}\n---\nDelegate.',
  'outer/nested/types.ts':'export type Value = string;',
  'outer/nested/child.nl':'---\nargs: {value: Value, label: Local}\nreturns: string\n---\nReturn label plus value.'
 };
 const outer=loadVirtualNatlang(files,'outer.nl');
 const child=loadVirtualNatlang(files,'outer/nested/child.nl');
 const requests=[];
 const runtime=createNatlangRuntime({model:async request=>{requests.push(request);return {text:'prefix-x'};}});
 assert.equal(await runtime.run(()=>child('x','prefix-')),'prefix-x');
 assert.equal(await runtime.run(()=>outer.nested.child('x','prefix-')),'prefix-x');
 assert.deepEqual(requests[0].messages,requests[1].messages);
});
