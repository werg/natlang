import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Folder } from '../dist/index.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { deriveSeed } from '../dist/native/trace.js';
import { MISSING } from '../dist/native/values.js';
import { NativeTraceRecorder } from '../dist/native/trace.js';
import { interpreter, lambda, nl, session as open, ts } from './support/natlang.mjs';

const run = (body, options) => interpreter(options).run(lambda(body));

test('a final expression is only shown; a top-level return stages a value of the declared type', async () => {
  const { lam, session } = open({ type: '() => number', instructions: 'Return nine.' });
  const shown = await session.applyAsync('eval', { code: '9' });
  assert.equal(shown.kind, 'ok'); assert.equal(shown.value, 9); assert.equal(lam.return, MISSING);
  const wrong = await session.applyAsync('eval', { code: 'return "oops"' });
  assert.equal(wrong.kind, 'ok'); assert.match(wrong.text, /This is not a valid number, so it is not the result/);
  assert.equal(lam.return, MISSING);
  const right = await session.applyAsync('eval', { code: 'return 9' });
  assert.equal(right.kind, 'ok'); assert.equal(lam.return, 9); assert.match(right.text, /Staged 9 as the result\..*reply done/);
  const callback = await session.applyAsync('eval', { code: 'const tens = [1, 2].map(value => { return value * 10; }); tens' });
  assert.equal(callback.kind, 'ok'); assert.equal(lam.return, 9, 'a return inside a callback is ordinary JavaScript');
});

test('an empty local assigned to the typed result keeps its list type', async () => {
  const { lam, session } = open({ type: '(items: string[]) => string[]', instructions: 'Select matching items.', args: { items: ['skip'] } });
  const outcome = await session.applyAsync('eval', { code:
    'const ids = items.filter(item => item.startsWith("take")).map(item => item.slice(5)); return ids;' });
  assert.equal(outcome.kind, 'ok', outcome.text);
  assert.deepEqual(lam.return, []); assert.deepEqual(lam.let.ids, []);
});

test('filter then map uses the mapped element type', async () => {
  const { lam, session } = open({ type: '(items: { id: string, status: string }[]) => string[]',
    instructions: 'Select active ids.', args: { items: [{ id: 'A1', status: 'active' }, { id: 'A2', status: 'closed' }] } });
  const outcome = await session.applyAsync('eval', { code:
    'const ids = items.filter(item => item.status === "active").map(item => item.id); return ids;' });
  assert.equal(outcome.kind, 'ok', outcome.text);
  assert.deepEqual(lam.return, ['A1']); assert.deepEqual(lam.let.ids, ['A1']);
});

test('a later eval may redeclare a prior local while correcting the result', async () => {
  const { lam, session } = open({ type: '() => string[]', instructions: 'Return selected ids.' });
  assert.equal((await session.applyAsync('eval', { code: 'const ids = []; return ids;' })).kind, 'ok');
  assert.equal((await session.applyAsync('eval', { code: 'const ids = ["A1"]; return ids;' })).kind, 'ok');
  assert.deepEqual(lam.return, ['A1']); assert.deepEqual(lam.let.ids, ['A1']);
});

test('a failed eval reports the error and says nothing from it was kept', async () => {
  const { session } = open({ type: '() => number', instructions: 'Return a number.' });
  const failed = await session.applyAsync('eval', { code: 'const values = []; values.noSuchMethod();' });
  assert.equal(failed.kind, 'error');
  assert.match(failed.text, /noSuchMethod/); assert.match(failed.text, /Nothing else from this eval was kept/);
});

test('eval returns console.log observations without changing the function result', async () => {
  const { lam, session } = open({ type: '(items: number[]) => number',
    instructions: 'Inspect the average, then return the total.', args: { items: [2, 4, 6] } });
  const observed = await session.applyAsync('eval', { code:
    'const average = items.reduce((sum, item) => sum + item, 0) / items.length; console.log("average", average);' });
  assert.equal(observed.kind, 'ok'); assert.match(observed.text, /console:\naverage 4/);
  assert.equal(lam.return, MISSING);
  const finished = await session.applyAsync('eval', { code: 'return items.reduce((sum, item) => sum + item, 0)' });
  assert.equal(finished.kind, 'ok'); assert.equal(lam.return, 12); assert.doesNotMatch(finished.text, /average 4/);
});

test('eval can use local functions within one call without persisting them', async () => {
  const { lam, session } = open({ type: '(items: number[]) => number', instructions: 'Return the average.', args: { items: [2, 4, 6] } });
  const arrow = await session.applyAsync('eval', { code:
    'const avg = (xs: number[]) => xs.reduce((sum, x) => sum + x, 0) / xs.length; const mean = avg(items); console.log("mean", mean);' });
  assert.equal(arrow.kind, 'ok'); assert.match(arrow.text, /console:\nmean 4/);
  assert.equal(Object.hasOwn(lam.let, 'avg'), false); assert.equal(lam.let.mean, 4);
  const declared = await session.applyAsync('eval', { code:
    'function average(xs: number[]) { return xs.reduce((sum, x) => sum + x, 0) / xs.length; } return average(items);' });
  assert.equal(declared.kind, 'ok'); assert.equal(lam.return, 4); assert.equal(Object.hasOwn(lam.let, 'average'), false);
});

test('an incompatible observation does not persist in the typed result slot', async () => {
  const { lam, session } = open({ type: '() => number', instructions: 'Return a count.' });
  assert.equal((await session.applyAsync('eval', { code: 'return { before: 2, after: 3 }' })).kind, 'ok');
  assert.equal(lam.return, MISSING); assert.equal(Object.hasOwn(lam.let, 'result'), false);
  assert.equal((await session.applyAsync('eval', { code: 'return 3' })).kind, 'ok');
  assert.equal(lam.return, 3);
});

test('return_result finishes with a typed value; an eval return only stages one', async () => {
  const unique = open({ type: '(items: string[]) => string[]', instructions: 'Remove duplicates.', args: { items: ['a', 'a', 'b'] } });
  assert.equal(unique.session.apply('return_result', { status: 'success', value: 'not a list' }).kind, 'rejected');
  const finished = unique.session.apply('return_result', { status: 'success', value: ['a', 'b'] });
  assert.equal(finished.kind, 'completed'); assert.deepEqual(unique.lam.return, ['a', 'b']); assert.equal(unique.session.completed, true);
  const doubled = open({ type: '(value: number) => number', instructions: 'Double the value.', args: { value: 7 } });
  assert.equal((await doubled.session.applyAsync('eval', { code: 'return value * 2;' })).kind, 'ok');
  assert.equal(doubled.lam.return, 14); assert.equal(doubled.session.completed, false);
});

test('a final text reply is the result of a string-typed call, but never a number, and done asks for the staged value', async () => {
  const replies = (texts, type) => { let turn = 0;
    return run({ type, instructions: 'Answer.' }, { agent: session => new NativeToolAgent(() => ({ text: texts[turn++] ?? 'done' }), { maxTurns: 3 }).run(session) }); };
  const label = await replies(['positive'], '() => "positive" | "negative"');
  assert.equal(label.outcome.kind, 'done'); assert.equal(label.value, 'positive');
  const summary = await replies(['The rollout is on track.'], '() => string');
  assert.equal(summary.value, 'The rollout is on track.');
  const number = await replies(['7', '7', '7'], '() => number');
  assert.equal(number.outcome.kind, 'quiesced');
  const early = await replies(['done', 'done', 'done'], '() => string');
  assert.equal(early.outcome.kind, 'quiesced', 'done is not an answer');
});

test('native collection temporaries can support a portable result within one eval', async () => {
  const { lam, session } = open({ type: '(items: string[]) => string[]', instructions: 'Remove duplicates.', args: { items: ['a', 'a', 'b'] } });
  const outcome = await session.applyAsync('eval', { code:
    'const seen = new Set<string>(); const result = items.filter(x => !seen.has(x) && Boolean(seen.add(x))); result' });
  assert.equal(outcome.kind, 'ok'); assert.deepEqual(outcome.value, ['a', 'b']);
  assert.equal(Object.prototype.toString.call(lam.let.seen), '[object Set]', 'a Set local persists by reference as a live value');
});

const counters = () => ({
  count_true: ts('count_true', 'export default function count_true(flags: boolean[]): number { return flags.filter(Boolean).length; }'),
  as_num: ts('as_num', 'export default async function as_num(flag: boolean): Promise<number> { return flag ? 1 : 0; }'),
});

test('scope eval persists locals, calls imports positionally, and treats result as an ordinary name', async () => {
  const { lam, session } = open({ type: '(flags: boolean[]) => number',
    instructions: 'function total(flags) -> number\n  Count the true flags.\n', args: { flags: [true, false, true] }, codebase: counters() });
  const pure = await session.applyAsync('eval', { code: 'const first = flags[0];\nfirst' });
  assert.equal(pure.kind, 'ok'); assert.equal(pure.value, true); assert.equal(lam.let.first, true);
  const call = await session.applyAsync('eval', { code: 'const count = count_true(flags);\ncount' });
  assert.equal(call.kind, 'ok', call.text); assert.equal(call.value, 2); assert.equal(lam.let.count, 2);
  assert.match(call.text, /Stored local count = 2\./);
  const resultLocal = await session.applyAsync('eval', { code: 'const result = count_true(flags);\nresult' });
  assert.equal(resultLocal.kind, 'ok'); assert.equal(lam.let.result, 2); assert.equal(lam.return, MISSING);
  const mapped = await session.applyAsync('eval', { code: 'const counts = await Promise.all(flags.map(flag => as_num(flag)));\ncounts' });
  assert.equal(mapped.kind, 'ok'); assert.deepEqual(mapped.value, [1, 0, 1]);
  const loopMapped = await session.applyAsync('eval', { code:
    'const loopCounts = [];\nfor (const flag of flags) {\n  const count = await as_num(flag);\n  loopCounts.push(count);\n}\nloopCounts' });
  assert.equal(loopMapped.kind, 'ok', loopMapped.text); assert.deepEqual(loopMapped.value, [1, 0, 1]);
  const repairedMap = await session.applyAsync('eval', { code: 'const repaired = flags.map(flag => await as_num(flag)); repaired' });
  assert.notEqual(repairedMap.kind, 'ok');
  const names = new NativeToolAgent(() => ({ calls: [] })).tools(session).map(entry => entry.function.name);
  assert.deepEqual(names, ['eval', 'read_page', 'read_function', 'edit_function', 'diff_functions',
    'compact_history', 'return_result']);
  const nullCase = open({ type: '() => null', instructions: 'Return null.' });
  assert.equal((await nullCase.session.applyAsync('eval', { code: 'return null' })).kind, 'ok');
  assert.equal(nullCase.lam.return, null);
});

test('synchronous TypeScript imports return values directly, including nested callable-folder imports', async () => {
  const { lam, session } = open({ type: '(value: number) => number', instructions: 'Compute the adjusted value.', args: { value: 4 },
    codebase: { adjusted: ts('adjusted', 'import double from "./adjusted/double.ts";\n' +
      'export default function adjusted(value: number): number { return double(value) + 1; }',
      { double: ts('double', 'export default function double(value: number): number { return value * 2; }') }) } });
  const result = await session.applyAsync('eval', { code: 'const answer = adjusted(value); return answer' });
  assert.equal(result.kind, 'ok', result.text); assert.equal(result.value, 9); assert.equal(lam.return, 9);
  const child = await session.applyAsync('eval', { code: 'adjusted.double(5)' });
  assert.equal(child.kind, 'ok', child.text); assert.equal(child.value, 10);
});

test('an unawaited asynchronous call is reported instead of escaping eval', async () => {
  const { session } = open({ type: '() => number', instructions: 'Get a number.',
    codebase: { get_number: ts('get_number', 'export default async function get_number(): Promise<number> { return 7; }') } });
  const forgotten = await session.applyAsync('eval', { code: 'const pending = get_number(); pending' });
  assert.equal(forgotten.kind, 'error'); assert.match(forgotten.text, /await the asynchronous function call/);
  const awaited = await session.applyAsync('eval', { code: 'const result = await get_number(); result' });
  assert.equal(awaited.kind, 'ok'); assert.equal(awaited.value, 7);
});

test('a failed natlang child bubbles to eval without leaving a local', async () => {
  const attempts = [];
  const { lam, session } = open({ type: '(value: number) => number', instructions: 'Ask the helper.', args: { value: 4 },
    codebase: { inspect: nl('inspect', { args: { value: 'number' }, returns: 'number', instructions: 'Inspect the value.' }) } },
  { seedPolicy: { mode: 'derived', root: 17 }, agent: child => {
    attempts.push(child.lam.attempts);
    return `original child failure, attempt ${child.lam.attempts}`;
  } });
  const first = await session.applyAsync('eval', { code: 'const answer = await inspect(value); answer' });
  assert.equal(first.kind, 'error'); assert.match(first.text, /original child failure/);
  assert.equal(Object.hasOwn(lam.let, 'answer'), false);
  const pure = await session.applyAsync('eval', { code: 'const other = value + 1; other' });
  assert.equal(pure.kind, 'ok'); assert.equal(pure.value, 5);
  assert.equal((await session.applyAsync('eval', { code: 'const answer = await inspect(value); answer' })).kind, 'error');
  assert.deepEqual(attempts, [1, 1]);
});

test('failed eval reports its console output and commits no partial locals', async () => {
  const { lam, session } = open({ type: '(item: { deep: { count: number } }) => number',
    instructions: 'Return the count plus one.', args: { item: { deep: { count: 4 } } } });
  assert.equal((await session.applyAsync('eval', { code: 'const earlier = item.deep.count; console.log("earlier", earlier);' })).kind, 'ok');
  const failed = await session.applyAsync('eval', { code:
    'const transient = item.deep.count * 2; console.log("before failure", transient); throw new Error("broken step");' });
  assert.equal(failed.kind, 'error'); assert.match(failed.text, /broken step\nconsole:\nbefore failure 8/);
  assert.equal(Object.hasOwn(lam.let, 'transient'), false); assert.equal(lam.return, MISSING);
  assert.equal(session.failureDebug.kind, 'runtime');
  assert.equal(session.failureDebug.scope.inputs.item.deep.count, 4);
  assert.equal(session.failureDebug.scope.locals.earlier, 4);
  assert.deepEqual(session.failureDebug.logs, ['before failure 8']);
  assert.ok(session.failureDebug.trace.some(event => event.kind === 'action'));
  assert.match(session.failureDebug.stack, /broken step/);
  assert.equal((await session.applyAsync('eval', { code: 'String(earlier)' })).value, '4');
  assert.ok(session.failureDebug);
  assert.equal((await session.applyAsync('eval', { code: 'return item.deep.count + 1' })).kind, 'ok');
  assert.equal(lam.return, 5); assert.equal(session.failureDebug, undefined);
});

test('compile failure reports source diagnostics and redeclaring a parameter says to use it', async () => {
  const { session } = open({ type: '(ticket: string) => number', instructions: 'Return two.', args: { ticket: 'x' } });
  const failed = await session.applyAsync('eval', { code: 'const answer = ;' });
  assert.equal(failed.kind, 'rejected'); assert.equal(session.failureDebug.kind, 'compile');
  assert.match(failed.text, /typescript-syntax/);
  const redeclared = await session.applyAsync('eval', { code: 'const ticket = "x"; return 2;' });
  assert.equal(redeclared.kind, 'rejected');
  assert.match(redeclared.text, /ticket is already defined in this scope; use it directly/);
});

test('model repairs an eval failure under an unchanged system prompt', async () => {
  const requests = [];
  const script = [['eval', { code: 'return String(items[9].value)' }], ['eval', { code: 'items.length' }],
    ['eval', { code: 'return String(items[0].value)' }]];
  const agent = new NativeToolAgent(request => {
    requests.push(structuredClone(request));
    return requests.length > script.length ? { text: 'done' } : { calls: [script[requests.length - 1]], completion_tokens: 1 };
  }, { maxTurns: 5 });
  const result = await run({ type: '(items: { value: number }[]) => string', instructions: 'Return the first value as text.',
    args: { items: [{ value: 7 }] } }, { agent: session => agent.run(session) });
  assert.equal(result.outcome.kind, 'done', result.outcome.detail); assert.equal(result.value, '7');
  assert.equal(requests.length, 4);
  assert.equal(requests[1].messages[0].content, requests[0].messages[0].content);
  assert.match(requests[1].messages.at(-1).content, /Nothing else from this eval was kept/);
});

test('repeated failed repairs stop at the configured limit, and probes do not reset it', async () => {
  let turns = 0;
  const failing = new NativeToolAgent(() => { turns++; return { calls: [['eval', { code: 'throw new Error("still broken")' }]], completion_tokens: 1 }; },
    { maxFailureRepairs: 1 });
  const first = await run({ type: '() => number', instructions: 'Return one.' }, { agent: session => failing.run(session) });
  assert.equal(first.outcome.kind, 'quiesced'); assert.match(first.outcome.detail, /eval repair limit reached/); assert.equal(turns, 2);
  let probeTurns = 0;
  const script = ['throw new Error("first failure")', '"probe"', 'throw new Error("second failure")'];
  const probing = new NativeToolAgent(() => ({ calls: [['eval', { code: script[probeTurns++] }]], completion_tokens: 1 }),
    { maxFailureRepairs: 1 });
  const second = await run({ type: '() => number', instructions: 'Return one.' }, { agent: session => probing.run(session) });
  assert.match(second.outcome.detail, /eval repair limit reached/); assert.equal(probeTurns, 3);
});

test('empty and mixed containers persist with loose types and can be filled later', async () => {
  const { lam, session } = open({ type: '() => number', instructions: 'Collect.' });
  const declared = await session.applyAsync('eval', { code: 'const found = {}; const names = []; const mixed = [1, "a"];' });
  assert.equal(declared.kind, 'ok', declared.text);
  assert.deepEqual(Object.keys(lam.let).sort(), ['found', 'mixed', 'names']);
  const filled = await session.applyAsync('eval', { code: 'found.port = 8080; names.push("x", 2); return names.length + mixed.length' });
  assert.equal(filled.kind, 'ok', filled.text); assert.equal(lam.return, 4); assert.deepEqual(lam.let.found, { port: 8080 });
});

test('in eval, return_result stages its value and blocked ends the call, after the eval succeeds', async () => {
  const typed = open({ type: '() => number', instructions: 'Return seven.' });
  const wrong = await typed.session.applyAsync('eval', { code: 'return_result("seven")' });
  assert.equal(wrong.kind, 'rejected'); assert.match(wrong.text, /return_result: /); assert.equal(typed.session.completed, false);
  const failed = await typed.session.applyAsync('eval', { code: 'return_result(7); throw new Error("after")' });
  assert.equal(failed.kind, 'error'); assert.equal(typed.session.completed, false, 'a failed eval finishes nothing');
  const staged = await typed.session.applyAsync('eval', { code: 'const n = 3 + 4; return_result(n)' });
  assert.equal(staged.kind, 'ok'); assert.match(staged.text, /Staged 7 as the result/);
  assert.equal(typed.lam.return, 7); assert.equal(typed.session.completed, false, 'a computed value is staged, not finished');
  const blocked = open({ type: '() => number', instructions: 'Convert with the rate in the notes.' });
  const reported = await blocked.session.applyAsync('eval', { code: 'return_result(undefined, "blocked", "No notes with an exchange rate were given.")' });
  assert.equal(reported.kind, 'blocked'); assert.match(reported.text, /blocked: No notes/);
});

test('long text output keeps its head and tail, names its transcript entry, and read_page continues after the head', async () => {
  const { session } = open({ type: '(text: string) => number', instructions: 'Inspect.', args: { text: 'x'.repeat(4500) } });
  const logged = await session.applyAsync('eval', { code: 'console.log("a".repeat(1400) + "b".repeat(2000) + "c".repeat(1100))' });
  assert.match(logged.text, /^console:\na{1400}b{100}\n<<cut off: 2500 of 4500 characters not shown; transcript\[0\]\.output holds all of it; read_page\("amber", 2\) shows the next part>>\nc{500}\n/);
  assert.match(session.transcript[0].output, /a{1400}b{2000}c{1100}/, 'the transcript entry holds all of it');
  const second = session.apply('read_page', { id: 'amber', page: 2 });
  assert.equal(second.kind, 'ok');
  assert.match(second.text, /^b{1900}c{100}\n<<page 2 of 3 shown; read_page\("amber", 3\) shows the next part>>$/, 'page 2 starts where the shown head ends');
  assert.match(session.apply('read_page', { id: 'amber', page: 3 }).text, /<<page 3 of 3, the last>>$/);
  assert.equal(session.apply('read_page', { id: 'amber', page: 4 }).kind, 'error');
  const value = await session.applyAsync('eval', { code: 'text' });
  assert.match(value.text, /^"x{2000}" <<cut off: 2500 of 4500 characters not shown; transcript\[4\]\.output holds all of it>>/,
    'a returned value is cut by structure and points at its transcript entry');
  assert.match(session.transcript[4].output, /^"x{4500}"/);
});

test('scope parameters are const: changing one is rejected and a copy is a new variable', async () => {
  const { lam, session } = open({ type: '(count: number, items: number[]) => number', instructions: 'Count the items.',
    args: { count: 2, items: [1] } });
  const rejected = await session.applyAsync('eval', { code: 'count += 3; count' });
  assert.equal(rejected.kind, 'rejected'); assert.match(rejected.text, /count is a parameter and cannot be changed/);
  const copied = await session.applyAsync('eval', { code: 'const more = [...items, 4]; return more.length + count' });
  assert.equal(copied.kind, 'ok'); assert.equal(lam.return, 4);
  assert.equal(lam.args.count, 2); assert.deepEqual(lam.args.items, [1]);
});

test('eval keeps static types for copies, reductions, helper returns and collection methods', async () => {
  const copied = open({ type: '(initial: { blocked: string[], done: boolean }) => boolean', instructions: 'Inspect.',
    args: { initial: { blocked: [], done: false } } });
  assert.equal((await copied.session.applyAsync('eval', { code: 'let state = initial; state' })).kind, 'ok');
  assert.ok(copied.lam.letTypes.state);
  const reduced = open({ type: '(items: string[]) => number', instructions: 'Count adjacent changes.', args: { items: ['a', 'a', 'b', 'c'] } });
  const reduction = await reduced.session.applyAsync('eval', { code:
    'let result = items.slice(1).reduce((count, value, i) => count + (value !== items[i] ? 1 : 0), 0); result' });
  assert.equal(reduction.value, 2); assert.equal(reduced.lam.letTypes.result.name, 'number');
  const helper = open({ type: '(values: string[]) => State', types: { State: '{ values: string[], done: string[] }' },
    instructions: 'Prepare the state.', args: { values: ['a'] }, codebase: {
      prepare: ts('prepare', 'export default function prepare(values: string[]): State { return { values, done: [] }; }', {},
        { State: '{ values: string[], done: string[] }' }) } });
  assert.equal((await helper.session.applyAsync('eval', { code: 'const initial = prepare(values); initial' })).kind, 'ok');
  assert.deepEqual(helper.lam.let.initial, { values: ['a'], done: [] }); assert.equal(helper.lam.letTypes.initial.name, 'State');
  const tasks = open({ type: '(tasks: Task[]) => string', types: { Task: '{ id: string, needs: string[] }' }, instructions: 'Choose a task.',
    args: { tasks: [{ id: 'a', needs: [] }, { id: 'b', needs: ['a'] }] } });
  const chosen = await tasks.session.applyAsync('eval', { code: "const copy = tasks;\nconst first = copy.find(task => task.id === 'a');\n" +
    'const filtered = copy.filter(task => task.needs.length === 0);\nconst sliced = copy.slice(0, 1);\nconst ids = copy.map(task => task.id);\n' +
    'const result: string = first.id;\nresult' });
  assert.equal(chosen.value, 'a'); assert.equal(tasks.lam.letTypes.first.name, 'Task');
  for (const name of ['filtered', 'sliced', 'ids']) assert.equal(tasks.lam.letTypes[name].kind, 'list');
});

test('codebase edits are live and reparsed; nested items are addressed by dotted path', async () => {
  const { lam, session } = open({ type: '() => string', instructions: 'Call label.', codebase: {
    label: ts('label', 'export default function label(): string { return "old"; }'),
    outer: nl('outer', { returns: 'string', instructions: 'Call inner.' },
      { inner: ts('inner', 'export default function inner(): string { return "old"; }') }) } });
  assert.match((await session.applyAsync('read_function', { name: 'label' })).value, /return "old";/);
  assert.equal((await session.applyAsync('edit_function', { name: 'label', find: 'return "old";', replace_with: 'return "new";' })).kind, 'ok');
  const called = await session.applyAsync('eval', { code: 'label()' });
  assert.equal(called.kind, 'ok', called.text); assert.equal(called.value, 'new');
  assert.match((await session.applyAsync('read_function', { name: 'outer' })).text, /Natural-language function source/);
  assert.match((await session.applyAsync('read_function', { name: 'inner' })).value, /return "old";/);
  assert.equal((await session.applyAsync('edit_function', { name: 'outer.inner', find: '"old"', replace_with: '"newer"' })).kind, 'ok');
  assert.match(lam.codebase.outer.codebase.inner.text, /"newer"/);
  assert.equal((await session.applyAsync('edit_function', { name: 'outer', find: 'Call inner.', replace_with: 'Call the inner helper.' })).kind, 'ok');
  assert.deepEqual(Object.keys(lam.codebase.outer.codebase), ['inner']);
  const invalid = await session.applyAsync('edit_function', { name: 'label', find: 'return "new";', replace_with: 'while (true) {}' });
  assert.equal(invalid.kind, 'error'); assert.match(invalid.text, /while/);
  assert.match(lam.codebase.label.text, /return "new";/, 'an invalid edit leaves the previous version live');
  assert.deepEqual(JSON.parse((await session.applyAsync('diff_functions', {})).text).map(item => item.function).sort(), ['inner.ts', 'label.ts', 'outer.nl']);
  assert.equal(new NativeToolAgent(() => ({ calls: [] })).tools(session).some(entry => entry.function.name === 'write_file'), false);
});

test('eval allows finite iteration and rejects open-ended loops', async () => {
  const { session } = open({ type: '(values: number[]) => number', instructions: 'Accumulate.', args: { values: [2, 3] }, codebase: {
    add: ts('add', 'export default function add(acc: number, item: number): number { return acc + item; }'),
    step: ts('step', 'export default function step(state: number): number { return state + 1; }'),
    finished: ts('finished', 'export default function finished(state: number): boolean { return state >= 3; }') } });
  const folded = await session.applyAsync('eval', { code: 'let total: number = 1; for (const item of values) { total = add(total, item); } total' });
  assert.equal(folded.kind, 'ok', folded.text); assert.equal(folded.value, 6);
  const repeated = await session.applyAsync('eval', { code:
    'let current: number = 0; for (let attempt = 0; attempt < 8; attempt++) { if (finished(current)) break; current = step(current); } current' });
  assert.equal(repeated.kind, 'ok'); assert.equal(repeated.value, 3);
  for (const code of ['while (true) {}', 'do {} while (false)', 'for (;;) {}', 'for (const key in values) {}',
    'function* gen() { yield 1; }', 'for (let i = 0; i < 3; i++) { i = 0; }']) {
    const rejected = await session.applyAsync('eval', { code });
    assert.equal(rejected.kind, 'rejected', code); assert.match(rejected.text, /forbidden-loop/, code);
  }
  const grown = await session.applyAsync('eval', { code: 'const xs = [1]; for (const x of xs) { xs.push(x); } xs' });
  assert.equal(grown.kind, 'error'); assert.match(grown.text, /grew while it was being iterated/);
  const recursive = await session.applyAsync('eval', { code: 'function f(n: number): number { return n ? f(n - 1) : 0; } f(3)' });
  assert.equal(recursive.kind, 'rejected'); assert.match(recursive.text, /recursion/);
  const mutual = await session.applyAsync('eval', { code: 'const a = (n: number): number => b(n); const b = (n: number): number => a(n); a(1)' });
  assert.equal(mutual.kind, 'rejected'); assert.match(mutual.text, /Mutual recursion/);
});

const directoryCodebase = () => ({ rewrite: nl('rewrite', { kind: 'directory-reducer', args: { replacement: 'string' }, returns: 'string',
  instructions: 'Replace the greeting in message.txt and report success.' }) });
const directoryAgent = async session => {
  assert.equal(session.lam.subtype, 'directory-reducer');
  assert.equal((await session.applyAsync('read_file', { path: 'message.txt' })).value, 'hello\n');
  assert.equal((await session.applyAsync('edit_file', { path: 'message.txt', find: 'hello', replace_with: session.lam.args.replacement })).kind, 'ok');
  assert.equal((await session.applyAsync('eval', { code: 'return "changed"' })).kind, 'ok');
};
async function reducerSession(files) {
  const folder = Folder.fromFiles(files);
  const opened = open({ type: '() => string', subtype: 'directory-reducer', instructions: 'Apply the rewrite and return its report.',
    codebase: directoryCodebase() }, { agent: directoryAgent });
  opened.lam.projectTransaction = await folder.beginTransaction(false); opened.lam.reducerMode = 'apply';
  return { ...opened, folder };
}

test('folder.apply installs directory reducer changes; a direct call discards them', async () => {
  const applied = await reducerSession({ 'message.txt': 'hello\n' });
  const result = await applied.session.applyAsync('eval', { code: 'const report = await folder.apply(rewrite, "hi"); report' });
  assert.equal(result.kind, 'ok', result.text); assert.equal(result.value, 'changed');
  assert.equal(await applied.lam.projectTransaction.folder.readText('message.txt'), 'hi\n');
  assert.equal(await applied.folder.readText('message.txt'), 'hello\n');
  applied.lam.projectTransaction.abort();
  const direct = await reducerSession({ 'message.txt': 'hello\n' });
  const discarded = await direct.session.applyAsync('eval', { code: 'const report = await rewrite(folder, "hi"); report' });
  assert.equal(discarded.kind, 'ok', discarded.text);
  assert.equal(await direct.lam.projectTransaction.folder.readText('message.txt'), 'hello\n');
  const missing = await direct.session.applyAsync('eval', { code: 'await rewrite("changed")' });
  assert.equal(missing.kind, 'error'); assert.match(missing.text, /directory reducer/);
  direct.lam.projectTransaction.abort();
});

test('folder handles apply a directory reducer within that subdirectory', async () => {
  const { lam, session } = await reducerSession({ 'packages/api/message.txt': 'hello\n', 'packages/web/message.txt': 'untouched\n' });
  const result = await session.applyAsync('eval', { code: 'const api = folder.dir("packages/api"); const report = await api.apply(rewrite, "changed"); report' });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(await lam.projectTransaction.folder.readText('packages/api/message.txt'), 'changed\n');
  assert.equal(await lam.projectTransaction.folder.readText('packages/web/message.txt'), 'untouched\n');
  lam.projectTransaction.abort();
});

test('normal lambdas do not list directory reducers or file tools', async () => {
  const { session } = open({ type: '() => string', instructions: 'Try the reducer.', codebase: directoryCodebase() });
  const agent = new NativeToolAgent(async () => ({ calls: [] }));
  const toolNames = agent.tools(session).map(tool => tool.function.name);
  assert.equal(toolNames.includes('read_file'), false); assert.equal(toolNames.includes('commit'), false);
  assert.equal((await session.applyAsync('eval', { code: 'await rewrite("hi")' })).kind, 'error');
});

test('folder and file handles persist as live scope values', async () => {
  const { lam, session } = await reducerSession({ 'notes/a.txt': 'alpha\n' });
  assert.equal((await session.applyAsync('eval', { code: 'const notes = folder.dir("notes"); notes' })).kind, 'ok');
  assert.equal(lam.let.notes.path, 'notes');
  assert.equal((await session.applyAsync('eval', { code: 'const note = notes.file("a.txt"); note' })).kind, 'ok');
  assert.equal(lam.let.note.path, 'notes/a.txt');
  const read = await session.applyAsync('eval', { code: 'const text: string = await note.readText(); text' });
  assert.equal(read.value, 'alpha\n');
  const listed = await session.applyAsync('eval', { code: '(await folder.files()).map(file => file.path)' });
  assert.deepEqual(listed.value, ['notes/a.txt']);
  lam.projectTransaction.abort();
});

test('a returned value is staged and a reply without a tool call returns it', async () => {
  let turn = 0;
  const requests = [];
  const agent = new NativeToolAgent(request => { requests.push(structuredClone(request));
    return ++turn === 1 ? { calls: [['eval', { code: 'return 11' }]], completion_tokens: 3 } : { text: 'done', completion_tokens: 1 }; });
  const result = await run({ type: '() => number', instructions: 'Return eleven.' }, { agent: session => agent.run(session) });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 11); assert.equal(turn, 2);
  assert.match(requests[1].messages.at(-1).content, /Staged 11 as the result\..*reply done/);
});

test('saying done before returning a value is answered, and only budgets the caller sets end a stuck call', async () => {
  const requests = [];
  const chatty = new NativeToolAgent(request => { requests.push(structuredClone(request)); return { calls: [], text: '7', completion_tokens: 1 }; },
    { maxTurns: 3 });
  const stuck = await run({ type: '() => number', instructions: 'Write a number.' }, { agent: session => chatty.run(session) });
  assert.equal(stuck.outcome.kind, 'quiesced'); assert.match(stuck.outcome.detail, /budget exhausted/);
  assert.match(requests[1].messages.at(-1).content, /There is no result yet\. Call return_result with status "success" and a number/);
  let turns = 0;
  const recovering = new NativeToolAgent(() => ++turns === 1 ? { calls: [], text: '7', completion_tokens: 1 } :
    turns === 2 ? { calls: [['eval', { code: 'return 7' }]], completion_tokens: 1 } : { text: 'done', completion_tokens: 1 });
  const second = await run({ type: '() => number', instructions: 'Write a number.' }, { agent: session => recovering.run(session) });
  assert.equal(second.outcome.kind, 'done'); assert.equal(second.value, 7); assert.equal(turns, 3);
});

test('traces reconstruct offline from their recorded events', async () => {
  const runtime = interpreter({ agent: async session => { await session.applyAsync('eval', { code: 'return 14' }); } });
  const result = await runtime.run(lambda({ type: '() => number', instructions: 'Return fourteen.' }));
  assert.equal(result.outcome.kind, 'done');
  const reconstructed = runtime.trace.reconstruct();
  assert.equal(reconstructed.$lambda.return, 14);
  assert.equal(NativeTraceRecorder.fromEvents(runtime.trace.events).replayObservations().outcome, 'done');
});

test('model seeds follow the derivation vectors', async () => {
  assert.equal(deriveSeed(43, '', 1, 'model-turn', 0), 1079124865);
  const seen = [];
  const agent = new NativeToolAgent(request => {
    seen.push(request.seed);
    return seen.length === 1 ? { calls: [['eval', { code: 'return true' }]], completion_tokens: 1 } : { text: 'done', completion_tokens: 1 };
  });
  const runtime = interpreter({ agent: session => agent.run(session), seedPolicy: { mode: 'derived', root: 43 }, runId: 'seed-run' });
  await runtime.run(lambda({ type: '() => boolean', instructions: 'Return true.' }));
  assert.deepEqual(seen, [deriveSeed(43, 'seed-run', 1, 'model-turn', 0), deriveSeed(43, 'seed-run', 1, 'model-turn', 1)]);
});
