import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Folder } from '../dist/index.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { deriveSeed } from '../dist/native/trace.js';
import { MISSING } from '../dist/native/values.js';
import { NativeTraceRecorder } from '../dist/native/trace.js';
import { interpreter, lambda, nl, session as open, ts } from './support/natlang.mjs';

const run = (body, options) => interpreter(options).run(lambda(body));

test('eval displays an incompatible expression without setting the typed result', async () => {
  const { lam, session } = open({ type: '() => number', instructions: 'Return nine.' });
  const wrong = await session.applyAsync('eval', { code: '"oops"' });
  assert.equal(wrong.kind, 'ok'); assert.equal(wrong.value, 'oops');
  assert.equal(lam.return, MISSING);
  const right = await session.applyAsync('eval', { code: '9' });
  assert.equal(right.kind, 'ok'); assert.equal(right.value, 9); assert.equal(lam.return, 9);
});

test('an empty local assigned to the typed result keeps its list type', async () => {
  const { lam, session } = open({ type: '(items: string[]) => string[]', instructions: 'Select matching items.', args: { items: ['skip'] } });
  const outcome = await session.applyAsync('eval', { code:
    'const ids = items.filter(item => item.startsWith("take")).map(item => item.slice(5)); result = ids;' });
  assert.equal(outcome.kind, 'ok', outcome.text);
  assert.deepEqual(lam.return, []); assert.deepEqual(lam.let.ids, []);
});

test('filter then map uses the mapped element type', async () => {
  const { lam, session } = open({ type: '(items: { id: string, status: string }[]) => string[]',
    instructions: 'Select active ids.', args: { items: [{ id: 'A1', status: 'active' }, { id: 'A2', status: 'closed' }] } });
  const outcome = await session.applyAsync('eval', { code:
    'const ids = items.filter(item => item.status === "active").map(item => item.id); result = ids;' });
  assert.equal(outcome.kind, 'ok', outcome.text);
  assert.deepEqual(lam.return, ['A1']); assert.deepEqual(lam.let.ids, ['A1']);
});

test('a later eval may redeclare a prior local while correcting the result', async () => {
  const { lam, session } = open({ type: '() => string[]', instructions: 'Return selected ids.' });
  assert.equal((await session.applyAsync('eval', { code: 'const ids = []; result = ids;' })).kind, 'ok');
  assert.equal((await session.applyAsync('eval', { code: 'const ids = ["A1"]; result = ids;' })).kind, 'ok');
  assert.deepEqual(lam.return, ['A1']); assert.deepEqual(lam.let.ids, ['A1']);
});

test('read_value can inspect the immutable debug snapshot after eval fails', async () => {
  const { session } = open({ type: '() => number', instructions: 'Return a number.' });
  const failed = await session.applyAsync('eval', { code: 'const values = []; values.noSuchMethod();' });
  assert.equal(failed.kind, 'error');
  const inspected = session.apply('read_value', { expression: `${session.failureBinding}.message` });
  assert.equal(inspected.kind, 'ok'); assert.match(inspected.text, /noSuchMethod/);
});

test('eval returns console.log observations without changing the function result', async () => {
  const { lam, session } = open({ type: '(items: number[]) => number',
    instructions: 'Inspect the average, then return the total.', args: { items: [2, 4, 6] } });
  const observed = await session.applyAsync('eval', { code:
    'const average = items.reduce((sum, item) => sum + item, 0) / items.length; console.log("average", average);' });
  assert.equal(observed.kind, 'ok'); assert.match(observed.text, /console:\naverage 4/);
  assert.equal(lam.return, MISSING);
  const finished = await session.applyAsync('eval', { code: 'result = items.reduce((sum, item) => sum + item, 0)' });
  assert.equal(finished.kind, 'ok'); assert.equal(lam.return, 12); assert.doesNotMatch(finished.text, /average 4/);
});

test('eval can use local functions within one call without persisting them', async () => {
  const { lam, session } = open({ type: '(items: number[]) => number', instructions: 'Return the average.', args: { items: [2, 4, 6] } });
  const arrow = await session.applyAsync('eval', { code:
    'const avg = (xs: number[]) => xs.reduce((sum, x) => sum + x, 0) / xs.length; const mean = avg(items); console.log("mean", mean);' });
  assert.equal(arrow.kind, 'ok'); assert.match(arrow.text, /console:\nmean 4/);
  assert.equal(Object.hasOwn(lam.let, 'avg'), false); assert.equal(lam.let.mean, 4);
  const declared = await session.applyAsync('eval', { code:
    'function average(xs: number[]) { return xs.reduce((sum, x) => sum + x, 0) / xs.length; } result = average(items);' });
  assert.equal(declared.kind, 'ok'); assert.equal(lam.return, 4); assert.equal(Object.hasOwn(lam.let, 'average'), false);
});

test('an incompatible observation does not persist in the typed result slot', async () => {
  const { lam, session } = open({ type: '() => number', instructions: 'Return a count.' });
  assert.equal((await session.applyAsync('eval', { code: 'result = { before: 2, after: 3 }' })).kind, 'ok');
  assert.equal(lam.return, MISSING); assert.equal(Object.hasOwn(lam.let, 'result'), false);
  assert.equal((await session.applyAsync('eval', { code: 'result = 3' })).kind, 'ok');
  assert.equal(lam.return, 3);
});

test('compatible result assignments supply the function value', async () => {
  const unique = open({ type: '(items: string[]) => string[]', instructions: 'Remove duplicates.', args: { items: ['a', 'a', 'b'] } });
  assert.equal((await unique.session.applyAsync('eval', { code: 'let result = [...new Set(items)];' })).kind, 'ok');
  assert.deepEqual(unique.lam.return, ['a', 'b']);
  const doubled = open({ type: '(value: number) => number', instructions: 'Double the value.', args: { value: 7 } });
  assert.equal((await doubled.session.applyAsync('eval', { code: 'result = value * 2;' })).kind, 'ok');
  assert.equal(doubled.lam.return, 14);
  assert.equal((await doubled.session.applyAsync('eval', { code: 'result' })).value, 14);
});

test('native collection temporaries can support a portable result within one eval', async () => {
  const { lam, session } = open({ type: '(items: string[]) => string[]', instructions: 'Remove duplicates.', args: { items: ['a', 'a', 'b'] } });
  const outcome = await session.applyAsync('eval', { code:
    'const seen = new Set<string>(); const result = items.filter(x => !seen.has(x) && Boolean(seen.add(x))); result' });
  assert.equal(outcome.kind, 'ok'); assert.deepEqual(outcome.value, ['a', 'b']);
  assert.equal(Object.prototype.toString.call(lam.let.seen), '[object Set]', 'a Set local persists by reference as a live value');
});

test('read_value inspects read-only inputs', () => {
  const { session } = open({ type: '(state: { head: string }) => string', instructions: 'Inspect the state.', args: { state: { head: 'manifest-1' } } });
  const result = session.apply('read_value', { expression: 'state' });
  assert.equal(result.kind, 'ok'); assert.deepEqual(result.value, { head: 'manifest-1' });
});

const counters = () => ({
  count_true: ts('count_true', 'export default function count_true(flags: boolean[]): number { return flags.filter(Boolean).length; }'),
  as_num: ts('as_num', 'export default async function as_num(flag: boolean): Promise<number> { return flag ? 1 : 0; }'),
});

test('scope eval persists locals, calls imports positionally and uses a compatible final value as the result', async () => {
  const { lam, session } = open({ type: '(flags: boolean[]) => number',
    instructions: 'function total(flags) -> number\n  Count the true flags.\n', args: { flags: [true, false, true] }, codebase: counters() });
  const pure = await session.applyAsync('eval', { code: 'const first = flags[0];\nfirst' });
  assert.equal(pure.kind, 'ok'); assert.equal(pure.value, true); assert.equal(lam.let.first, true);
  const call = await session.applyAsync('eval', { code: 'const count = count_true(flags);\ncount' });
  assert.equal(call.kind, 'ok', call.text); assert.equal(call.value, 2); assert.equal(lam.let.count, 2);
  assert.match(call.text, /Stored local count = 2\./);
  const resultLocal = await session.applyAsync('eval', { code: 'const result = count_true(flags);\nresult' });
  assert.equal(resultLocal.kind, 'ok'); assert.equal(lam.let.result, 2); assert.equal(lam.return, 2);
  assert.equal(session.apply('read_value', { expression: 'flags[1]' }).value, false);
  assert.deepEqual(session.apply('read_value', { expression: 'flags', start: 1, end: 3 }).value, [false, true]);
  const mapped = await session.applyAsync('eval', { code: 'const counts = await Promise.all(flags.map(flag => as_num(flag)));\ncounts' });
  assert.equal(mapped.kind, 'ok'); assert.deepEqual(mapped.value, [1, 0, 1]);
  const loopMapped = await session.applyAsync('eval', { code:
    'const loopCounts = [];\nfor (const flag of flags) {\n  const count = await as_num(flag);\n  loopCounts.push(count);\n}\nloopCounts' });
  assert.equal(loopMapped.kind, 'ok', loopMapped.text); assert.deepEqual(loopMapped.value, [1, 0, 1]);
  const repairedMap = await session.applyAsync('eval', { code: 'const repaired = flags.map(flag => await as_num(flag)); repaired' });
  assert.notEqual(repairedMap.kind, 'ok');
  const names = new NativeToolAgent(() => ({ calls: [] })).tools(session).map(entry => entry.function.name);
  assert.deepEqual(names, ['eval', 'read_value', 'read_function', 'edit_function', 'diff_functions', 'mark_lines',
    'report_blocker', 'report_error']);
  const nullCase = open({ type: '() => null', instructions: 'Return null.' });
  assert.equal((await nullCase.session.applyAsync('eval', { code: 'const result: null = null; result' })).kind, 'ok');
  assert.equal(nullCase.lam.return, null);
  nullCase.session.apply('mark_lines', { start: 1 });
  assert.deepEqual(new NativeToolAgent(() => ({ calls: [] })).tools(nullCase.session), []);
});

test('synchronous TypeScript imports return values directly, including nested callable-folder imports', async () => {
  const { lam, session } = open({ type: '(value: number) => number', instructions: 'Compute the adjusted value.', args: { value: 4 },
    codebase: { adjusted: ts('adjusted', 'import double from "./adjusted/double.ts";\n' +
      'export default function adjusted(value: number): number { return double(value) + 1; }',
      { double: ts('double', 'export default function double(value: number): number { return value * 2; }') }) } });
  const result = await session.applyAsync('eval', { code: 'const answer = adjusted(value); answer' });
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

test('failed eval exposes an immutable, probeable scope and trace without committing partial locals', async () => {
  const { lam, session } = open({ type: '(item: { deep: { count: number } }) => number',
    instructions: 'Return the count plus one.', args: { item: { deep: { count: 4 } } } });
  assert.equal((await session.applyAsync('eval', { code: 'const earlier = item.deep.count; console.log("earlier", earlier);' })).kind, 'ok');
  const failed = await session.applyAsync('eval', { code:
    'const transient = item.deep.count * 2; console.log("before failure", transient); throw new Error("broken step");' });
  assert.equal(failed.kind, 'error'); assert.match(failed.text, /immutable debug/);
  assert.equal(Object.hasOwn(lam.let, 'transient'), false); assert.equal(lam.return, MISSING);
  assert.equal(session.failureDebug.kind, 'runtime');
  assert.equal(session.failureDebug.scope.inputs.item.deep.count, 4);
  assert.equal(session.failureDebug.scope.locals.earlier, 4);
  assert.deepEqual(session.failureDebug.logs, ['before failure 8']);
  assert.ok(session.failureDebug.trace.some(event => event.kind === 'action'));
  assert.match(session.failureDebug.stack, /broken step/);
  const probe = await session.applyAsync('eval', { code: 'console.log("count", debug.scope.inputs.item.deep.count); debug.kind' });
  assert.equal(probe.kind, 'ok'); assert.match(probe.text, /count 4/);
  assert.ok(session.failureDebug);
  assert.equal((await session.applyAsync('eval', { code: 'result = item.deep.count + 1' })).kind, 'ok');
  assert.equal(lam.return, 5); assert.equal(session.failureDebug, undefined);
});

test('compile failure retains source diagnostics for a subsequent eval', async () => {
  const { session } = open({ type: '() => number', instructions: 'Return two.' });
  const failed = await session.applyAsync('eval', { code: 'const answer = ;' });
  assert.equal(failed.kind, 'rejected'); assert.equal(session.failureDebug.kind, 'compile');
  const probe = await session.applyAsync('eval', { code: 'debug.diagnostics[0].code' });
  assert.equal(probe.kind, 'ok'); assert.equal(probe.value, 'typescript-syntax');
  const immutable = await session.applyAsync('eval', { code: 'debug.message = "forged";' });
  assert.equal(immutable.kind, 'rejected'); assert.match(immutable.text, /immutable in eval/);
});

test('failure debug uses a distinct binding when an input is named debug', async () => {
  const { session } = open({ type: '(debug: number) => number', instructions: 'Return debug plus one.', args: { debug: 5 } });
  assert.equal((await session.applyAsync('eval', { code: 'throw new Error("failed")' })).kind, 'error');
  assert.equal(session.failureBinding, '__natlangDebug');
  const probe = await session.applyAsync('eval', { code: '__natlangDebug.scope.inputs.debug' });
  assert.equal(probe.kind, 'ok', probe.text); assert.equal(probe.value, 5);
});

test('model repairs an eval failure in caller-feedback mode using the debug snapshot', async () => {
  const requests = [];
  const script = [['eval', { code: 'result = String(items[9].value)' }], ['eval', { code: 'debug.scope.inputs.items.length' }],
    ['eval', { code: 'result = String(items[0].value)' }], ['mark_lines', { start: 1 }]];
  const agent = new NativeToolAgent(request => {
    requests.push(structuredClone(request));
    return { calls: [script[requests.length - 1]], completion_tokens: 1 };
  }, { validationFeedback: 'caller', maxTurns: 5 });
  const result = await run({ type: '(items: { value: number }[]) => string', instructions: 'Return the first value as text.',
    args: { items: [{ value: 7 }] } }, { agent: session => agent.run(session) });
  assert.equal(result.outcome.kind, 'done', result.outcome.detail); assert.equal(result.value, '7');
  assert.equal(requests.length, 4);
  assert.match(requests[1].messages[0].content, /immutable debug/);
  assert.match(requests[1].messages.at(-1).content, /Debug snapshot available/);
});

test('repeated failed repairs stop at the configured limit, and probes do not reset it', async () => {
  let turns = 0;
  const failing = new NativeToolAgent(() => { turns++; return { calls: [['eval', { code: 'throw new Error("still broken")' }]], completion_tokens: 1 }; },
    { validationFeedback: 'caller', maxFailureRepairs: 1 });
  const first = await run({ type: '() => number', instructions: 'Return one.' }, { agent: session => failing.run(session) });
  assert.equal(first.outcome.kind, 'quiesced'); assert.match(first.outcome.detail, /eval repair limit reached/); assert.equal(turns, 2);
  let probeTurns = 0;
  const script = ['throw new Error("first failure")', 'debug.kind', 'throw new Error("second failure")'];
  const probing = new NativeToolAgent(() => ({ calls: [['eval', { code: script[probeTurns++] }]], completion_tokens: 1 }),
    { validationFeedback: 'caller', maxFailureRepairs: 1 });
  const second = await run({ type: '() => number', instructions: 'Return one.' }, { agent: session => probing.run(session) });
  assert.match(second.outcome.detail, /eval repair limit reached/); assert.equal(probeTurns, 3);
});

test('read_value slices text by characters and lists by items, and pages large values', () => {
  const { session } = open({ type: '(text: string, flags: boolean[]) => string', instructions: 'Return part of the text.',
    args: { text: 'alpha\nbeta', flags: [true, false, true] } });
  const text = session.apply('read_value', { expression: 'text', start: 4, end: 8 });
  assert.equal(text.value, 'a\nbe'); assert.equal(text.text, 'a\nbe');
  assert.deepEqual(session.apply('read_value', { expression: 'flags', start: 1, end: 3 }).value, [false, true]);
  assert.equal(session.apply('read_value', { expression: 'text', start: 2000, end: 3000 }).value, '');
  const large = open({ type: '(text: string, items: number[], fields: Record<string, number>) => number', instructions: 'Inspect.',
    args: { text: 'x'.repeat(5000), items: Array.from({ length: 30 }, (_, i) => i),
      fields: Object.fromEntries(Array.from({ length: 25 }, (_, i) => [`field${i}`, i])) } }).session;
  const textPage = large.apply('read_value', { expression: 'text' });
  assert.equal(textPage.value.length, 2000); assert.match(textPage.text, /characters \[0, 2000\) of 5000; next start=2000/);
  const listPage = large.apply('read_value', { expression: 'items' });
  assert.deepEqual(listPage.value, Array.from({ length: 12 }, (_, i) => i)); assert.match(listPage.text, /items \[0, 12\) of 30; next start=12/);
  const recordPage = large.apply('read_value', { expression: 'fields' });
  assert.equal(Object.keys(recordPage.value).length, 12); assert.match(recordPage.text, /fields \[0, 12\) of 25; next start=12/);
});

test('scope eval reports every still-open instruction line after setting the result', async () => {
  const { lam, session } = open({ type: '() => number', instructions: 'First step.\n# explanation\nSecond step.' });
  const result = await session.applyAsync('eval', { code: '2' });
  assert.match(result.text, /Function result set; lines still open: 1, 3\./); assert.equal(lam.return, 2);
});

test('scope parameters are mutable function-local bindings', async () => {
  const { lam, session } = open({ type: '(count: number, items: number[]) => number', instructions: 'Update the working inputs.',
    args: { count: 2, items: [1] } });
  const result = await session.applyAsync('eval', { code: 'count += 3; items.push(4); count' });
  assert.equal(result.kind, 'ok'); assert.equal(lam.return, 5);
  assert.equal(lam.args.count, 5); assert.deepEqual(lam.args.items, [1, 4]);
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
  assert.equal(session.apply('commit', { value: 'changed' }).kind, 'ok');
  assert.equal(session.apply('mark_lines', { start: 1 }).kind, 'ok');
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

test('the model-turn loop drives tool actions and nudges locally when configured', async () => {
  let turn = 0;
  const agent = new NativeToolAgent(() => ++turn === 1 ? { calls: [['eval', { code: '11' }]], completion_tokens: 3 } :
    { calls: [['mark_lines', { start: 1 }]], completion_tokens: 2 });
  const result = await run({ type: '() => number', instructions: 'Return eleven.' }, { agent: session => agent.run(session) });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 11); assert.equal(turn, 2);
  const caller = new NativeToolAgent(() => ({ calls: [], text: '7', completion_tokens: 1 }));
  const first = await run({ type: '() => number', instructions: 'Write a number.' }, { agent: session => caller.run(session) });
  assert.match(first.outcome.detail, /validation failed: `return` has not been written yet/);
  let localTurns = 0;
  const local = new NativeToolAgent(() => ++localTurns === 1 ? { calls: [], text: '7', completion_tokens: 1 } :
    localTurns === 2 ? { calls: [['eval', { code: '7' }]], completion_tokens: 1 } : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 },
  { validationFeedback: 'local' });
  const second = await run({ type: '() => number', instructions: 'Write a number.' }, { agent: session => local.run(session) });
  assert.equal(second.outcome.kind, 'done'); assert.equal(second.value, 7); assert.equal(localTurns, 3);
});

test('traces reconstruct offline from their recorded events', async () => {
  const runtime = interpreter({ agent: async session => { await session.applyAsync('eval', { code: '14' }); session.apply('mark_lines', { start: 1 }); } });
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
    return seen.length === 1 ? { calls: [['eval', { code: 'true' }]], completion_tokens: 1 } : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
  });
  const runtime = interpreter({ agent: session => agent.run(session), seedPolicy: { mode: 'derived', root: 43 }, runId: 'seed-run' });
  await runtime.run(lambda({ type: '() => boolean', instructions: 'Return true.' }));
  assert.deepEqual(seen, [deriveSeed(43, 'seed-run', 1, 'model-turn', 0), deriveSeed(43, 'seed-run', 1, 'model-turn', 1)]);
});
