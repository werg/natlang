import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Folder } from '../dist/index.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { renderValue } from '../dist/native/agent.js';
import { deriveSeed } from '../dist/native/trace.js';
import { MISSING } from '../dist/native/values.js';
import { NativeTraceRecorder } from '../dist/native/trace.js';
import { interpreter, lambda, nl, session as open, ts } from './support/natlang.mjs';

const run = (body, options) => interpreter(options).run(lambda(body));

test('fresh native executions show identical live-value observations and preserve aliases', async () => {
  const replay = async () => {
    const { session } = open({ type: '() => number', instructions: 'Inspect the store.' }, {
      services: { store: { open: () => new Map([['stock', 3]]) } },
      declarations: { store: 'declare namespace store { export function open(): Map<string, number>; }' },
    });
    const first = await session.applyAsync('eval', { code: 'const inventory = store.open(); const alias = inventory; inventory;' });
    const second = await session.applyAsync('eval', { code: 'alias;' });
    assert.equal(first.kind, 'ok', first.text); assert.equal(second.kind, 'ok', second.text);
    assert.match(first.text, /Map #1/); assert.match(second.text, /Map #1/);
    return [first.text, second.text];
  };
  assert.deepEqual(await replay(), await replay());
});

test('optional undefined properties in unknown evidence do not poison the next eval',async()=>{
 const {lam,session}=open({type:'() => number',instructions:'Return one.'});
 const first=await session.applyAsync('eval',{code:'const evidence:unknown[]=[{optional:undefined,value:1}]; evidence;'});
 assert.equal(first.kind,'ok',first.text);
 const second=await session.applyAsync('eval',{code:'const copied = evidence; return 1;'});
 assert.equal(second.kind,'ok',second.text);assert.equal(lam.return,1);
});

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

test('a final text reply is the result: text for a string, a JSON value for any other type; prose and done are not', async () => {
  const replies = (texts, type) => { let turn = 0;
    return run({ type, instructions: 'Answer.' }, { agent: session => new NativeToolAgent(() => ({ text: texts[turn++] ?? 'done' }), { maxTurns: 3 }).run(session) }); };
  const label = await replies(['positive'], '() => "positive" | "negative"');
  assert.equal(label.outcome.kind, 'done'); assert.equal(label.value, 'positive');
  const summary = await replies(['The rollout is on track.'], '() => string');
  assert.equal(summary.value, 'The rollout is on track.');
  const number = await replies(['7'], '() => number');
  assert.equal(number.outcome.kind, 'done'); assert.equal(number.value, 7);
  const prose = await replies(['Seven.', 'Seven.', 'Seven.'], '() => number');
  assert.equal(prose.outcome.kind, 'quiesced');
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
  assert.deepEqual(names, ['eval', 'read_page', 'read_code', 'edit_code', 'diff_code',
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

test('what an eval leaves unawaited is awaited before it is kept', async () => {
  const { session } = open({ type: '() => number', instructions: 'Get a number.',
    codebase: { get_number: ts('get_number', 'export default async function get_number(): Promise<number> { return 7; }') } });
  const forgotten = await session.applyAsync('eval', { code: 'const pending = get_number(); pending' });
  assert.equal(forgotten.kind, 'ok', forgotten.text); assert.equal(forgotten.value, 7);
  const kept = await session.applyAsync('eval', { code: 'pending + 1' });
  assert.equal(kept.kind, 'ok', kept.text); assert.equal(kept.value, 8);
  const many = await session.applyAsync('eval', { code: 'const all = [1, 2].map(() => get_number()); const box = { n: get_number() }; [all, box]' });
  assert.equal(many.kind, 'ok', many.text); assert.deepEqual(many.value, [[7, 7], { n: 7 }]);
  const chained = await session.applyAsync('eval', { code: 'get_number().then(n => n * 2)' });
  assert.equal(chained.kind, 'ok', chained.text); assert.equal(chained.value, 14);
  const wrapped = await session.applyAsync('eval', { code: '(async () => { const n = await get_number(); return n + 3; })()' });
  assert.equal(wrapped.kind, 'ok', wrapped.text); assert.equal(wrapped.value, 10);
  const awaited = await session.applyAsync('eval', { code: 'const result = await get_number(); result' });
  assert.equal(awaited.kind, 'ok'); assert.equal(awaited.value, 7);
});

test('a for...of over a missing value names the loop and the fallback', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count.', codebase: {} });
  const missing = await session.applyAsync('eval', { code: 'let total = 0;\nfor (const m of "no digits".match(/\\d/g)) total++;\ntotal' });
  assert.equal(missing.kind, 'error');
  assert.match(missing.text, /`for \(… of "no digits"\.match\(\/\\d\/g\)\)` iterates an array, string, Map or Set, but got null/);
  assert.match(missing.text, /\?\? \[\]/);
  const plain = await session.applyAsync('eval', { code: 'const o = { a: 1 };\nlet n = 0;\nfor (const k of o as any) n++;\nn' });
  assert.match(plain.text, /got a plain object; iterate Object\.keys, Object\.values or Object\.entries of it/);
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

test('natlang calls an eval starts and never awaits fail without ending the process', async () => {
  // In a process of its own: the test runner reports every unhandled rejection, where a host would end.
  const { execFile } = await import('node:child_process');
  const script = `
    import { nl, session as open } from ${JSON.stringify(new URL('./support/natlang.mjs', import.meta.url).href)};
    const { session } = open({ type: '(value: number) => number', instructions: 'Ask the helper.', args: { value: 4 },
      codebase: { inspect: nl('inspect', { args: { value: 'number' }, returns: 'number', instructions: 'Inspect the value.' }) } },
    { agent: async () => { await new Promise(resolve => setTimeout(resolve, 20)); return 'the child gave up'; } });
    // A call left in a variable by an eval that fails, and calls in async callbacks forEach drops.
    const failed = await session.applyAsync('eval', { code: 'const pending = inspect(value); throw new Error("first")' });
    const dropped = await session.applyAsync('eval', { code: '[1, 2].forEach(async v => { await inspect(v); }); 1' });
    await new Promise(resolve => setTimeout(resolve, 200));
    console.log(JSON.stringify([failed.kind, dropped.kind]));`;
  const output = await new Promise((resolve, reject) => execFile(process.execPath, ['--input-type=module', '-e', script],
    (error, stdout, stderr) => error ? reject(new Error(stderr || error.message)) : resolve(stdout)));
  assert.deepEqual(JSON.parse(output.trim()), ['error', 'error']);
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
  assert.match(redeclared.text, /ticket is this call's input and already holds the caller's value; use it directly/);
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

test('long text logs and safely sized values retain their read_page route', async () => {
  const { session } = open({ type: '(text: string) => number', instructions: 'Inspect.', args: { text: 'x'.repeat(4500) } });
  const logged = await session.applyAsync('eval', { code: 'console.log("a".repeat(1400) + "b".repeat(2000) + "c".repeat(1100))' });
  assert.match(logged.text, /^console:\na{1400}b{100}\n<<cut off: 2500 of 4500 characters not shown; transcript\.entry\(0\)\.output holds all of it; read_page\("amber", 2\) shows the next part>>\nc{500}\n/);
  assert.match(session.transcript[0].output, /a{1400}b{2000}c{1100}/, 'the transcript entry holds all of it');
  const second = session.apply('read_page', { id: 'amber', page: 2 });
  assert.equal(second.kind, 'ok');
  assert.match(second.text, /^b{1900}c{100}\n<<page 2 of 3 shown; read_page\("amber", 3\) shows the next part>>$/, 'page 2 starts where the shown head ends');
  assert.match(session.apply('read_page', { id: 'amber', page: 3 }).text, /<<page 3 of 3, the last>>$/);
  assert.equal(session.apply('read_page', { id: 'amber', page: 4 }).kind, 'error');
  const value = await session.applyAsync('eval', { code: 'text' });
  const fullValueLink = value.text.match(/<<full value: 3 pages; read_page\("([^"]+)", 1\) shows the first page>>/);
  assert.ok(fullValueLink, 'a modest value is paged after its bounded preview');
  assert.match(session.transcript[4].output, /^"x{4500}"/);
  assert.match(session.apply('read_page', { id: fullValueLink[1], page: 2 }).text, /page 2 of 3 shown/);
  assert.equal(value.value, 'x'.repeat(4500), 'bounded diagnostics leave the returned computation intact');
});

test('a failed eval reports the service calls it already made, whoever built the runtime', async () => {
  let revision = 25;
  const board = { commit_move: async () => ({ revision: ++revision }), render: async () => { throw new Error('view renderer timed out; the board state is unaffected'); } };
  const { session } = open({ type: '() => number', instructions: 'Move the card.' }, { services: { board } });
  const failed = await session.applyAsync('eval', { code: 'await board.commit_move({ card: "fub", to: "done" }); await board.render(); 1' });
  assert.match(failed.text, /Already performed before the failure \(not undone\): board\.commit_move/);
  assert.match(failed.text, /board\.commit_move completed:.*"revision":26/);
  assert.match(failed.text, /without repeating an already completed write/);
  assert.equal(revision, 26);
});

test('tool-call markup in a reply is never taken as a string result', () => {
  const { session, lam } = open({ type: '() => string', instructions: 'Say something.' });
  assert.equal(session.acceptTextResult('</parameter>\n</function>\n</tool_call>'), false);
  assert.equal(session.acceptTextResult('A plain answer.'), true); assert.equal(lam.return, 'A plain answer.');
});

test('an object keyed by data is stored as a dictionary that later evals can extend', async () => {
  const { session } = open({ type: '() => number', instructions: 'Group.' });
  const grouped = await session.applyAsync('eval', { code: 'const byEmail: Record<string, string[]> = {}; byEmail["meshuk.zubux@mail.example"] = ["C01"]; byEmail["trim@x.example"] = ["C02"]; 1' });
  assert.equal(grouped.kind, 'ok', grouped.text);
  const later = await session.applyAsync('eval', { code: 'byEmail["new@x.example"] = ["C09"]; Object.keys(byEmail).length' });
  assert.equal(later.kind, 'ok', later.text); assert.match(later.text, /^3\b/);
  const many = await session.applyAsync('eval', { code: 'const byId = Object.fromEntries(Array.from({ length: 20 }, (_, i) => ["C" + i, i])); 1' });
  assert.equal(many.kind, 'ok', many.text);
  const added = await session.applyAsync('eval', { code: 'byId["C99"] = 99; byId["C99"]' });
  assert.equal(added.kind, 'ok', added.text); assert.match(added.text, /^99\b/);
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
  assert.match((await session.applyAsync('read_code', { name: 'label' })).value, /return "old";/);
  assert.equal((await session.applyAsync('edit_code', { name: 'label', find: 'return "old";', replace_with: 'return "new";' })).kind, 'ok');
  const called = await session.applyAsync('eval', { code: 'label()' });
  assert.equal(called.kind, 'ok', called.text); assert.equal(called.value, 'new');
  assert.match((await session.applyAsync('read_code', { name: 'outer' })).text, /Natural-language function source/);
  assert.match((await session.applyAsync('read_code', { name: 'inner' })).value, /return "old";/);
  assert.equal((await session.applyAsync('edit_code', { name: 'outer.inner', find: '"old"', replace_with: '"newer"' })).kind, 'ok');
  assert.match(lam.codebase.outer.codebase.inner.text, /"newer"/);
  assert.equal((await session.applyAsync('edit_code', { name: 'outer', find: 'Call inner.', replace_with: 'Call the inner helper.' })).kind, 'ok');
  assert.deepEqual(Object.keys(lam.codebase.outer.codebase), ['inner']);
  const invalid = await session.applyAsync('edit_code', { name: 'label', find: 'return "new";', replace_with: 'while (true) {}' });
  assert.equal(invalid.kind, 'error'); assert.match(invalid.text, /while/);
  assert.match(lam.codebase.label.text, /return "new";/, 'an invalid edit leaves the previous version live');
  assert.deepEqual(JSON.parse((await session.applyAsync('diff_code', {})).text).map(item => item.function).sort(), ['inner.ts', 'label.ts', 'outer.nl']);
  assert.equal(new NativeToolAgent(() => ({ calls: [] })).tools(session).some(entry => entry.function.name === 'write_file'), false);
});

test('eval allows finite iteration and rejects open-ended loops', async () => {
  const { session } = open({ type: '(values: number[]) => number', instructions: 'Accumulate.', args: { values: [2, 3] }, codebase: {
    add: ts('add', 'export default function add(acc: number, item: number): number { return acc + item; }'),
    step: ts('step', 'export default function step(state: number): number { return state + 1; }'),
    finished: ts('finished', 'export default function finished(state: number): boolean { return state >= 3; }') } });
  const folded = await session.applyAsync('eval', { code: 'let total: number = 1; for (const item of values) { total = add(total, item); } total' });
  assert.equal(folded.kind, 'ok', folded.text); assert.equal(folded.value, 6);
  const propertyKeys = await session.applyAsync('eval', { code: `
    const parent = { inherited: 'parent', shared: 'parent' };
    const record = Object.assign(Object.create(parent), { first: 'a', shared: 'own', second: 'b' });
    const enumerable: string[] = [];
    for (const key in record) enumerable.push(key);
    enumerable.join(',') + '|' + Object.keys(record).join(',')` });
  assert.equal(propertyKeys.kind, 'ok', propertyKeys.text);
  assert.equal(propertyKeys.value, 'first,shared,second,inherited|first,shared,second');
  const repeated = await session.applyAsync('eval', { code:
    'let current: number = 0; for (let attempt = 0; attempt < 8; attempt++) { if (finished(current)) break; current = step(current); } current' });
  assert.equal(repeated.kind, 'ok'); assert.equal(repeated.value, 3);
  for (const code of ['while (true) {}', 'do {} while (false)', 'for (;;) {}',
    'function* gen() { yield 1; }', 'for (let i = 0; i < 3; i++) { i = 0; }']) {
    const rejected = await session.applyAsync('eval', { code });
    assert.equal(rejected.kind, 'rejected', code); assert.match(rejected.text, /forbidden-loop/, code);
  }
  const grown = await session.applyAsync('eval', { code: 'const xs = [1]; for (const x of xs) { xs.push(x); } xs' });
  assert.equal(grown.kind, 'error'); assert.match(grown.text, /grew while it was being iterated/);
  // A function may call itself on a smaller argument; anything else fails when the call happens.
  const recursive = await session.applyAsync('eval', { code: 'function f(n: number): number { return n ? f(n - 1) : 0; } f(3)' });
  assert.equal(recursive.kind, 'ok', recursive.text); assert.equal(recursive.value, 0);
  const same = await session.applyAsync('eval', { code: 'function g(n: number): number { return n ? g(n) : 0; } g(3)' });
  assert.equal(same.kind, 'error'); assert.match(same.text, /`g` called itself without a smaller argument/);
  const mutual = await session.applyAsync('eval', { code: 'const a = (n: number): number => b(n); const b = (n: number): number => a(n); a(1)' });
  assert.equal(mutual.kind, 'error'); assert.match(mutual.text, /called itself without a smaller argument/);
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

test('an inline child given a file receives a rebased one-file folder', async () => {
  const folder = Folder.fromFiles({ 'inbox/a.txt': 'urgent request', 'inbox/b.txt': 'private sibling' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Classify the first file.' }, { agent: async child => {
    assert.deepEqual(child.lam.projectTransaction.folder.listFiles().map(entry => entry.path), ['a.txt']);
    assert.equal((await child.applyAsync('read_file', { path: 'a.txt' })).value, 'urgent request');
    assert.equal((await child.applyAsync('read_file', { path: 'b.txt' })).kind, 'error');
    child.apply('return_result', { status: 'success', value: true });
  } });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const verdict: boolean = await nl<boolean>`Is this file urgent?`(folder.file("inbox/a.txt")); verdict' });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(result.value, true);
  lam.projectTransaction.abort();
});

test('an inline child given a nested file handle receives a scoped copy-on-write handle', async () => {
  const folder = Folder.fromFiles({ 'records/MU-13.md': 'museum record', 'records/MU-14.md': 'private sibling' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Pass the selected record and criterion to a child.' }, { agent: async child => {
    const input = Object.values(child.lam.args)[0];
    assert.equal(input.criterion, 'receipt');
    assert.equal(input.file.path, 'MU-13.md');
    assert.deepEqual(child.lam.projectTransaction.folder.listFiles().map(entry => entry.path), ['MU-13.md']);
    await assert.rejects(() => input.file.folder.readText('MU-14.md'));
    assert.equal(await input.file.readText(), 'museum record');
    await input.file.writeText('updated record');
    child.apply('return_result', { status: 'success', value: true });
  } });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const criterion = "receipt"; const file = folder.file("records/MU-13.md"); const input = {criterion, file}; const ok: boolean = await nl<boolean>`Inspect the file.`(input); ok' });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(result.value, true);
  assert.equal(await lam.projectTransaction.folder.readText('records/MU-13.md'), 'updated record');
  assert.equal(await lam.projectTransaction.folder.readText('records/MU-14.md'), 'private sibling');
  lam.projectTransaction.abort();
});

test('a nested inline call reuses its parent scoped FileHandle instead of waiting on its own lease', async () => {
  const folder = Folder.fromFiles({ 'records/TC-3.md': 'attendance signed' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Pass the selected file to an inline child.' }, { agent: async child => {
    const depth = child.runtime.frame?.chain.length ?? 0;
    const code = depth === 1 ?
      'const nested = nl.with({file: live(file)})`Check and update the supplied record.`; const result: boolean = await nested({supplied: file}); const text = await file.readText(); return result && text === "reviewed";' :
      'file = file; const text = await file.readText(); if (text !== "attendance signed") return false; await file.writeText("reviewed"); return true;';
    const result = await child.applyAsync('eval', { code, finish: true });
    assert.ok(['ok', 'completed'].includes(result.kind), result.text);
  } });
  lam.projectTransaction = await folder.beginTransaction(false); lam.reducerMode = 'apply';
  const result = await Promise.race([
    session.applyAsync('eval', { code:
      'let file = folder.file("records/TC-3.md"); const inspect = nl.with({file: live(file)})`Inspect this record.`; const ok: boolean = await inspect({supplied: file}); return ok && (await file.readText()) === "reviewed";' }),
    new Promise((_, reject) => setTimeout(() => reject(new Error('nested FileHandle call deadlocked')), 1000)),
  ]);
  assert.ok(['ok', 'completed'].includes(result.kind), result.text);
  assert.equal(lam.return, true);
  assert.equal(await lam.projectTransaction.folder.readText('records/TC-3.md'), 'reviewed');
  assert.equal(await folder.readText('records/TC-3.md'), 'attendance signed');
  lam.projectTransaction.abort();
});

test('nl tag suffix .with uses explicit snapshot and live capture semantics after interpolation', async () => {
  const order = [];
  const draftType = '{ recordingId: string; permittedChannels: string; requestedChannel: string; decision: string }';
  const { lam, session } = open({ type: `(nextPass: string) => ${draftType}`, args: { nextPass: 'notes' },
    instructions: 'Call the child and return its answer.' }, {
    services: { recordOrder: value => order.push(value) },
    declarations: { recordOrder: 'declare function recordOrder(value: string): void;' },
    agent: async child => {
      assert.equal(child.lam.captures?.decisionRule?.get(), 'before');
      assert.equal(child.lam.captures?.outputContract?.get(), 'four-field Draft');
      assert.equal(child.lam.captures?.counter?.get(), 0);
      const result = await child.applyAsync('eval', { code:
        'counter += 1; return { recordingId: "ARC-218", permittedChannels: "unknown", requestedChannel: "unknown", decision: "withhold" };' });
      assert.ok(['ok', 'completed'].includes(result.kind), result.text);
    },
  });
  const result = await session.applyAsync('eval', { code:
    'let decisionRule = "before"; const outputContract = "four-field Draft"; let counter = 0; ' +
    'const interpolation = () => { recordOrder("interpolation"); return "current"; }; ' +
    'const decisionRuleSnapshot = () => { recordOrder("capture"); return decisionRule; }; ' +
    'const child = nl<{ recordingId: string; permittedChannels: string; requestedChannel: string; decision: string }>' +
    '`Create the final Draft from nextPass using ${interpolation()}, decisionRule, and outputContract.`' +
    '.with({ decisionRule: decisionRuleSnapshot(), outputContract, counter: live(counter) })(nextPass); ' +
    'decisionRule = "after"; return await child;' });
  assert.ok(['ok', 'completed'].includes(result.kind), result.text);
  assert.deepEqual(lam.return, { recordingId: 'ARC-218', permittedChannels: 'unknown', requestedChannel: 'unknown', decision: 'withhold' });
  assert.deepEqual(order, ['interpolation', 'capture'], 'template interpolation runs before suffix capture snapshots');
  assert.equal(lam.let.counter, 1, 'live capture writes the child update back to the parent binding');
});

test('a saved inline nl value can be rebound with the same finite captures and keeps interpolation snapshots', async () => {
  let childCalls = 0;
  const { lam, session } = open({ type: '() => boolean', instructions: 'Create a rebound child and verify it.' }, {
    agent: async child => {
      childCalls += 1;
      assert.equal(child.lam.captures?.policy?.get(), 'replacement');
      assert.match(child.lam.body, /original interpolation/);
      child.apply('return_result', { status: 'success', value: 'done' });
    },
  });
  const wrongType = await session.applyAsync('eval', { code:
    'let policy = "original"; const text = () => "original interpolation"; ' +
    'const update = nl<string>`Use policy and ${text()}.`; ' +
    'update.with({ policy: 42 as unknown as string, text });' });
  assert.equal(wrongType.kind, 'error');
  assert.match(wrongType.text, /policy|capture/i);
  assert.equal(childCalls, 0, 'a runtime type mismatch is rejected before a child invocation');
  const wrongMethod = await session.applyAsync('eval', { code:
    'let policy = "original"; const text = () => "original interpolation"; ' +
    'const update = nl<string>`Use policy and ${text()}.`; ' +
    '(update as any)["with"]({ policy: 42 as unknown as string, text });' });
  assert.equal(wrongMethod.kind, 'error', wrongMethod.text);
  assert.match(wrongMethod.text, /policy|capture/i);
  assert.equal(childCalls, 0, 'direct callable method access uses the same runtime capture validator');

  const result = await session.applyAsync('eval', { code:
    'let policy = "original"; let replacement = "replacement"; let interpolationCalls = 0; ' +
    'const text = () => { interpolationCalls += 1; return "original interpolation"; }; ' +
    'const update = nl<string>`Use policy and ${text()}.`; ' +
    'const direct = (update as any)["with"]({ policy: replacement, text }); ' +
    'const saved = update; const rebound = saved.with({ policy: replacement, text }).with({ policy: replacement, text }); ' +
    'replacement = "changed later"; ' +
    'const answer = await rebound(); const directAnswer = await direct(); ' +
    'return answer === "done" && directAnswer === "done" && interpolationCalls === 1;' });
  assert.ok(['ok', 'completed'].includes(result.kind), result.text);
  assert.equal(lam.return, true);
  assert.equal(childCalls, 2);
  assert.equal(lam.let.interpolationCalls, 1);
});

test('repeated child calls rebase derived captured file handles onto the current parent transaction', async () => {
  const folder = Folder.fromFiles({ 'records/TC-4.md': 'base' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Apply two sequential updates to one selected file.' }, { agent: async child => {
    const depth = child.runtime.frame?.chain.length ?? 0;
    const code = depth === 1 ?
      'const update = nl.with({file})`Append one marker to the file.`; await update(); await update(); return true;' :
      'const current = await file.readText(); await file.writeText(current + "x"); return true;';
    const result = await child.applyAsync('eval', { code, finish: true });
    assert.ok(['ok', 'completed'].includes(result.kind), result.text);
  } });
  lam.projectTransaction = await folder.beginTransaction(false); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const file = folder.file("records/TC-4.md"); const okay: boolean = await nl<boolean>`Update this file twice.`(file); return okay;' });
  assert.ok(['ok', 'completed'].includes(result.kind), result.text);
  assert.equal(lam.return, true);
  assert.equal(await lam.projectTransaction.folder.readText('records/TC-4.md'), 'basexx');
  assert.equal(await folder.readText('records/TC-4.md'), 'base');
  lam.projectTransaction.abort();
});

test('a directory reducer rebases its scoped FolderHandle before nested inline calls', async () => {
  const folder = Folder.fromFiles({ 'records/TC-3.md': 'attendance signed' });
  const inspect = nl('inspect', { kind: 'directory-reducer', returns: 'boolean',
    instructions: 'Inspect the supplied directory record.' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Apply the inspection to the records folder.', codebase: { inspect } }, { agent: async child => {
    const depth = child.runtime.frame?.chain.length ?? 0;
    const code = depth === 1 ?
      'const nested = nl.with({scope: folder})`Read the scoped record.`; const okay: boolean = await nested({scope: folder}); return okay;' :
      'const text = await scope.file("TC-3.md").readText(); return text === "attendance signed";';
    const result = await child.applyAsync('eval', { code, finish: true });
    assert.ok(['ok', 'completed'].includes(result.kind), result.text);
  } });
  lam.projectTransaction = await folder.beginTransaction(false); lam.reducerMode = 'apply';
  const result = await Promise.race([
    session.applyAsync('eval', { code:
      'const records = folder.dir("records"); const okay: boolean = await records.apply(inspect); return okay;' }),
    new Promise((_, reject) => setTimeout(() => reject(new Error('nested FolderHandle call deadlocked')), 1000)),
  ]);
  assert.ok(['ok', 'completed'].includes(result.kind), result.text);
  assert.equal(result.value, true);
  assert.equal(await lam.projectTransaction.folder.readText('records/TC-3.md'), 'attendance signed');
  assert.equal(await folder.readText('records/TC-3.md'), 'attendance signed');
  lam.projectTransaction.abort();
});

test('a directory reducer rebases a derived FileHandle through its FolderHandle while keeping it file-scoped', async () => {
  const folder = Folder.fromFiles({ 'records/TC-5.md': 'selected', 'records/TC-6.md': 'sibling' });
  const inspect = nl('inspect', { kind: 'directory-reducer', args: { file: 'FileHandle' }, returns: 'boolean',
    instructions: 'Update the supplied file.' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Apply the reducer to the records folder.', codebase: { inspect } }, { agent: async child => {
    const file = Object.values(child.lam.args)[0];
    assert.equal(file.path, 'TC-5.md');
    assert.deepEqual(child.lam.projectTransaction.folder.listFiles().map(entry => entry.path), ['TC-5.md', 'TC-6.md']);
    await file.writeText('updated');
    child.apply('return_result', { status: 'success', value: true });
  } });
  lam.projectTransaction = await folder.beginTransaction(false); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const records = folder.dir("records"); const file = folder.file("records/TC-5.md"); await records.apply(inspect, file); true' });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(result.value, true);
  assert.equal(await lam.projectTransaction.folder.readText('records/TC-5.md'), 'updated');
  assert.equal(await lam.projectTransaction.folder.readText('records/TC-6.md'), 'sibling');
  assert.equal(await folder.readText('records/TC-5.md'), 'selected');
  lam.projectTransaction.abort();
});

test('nested arrays preserve repeated file-handle aliases and safely copy __proto__ keys', async () => {
  const folder = Folder.fromFiles({ 'records/MU-13.md': 'museum record', 'records/MU-14.md': 'private sibling' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Inspect the nested record references.' }, { agent: async child => {
    const input = Object.values(child.lam.args)[0];
    assert.equal(input.files[0], input.files[1]);
    assert.equal(input.files[0], input.primary);
    assert.equal(input.meta['__proto__'].path, 'MU-13.md');
    assert.deepEqual(child.lam.projectTransaction.folder.listFiles().map(entry => entry.path), ['MU-13.md']);
    await input.files[0].writeText('array update');
    child.apply('return_result', { status: 'success', value: true });
  } });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const file = folder.file("records/MU-13.md"); const meta = Object.fromEntries([[ ["__", "proto__"].join(""), file ]]); const input = {files: [file, file], primary: file, meta}; const ok: boolean = await nl<boolean>`Inspect these references.`(input); ok' });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(result.value, true);
  assert.equal(await lam.projectTransaction.folder.readText('records/MU-13.md'), 'array update');
  assert.equal(await lam.projectTransaction.folder.readText('records/MU-14.md'), 'private sibling');
  lam.projectTransaction.abort();
});

test('a failed nested-handle child rolls back its file writes', async () => {
  const folder = Folder.fromFiles({ 'records/MU-13.md': 'museum record', 'records/MU-14.md': 'private sibling' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Update then fail on the supplied record.' }, { agent: async child => {
    const input = Object.values(child.lam.args)[0];
    assert.deepEqual(child.lam.projectTransaction.folder.listFiles().map(entry => entry.path), ['MU-13.md']);
    await input.file.writeText('must roll back');
    child.apply('return_result', { status: 'failed', reason: 'deliberate rollback check' });
  } });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const file = folder.file("records/MU-13.md"); const input = {file}; const ok: boolean = await nl<boolean>`Inspect the file.`(input); ok' });
  assert.equal(result.kind, 'error');
  assert.equal(await lam.projectTransaction.folder.readText('records/MU-13.md'), 'museum record');
  assert.equal(await lam.projectTransaction.folder.readText('records/MU-14.md'), 'private sibling');
  const next = await lam.projectTransaction.folder.beginFileTransaction('records/MU-13.md', false);
  next.abort();
  lam.projectTransaction.abort();
});

test('a denied child file write stays auditable and a later eval returns successfully', async () => {
  const folder = Folder.fromFiles({ 'records/packet.md': 'approved evidence', 'records/decision.json': 'parent-owned result' });
  let denied, recovered, deniedTrace;
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Inspect the supplied record and report whether it has content.' }, { agent: async child => {
    denied = await child.applyAsync('eval', { code:
      'await folder.file("decision.json").writeText("unauthorized child output"); return true;' });
    deniedTrace = child.runtime.trace.events.filter(event => event.kind === 'scope_failure');
    recovered = await child.applyAsync('eval', { code:
      'const packet = await folder.file("packet.md").readText(); console.log(packet); return packet.length > 0;' });
    assert.equal(recovered.kind, 'ok', recovered.text);
    child.apply('return_result', { status: 'success', value: recovered.value });
  } });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const file = folder.file("records/packet.md"); const found: boolean = await nl<boolean>`Inspect this record.`(file); found' });
  assert.equal(denied.kind, 'error', denied.text);
  assert.match(denied.text, /outside the supplied FileHandle scope/);
  assert.equal(recovered.value, true);
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(result.value, true);
  assert.equal(await lam.projectTransaction.folder.file('records/decision.json').readText(), 'parent-owned result');
  assert.ok(deniedTrace.some(event => String(event.message ?? '').includes('outside the supplied FileHandle scope')),
    'the denied attempt remains in the failure trace');
  lam.projectTransaction.abort();
});

test('a child given two file handles receives two disjoint roots', async () => {
  const folder = Folder.fromFiles({ 'a/one.txt': 'one', 'b/two.txt': 'two' });
  const { lam, session } = open({ type: '() => boolean', subtype: 'directory-reducer',
    instructions: 'Hand both files to a child.' }, { agent: async child => {
    const [first, second] = Object.values(child.lam.args);
    assert.deepEqual(first.folder.listFiles().map(entry => entry.path), ['one.txt']);
    assert.deepEqual(second.folder.listFiles().map(entry => entry.path), ['two.txt']);
    await first.writeText('ONE'); await second.writeText('TWO');
    child.apply('return_result', { status: 'success', value: true });
  } });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';
  const result = await session.applyAsync('eval', { code:
    'const both: boolean = await nl<boolean>`Read both files.`(folder.file("a/one.txt"), folder.file("b/two.txt")); both' });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(await lam.projectTransaction.folder.readText('a/one.txt'), 'ONE');
  assert.equal(await lam.projectTransaction.folder.readText('b/two.txt'), 'TWO');
  lam.projectTransaction.abort();
});

test('the Python tool shares the reducer folder and records its writes', async () => {
  const { lam, session } = await reducerSession({ 'message.txt': 'hello\n' });
  const result = await session.applyAsync('python', { code:
    'from pathlib import Path\nPath("answer.txt").write_text(Path("message.txt").read_text().upper())\nPath("answer.txt").read_text()' });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(result.value.value, 'HELLO\n');
  assert.equal(await lam.projectTransaction.folder.readText('answer.txt'), 'HELLO\n');
  lam.projectTransaction.abort();
});

test('delegate runs a child on one subfolder and merges its successful edits', async () => {
  const folder = Folder.fromFiles({ 'a/note.txt': 'old', 'b/note.txt': 'untouched' });
  const { lam, session } = open({ type: '() => string', subtype: 'directory-reducer',
    instructions: 'Delegate the first folder.' }, { agent: async child => {
    assert.deepEqual(child.lam.projectTransaction.folder.listFiles().map(entry => entry.path), ['note.txt']);
    assert.equal((await child.applyAsync('read_file', { path: 'note.txt' })).value, 'old');
    await child.applyAsync('write_file', { path: 'note.txt', content: 'new' });
    child.apply('return_result', { status: 'success', value: 'done' });
  } });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';
  const delegated = await session.applyAsync('delegate',
    { path: 'a', instructions: 'Update note.txt.', returns: 'string' });
  assert.equal(delegated.kind, 'ok', delegated.text);
  assert.equal(await lam.projectTransaction.folder.readText('a/note.txt'), 'new');
  assert.equal(await lam.projectTransaction.folder.readText('b/note.txt'), 'untouched');
  lam.projectTransaction.abort();
});

test('a delegate with an invalid return type releases its folder lease before another call', async () => {
  const folder = Folder.fromFiles({ 'jobs/a.json': '{}', 'jobs/b.json': '{}' });
  const { lam, session } = open({ type: '() => string', subtype: 'directory-reducer', instructions: 'Delegate a subproblem.' },
    { agent: async child => child.apply('return_result', { status: 'success', value: 'ok' }) });
  lam.projectTransaction = await folder.beginTransaction(); lam.reducerMode = 'apply';

  const malformed = await session.applyAsync('delegate', { path: 'jobs', instructions: 'Read a.json.',
    returns: 'The answer to the question in a.json, as plain text.' });
  assert.equal(malformed.kind, 'error');
  assert.match(malformed.text, /bad character/);

  const lease = await lam.projectTransaction.folder.dir('jobs').beginTransaction(false);
  lease.abort();
  const valid = await session.applyAsync('delegate', { path: 'jobs', instructions: 'Read a.json.', returns: 'string' });
  assert.equal(valid.kind, 'ok', valid.text);
  lam.projectTransaction.abort();
});

test('editor views numbered lines and applies exact edits to the folder', async () => {
  const { lam, session } = await reducerSession({ 'message.txt': 'hello\nworld\n' });
  const view = await session.applyAsync('editor', { command: 'view', path: 'message.txt', start_line: 2 });
  assert.equal(view.value, '2\tworld');
  assert.equal((await session.applyAsync('editor',
    { command: 'str_replace', path: 'message.txt', old_str: 'world', new_str: 'earth' })).kind, 'ok');
  assert.equal((await session.applyAsync('editor',
    { command: 'create', path: 'new.txt', file_text: 'created' })).kind, 'ok');
  assert.equal(await lam.projectTransaction.folder.readText('message.txt'), 'hello\nearth\n');
  assert.equal(await lam.projectTransaction.folder.readText('new.txt'), 'created');
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
  const chatty = new NativeToolAgent(request => { requests.push(structuredClone(request)); return { calls: [], text: 'Seven.', completion_tokens: 1 }; },
    { maxTurns: 3 });
  const stuck = await run({ type: '() => number', instructions: 'Write a number.' }, { agent: session => chatty.run(session) });
  assert.equal(stuck.outcome.kind, 'quiesced'); assert.match(stuck.outcome.detail, /budget exhausted/);
  assert.match(requests[1].messages.at(-1).content, /There is no result yet\. Call return_result with status "success" and a number/);
  let turns = 0;
  const recovering = new NativeToolAgent(() => ++turns === 1 ? { calls: [], text: 'Seven.', completion_tokens: 1 } :
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

test('a let declared without a value is reported as not kept, not left to fail a later eval', async () => {
  const { lam, session } = open({ type: '() => number', instructions: 'Collect.' });
  const declared = await session.applyAsync('eval', { code: 'let budget: number;\nconst f = (x: number) => x; let later: number; later = f(4);' });
  assert.match(declared.text, /Not kept: budget has no value.*\(let budget = …\)/);
  assert.doesNotMatch(declared.text, /Not kept: .*(f|later)\b/);
  assert.deepEqual(Object.keys(lam.let), ['later']);
});

test('a saved inline function without parameters takes whatever each call passes, in a later eval too', async () => {
  const seen = [];
  const { session } = open({ type: '() => number', instructions: 'Collect.' }, { agent: child => {
    seen.push(child.lam.args); child.lam.return = true; } });
  await session.applyAsync('eval', { code: 'let budget = 3000; const fits = nl<boolean>`Does the amount application asks for fit within budget?`;' });
  const first = await session.applyAsync('eval', { code: 'const v = await fits({ id: "A1", ask: "2,500 dollars" }); v' });
  assert.equal(first.kind, 'ok', first.text); assert.deepEqual(seen, [{ input: { id: 'A1', ask: '2,500 dollars' } }]);
  const again = await session.applyAsync('eval', { code: 'const w = await fits(1, 2); w' });
  assert.equal(again.kind, 'ok', again.text); assert.deepEqual(seen[1], { input: 1, input2: 2 }, 'each call binds its own arguments');
  await session.applyAsync('eval', { code: 'const pick = nl<number>`Pick a number.`; const n = await pick(); n' });
  // Called with a value its first call did not pass, it takes that too.
  const extra = await session.applyAsync('eval', { code: 'const m = await pick(3); m' });
  assert.equal(extra.kind, 'ok', extra.text); assert.deepEqual(seen.at(-1), { input: 3 });
});

test('an external service is called and read by its declaration, and cannot be edited', async () => {
  const { externalModule } = await import('../dist/native/external.js');
  const board = externalModule('board', 'const REJECT = true;\n/** Move a card. */\nexport function commit_move(event: { card: string, to: string }): { revision: number } {\n' +
    '  if (REJECT) throw new Error("event rejected: card " + event.card + " is locked");\n  return { revision: 1 };\n}\n');
  assert.equal(board.declaration, 'declare namespace board {\n  /** Move a card. */\n  export function commit_move(event: {\n      card: string;\n      to: string;\n  }): {\n      revision: number;\n  };\n}');
  const { session } = open({ type: '() => number', instructions: 'Move the card.' },
    { services: { board: board.exports }, declarations: { board: board.declaration } });
  const moved = await session.applyAsync('eval', { code: 'await board.commit_move({ card: "fub", to: "done" })' });
  assert.match(moved.text, /event rejected: card fub is locked/);
  const read = session.apply('read_code', { name: 'board.commit_move' });
  assert.equal(read.kind, 'ok', read.text); assert.match(read.text, /^declare namespace board \{/); assert.doesNotMatch(read.text, /REJECT/);
  const edited = session.apply('edit_code', { name: 'board', find: 'x', replace_with: 'y' });
  assert.equal(edited.kind, 'rejected'); assert.match(edited.text, /board is an external service/);
});

test('a call made from eval is shown the services by their declarations too', async () => {
  const { externalModule } = await import('../dist/native/external.js');
  const facts = externalModule('facts', '/** The number of pages. */\nexport function pages(): number { return 2; }\n');
  const { WorldBridge } = await import('../dist/teacher/world-bridge.js');
  const world = WorldBridge.declaration('alfworld');
  const seen = [];
  const { session } = open({ type: '() => number', instructions: 'Count the pages.' }, {
    services: { facts: facts.exports, world: { look: async () => 'a room' } }, declarations: { facts: facts.declaration, world },
    agent: child => { seen.push({ ...child.runtime.declarations }); child.lam.return = 2; } });
  const called = await session.applyAsync('eval', { code: 'const n = await nl<number>`Count the pages of facts.`(); n' });
  assert.equal(called.kind, 'ok', called.text);
  assert.equal(seen[0].facts, facts.declaration);
  assert.equal(seen[0].world, world, 'a declared const is kept as it is');
  assert.match(world, /look\(\): Promise<string>;/); assert.match(world, /actions\(\): Promise<\{ commands: string\[\] \}>/);
});

test('inline children can delegate a subproblem to another inline child', async () => {
  const texts = [];
  let children = 0;
  const { session } = open({ type: '() => boolean', instructions: 'Judge.' }, { agent: async child => {
    children++;
    if (children === 1) {
      const handed = await child.applyAsync('eval', { code: 'const again = await nl<boolean>`Check whether the sky is blue.`(); return again' });
      texts.push(handed.text);
    } else child.lam.return = true;
  } });
  const judged = await session.applyAsync('eval', { code: 'const v = await nl<boolean>`Is the sky blue?`(); v' });
  assert.equal(judged.kind, 'ok', judged.text); assert.equal(judged.value, true);
  assert.equal(children, 2);
  assert.match(texts[0], /Staged true/);
});

test('inline children inherit callable namespaces without capturing a duplicate binding', async () => {
  const results = [];
  const { session } = open({ type: '() => number', instructions: 'Count the pages.',
    codebase: { facts: ts('facts', 'export function pages(): number { return 3; }') } }, {
    agent: async child => {
      assert.equal(child.lam.captures?.facts, undefined);
      const read = await child.applyAsync('eval', { code: 'return facts.pages()' });
      results.push(read);
    } });
  const called = await session.applyAsync('eval', { code: 'const n = await nl<number>`Count the pages of facts.`(); n' });
  assert.equal(called.kind, 'ok', called.text);
  assert.equal(called.value, 3);
  assert.equal(results[0].kind, 'ok', results[0].text);
});

test('five ad hoc layers are allowed, and the sixth is refused with matching prompt and help', async () => {
  const depths = [], prompts = [];
  const { session } = open({ type: '() => number', instructions: 'Delegate.' }, { agent: async child => {
    const depth = child.runtime.frame.adHocDepth;
    depths.push(depth);
    if (depth < 5) {
      const next = await child.applyAsync('eval', { code: 'return await nl<number>`Continue this subproblem.`()' });
      assert.equal(next.kind, 'ok', next.text);
    } else {
      const refused = await child.applyAsync('eval', { code: 'return await nl<number>`Sixth layer.`()' });
      assert.equal(refused.kind, 'error');
      assert.match(refused.text, /limited to 5 nested layers/);
      const docs = child.apply('read_code', { name: 'nl' });
      assert.match(docs.text, /unavailable at this fifth layer/);
      assert.doesNotMatch(child.apply('read_code', { name: 'iterateOn' }).text, /nl`/);
      const guide = await child.applyAsync('eval', { code: 'missingThing' });
      assert.doesNotMatch(guide.text, /built-ins nl/);
      const agent = new NativeToolAgent(request => {
        prompts.push(request);
        return { calls: [['return_result', { status: 'success', value: 7 }]] };
      });
      await agent.run(child);
    }
  } });
  const called = await session.applyAsync('eval', { code: 'return await nl<number>`Continue this subproblem.`()' });
  assert.equal(called.kind, 'ok', called.text);
  assert.equal(called.value, 7);
  assert.deepEqual(depths, [1, 2, 3, 4, 5]);
  assert.match(prompts[0].messages[0].content, /fifth and final layer/);
  assert.doesNotMatch(prompts[0].messages[0].content, /nl`|nl<|creates one inline/);
  assert.doesNotMatch(prompts[0].messages[1].content, /built-ins nl/);
  assert.doesNotMatch(JSON.stringify(prompts[0].tools), /\(nl, iterateOn/);
});

test('a named function from a file starts a fresh five-layer ad hoc budget', async () => {
  const depths = [];
  let enteredHelper = false;
  const { session } = open({ type: '() => number', instructions: 'Delegate.',
    codebase: { helper: nl('helper', { returns: 'number', instructions: 'Delegate another part.' }) } }, { agent: async child => {
    const depth = child.runtime.frame.adHocDepth;
    depths.push(depth);
    let code;
    if (child.lam.functionName === 'helper') {
      enteredHelper = true;
      assert.equal(depth, 0);
      code = 'return await nl<number>`New root layer 1.`()';
    } else if (depth < 5) code = `return await nl<number>\`${enteredHelper ? 'New root' : 'Original root'} layer ${depth + 1}.\`()`;
    else if (!enteredHelper) code = 'return await helper()';
    else code = 'return 9';
    const result = await child.applyAsync('eval', { code });
    assert.equal(result.kind, 'ok', result.text);
  } });
  const called = await session.applyAsync('eval', { code: 'return await nl<number>`Original root layer 1.`()' });
  assert.equal(called.kind, 'ok', called.text);
  assert.equal(called.value, 9);
  assert.deepEqual(depths, [1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]);
});

test('types an eval declares annotate its locals, then and in later evals', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count.' });
  const declared = await session.applyAsync('eval', { code: 'type State = { n: number };\nconst s: State = { n: 1 };\ns' });
  assert.equal(declared.kind, 'ok', declared.text);
  const later = await session.applyAsync('eval', { code: 'const t: State = { n: 2 };\nt.n' });
  assert.equal(later.kind, 'ok', later.text); assert.equal(later.value, 2);
  const wrong = await session.applyAsync('eval', { code: 'const u: State = { n: "two" } as any;\nu' });
  assert.notEqual(wrong.kind, 'ok', 'a declared type is checked like any other');
  const shaped = await session.applyAsync('eval', { code: 'interface Row { id: string; n: number }\nconst rows: Row[] = [{ id: "a", n: 1 }];\nrows.length' });
  assert.equal(shaped.kind, 'ok', shaped.text);
  const open_ = await session.applyAsync('eval', { code: 'type Pair = Array<number>;\nconst p: Pair = [1, 2];\np' });
  assert.equal(open_.kind, 'ok', 'a type the runtime cannot express is left open');
});

test('a type alias from an earlier eval preserves an inline child result type', async () => {
  const childResults = [];
  const { lam, session } = open({ type: '() => { n: number }', instructions: 'Return the child record.' }, {
    agent: async child => {
      const result = await child.applyAsync('eval', { code: 'const draft = { n: 2 }; return JSON.stringify(draft);', finish: true });
      assert.equal(result.kind, 'completed', result.text);
      childResults.push({ returns: child.lam.type.returns, typesSrc: child.lam.typesSrc, value: child.lam.return });
    },
  });
  const declared = await session.applyAsync('eval', { code: 'type Draft = { n: number }; const note = "notes";' });
  assert.equal(declared.kind, 'ok', declared.text);
  const called = await session.applyAsync('eval', { code:
    'const interpret: Neuralese<(text: string) => Promise<Draft>> = nl.with<Draft>({})`Return a Draft.`;\n' +
    'const draft = await interpret(note);\nreturn draft;' });
  assert.equal(called.kind, 'ok', called.text);
  assert.equal(childResults.length, 1);
  assert.equal(childResults[0].returns.kind, 'name');
  assert.equal(childResults[0].returns.name, 'Draft');
  assert.ok(childResults[0].typesSrc.Draft, 'the child receives the alias definition used by its result type');
  assert.deepEqual(childResults[0].value, { n: 2 });
  assert.deepEqual(lam.return, { n: 2 });
});

test('an iteration that was never run says how to run it', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  const shown = await session.applyAsync('eval', { code: 'const it = await iterateOn((n: number) => n + 1, 0, (n: number) => n > 3);\nconsole.log(String(it));\nit' });
  assert.equal(shown.kind, 'ok', shown.text);
  assert.match(shown.text, /iterateOn\(…\) that has not run: it runs when you await \.until\(done\)/);
  assert.doesNotMatch(shown.text, /observers/, 'its internals are not listed');
});

test('any function runs with .iterateOn(initial), as a natural-language function does', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  const run = await session.applyAsync('eval', { code: 'const step = (n: number): number => n + 1;\nconst last = await step.iterateOn(0).withMeasure(n => Math.max(0,4-n)).until(n => n > 3);\nlast' });
  assert.equal(run.kind, 'ok', run.text); assert.equal(run.value, 4);
});

test('read_code shows an importable package by its declarations, and will not edit it', async () => {
  const { session } = open({ type: '() => number', instructions: 'Look something up.' });
  const listed = session.apply('read_code', { name: 'undici' });
  assert.equal(listed.kind, 'ok', listed.text); assert.match(listed.text, /^declare module "undici" \{ {2}\/\/ \d+ exports; read_code\("undici\.<name>"\)/);
  const one = session.apply('read_code', { name: 'undici.fetch' });
  assert.match(one.text, /^\/\/ fetch, from "undici" .*\nexport declare function fetch/);
  const edited = session.apply('edit_code', { name: 'undici.fetch', find: 'a', replace_with: 'b' });
  assert.equal(edited.kind, 'rejected'); assert.match(edited.text, /belongs to an imported package/);
  assert.equal(session.apply('read_code', { name: 'no_such_package' }).kind, 'rejected');
});

test('read_page on a name in scope says to use it in eval', () => {
  const facts = { page: () => [{ id: 'F1', text: 'Leaves make sugar.' }], pages: () => 1 };
  const { session } = open({ type: '() => number', instructions: 'Read the facts.' }, { services: { facts } });
  const paged = session.apply('read_page', { id: 'facts', page: 1 });
  assert.equal(paged.kind, 'rejected'); assert.match(paged.text, /facts is a name in eval's scope, so use it in eval \(for example facts\.page\(1\)/);
  assert.match(session.apply('read_page', { id: 'amber', page: 1 }).text, /no cut-off output is named "amber"/);
});

test('code written into a reply is answered with a note that it did not run', async () => {
  const requests = [];
  const replies = [{ text: 'The answer:\n```ts\nreturn 6 * 7;\n```' }, { calls: [['return_result', { status: 'success', value: 42 }]] }];
  const result = await run({ type: '() => number', instructions: 'Compute six times seven.' }, { agent: session =>
    new NativeToolAgent(request => { requests.push(structuredClone(request.messages)); return replies[requests.length - 1]; }, { maxTurns: 3 }).run(session) });
  assert.equal(result.value, 42);
  assert.match(requests[1].at(-1).content, /^The code in your reply was not run: code runs only when you call eval with it\. There is no result yet/);
});

test('a service scoped to a function is usable in its calls only, and named elsewhere with who can use it', async () => {
  const tables = { page: () => ['Gigabut threw 60.4.'] };
  const seen = {};
  const agent = async session => {
    const name = session.lam.functionName || 'root';
    seen[name] = (await session.applyAsync('eval', { code: 'const rows = tables.page(1); rows' })).text;
    if (name === 'root') {
      const asked = await session.applyAsync('eval', { code: 'const answer = await expert("How far did Gigabut throw?"); answer' });
      seen.asked = asked.text; session.lam.return = 1;
    } else session.lam.return = ['60.4'];
  };
  const { session } = open({ type: '() => number', instructions: 'Ask the expert.',
    codebase: { expert: nl('expert', { args: { question: 'string' }, returns: 'string[]', instructions: 'Answer question from tables.page(n).' }) } },
  { services: { tables }, serviceScopes: { tables: ['expert.nl'] }, agent });
  await agent(session);
  assert.match(seen.root, /tables is not defined|Cannot find name 'tables'/, 'the root call cannot use it');
  assert.match(seen.expert, /Gigabut threw 60\.4/, 'the expert call can');
  assert.match(seen.asked, /60\.4/);
});

test('a reply that is only a JSON value of the declared type is the result; prose is not', () => {
  const record = open({ type: '() => { verdict: "entailed" | "unknown" }', instructions: 'Judge.' });
  assert.equal(record.session.acceptTextResult('\n\n{\n  "verdict": "unknown"\n}'), true);
  assert.deepEqual(record.lam.return, { verdict: 'unknown' });
  const fenced = open({ type: '() => number[]', instructions: 'List.' });
  assert.equal(fenced.session.acceptTextResult('```json\n[1, 2]\n```'), true); assert.deepEqual(fenced.lam.return, [1, 2]);
  const number = open({ type: '() => number', instructions: 'Count.' });
  assert.equal(number.session.acceptTextResult('The total is 1523.'), false, 'prose is a reply');
  assert.equal(number.session.acceptTextResult('1523'), true); assert.equal(number.lam.return, 1523);
  const wrong = open({ type: '() => { verdict: "entailed" | "unknown" }', instructions: 'Judge.' });
  assert.equal(wrong.session.acceptTextResult('{"verdict": "maybe"}'), false, 'a value of another type is not the result');
});


test('portable dictionary keys preserve __proto__ through eval, locals and typed results', async () => {
  const { lam, session } = open({ type: '() => { totals: Record<string, number> }', instructions: 'Return totals.' });
  const result = await session.applyAsync('eval', { code:
    'const totals: Record<string, number> = JSON.parse(\'{"__proto__":56,"constructor":177,"toString":9}\'); return { totals };' });
  assert.equal(result.kind, 'ok', result.text);
  const expected = JSON.parse('{"__proto__":56,"constructor":177,"toString":9}');
  assert.deepEqual(lam.let.totals, expected);
  assert.deepEqual(lam.return, { totals: expected });
  assert.equal((await session.applyAsync('eval', { code: 'totals["__proto__"]' })).value, 56);
});

test('qualified service aliases preserve structural validation for eval bindings', async () => {
  const { externalModule } = await import('../dist/native/external.js');
  const proof = externalModule('proof', 'type Claim = string; type Step = { claim: Claim, from?: string }; export function verify(steps: Step[]): boolean { return steps[0].claim === "fact"; }');
  const { lam, session } = open({ type: '() => boolean', instructions: 'Verify.' },
    { services: { proof: proof.exports }, declarations: { proof: proof.declaration } });
  const result = await session.applyAsync('eval', { code:
    'const steps: (proof.Step)[] = [{ claim: "fact", from: "F1" }]; return proof.verify(steps);' });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(lam.return, true);
  assert.deepEqual(lam.let.steps, [{ claim: 'fact', from: 'F1' }]);
  const bad = await session.applyAsync('eval', { code: 'const invalid: proof.Step[] = [{ claim: 3 }];' });
  assert.equal(bad.kind, 'rejected', bad.text);
});


test('saved inline functions read the current mutable local across eval transactions', async () => {
  const seen = [];
  const { session } = open({ type: '() => boolean', instructions: 'Judge.' }, { agent: async child => {
    const read = await child.applyAsync('eval', { code: 'budget' });
    seen.push(read.value);
    child.lam.return = read.value >= 4600;
  } });
  assert.equal((await session.applyAsync('eval', { code: 'let budget = 0; const fits = nl<boolean>`Does the amount fit within budget?`; budget = 5000; const first = await fits();' })).kind, 'ok');
  const second = await session.applyAsync('eval', { code: 'budget = 4000; const second = await fits(); second' });
  assert.equal(second.kind, 'ok', second.text);
  assert.equal(second.value, false);
  assert.deepEqual(seen, [5000, 4000]);
});

test('eval finish computes and completes a fresh typed result atomically',async()=>{
 const {session:call,lam}=open({instructions:'Count exactly.',type:'() => number',args:{}});
 const result=await call.applyAsync('eval',{code:'return [true,false,true].filter(Boolean).length;',finish:true});
 assert.equal(result.kind,'completed');assert.equal(result.value,2);assert.equal(lam.return,2);assert.equal(call.completed,true);
});

test('eval finish never publishes an older staged result after inspection, invalid return or failure',async()=>{
 for(const code of ['const inspection = 99;','return "wrong type";','throw new Error("computation failed");']){
  const {session:call,lam}=open({instructions:'Return a number.',type:'() => number',args:{}});
  await call.applyAsync('eval',{code:'return 7;'});
  const result=await call.applyAsync('eval',{code,finish:true});
  assert.ok(['rejected','error'].includes(result.kind),result.text);assert.equal(call.completed,false);assert.equal(lam.return,7);
 }
 const {session:call}=open({instructions:'Return a number.',type:'() => number',args:{}});
 const invalid=await call.applyAsync('eval',{code:'return 1;',finish:'yes'});
 assert.equal(invalid.kind,'rejected');assert.equal(call.completed,false);
});

test('eval finish accepts a fresh final expression without a redundant return statement',async()=>{
 const {session:call}=open({instructions:'Judge the statement.',type:'() => boolean',args:{}});
 const result=await call.applyAsync('eval',{code:'true',finish:true});assert.equal(result.kind,'completed');assert.equal(result.value,true);
 const {session:next}=open({instructions:'Compute a number.',type:'() => number',args:{}});await next.applyAsync('eval',{code:'return 7;'});
 const updated=await next.applyAsync('eval',{code:'99',finish:true});assert.equal(updated.kind,'completed');assert.equal(updated.value,99);
});

test('large record previews retain later fields and direct paths to clipped values',async()=>{
 const {renderValue}=await import('../dist/native/agent.js');
 const value={source:'x'.repeat(10000),evidence:[{trace:'y'.repeat(10000),passed:true}],opportunity:'efficiency',status:'ready',history:[]};
 const shown=renderValue(value,{holder:'context'});
 assert.match(shown,/opportunity: "efficiency"/);assert.match(shown,/status: "ready"/);assert.match(shown,/context\.source holds all of it/);assert.match(shown,/history: \[\]/);assert.ok(shown.length<4000);
 const fits={source:'x'.repeat(1100),status:'ready'};assert.equal(renderValue(fits,{holder:'context'}),renderValue(fits,{holder:'context',budget:Infinity}));
 const full=renderValue(value,{budget:Infinity});assert.doesNotMatch(full,/cut off|not shown/);assert.match(full,/x{10000}/);
 const combinatorial=Array.from({length:1_000_000},(_,index)=>[index]);
 const bounded=renderValue(combinatorial,{holder:'allSubsets'});
 assert.ok(bounded.length<5000,`bounded diagnostic was ${bounded.length} characters`);
 assert.match(bounded,/allSubsets holds all of it/);
});

test('large persistent bindings stay computationally intact while durable capture stays bounded', async () => {
  const { lam, session, runtime } = open({ type: '() => number', instructions: 'Compute from the stored data.' });
  const built = await session.applyAsync('eval', { code: 'const allSubsets = Array.from({ length: 200000 }, (_, i) => [i]); 1' });
  assert.equal(built.kind, 'ok', built.text);
  assert.equal(lam.let.allSubsets.length, 200000);
  assert.deepEqual(lam.let.allSubsets[199999], [199999]);
  const finished = await session.applyAsync('eval', { code: 'return allSubsets.length + allSubsets[199999][0]', finish: true });
  assert.equal(finished.kind, 'completed', finished.text);
  assert.equal(finished.value, 399999);
  assert.equal(lam.let.allSubsets.length, 200000, 'diagnostic projection does not truncate computation state');
  assert.ok(JSON.stringify(runtime.trace.events).length < 300000, 'trace stores bounded state projections');
  const failed = open({ type: '() => number', instructions: 'Return a computed number.' });
  const allocationFailure = await failed.session.applyAsync('eval', {
    code: 'throw new RangeError("Invalid string length")', finish: true,
  });
  assert.equal(allocationFailure.kind, 'error');
  assert.equal(failed.session.completed, false, 'diagnostic protection cannot turn failed computation into a result');
  assert.equal(failed.lam.return, MISSING);
});

test('unknown oversized equality reports a stored-local change conservatively', async () => {
  const { session } = open({ type: '() => number', instructions: 'Compute from stored values.' });
  const first = await session.applyAsync('eval', { code: 'let large = Array.from({ length: 5000 }, (_, i) => i); large.length' });
  assert.equal(first.kind, 'ok', first.text);
  const replacement = await session.applyAsync('eval', { code: 'large = Array.from(large); large.length' });
  assert.equal(replacement.kind, 'ok', replacement.text);
  assert.match(replacement.text, /Stored local large =/,
    'when bounded equality cannot decide, report the binding as changed rather than suppressing it');
});

test('cyclic and deeply nested diagnostic values render finitely with an explicit marker', () => {
  const cyclic = []; cyclic.push(cyclic);
  const cyclicText = renderValue(cyclic, { holder: 'cycle' });
  assert.ok(cyclicText.length < 2000);
  assert.match(cyclicText, /inspection limit reached/);
  const deep = []; let cursor = deep;
  for (let i = 0; i < 10000; i++) { const child = []; cursor.push(child); cursor = child; }
  const deepText = renderValue(deep, { holder: 'deep' });
  assert.ok(deepText.length < 2000);
  assert.match(deepText, /inspection limit reached/);
});
