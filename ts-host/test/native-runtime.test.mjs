import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Folder, NativeRuntime } from '../dist/index.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { checkedDefinitions } from '../dist/native/codebase.js';
import { readTrace } from '../../web/natlang_lite.mjs';
import { deriveSeed } from '../dist/native/trace.js';
import { admitNativeTrace } from '../dist/native/scenario.js';
import { buildPending, dumpState, MISSING } from '../dist/native/values.js';
import { NativeSession } from '../dist/native/runtime.js';
import { TypeEnv } from '../dist/native/types.js';

const crisp = (type, code, extra = {}) => ({ $lambda: { type, code, engine: 'typescript-host', ...extra } });

test('native eval runs TypeScript collection operations without Python', async () => {
  const one = await new NativeRuntime().runRoot(crisp('() => number', 'return 7;'));
  assert.equal(one.outcome.kind, 'done'); assert.equal(one.value, 7);
  const collection = await new NativeRuntime().runRoot(crisp('() => number[]',
    'const doubled = [2, 4].map(item => item * 3); return doubled;'));
  assert.equal(collection.outcome.kind, 'done'); assert.deepEqual(collection.value, [6, 12]);
});

test('eval displays an incompatible expression without setting the typed result', async () => {
  const lam = buildPending({ $lambda: { type: '() => number', instructions: 'Return nine.' } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const wrong = await session.applyAsync('eval', { code: '"oops"' });
  assert.equal(wrong.kind, 'ok'); assert.equal(wrong.value, 'oops');
  assert.equal(lam.return, MISSING);
  const right = await session.applyAsync('eval', { code: '9' });
  assert.equal(right.kind, 'ok'); assert.equal(right.value, 9); assert.equal(lam.return, 9);
});

test('a compatible result assignment supplies the function value without an extra expression', async () => {
  const lam = buildPending({ $lambda: { type: '(items: string[]) => string[]',
    instructions: 'Remove duplicates.', args: { items: ['a', 'a', 'b'] } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'let result = [...new Set(items)];' });
  assert.equal(result.kind, 'ok');
  assert.deepEqual(lam.return, ['a', 'b']);
});

test('native reads may inspect read-only inputs', () => {
  const lam = buildPending({ $lambda: { type: '(state: { head: string }) => string',
    instructions: 'Inspect the state.', args: { state: { head: 'manifest-1' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = session.apply('read_value', { expression: 'state' });
  assert.equal(result.kind, 'ok');
  assert.deepEqual(result.value, { head: 'manifest-1' });
});

test('scope execution can await checked host bridge calls', async () => {
  const lam = buildPending({ $lambda: { type: '() => number', instructions: 'Return a number.' } });
  const runtime = new NativeRuntime();
  const result = await runtime.evalScopeFor(lam,
    'const helper = async (value: number) => fx.natlang.scope(self.__natlangScopeToken, "call", [value]);\n' +
    'return await helper(6);', 'eval', { args: {}, let: {} }, async (operation, args) => {
      assert.equal(operation, 'call'); assert.deepEqual(args, [6]); return 7;
    });
  assert.equal(result.result, 7);
});

test('scope eval persists locals, calls imports positionally and uses a compatible final value as the result', async () => {
  const lam = buildPending({ $lambda: { type: '(flags: boolean[]) => number',
    instructions: 'function total(flags) -> number\n  Count the true flags.\n', args: { flags: [true, false, true] },
    codebase: { count_true: { args: { flags: 'boolean[]' }, returns: 'number',
      code: 'return flags.filter(Boolean).length;' },
      as_num: { args: { flag: 'boolean' }, returns: 'number', code: 'return flag ? 1 : 0;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const pure = await session.applyAsync('eval', { code: 'const first = flags[0];\nfirst' });
  assert.equal(pure.kind, 'ok'); assert.equal(pure.value, true); assert.equal(lam.let.first, true);
  const call = await session.applyAsync('eval', { code: 'const count = await count_true(flags);\ncount' });
  assert.equal(call.kind, 'ok'); assert.equal(call.value, 2); assert.equal(lam.let.count, 2);
  assert.match(call.text, /Stored local count = 2\./);
  const resultLocal = await session.applyAsync('eval', {
    code: 'const result = await count_true(flags);\nresult' });
  assert.equal(resultLocal.kind, 'ok'); assert.equal(resultLocal.value, 2);
  assert.equal(lam.let.result, 2); assert.equal(lam.return, 2);
  assert.equal(session.apply('read_value', { expression: 'flags[1]' }).value, false);
  assert.deepEqual(session.apply('read_value', { expression: 'flags', start: 1, end: 3 }).value, [false, true]);
  assert.equal(lam.return, 2);
  const mapped = await session.applyAsync('eval', {
    code: 'const counts = await Promise.all(flags.map(flag => as_num(flag)));\ncounts' });
  assert.equal(mapped.kind, 'ok'); assert.deepEqual(mapped.value, [1, 0, 1]);
  const loopMapped = await session.applyAsync('eval', { code:
    'const loopCounts = [];\nfor (const flag of flags) {\n' +
    '  const count = await as_num(flag);\n  loopCounts.push(count);\n}\nloopCounts' });
  assert.equal(loopMapped.kind, 'ok'); assert.deepEqual(loopMapped.value, [1, 0, 1]);
  const repairedMap = await session.applyAsync('eval', { code:
    'const repaired = flags.map(flag => await as_num(flag)); repaired' });
  assert.equal(repairedMap.kind, 'error');
  const sequenced = await session.applyAsync('eval', { code:
    'const again: number[] = await Promise.all(flags.map(flag => as_num(flag)));\n' +
    'const finalCount: number = await count_true(flags);\nfinalCount' });
  assert.equal(sequenced.kind, 'ok'); assert.equal(sequenced.value, 2);
  const names = new NativeToolAgent(() => ({ calls: [] }))
    .tools(session).map(entry => entry.function.name);
  assert.deepEqual(names, ['eval', 'read_value', 'read_function', 'edit_function',
    'diff_functions', 'mark_lines',
    'report_blocker', 'report_error']);
  const nullLam = buildPending({ $lambda: { type: '() => null', instructions: 'Return null.' } });
  const nullSession = new NativeSession(new NativeRuntime(), nullLam, new TypeEnv());
  const nullResult = await nullSession.applyAsync('eval', { code: 'const result: null = null; result' });
  assert.equal(nullResult.kind, 'ok'); assert.equal(nullLam.let.result, null);
  assert.equal(nullLam.return, null);
  nullSession.apply('mark_lines', { start: 1 });
  assert.deepEqual(new NativeToolAgent(() => ({ calls: [] })).tools(nullSession), []);
});

test('ordinary synchronous TypeScript imports return values without await, including nested imports', async () => {
  const lam = buildPending({ $lambda: { type: '(value: number) => number',
    instructions: 'Compute the adjusted value.', args: { value: 4 },
    codebase: { adjusted: { args: { value: 'number' }, returns: 'number', async: false,
      code: 'return double(value) + 1;', codebase: { double: { args: { value: 'number' },
        returns: 'number', async: false, code: 'return value * 2;' } } } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'const answer = adjusted(value); answer' });
  assert.equal(result.kind, 'ok');
  assert.equal(result.value, 9);
  assert.equal(lam.return, 9);
});

test('failed scope child bubbles to eval without leaving a resumable model-facing local', async () => {
  const lam = buildPending({ $lambda: { type: '(value: number) => number',
    instructions: 'Ask the helper.', args: { value: 4 },
    codebase: { inspect: { args: { value: 'number' }, returns: 'number', instructions: 'Inspect the value.' } } } });
  const attempts = [];
  const runtime = new NativeRuntime({ seedPolicy: { mode: 'derived', root: 17 }, agent: session => {
    attempts.push(session.lam.attempts);
    return `original child failure, attempt ${session.lam.attempts}`;
  } });
  const session = new NativeSession(runtime, lam, new TypeEnv());
  const first = await session.applyAsync('eval', { code: 'const answer = await inspect(value); answer' });
  assert.equal(first.kind, 'error');
  assert.match(first.text, /original child failure/);
  assert.equal(Object.hasOwn(lam.let, 'answer'), false);

  const pure = await session.applyAsync('eval', { code: 'const other = value + 1; other' });
  assert.equal(pure.kind, 'ok'); assert.equal(pure.value, 5);
  assert.equal(Object.hasOwn(lam.let, 'answer'), false);

  const unchanged = await session.applyAsync('eval', { code: 'const answer = await inspect(value); answer' });
  assert.equal(unchanged.kind, 'error');
  assert.deepEqual(attempts, [1, 1]);
  assert.equal(Object.hasOwn(lam.let, 'answer'), false);
});

test('scope read_value slices string by zero-based characters and lists by items', () => {
  const lam = buildPending({ $lambda: { type: '(text: string, flags: boolean[]) => string',
    instructions: 'Return part of the text.', args: { text: 'alpha\nbeta', flags: [true, false, true] } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const text = session.apply('read_value', { expression: 'text', start: 4, end: 8 });
  assert.equal(text.kind, 'ok'); assert.equal(text.value, 'a\nbe'); assert.equal(text.text, 'a\nbe');
  assert.deepEqual(session.apply('read_value', { expression: 'flags', start: 1, end: 3 }).value, [false, true]);
  const clamped = session.apply('read_value', { expression: 'text', start: 0, end: 2000 });
  assert.equal(clamped.kind, 'ok'); assert.equal(clamped.value, 'alpha\nbeta');
  assert.equal(session.apply('read_value', { expression: 'text', start: 2000, end: 3000 }).value, '');
});

test('scope eval reports every still-open instruction line after setting the result', async () => {
  const lam = buildPending({ $lambda: { type: '() => number',
    instructions: 'First step.\n# explanation\nSecond step.' } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code: '2' });
  assert.equal(result.kind, 'ok');
  assert.match(result.text, /Function result set; lines still open: 1, 3\./);
  assert.equal(lam.return, 2);
});

test('scope parameters are mutable function-local bindings', async () => {
  const lam = buildPending({ $lambda: { type: '(count: number, items: number[]) => number',
    instructions: 'Update the working inputs.', args: { count: 2, items: [1] } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'count += 3; items.push(4); count' });
  assert.equal(result.kind, 'ok'); assert.equal(lam.return, 5);
  assert.equal(lam.args.count, 5); assert.deepEqual(lam.args.items, [1, 4]);
});

test('scope eval preserves static type when copying an ambiguous value', async () => {
  const lam = buildPending({ $lambda: {
    type: '(initial: { blocked: string[], done: boolean }) => boolean',
    instructions: 'Inspect the initial state.', args: { initial: { blocked: [], done: false } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'let state = initial; state' });
  assert.equal(result.kind, 'ok'); assert.equal(JSON.stringify(result.value), '{"blocked":[],"done":false}');
  assert.ok(lam.letTypes.state);
});

test('scope eval infers a numeric result from a sliced string reduction', async () => {
  const lam = buildPending({ $lambda: { type: '(items: string[]) => number',
    instructions: 'Count adjacent changes.', args: { items: ['a', 'a', 'b', 'c'] } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code:
    'let result = items.slice(1).reduce((count, value, i) => count + (value !== items[i] ? 1 : 0), 0); result' });
  assert.equal(result.kind, 'ok');
  assert.equal(result.value, 2);
  assert.equal(lam.letTypes.result.kind, 'prim');
  assert.equal(lam.letTypes.result.name, 'number');
});

test('scope eval uses a checked helper return type for nested empty collections', async () => {
  const lam = buildPending({ $lambda: {
    type: '(values: string[]) => State', types: { State: '{ values: string[], done: string[] }' },
    instructions: 'Prepare the state.', args: { values: ['a'] }, codebase: {
      prepare: { args: { values: 'string[]' }, returns: 'State',
        code: 'return { values: values, done: [] };' },
    } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'const initial = await prepare(values); initial' });
  assert.equal(result.kind, 'ok');
  assert.deepEqual(lam.let.initial, { values: ['a'], done: [] });
  assert.equal(lam.letTypes.initial.kind, 'name');
  assert.equal(lam.letTypes.initial.name, 'State');
});

test('scope eval preserves collection types through find, filter, slice and map', async () => {
  const lam = buildPending({ $lambda: {
    type: '(tasks: Task[]) => string', types: { Task: '{ id: string, needs: string[] }' },
    instructions: 'Choose a task.', args: { tasks: [{ id: 'a', needs: [] }, { id: 'b', needs: ['a'] }] } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code:
    "const copy = tasks;\n" +
    "const first = copy.find(task => task.id === 'a');\n" +
    'const filtered = copy.filter(task => task.needs.length === 0);\n' +
    'const sliced = copy.slice(0, 1);\n' +
    'const ids = copy.map(task => task.id);\n' +
    'const result: string = first.id;\nresult' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 'a');
  assert.deepEqual(lam.let.first, { id: 'a', needs: [] });
  assert.equal(lam.letTypes.first.kind, 'name'); assert.equal(lam.letTypes.first.name, 'Task');
  for (const name of ['filtered', 'sliced', 'ids']) assert.equal(lam.letTypes[name].kind, 'list');
});

test('scope imports are callable synchronously in branches and as effects', async () => {
  const lam = buildPending({ $lambda: { type: '(flag: boolean) => number',
    instructions: 'Choose and record a number.', args: { flag: false }, codebase: {
      one: { args: {}, returns: 'number', code: 'return 1;' },
      two: { args: {}, returns: 'number', code: 'return 2;' },
      observe: { args: { value: 'number' }, returns: 'boolean', code: 'return value > 0;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code:
    'let chosen: number;\n' +
    'if (flag) { chosen = one(); } else { chosen = await two(); }\n' +
    'await observe(chosen);\n' +
    'chosen' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 2); assert.equal(lam.let.chosen, 2);
});

test('native codebase edits are live while the codebase file set stays fixed', async () => {
  const lam = buildPending({ $lambda: { type: '() => string', instructions: 'Call label.',
    codebase: { label: { args: {}, returns: 'string', code: 'return "old";' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const source = (await session.applyAsync('read_function', { name: 'label' })).value;
  assert.match(source, /return "old";/);
  assert.equal((await session.applyAsync('edit_function', { name: 'label', find: 'return "old";',
    replace_with: 'return "new";' })).kind, 'ok');
  const called = await session.applyAsync('eval', { code: 'const answer = await label(); answer' });
  assert.equal(called.kind, 'ok'); assert.equal(called.value, 'new');
  assert.equal(new NativeToolAgent(() => ({ calls: [] })).tools(session)
    .some(entry => entry.function.name === 'write_file'), false);
});

test('native codebase folder exposes and live edits nested lexical sources', async () => {
  const lam = buildPending({ $lambda: { type: '() => string', instructions: 'Inspect helpers.',
    codebase: { outer: { args: {}, returns: 'string', instructions: 'Call inner.', codebase: {
      inner: { args: {}, returns: 'string', code: 'return "old";' } } } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.match((await session.applyAsync('read_function', { name: 'outer' })).value,
    /import inner from "\.\/outer\/inner\.ts";/);
  assert.equal((await session.applyAsync('edit_function', { name: 'outer.inner', find: 'return "old";',
    replace_with: 'return "new";' })).kind, 'ok');
  assert.equal(lam.codebase.outer.codebase.inner.code, 'return "new";\n');
  assert.equal((await session.applyAsync('edit_function', { name: 'outer', find: 'import inner from',
    replace_with: 'import renamed from' })).kind, 'ok');
  assert.deepEqual(Object.keys(lam.codebase.outer.codebase), ['renamed']);
  assert.equal((await session.applyAsync('edit_function', { name: 'outer.inner', find: 'return "new";',
    replace_with: 'return "newer";' })).kind, 'ok');
  assert.equal(lam.codebase.outer.codebase.renamed.code, 'return "newer";\n');
  assert.equal((await session.applyAsync('edit_function', { name: 'outer', find: './outer/inner.ts',
    replace_with: './missing.ts' })).kind, 'rejected');
  assert.deepEqual(Object.keys(lam.codebase.outer.codebase), ['renamed']);
});

test('normal lambdas edit imports as functions rather than through a filesystem', async () => {
  const lam = buildPending({ $lambda: { type: '() => string', instructions: 'Inspect label.',
    codebase: { label: { args: {}, returns: 'string', code: 'return "old";' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const unavailable = await session.applyAsync('eval', { code:
    'await fs.readText("label.ts")' });
  assert.equal(unavailable.kind, 'error');
  assert.match((await session.applyAsync('read_function', { name: 'label' })).value, /return "old";/);
});

test('scope eval lowers ordinary accumulation and bounded repeat', async () => {
  const lam = buildPending({ $lambda: { type: '(values: number[]) => number', instructions: 'Accumulate.',
    args: { values: [2, 3] }, codebase: {
      add: { args: { acc: 'number', item: 'number' }, returns: 'number', code: 'return acc + item;' },
      step: { args: { state: 'number' }, returns: 'number', code: 'return state + 1;' },
      finished: { args: { state: 'number' }, returns: 'boolean', code: 'return state >= 3;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const folded = await session.applyAsync('eval', { code:
    'let total: number = 1; for (const item of values) { total = await add(total, item); } total' });
  assert.equal(folded.kind, 'ok'); assert.equal(folded.value, 6);
  const repeated = await session.applyAsync('eval', { code:
    'let current: number = 0; for (let attempt = 0; attempt < 8; attempt++) { ' +
    'if (await finished(current)) break; current = await step(current); } current' });
  assert.equal(repeated.kind, 'ok'); assert.equal(repeated.value, 3);
  const whileSession = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const whileRepeated = await whileSession.applyAsync('eval', { code:
    'current = 0; let rounds = 0; while (!(await finished(current)) && rounds < 8) { ' +
    'current = await step(current); rounds++; } current' });
  assert.equal(whileRepeated.kind, 'ok'); assert.equal(whileRepeated.value, 3);
});

test('checked directory reducer metadata survives graph instantiation', () => {
  const graph = checkedDefinitions({ inspect: { kind: 'directory-reducer', args: { request: 'string' },
    returns: 'string', instructions: 'Inspect project/ and return a report.' } }, 'inspect');
  assert.equal(graph.instantiate({ request: 'audit' }).subtype, 'directory-reducer');
});

const directoryProgram = () => ({ $lambda: {
  type: '() => string', subtype: 'directory-reducer', instructions: 'Apply the rewrite and return its report.',
  codebase: { rewrite: { subtype: 'directory-reducer', args: { replacement: 'string' },
    returns: 'string', instructions: 'Replace the greeting in message.txt and report success.' } } } });

const directoryAgent = async session => {
  assert.equal(session.lam.subtype, 'directory-reducer');
  assert.equal((await session.applyAsync('read_file', { path: 'message.txt' })).value, 'hello\n');
  assert.equal((await session.applyAsync('edit_file', { path: 'message.txt', find: 'hello',
    replace_with: session.lam.args.replacement })).kind, 'ok');
  assert.equal((await session.applyAsync('eval', { code: 'const report: string = "changed"; report' })).kind, 'ok');
  assert.equal(session.apply('commit', { value: 'changed' }).kind, 'ok');
  assert.equal(session.apply('mark_lines', { start: 1 }).kind, 'ok');
  assert.equal(session.finish(), true);
};

test('native folder.apply installs directory reducer changes atomically', async () => {
  const folder = Folder.fromFiles({ 'message.txt': 'hello\n' });
  const runtime = new NativeRuntime({ agent: directoryAgent });
  const root = buildPending(directoryProgram());
  root.projectTransaction = await folder.beginTransaction(false); root.reducerMode = 'apply';
  const session = new NativeSession(runtime, root, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'const report = await folder.apply(rewrite, "hi"); report' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 'changed');
  assert.equal(await root.projectTransaction.folder.readText('message.txt'), 'hi\n');
  assert.equal(await folder.readText('message.txt'), 'hello\n');
  root.projectTransaction.abort();
});

test('folder handles apply a directory reducer within that subdirectory', async () => {
  const folder = Folder.fromFiles({
    'packages/api/message.txt': 'hello\n',
    'packages/web/message.txt': 'untouched\n',
  });
  const runtime = new NativeRuntime({ agent: directoryAgent });
  const root = buildPending(directoryProgram());
  root.projectTransaction = await folder.beginTransaction(false); root.reducerMode = 'apply';
  const session = new NativeSession(runtime, root, new TypeEnv());
  const result = await session.applyAsync('eval', { code:
    'const api = folder.dir("packages/api"); const report = await api.apply(rewrite, "changed"); report' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 'changed');
  assert.equal(await root.projectTransaction.folder.readText('packages/api/message.txt'), 'changed\n');
  assert.equal(await root.projectTransaction.folder.readText('packages/web/message.txt'), 'untouched\n');
  root.projectTransaction.abort();
});

test('native direct directory reducer call discards its private changes', async () => {
  const folder = Folder.fromFiles({ 'message.txt': 'hello\n' });
  const runtime = new NativeRuntime({ agent: directoryAgent });
  const root = buildPending(directoryProgram());
  root.projectTransaction = await folder.beginTransaction(false); root.reducerMode = 'apply';
  const session = new NativeSession(runtime, root, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'const report = await rewrite(folder, "hi"); report' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 'changed');
  assert.equal(await root.projectTransaction.folder.readText('message.txt'), 'hello\n');
  assert.equal(await folder.readText('message.txt'), 'hello\n');
  root.projectTransaction.abort();
});

test('direct directory reducer calls require and honor an explicit folder handle', async () => {
  const folder = Folder.fromFiles({ 'nested/message.txt': 'hello\n' });
  const runtime = new NativeRuntime({ agent: directoryAgent });
  const root = buildPending(directoryProgram());
  root.projectTransaction = await folder.beginTransaction(false); root.reducerMode = 'apply';
  const session = new NativeSession(runtime, root, new TypeEnv());
  const missingFolder = await session.applyAsync('eval', { code: 'await rewrite("changed")' });
  assert.equal(missingFolder.kind, 'error');
  const result = await session.applyAsync('eval', { code:
    'const nested = folder.dir("nested"); const report = await rewrite(nested, "changed"); report' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 'changed');
  assert.equal(await root.projectTransaction.folder.readText('nested/message.txt'), 'hello\n');
  root.projectTransaction.abort();
});

test('normal lambdas cannot call directory reducers', async () => {
  const root = buildPending({ $lambda: { type: '() => string', instructions: 'Try the reducer.',
    codebase: { rewrite: { subtype: 'directory-reducer', args: { replacement: 'string' },
      returns: 'string', instructions: 'Change a file.' } } } });
  const session = new NativeSession(new NativeRuntime(), root, new TypeEnv());
  const agent = new NativeToolAgent(async () => ({ calls: [] }));
  const toolNames = agent.tools(session).map(tool => tool.function.name);
  assert.equal(toolNames.includes('read_file'), false);
  assert.equal(toolNames.includes('commit'), false);
  assert.equal(agent.opening(session).includes('rewrite'), false);
  const result = await session.applyAsync('eval', { code: 'await rewrite("hi")' });
  assert.equal(result.kind, 'error');
});

test('native folder and file handles persist as typed scope values', async () => {
  const folder = Folder.fromFiles({ 'notes/a.txt': 'alpha\n' });
  const root = buildPending(directoryProgram());
  root.projectTransaction = await folder.beginTransaction(false); root.reducerMode = 'apply';
  const session = new NativeSession(new NativeRuntime(), root, new TypeEnv());
  const madeDir = await session.applyAsync('eval', { code: 'const notes: Folder = folder.dir("notes"); notes' });
  assert.equal(madeDir.kind, 'ok'); assert.equal(root.let.notes.path, 'notes');
  const madeFile = await session.applyAsync('eval', { code: 'const note: FileHandle = notes.file("a.txt"); note' });
  assert.equal(madeFile.kind, 'ok'); assert.equal(root.let.note.path, 'notes/a.txt');
  const read = await session.applyAsync('eval', { code: 'const text: string = await note.readText(); text' });
  assert.equal(read.kind, 'ok'); assert.equal(read.value, 'alpha\n');
  const listed = await session.applyAsync('eval', { code: 'await fs.list()' });
  assert.equal(listed.kind, 'ok', listed.text); assert.equal(listed.value[0].path, 'notes/a.txt');
  const diff = await session.applyAsync('eval', { code: 'await fs.diff()' });
  assert.equal(diff.kind, 'ok', diff.text); assert.deepEqual(diff.value, []);
  root.projectTransaction.abort();
});

test('native eval calls an imported function positionally', async () => {
  const actions = [];
  const runtime = new NativeRuntime({ agent: async session => {
    actions.push(await session.applyAsync('eval', { code: 'await double(item)' }));
    actions.push(session.apply('mark_lines', { start: 1 }));
  } });
  const result = await runtime.runRoot({ $lambda: {
    type: '(item: number) => number', instructions: 'Double the item.', args: { item: 8 },
    codebase: { double: { args: { item: 'number' }, returns: 'number', engine: 'typescript-host',
      code: 'return item * 2;' } },
  } });
  assert.equal(result.outcome.kind, 'done', `${result.outcome.detail}; actions=${JSON.stringify(actions)}`); assert.equal(result.value, 16);
  assert.deepEqual(actions.map(action => action.kind), ['ok', 'ok']);
});

test('native model-turn loop drives tool actions without Python', async () => {
  let turn = 0;
  const agent = new NativeToolAgent(() => ++turn === 1 ? {
    calls: [['eval', { code: '11' }]], completion_tokens: 3,
  } : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 2 });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  const result = await runtime.runRoot({ $lambda: { type: '() => number', instructions: 'Return eleven.' } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 11); assert.equal(turn, 2);
});

test('native model loop returns validation to caller by default and can nudge locally', async () => {
  const root = { $lambda: { type: '() => number', instructions: 'Write a number.' } };
  let defaultTurns = 0;
  const caller = new NativeToolAgent(() => { defaultTurns++; return { calls: [], text: '7', completion_tokens: 1 }; });
  const first = await new NativeRuntime({ agent: session => caller.run(session) }).runRoot(root);
  assert.equal(first.outcome.kind, 'quiesced');
  assert.match(first.outcome.detail, /validation failed: `return` has not been written yet/);
  assert.equal(defaultTurns, 1);
  let localTurns = 0;
  const local = new NativeToolAgent(() => ++localTurns === 1 ? { calls: [], text: '7', completion_tokens: 1 } :
    localTurns === 2 ? { calls: [['eval', { code: '7' }]], completion_tokens: 1 } :
      { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 }, { validationFeedback: 'local' });
  const second = await new NativeRuntime({ agent: session => local.run(session) }).runRoot(root);
  assert.equal(second.outcome.kind, 'done'); assert.equal(second.value, 7); assert.equal(localTurns, 3);
});

test('native checked definitions snapshot and link source calls', async () => {
  const source = { main: { args: { price: 'number' }, returns: 'number',
    instructions: 'Use double.', uses: { double: 'helper' } },
    helper: { args: { item: 'number' }, returns: 'number', code: 'return item * 2;' } };
  const graph = checkedDefinitions(source, 'main');
  source.helper.code = 'return 999;';
  const runtime = new NativeRuntime({ agent: async session => {
    const out = await session.applyAsync('eval', { code: 'await double(price)' });
    assert.equal(out.kind, 'ok');
    assert.equal(session.apply('mark_lines', { start: 1 }).kind, 'ok');
  } });
  const result = await runtime.runRoot(graph.instantiate({ price: 8 }));
  assert.equal(result.outcome.kind, 'done', result.outcome.detail); assert.equal(result.value, 16);
  assert.throws(() => checkedDefinitions({ a: { returns: 'number', code: 'return 1;', uses: { a: 'a' } } }, 'a'), /recursion/);
});

test('native eval combines loops with imported TypeScript functions', async () => {
  const actions = [];
  const runtime = new NativeRuntime({ agent: async session => {
    actions.push(await session.applyAsync('eval', { code:
      'let total = 0;\nfor (const item of items) {\n' +
      '  const doubled = await double(item);\n  total = await add(total, doubled);\n}\ntotal' }));
    actions.push(session.apply('mark_lines', { start: 1 }));
  } });
  const result = await runtime.runRoot({ $lambda: { type: '(items: number[]) => number',
    instructions: 'Double then add.', args: { items: [1, 2, 3] }, codebase: {
      double: { args: { item: 'number' }, returns: 'number', code: 'return item * 2;' },
      add: { args: { acc: 'number', item: 'number' }, returns: 'number', code: 'return acc + item;' },
    } } });
  assert.equal(result.outcome.kind, 'done', `${result.outcome.detail}; actions=${JSON.stringify(actions)}`); assert.equal(result.value, 12);
  assert.deepEqual(actions.map(x => x.kind), ['ok', 'ok']);
});

test('native traces reconstruct in the shared offline reader', async () => {
  const runtime = new NativeRuntime();
  const result = await runtime.runRoot(crisp('() => number', 'return 14;'));
  assert.equal(result.outcome.kind, 'done');
  assert.equal(runtime.trace.reconstruct(), 14);
  const inspected = readTrace(runtime.trace.events);
  assert.equal(inspected.outcome, 'done');
  assert.equal(inspected.reconstructed, 14);
  assert.equal(runtime.trace.events[0].engine_contracts['typescript-host'].native_state_replayable, false);
});

test('native model seeds match Python derivation vectors', async () => {
  assert.equal(deriveSeed(43, '', 1, 'model-turn', 0), 1079124865);
  assert.equal(deriveSeed(43, 'return/0', 1, 'model-turn', 0), 365401298);
  const seen = [];
  const agent = new NativeToolAgent(request => {
    seen.push(request.seed);
    return seen.length === 1 ? { calls: [['eval', { code: 'true' }]], completion_tokens: 1 } :
      { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
  });
  const runtime = new NativeRuntime({ agent: session => agent.run(session), seedPolicy: { mode: 'derived', root: 43 } });
  const result = await runtime.runRoot({ $lambda: { type: '() => boolean', instructions: 'Return true.' } });
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(seen, [1079124865, deriveSeed(43, '', 1, 'model-turn', 1)]);
});

test('eval passes ordinary positional arguments and checks imported function arity', async () => {
  const lam = buildPending({ $lambda: { type: '(item: number) => number',
    instructions: 'Return a selected value.', args: { item: 2 }, codebase: {
      add: { args: { left: 'number', right: 'number' }, returns: 'number', code: 'return left + right;' },
    } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'await add(item, 5)' });
  assert.equal(result.kind, 'ok');
  assert.equal(lam.return, 7);
  const wrongArity = await session.applyAsync('eval', { code: 'await add(item)' });
  assert.equal(wrongArity.kind, 'error');
});

test('crisp code may replace itself with an unreduced typed task', async () => {
  const result = await new NativeRuntime().runRoot(crisp('() => number',
    'return lambda({ type: "() => number", instructions: "Compute seven." });'));
  assert.equal(result.outcome.kind, 'replaced');
  assert.equal(result.value.nodeKind, 'lambda');
  assert.equal(result.value.status, 'unreduced');
});
