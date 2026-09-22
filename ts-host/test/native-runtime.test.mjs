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

test('native reducer executes crisp root, Map and Fold without Python', async () => {
  const one = await new NativeRuntime().runRoot(crisp('Lambda<{}, Num>', 'return 7;'));
  assert.equal(one.outcome.kind, 'done'); assert.equal(one.value, 7);
  const map = await new NativeRuntime().runRoot({ $map: { type: 'Map<Num, Num>', over: [2, 4],
    fn: crisp('Lambda<{ item: Num }, Num>', 'return args.item * 3;') } });
  assert.equal(map.outcome.kind, 'done'); assert.deepEqual(map.value, [6, 12]);
  const fold = await new NativeRuntime().runRoot({ $fold: { type: 'Fold<Num, Num>', over: [2, 4], init: 1,
    step: crisp('Lambda<{ acc: Num, item: Num }, Num>', 'return args.acc + args.item;') } });
  assert.equal(fold.outcome.kind, 'done'); assert.equal(fold.value, 7);
});

test('native episodes write through typed actions and reject wrong values', async () => {
  let actions;
  const runtime = new NativeRuntime({ agent: session => {
    const wrong = session.apply('write', { path: 'return', type: 'Num', value: 'oops' });
    const right = session.apply('write', { path: 'return', type: 'Num', value: 9 });
    actions = [wrong, right];
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return nine.' } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 9);
  assert.equal(actions[0].kind, 'rejected'); assert.deepEqual(actions[0].codes, ['type-mismatch']);
  assert.equal(actions[1].kind, 'ok');
});

test('native reads may inspect read-only inputs', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ state: { head: Text } }, Text>',
    instructions: 'Inspect the state.', args: { state: { head: 'manifest-1' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = session.apply('read', { path: 'args/state' });
  assert.equal(result.kind, 'ok');
  assert.deepEqual(result.value, { head: 'manifest-1' });
});

test('scope execution can await checked host bridge calls', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return a number.' } });
  const runtime = new NativeRuntime();
  const result = await runtime.evalScopeFor(lam,
    'const helper = async (value: Num) => fx.natlang.scope(self.__natlangScopeToken, "call", [value]);\n' +
    'return await helper(6);', 'eval', { args: {}, let: {} }, async (operation, args) => {
      assert.equal(operation, 'call'); assert.deepEqual(args, [6]); return 7;
    });
  assert.equal(result.result, 7);
});

test('scope-eval-v1 persists locals, calls imports positionally and stages a named result', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ flags: Bool[] }, Num>',
    instructions: 'function total(flags) -> Num\n  Count the true flags.\n', args: { flags: [true, false, true] },
    codebase: { count_true: { args: { flags: 'Bool[]' }, returns: 'Num',
      code: 'return args.flags.filter(Boolean).length;' },
      as_num: { args: { flag: 'Bool' }, returns: 'Num', code: 'return args.flag ? 1 : 0;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const pure = await session.applyAsync('eval', { code: 'const first = flags[0];\nfirst' });
  assert.equal(pure.kind, 'ok'); assert.equal(pure.value, true); assert.equal(lam.let.first, true);
  const call = await session.applyAsync('eval', { code: 'const count = await count_true(flags);\ncount' });
  assert.equal(call.kind, 'ok'); assert.equal(call.value, 2); assert.equal(lam.let.count, 2);
  assert.equal(session.apply('read_value', { expression: 'flags[1]' }).value, false);
  assert.deepEqual(session.apply('read_value', { expression: 'flags', start: 1, end: 3 }).value, [false, true]);
  assert.equal(session.apply('return_value', { variable: 'count' }).kind, 'ok');
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
    'const again: Num[] = await Promise.all(flags.map(flag => as_num(flag)));\n' +
    'const finalCount: Num = await count_true(flags);\nfinalCount' });
  assert.equal(sequenced.kind, 'ok'); assert.equal(sequenced.value, 2);
  const names = new NativeToolAgent(() => ({ calls: [] }), { toolSchema: 'scope-eval-v1' })
    .tools(session).map(entry => entry.function.name);
  assert.deepEqual(names, ['eval', 'read_value', 'return_value', 'write_value', 'list_files', 'search_files',
    'read_file', 'write_file', 'edit_file', 'diff_files', 'mark_lines',
    'report_blocker', 'report_error']);
  const existingWrite = session.apply('write_value', { name: 'count', value: 3 });
  assert.equal(existingWrite.kind, 'rejected');
  assert.match(existingWrite.text, /new scope variable name/);
  const nullLam = buildPending({ $lambda: { type: 'Lambda<{}, Null>', instructions: 'Return null.' } });
  const nullSession = new NativeSession(new NativeRuntime(), nullLam, new TypeEnv());
  const nullResult = await nullSession.applyAsync('eval', { code: 'const result: Null = null; result' });
  assert.equal(nullResult.kind, 'ok'); assert.equal(nullLam.let.result, null);
  assert.equal(nullSession.apply('return_value', { variable: 'result' }).kind, 'ok');
  nullSession.apply('mark_lines', { start: 1 });
  assert.deepEqual(new NativeToolAgent(() => ({ calls: [] }), { toolSchema: 'scope-eval-v1' }).tools(nullSession), []);
});

test('failed scope child bubbles to eval without leaving a resumable model-facing local', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ value: Num }, Num>',
    instructions: 'Ask the helper.', args: { value: 4 },
    codebase: { inspect: { args: { value: 'Num' }, returns: 'Num', instructions: 'Inspect the value.' } } } });
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

test('scope read_value slices Text by zero-based characters and lists by items', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ text: Text, flags: Bool[] }, Text>',
    instructions: 'Return part of the text.', args: { text: 'alpha\nbeta', flags: [true, false, true] } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const text = session.apply('read_value', { expression: 'text', start: 4, end: 8 });
  assert.equal(text.kind, 'ok'); assert.equal(text.value, 'a\nbe'); assert.equal(text.text, 'a\nbe');
  assert.deepEqual(session.apply('read_value', { expression: 'flags', start: 1, end: 3 }).value, [false, true]);
  const clamped = session.apply('read_value', { expression: 'text', start: 0, end: 2000 });
  assert.equal(clamped.kind, 'ok'); assert.equal(clamped.value, 'alpha\nbeta');
  assert.equal(session.apply('read_value', { expression: 'text', start: 2000, end: 3000 }).value, '');
});

test('scope return_value reports every still-open instruction line', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Num>',
    instructions: 'First step.\n# explanation\nSecond step.' } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  session.apply('write_value', { name: 'answer', value: 2 });
  const result = session.apply('return_value', { variable: 'answer' });
  assert.equal(result.kind, 'ok');
  assert.match(result.text, /Result staged; lines still open: 1, 3\./);
  assert.equal(lam.return, 2);
});

test('scope eval preserves static type when copying an ambiguous value', async () => {
  const lam = buildPending({ $lambda: {
    type: 'Lambda<{ initial: { blocked: Text[], done: Bool } }, Bool>',
    instructions: 'Inspect the initial state.', args: { initial: { blocked: [], done: false } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'let state = initial; state' });
  assert.equal(result.kind, 'ok'); assert.equal(JSON.stringify(result.value), '{"blocked":[],"done":false}');
  assert.ok(lam.letTypes.state);
});

test('scope eval uses a checked helper return type for nested empty collections', async () => {
  const lam = buildPending({ $lambda: {
    type: 'Lambda<{ values: Text[] }, State>', types: { State: '{ values: Text[], done: Text[] }' },
    instructions: 'Prepare the state.', args: { values: ['a'] }, codebase: {
      prepare: { args: { values: 'Text[]' }, returns: 'State',
        code: 'return { values: args.values, done: [] };' },
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
    type: 'Lambda<{ tasks: Task[] }, Text>', types: { Task: '{ id: Text, needs: Text[] }' },
    instructions: 'Choose a task.', args: { tasks: [{ id: 'a', needs: [] }, { id: 'b', needs: ['a'] }] } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code:
    "const copy = tasks;\n" +
    "const first = copy.find(task => task.id === 'a');\n" +
    'const filtered = copy.filter(task => task.needs.length === 0);\n' +
    'const sliced = copy.slice(0, 1);\n' +
    'const ids = copy.map(task => task.id);\n' +
    'const result: Text = first.id;\nresult' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 'a');
  assert.deepEqual(lam.let.first, { id: 'a', needs: [] });
  assert.equal(lam.letTypes.first.kind, 'name'); assert.equal(lam.letTypes.first.name, 'Task');
  for (const name of ['filtered', 'sliced', 'ids']) assert.equal(lam.letTypes[name].kind, 'list');
});

test('scope imports are callable synchronously in branches and as effects', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ flag: Bool }, Num>',
    instructions: 'Choose and record a number.', args: { flag: false }, codebase: {
      one: { args: {}, returns: 'Num', code: 'return 1;' },
      two: { args: {}, returns: 'Num', code: 'return 2;' },
      observe: { args: { value: 'Num' }, returns: 'Bool', code: 'return args.value > 0;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('eval', { code:
    'let chosen: Num;\n' +
    'if (flag) { chosen = one(); } else { chosen = await two(); }\n' +
    'await observe(chosen);\n' +
    'chosen' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 2); assert.equal(lam.let.chosen, 2);
});

test('native codebase edits are live while the codebase file set stays fixed', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Text>', instructions: 'Call label.',
    codebase: { label: { args: {}, returns: 'Text', code: 'return "old";' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const source = (await session.applyAsync('read_file', { path: 'codebase/label.ts' })).value;
  assert.match(source, /return "old";/);
  assert.equal((await session.applyAsync('edit_file', { path: 'codebase/label.ts', find: 'return "old";',
    replace_with: 'return "new";' })).kind, 'ok');
  const called = await session.applyAsync('eval', { code: 'const answer = await label(); answer' });
  assert.equal(called.kind, 'ok'); assert.equal(called.value, 'new');
  assert.equal((await session.applyAsync('write_file', { path: 'codebase/extra.nl', content: source })).kind, 'rejected');
});

test('native codebase folder exposes and live edits nested lexical sources', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Text>', instructions: 'Inspect helpers.',
    codebase: { outer: { args: {}, returns: 'Text', instructions: 'Call inner.', codebase: {
      inner: { args: {}, returns: 'Text', code: 'return "old";' } } } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const listed = (await session.applyAsync('list_files', {})).value;
  assert.deepEqual(listed.map(item => item.path), ['codebase/outer.nl', 'codebase/outer/inner.ts']);
  assert.match((await session.applyAsync('read_file', { path: 'codebase/outer.nl' })).value,
    /import \{ inner \} from "\.\/outer\/inner\.ts";/);
  assert.equal((await session.applyAsync('edit_file', { path: 'codebase/outer/inner.ts', find: 'return "old";',
    replace_with: 'return "new";' })).kind, 'ok');
  assert.equal(lam.codebase.outer.codebase.inner.code, 'return "new";\n');
  const source = (await session.applyAsync('read_file', { path: 'codebase/outer.nl' })).value;
  const relinked = source.replace('import { inner }', 'import { inner as renamed }');
  assert.equal((await session.applyAsync('write_file', { path: 'codebase/outer.nl', content: relinked })).kind, 'ok');
  assert.deepEqual(Object.keys(lam.codebase.outer.codebase), ['renamed']);
  assert.equal((await session.applyAsync('edit_file', { path: 'codebase/outer/inner.ts', find: 'return "new";',
    replace_with: 'return "newer";' })).kind, 'ok');
  assert.equal(lam.codebase.outer.codebase.renamed.code, 'return "newer";\n');
  const invalid = relinked.replace('./outer/inner.ts', './missing.ts');
  assert.equal((await session.applyAsync('write_file', { path: 'codebase/outer.nl', content: invalid })).kind, 'rejected');
  assert.deepEqual(Object.keys(lam.codebase.outer.codebase), ['renamed']);
});

test('native injected fs and file tools share the codebase overlay', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Text>', instructions: 'Inspect label.',
    codebase: { label: { args: {}, returns: 'Text', code: 'return "old";' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const read = await session.applyAsync('eval', { code:
    'const source: Text = await fs.readText("codebase/label.ts"); source' });
  assert.equal(read.kind, 'ok'); assert.match(read.value, /return "old";/);
  const edited = await session.applyAsync('eval', { code:
    'const receipt = await fs.editText("codebase/label.ts", { find: "old", replaceWith: "new" }); receipt' });
  assert.equal(edited.kind, 'ok');
  assert.match((await session.applyAsync('read_file', { path: 'codebase/label.ts' })).value, /new/);
});

test('scope eval lowers ordinary accumulation and bounded repeat', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ values: Num[] }, Num>', instructions: 'Accumulate.',
    args: { values: [2, 3] }, codebase: {
      add: { args: { acc: 'Num', item: 'Num' }, returns: 'Num', code: 'return args.acc + args.item;' },
      step: { args: { state: 'Num' }, returns: 'Num', code: 'return args.state + 1;' },
      finished: { args: { state: 'Num' }, returns: 'Bool', code: 'return args.state >= 3;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const folded = await session.applyAsync('eval', { code:
    'let total: Num = 1; for (const item of values) { total = await add(total, item); } total' });
  assert.equal(folded.kind, 'ok'); assert.equal(folded.value, 6);
  const repeated = await session.applyAsync('eval', { code:
    'let current: Num = 0; for (let attempt = 0; attempt < 8; attempt++) { ' +
    'if (await finished(current)) break; current = await step(current); } current' });
  assert.equal(repeated.kind, 'ok'); assert.equal(repeated.value, 3);
  const whileSession = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const whileRepeated = await whileSession.applyAsync('eval', { code:
    'current = 0; let rounds = 0; while (!(await finished(current)) && rounds < 8) { ' +
    'current = await step(current); rounds++; } current' });
  assert.equal(whileRepeated.kind, 'ok'); assert.equal(whileRepeated.value, 3);
});

test('checked directory reducer metadata survives graph instantiation', () => {
  const graph = checkedDefinitions({ inspect: { kind: 'directory-reducer', args: { request: 'Text' },
    returns: 'Text', instructions: 'Inspect project/ and return a report.' } }, 'inspect');
  assert.equal(graph.instantiate({ request: 'audit' }).subtype, 'directory-reducer');
});

const directoryProgram = folder => ({ $lambda: {
  type: 'Lambda<{ folder: Folder }, Text>', instructions: 'Apply the rewrite and return its report.',
  args: { folder }, codebase: { rewrite: { subtype: 'directory-reducer', args: { replacement: 'Text' },
    returns: 'Text', instructions: 'Replace the greeting in project/message.txt and report success.' } } } });

const directoryAgent = async session => {
  assert.equal(session.lam.subtype, 'directory-reducer');
  assert.equal((await session.applyAsync('read_file', { path: 'project/message.txt' })).value, 'hello\n');
  assert.equal((await session.applyAsync('edit_file', { path: 'project/message.txt', find: 'hello',
    replace_with: session.lam.args.replacement })).kind, 'ok');
  assert.equal((await session.applyAsync('eval', { code: 'const report: Text = "changed"; report' })).kind, 'ok');
  assert.equal(session.apply('commit', { value: 'report' }).kind, 'ok');
  assert.equal(session.apply('mark_lines', { start: 1 }).kind, 'ok');
  assert.equal(session.finish(), true);
};

test('native folder.apply installs directory reducer changes atomically', async () => {
  const folder = Folder.fromFiles({ 'message.txt': 'hello\n' });
  const runtime = new NativeRuntime({ agent: directoryAgent });
  const root = buildPending(directoryProgram(folder));
  const session = new NativeSession(runtime, root, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'const report = await folder.apply(rewrite, "hi"); report' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 'changed');
  assert.equal(await folder.readText('message.txt'), 'hi\n');
});

test('native direct directory reducer call discards its private changes', async () => {
  const folder = Folder.fromFiles({ 'message.txt': 'hello\n' });
  const runtime = new NativeRuntime({ agent: directoryAgent });
  const root = buildPending(directoryProgram(folder));
  const session = new NativeSession(runtime, root, new TypeEnv());
  const result = await session.applyAsync('eval', { code: 'const report = await rewrite(folder, "hi"); report' });
  assert.equal(result.kind, 'ok'); assert.equal(result.value, 'changed');
  assert.equal(await folder.readText('message.txt'), 'hello\n');
});

test('native folder and file handles persist as typed scope values', async () => {
  const folder = Folder.fromFiles({ 'notes/a.txt': 'alpha\n' });
  const root = buildPending(directoryProgram(folder));
  const session = new NativeSession(new NativeRuntime(), root, new TypeEnv());
  const madeDir = await session.applyAsync('eval', { code: 'const notes: Folder = folder.dir("notes"); notes' });
  assert.equal(madeDir.kind, 'ok'); assert.equal(root.let.notes.path, 'notes');
  const madeFile = await session.applyAsync('eval', { code: 'const note: FileHandle = notes.file("a.txt"); note' });
  assert.equal(madeFile.kind, 'ok'); assert.equal(root.let.note.path, 'notes/a.txt');
  const read = await session.applyAsync('eval', { code: 'const text: Text = await note.readText(); text' });
  assert.equal(read.kind, 'ok'); assert.equal(read.value, 'alpha\n');
});

test('explicit inference and action limits still apply', async () => {
  const program = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return seven.' } };
  const runtime = new NativeRuntime();
  assert.equal(runtime.options.maxEpisodes, undefined);
  assert.equal(runtime.options.maxDepth, undefined);
  const capped = new NativeSession(new NativeRuntime({ maxActions: 2, maxToolCalls: 3 }),
    buildPending(program), new TypeEnv());
  assert.equal(capped.apply('read', { path: 'instructions' }).kind, 'ok');
  assert.equal(capped.apply('read', { path: 'instructions' }).kind, 'ok');
  assert.equal(capped.apply('read', { path: 'instructions' }).kind, 'budget');
  const cappedAgent = new NativeToolAgent(() => ({ calls: [['read', { path: 'instructions' }]],
    completion_tokens: 1 }), { maxTurns: 2 });
  const stopped = await new NativeRuntime({ agent: session => cappedAgent.run(session) }).runRoot(program);
  assert.equal(stopped.outcome.kind, 'quiesced');
  assert.match(stopped.outcome.detail, /budget exhausted/);
});

test('native Fold can wait for more input and resume', async () => {
  let poll = 0;
  const source = { poll: () => ++poll === 1 ? { kind: 'item', value: 2 } :
    poll === 2 ? { kind: 'empty' } : poll === 3 ? { kind: 'item', value: 3 } : { kind: 'closed' } };
  const runtime = new NativeRuntime({ stream: source });
  const root = { $fold: { type: 'Fold<Num, Num>', over: [], init: 1,
    step: crisp('Lambda<{ acc: Num, item: Num }, Num>', 'return args.acc + args.item;') } };
  const first = await runtime.runRoot(root);
  assert.equal(first.outcome.kind, 'waiting');
  const second = await runtime.runRoot(first.value);
  assert.equal(second.outcome.kind, 'done'); assert.equal(second.value, 6);
  assert.equal(runtime.trace.coverage().live_source_reconstructable, false);
  assert.equal(runtime.trace.replayObservations().outcome, 'done');
});

test('native Iterate uses a checked step and Boolean completion condition', async () => {
  const result = await new NativeRuntime().runRoot({ $iterate: {
    type: 'Iterate<Num>', init: 0, max: 5, state_name: 'value', check_name: 'value',
    step: crisp('Lambda<{ value: Num }, Num>', 'return args.value + 1;'),
    check: crisp('Lambda<{ value: Num }, Bool>', 'return args.value >= 3;'),
  } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 3);
});

test('native codebase call binds typed inputs and reduces a child', async () => {
  const actions = [];
  const runtime = new NativeRuntime({ agent: async session => {
    actions.push(await session.applyAsync('call', { function: 'double', to: 'return',
      inputs: { item: 'args/item' } }));
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: {
    type: 'Lambda<{ item: Num }, Num>', instructions: 'Double the item.', args: { item: 8 },
    codebase: { double: { args: { item: 'Num' }, returns: 'Num', engine: 'typescript-host',
      code: 'return args.item * 2;' } },
  } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 16);
  assert.equal(actions[0].kind, 'done');
});

test('native model-turn loop drives tool actions without Python', async () => {
  let turn = 0;
  const agent = new NativeToolAgent(() => ++turn === 1 ? {
    calls: [['write', { path: 'return', type: 'Num', value: 11 }]], completion_tokens: 3,
  } : { calls: [], text: 'done', completion_tokens: 2 });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return eleven.' } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 11); assert.equal(turn, 2);
});

test('native model loop returns validation to caller by default and can nudge locally', async () => {
  const root = { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Write a number.' } };
  let defaultTurns = 0;
  const caller = new NativeToolAgent(() => { defaultTurns++; return { calls: [], text: '7', completion_tokens: 1 }; });
  const first = await new NativeRuntime({ agent: session => caller.run(session) }).runRoot(root);
  assert.equal(first.outcome.kind, 'quiesced');
  assert.match(first.outcome.detail, /validation failed: `return` has not been written yet/);
  assert.equal(defaultTurns, 1);
  let localTurns = 0;
  const local = new NativeToolAgent(() => ++localTurns === 1 ? { calls: [], text: '7', completion_tokens: 1 } :
    localTurns === 2 ? { calls: [['write', { path: 'return', type: 'Num', value: 7 }]], completion_tokens: 1 } :
      { calls: [], text: 'done', completion_tokens: 1 }, { validationFeedback: 'local' });
  const second = await new NativeRuntime({ agent: session => local.run(session) }).runRoot(root);
  assert.equal(second.outcome.kind, 'done'); assert.equal(second.value, 7); assert.equal(localTurns, 3);
});

test('native checked definitions snapshot and link source calls', async () => {
  const source = { main: { args: { price: 'Num' }, returns: 'Num',
    instructions: 'Use double.', uses: { double: 'helper' } },
    helper: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' } };
  const graph = checkedDefinitions(source, 'main');
  source.helper.code = 'return 999;';
  const runtime = new NativeRuntime({ agent: async session => {
    const out = await session.applyAsync('call', { function: 'double', to: 'return',
      inputs: { item: 'args/price' } });
    assert.equal(out.kind, 'done'); session.finish();
  } });
  const result = await runtime.runRoot(graph.instantiate({ price: 8 }));
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 16);
  assert.throws(() => checkedDefinitions({ a: { returns: 'Num', code: 'return 1;', uses: { a: 'a' } } }, 'a'), /recursion/);
});

test('native source calls lower to Map and Fold combinators', async () => {
  const actions = [];
  const runtime = new NativeRuntime({ agent: async session => {
    actions.push(await session.applyAsync('call', { function: 'double', to: 'let/doubled', over: 'args/items' }));
    actions.push(await session.applyAsync('call', { function: 'add', to: 'return', over: 'let/doubled', init: 0 }));
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{ items: Num[] }, Num>',
    instructions: 'Double then add.', args: { items: [1, 2, 3] }, codebase: {
      double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' },
      add: { args: { acc: 'Num', item: 'Num' }, returns: 'Num', code: 'return args.acc + args.item;' },
    } } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 12);
  assert.deepEqual(actions.map(x => x.kind), ['done', 'done']);
});

test('tools-v4 binds ordinary positional paths for calls and folds', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ items: Num[] }, Num>',
    instructions: 'Add the items.', args: { items: [2, 4] }, codebase: {
      add: { args: { total: 'Num', entry: 'Num' }, returns: 'Num',
        code: 'return args.total + args.entry;' },
    } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.equal(session.apply('write_value', { destination: 'let/zero', type: 'Num', value: 0 }).kind, 'ok');
  const result = await session.applyAsync('fold', { function: 'add', items: 'args/items',
    initial: 'let/zero', save_as: 'return' });
  assert.equal(result.kind, 'done');
  assert.equal(lam.return, 6);
});

test('native traces reconstruct in the shared offline reader', async () => {
  const runtime = new NativeRuntime();
  const result = await runtime.runRoot(crisp('Lambda<{}, Num>', 'return 14;'));
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
    return seen.length === 1 ? { calls: [['write', { path: 'return', type: 'Bool', value: true }]], completion_tokens: 1 } :
      { calls: [], text: 'done', completion_tokens: 1 };
  });
  const runtime = new NativeRuntime({ agent: session => agent.run(session), seedPolicy: { mode: 'derived', root: 43 } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Bool>', instructions: 'Return true.' } });
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(seen, [1079124865, deriveSeed(43, '', 1, 'model-turn', 1)]);
});

test('nested pending paths preserve types, frozen slots, and writable child results', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, { score: Num }>', instructions: 'Compute a score.',
    return: { score: { $lambda: { type: 'Lambda<{}, Num>', instructions: 'Find it.' } } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.equal(session.apply('read', { path: 'return/score/instructions' }).kind, 'ok');
  assert.equal(session.apply('write', { path: 'return/score/return', type: 'Num', value: 5 }).kind, 'ok');
  assert.equal(session.apply('write', { path: 'return/score/args/unknown', type: 'Num', value: 5 }).kind, 'rejected');
  assert.equal(session.apply('edit', { path: 'args/name', old: 'a', new: 'b' }).kind, 'rejected');
  assert.equal(lam.return.score.return, 5);
});

test('failed local writes roll back declaration; source copy, delete and effect scope are checked', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ item: Num }, Num>', instructions: 'Use item.', args: { item: 3 } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.equal(session.apply('write', { path: 'let/a', type: 'Num', value: 'bad' }).kind, 'rejected');
  assert.equal(Object.hasOwn(lam.letTypes, 'a'), false);
  assert.equal(session.apply('write', { path: 'let/a', type: 'Num', source: 'args/item' }).kind, 'ok');
  assert.equal(lam.let.a, 3);
  assert.equal(session.apply('copy', { from: 'let/a', to: 'return' }).kind, 'ok');
  assert.equal(lam.return, 3);
  assert.equal(session.apply('delete', { path: 'let/a' }).kind, 'ok');
  assert.equal(Object.hasOwn(lam.let, 'a'), false);
  assert.equal(session.apply('write', { path: 'return', type: 'Num', value: { $lambda: {
    type: 'Lambda<{}, Num>', code: 'return 1;', effects: ['out.emit'] } } }).kind, 'rejected');
  assert.equal(lam.return, 3);
  assert.notEqual(lam.return, MISSING);
});

test('native tool schemas narrow to typed slots and expand as workspace values appear', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ item: Num }, { label: "yes" | "no", count: Num }>',
    instructions: 'Classify and count.', args: { item: 2 }, codebase: {
      count: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item;' },
    } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const agent = new NativeToolAgent(() => ({ calls: [] }), { toolSchema: 'tools-v3' });
  const tools = agent.tools(session);
  const write = tools.find(item => item.function.name === 'write').function.parameters;
  assert.ok(write['x-natlang-alternatives'].some(alt => alt.path?.const === 'return/label'));
  assert.ok(write['x-natlang-alternatives'].some(alt => alt.path?.const === 'return/count'));
  const read = tools.find(item => item.function.name === 'read').function.parameters;
  assert.ok(read.properties.path.enum.includes('args/item'));
  assert.ok(!read.properties.path.enum.includes('return/label'));
  assert.equal(session.apply('write', { path: 'return/label', type: '"yes" | "no"', value: 'yes' }).kind, 'ok');
  const readAfter = agent.tools(session).find(item => item.function.name === 'read').function.parameters;
  assert.ok(readAfter.properties.path.enum.includes('return/label'));
});

test('native tool alternatives bind paths to their declared types and readable sources', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ number: Num, words: Text[] }, { count: Num, label: Text }>',
    instructions: 'Copy the number and label the words.', args: { number: 3, words: ['a', 'b'] },
    codebase: { identity: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const definitions = new NativeToolAgent(() => ({ calls: [] }), { toolSchema: 'tools-v3' }).tools(session);
  const parameters = name => definitions.find(item => item.function.name === name).function.parameters;
  const writes = parameters('write')['x-natlang-alternatives'];
  assert.ok(writes.some(alt => alt.path?.const === 'return/count' && alt.type?.const === 'Num' &&
    alt.value?.type === 'number'));
  assert.ok(writes.some(alt => alt.path?.const === 'return/count' && alt.source?.enum.includes('args/number')));
  assert.ok(!writes.some(alt => alt.path?.const === 'return/label' && alt.source?.enum.includes('args/number')));
  assert.ok(parameters('read')['x-natlang-alternatives'].some(alt => alt.path?.const === 'args/words' &&
    alt.start?.enum.includes(0) && alt.end?.enum.includes(1)));
  assert.ok(parameters('call')['x-natlang-alternatives'].some(alt => alt.function?.const === 'identity' &&
    alt.inputs?.properties?.item?.description === 'workspace path to Num'));
  assert.ok(parameters('call')['x-natlang-alternatives'].some(alt => alt.function?.const === 'identity' &&
    alt.values?.properties?.item?.type === 'number'));
});

test('tools-v4 presents split operations and positional path-only function inputs', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ text: Text, old: Text, replacement: Text, nums: Num[] }, Text>',
    instructions: 'Replace text and total the numbers.', args: { text: 'alpha beta', old: 'beta', replacement: 'gamma', nums: [2, 3] },
    codebase: {
      replace: { args: { text: 'Text', old: 'Text', replacement: 'Text' }, returns: 'Text',
        code: 'return args.text.replace(args.old, args.replacement);' },
      add: { args: { total: 'Num', number: 'Num' }, returns: 'Num', code: 'return args.total + args.number;' },
    } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const tools = new NativeToolAgent(() => ({ calls: [] }), { toolSchema: 'tools-v4' }).tools(session);
  const names = tools.map(item => item.function.name);
  assert.ok(names.includes('write_value')); assert.ok(names.includes('copy_value'));
  assert.ok(names.includes('run_function')); assert.ok(names.includes('for_each')); assert.ok(names.includes('fold'));
  assert.ok(names.includes('mark_lines')); assert.ok(!names.includes('write')); assert.ok(!names.includes('call'));
  const run = tools.find(item => item.function.name === 'run_function').function.parameters;
  const replace = run['x-natlang-alternatives'].find(alt => alt.function.const === 'replace');
  assert.deepEqual(replace['x-natlang-parameters'], ['text', 'old', 'replacement']);
  assert.equal(replace.inputs.type, 'array'); assert.equal(replace.inputs.minItems, 3);
  assert.equal(replace.inputs.maxItems, 3); assert.equal(replace.inputs.prefixItems.length, 3);
  const fold = tools.find(item => item.function.name === 'fold').function.parameters;
  const add = fold['x-natlang-alternatives'].find(alt => alt.function.const === 'add');
  assert.deepEqual(add['x-natlang-parameters'], ['total', 'number']);
  assert.equal(add.inputs, undefined); assert.ok(add.items); assert.ok(add.initial);
});

test('native calls accept typed literal values and reject double binding', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ item: Num }, Num>',
    instructions: 'Return a selected value.', args: { item: 2 }, codebase: {
      add: { args: { left: 'Num', right: 'Num' }, returns: 'Num', code: 'return args.left + args.right;' },
    } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('call', { function: 'add', to: 'return',
    inputs: { left: 'args/item' }, values: { right: 5 } });
  assert.equal(result.kind, 'done');
  assert.equal(lam.return, 7);
  const overlap = await session.applyAsync('call', { function: 'add', to: 'let/nope',
    inputs: { left: 'args/item' }, values: { left: 3, right: 4 } });
  assert.equal(overlap.kind, 'rejected');
  assert.ok(overlap.codes.includes('bad-call'));
});

test('native opening state distinguishes absent inputs, empty text, partial records, and pending tasks', () => {
  const agent = new NativeToolAgent(() => ({ calls: [] }));
  const lam = buildPending({ $lambda: { type: 'Lambda<{ text?: Text }, { label: Text, count: Num }>',
    instructions: 'Summarize.', args: {} } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.match(agent.opening(session), /args\/text \(Text, read-only\): not supplied/);
  assert.equal(agent.missing(session), '`return` has not been written yet. Write a { label: Text, count: Num } to `return`.');
  lam.args.text = '';
  assert.match(agent.opening(session), /args\/text \(Text, read-only\): ""/);
  session.apply('write', { path: 'return/label', type: 'Text', value: 'ok' });
  assert.match(agent.opening(session), /return \(\{ label: Text, count: Num \}\): partly written/);
  assert.match(agent.opening(session), /still missing: return\/count \(Num\)/);
  assert.match(agent.missing(session), /`return` is missing: return\/count \(Num\)\./);
});

test('complete native state snapshot resumes only unfinished Map slots', async () => {
  const program = { $map: { type: 'Map<Text, Text>', over: ['a', 'bad', 'c'],
    fn: { $lambda: { type: 'Lambda<{ item: Text }, Text>', instructions: 'Echo item.' } } } };
  const firstRuntime = new NativeRuntime({ agent: session => {
    if (session.lam.args.item === 'bad') return 'unreadable item';
    session.apply('write', { path: 'return', value: session.lam.args.item });
    session.finish();
  } });
  const first = await firstRuntime.runRoot(program);
  assert.equal(first.outcome.kind, 'quiesced');
  const snapshot = dumpState(first.value);
  assert.equal(snapshot.$map.slots.length, 3);
  const resumed = new NativeRuntime({ agent: session => {
    session.apply('write', { path: 'return', value: session.lam.args.item === 'bad' ? 'b' : session.lam.args.item });
    session.finish();
  } });
  const result = await resumed.runRoot(snapshot);
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(result.value, ['a', 'b', 'c']);
  assert.equal(resumed.episodesStarted, 1);
});

test('crisp code may replace itself with an unreduced typed task', async () => {
  const result = await new NativeRuntime().runRoot(crisp('Lambda<{}, Num>',
    'return lambda({ type: "Lambda<{}, Num>", instructions: "Compute seven." });'));
  assert.equal(result.outcome.kind, 'replaced');
  assert.equal(result.value.nodeKind, 'lambda');
  assert.equal(result.value.status, 'unreduced');
});

test('model write cannot invent an unchecked anonymous task', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return one.' } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = session.apply('write', { path: 'return', type: 'Num', value: {
    $lambda: { type: 'Lambda<{}, Num>', instructions: 'Guess one.' },
  } });
  assert.deepEqual(result.codes, ['anonymous-lambda']);
  assert.equal(lam.return, MISSING);
});

test('review withdrawal prevents a proposed write and permits one corrected retry', async () => {
  let proposed = 0, reviewed = 0;
  const agent = new NativeToolAgent(() => {
    proposed++;
    if (proposed === 1) return { calls: [['write', { path: 'return', type: 'Num', value: 99 }]],
      value_confidence: [0.1], completion_tokens: 1 };
    if (proposed === 2) return { calls: [['write', { path: 'return', type: 'Num', value: 7 }]],
      value_confidence: [0.9], completion_tokens: 1 };
    return { calls: [], text: 'done', completion_tokens: 1 };
  }, { review: { threshold: 0.5, withdrawalPolicy: 'retry', driver: () => {
    reviewed++;
    return { calls: [['review_write', { reason: 'The proposed value ignores the instruction.', decision: 'withdraw' }]],
      completion_tokens: 1 };
  } } });
  const runtime = new NativeRuntime({ agent: session => agent.run(session) });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Return seven.' } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 7);
  assert.equal(reviewed, 1);
  assert.ok(runtime.trace.events.some(event => event.kind === 'proposal' && event.phase === 'withdrawn'));
  assert.equal(runtime.trace.events.filter(event => event.kind === 'action' && event.name === 'write').length, 1);
});

test('native scenario admission checks exact calls and ordered effects without replay', async () => {
  const runtime = new NativeRuntime({ agent: async session => {
    await session.applyAsync('call', { function: 'double', to: 'return', inputs: { item: 'args/item' } });
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{ item: Num }, Num>',
    instructions: 'Call double.', args: { item: 2 }, codebase: {
      double: { args: { item: 'Num' }, returns: 'Num', code: 'return args.item * 2;' },
    } } });
  assert.equal(result.value, 4);
  const contract = { outcome: 'done', value: 4, constrainedCalls: [
    { function: 'double', to: 'return', inputs: { item: 'args/item' } }], requiredActions: [
    { name: 'call', arguments: { function: 'double', to: 'return' } }] };
  assert.equal(admitNativeTrace(runtime.trace, contract).admitted, true);
  assert.throws(() => admitNativeTrace(runtime.trace, { ...contract, constrainedCalls: [
    { function: 'double', to: 'return/size', inputs: { item: 'args/item' } }] }), /destination or inputs/);
  const effects = new NativeRuntime();
  const effected = await effects.runRoot({ $lambda: { type: 'Lambda<{}, Num>', code: 'fx.out.emit({ x: 1 }); return 3;',
    effects: ['out.emit'] } });
  assert.equal(effected.value, 3);
  assert.equal(admitNativeTrace(effects.trace, { outcome: 'done', value: 3,
    effects: [['out.emit', [{ x: 1 }]]] }).admitted, true);
});

test('native run_code can await a declared asynchronous capability', async () => {
  const runtime = new NativeRuntime({ capabilities: { 'counter.add': async ([n]) => n + 4 },
    agent: async session => {
      const result = await session.applyAsync('run_code', { engine: 'typescript-host', code: 'await fx.counter.add(3)' });
      assert.equal(result.kind, 'ok'); assert.equal(result.value, 7);
      session.apply('write', { path: 'return', value: result.value });
      session.finish();
    } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Compute seven.',
    effects: ['counter.add'] } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 7);
});

test('bounded independent Map work overlaps while preserving slot order and shared budget', async () => {
  let active = 0, peak = 0;
  const runtime = new NativeRuntime({ mapWorkers: 2, parallelModelSafe: true, maxEpisodes: 4,
    agent: async session => {
      active++; peak = Math.max(peak, active);
      await new Promise(resolve => setTimeout(resolve, 8));
      session.apply('write', { path: 'return', value: session.lam.args.item * 2 });
      session.finish(); active--;
    } });
  const result = await runtime.runRoot({ $map: { type: 'Map<Num, Num>', over: [1, 2, 3, 4],
    fn: { $lambda: { type: 'Lambda<{ item: Num }, Num>', instructions: 'Double item.' } } } });
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(result.value, [2, 4, 6, 8]);
  assert.equal(runtime.episodesStarted, 4);
  assert.equal(peak, 2);
  assert.equal(runtime.trace.events.filter(event => event.kind === 'map_slot').length, 4);
});

test('native en-passant done marks validate before work and follow successful writes', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'First step.\nSecond step.',
    codebase: { one: { returns: 'Num', code: 'return 1;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  assert.deepEqual(session.apply('write', { path: 'return', value: 7, done: [1, 9] }).codes, ['bad-range']);
  assert.equal(lam.return, MISSING);
  assert.equal(session.apply('write', { path: 'return', value: 'bad', done: 1 }).kind, 'rejected');
  assert.deepEqual(lam.marks, {});
  assert.equal(session.apply('write', { path: 'return', value: 7, done: 1 }).kind, 'ok');
  assert.equal(lam.marks[1], 'done');
  const tools = new NativeToolAgent(() => ({ calls: [] }), { toolSchema: 'tools-v3' }).tools(session);
  const mark = tools.find(item => item.function.name === 'mark_done');
  assert.deepEqual(mark.function.parameters['x-natlang-alternatives'][0].start.enum, [2]);
  assert.ok(tools.find(item => item.function.name === 'write').function.parameters.properties.done);
});

test('native retry reopens a completed natural-language child with its draft value', async () => {
  const runtime = new NativeRuntime({ agent: async session => {
    if (session.lam.functionName === 'draft') {
      session.apply('write', { path: 'return', value: session.lam.body.includes('short') ? 'short' : 'too long' });
      session.finish(); return;
    }
    assert.equal((await session.applyAsync('call', { function: 'draft', to: 'let/headline' })).kind, 'done');
    assert.equal(session.apply('retry', { path: 'let/headline', feedback: 'Write a short headline.' }).kind, 'ok');
    assert.equal((await session.applyAsync('run', { paths: 'let/headline' })).kind, 'done');
    session.apply('write', { path: 'return', source: 'let/headline' });
    session.finish();
  } });
  const result = await runtime.runRoot({ $lambda: { type: 'Lambda<{}, Text>', instructions: 'Draft then refine.',
    codebase: { draft: { returns: 'Text', instructions: 'Write a headline.' } } } });
  assert.equal(result.outcome.kind, 'done'); assert.equal(result.value, 'short');
});

test('failed checked call does not leave an empty local declaration', async () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{}, Num>', instructions: 'Do the work.',
    codebase: { emit: { returns: 'Num', effects: ['out.emit'], code: 'fx.out.emit(1); return 1;' } } } });
  const session = new NativeSession(new NativeRuntime(), lam, new TypeEnv());
  const result = await session.applyAsync('call', { function: 'emit', to: 'let/draft' });
  assert.deepEqual(result.codes, ['effect-wider-than-parent']);
  assert.equal(Object.hasOwn(lam.letTypes, 'draft'), false);
});
