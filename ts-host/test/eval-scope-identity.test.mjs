// Language and runtime gaps the authored-program wave 1 hit (2026-11-25): object identity across evals, saved helpers
// passed as arguments, recursive eval-local types, interfaces in types.ts, reserved argument names, and the recursion
// error's graph hint.
import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import { workerdPlatform } from '../dist/core/workerd.js';
import { compileScopeSnippet, evalDeclaredNames } from '../dist/scope-compiler.js';
import { readTypeAliases } from '../dist/native/type-aliases.js';

const platforms = {
  node: () => core.nodePlatform(),
  workerd: () => workerdPlatform({ realm: { call(body, params, args) { return Function(...params, body)(...args); } } }),
};

/** Runs `owner.nl` (with `files` beside it) whose session runs `evals` in order; children answer with `answer(session)`. */
async function runEvals(platformName, evals, { args = '  items: { what: string }[]', input = '[{what:"a"},{what:"b"},{what:"c"}]',
  files = {}, answer } = {}) {
  const platform = platforms[platformName]();
  const results = [];
  const rt = core.createNatlangRuntime({ platform, calls: false, agent: async session => {
    if (session.lam.functionName !== 'owner') return answer?.(session);
    for (const [index, code] of evals.entries())
      results.push(await session.applyAsync('eval', { code, finish: index === evals.length - 1 }));
  } });
  try {
    const project = core.compileVirtualProject({ root: '/gaps', files: {
      'entry.ts': `import owner from "./owner.nl"; export default async function entry():Promise<string>{return await owner(${input});}`,
      'owner.nl': `---\nargs:\n${args}\nreturns: string\n---\nDo it.`, ...files } }, core, { platform });
    assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
    const value = await rt.run(() => project.require('entry.ts').default());
    return { value, results };
  } finally { rt.close(); }
}

for (const platformName of Object.keys(platforms)) {
  test(`${platformName}: an object two locals hold, or a local and an input, stays one object in later evals`, async () => {
    const { value, results } = await runEvals(platformName, [
      'const events = items.map(i => ({ what: i.what }));',
      'const needs = events.filter(e => e.what !== "b"); const first = items[0]; const pair = { right: items[1] };',
      'return JSON.stringify([needs.indexOf(events[0]), needs.indexOf(events[2]), first === items[0], pair.right === items[1], ' +
        'items.filter(i => i !== pair.right).length]);',
    ]);
    for (const result of results) assert.notEqual(result.kind, 'error', result.text);
    assert.equal(value, '[0,1,true,true,2]');
  });

  test(`${platformName}: a local's mutation shows through every name that holds the object, and inputs stay read-only`, async () => {
    const { value } = await runEvals(platformName, [
      'const a = [{ n: 1 }]; const b = a;',
      'a.push({ n: 2 }); a[0].n = 5;',
      'const part = items[0]; try { (part as any).what = "x"; } catch { /* strict-mode realms throw */ }\n' +
        'return JSON.stringify([b.length, b[0].n, a === b, Object.isFrozen(part), items[0].what]);',
    ]);
    assert.equal(value, '[2,5,true,true,"a"]');
  });

  test(`${platformName}: a saved helper passed as an argument in a later eval stays bound`, async () => {
    const { value, results } = await runEvals(platformName, [
      'const graders = items.map(i => (o: string) => o + i.what);\nconst grade = (w: string, o: string) => graders[items.findIndex(c => c.what === w)](o);',
      'const apply = (f: (w: string, o: string) => string, w: string) => f(w, "x");\nconst step = (n: number) => apply(grade, items[n].what);',
      'return [step(1), [1, 2].map(n => apply(grade, items[n].what)).join("")].join(":");',
    ]);
    for (const result of results) assert.notEqual(result.kind, 'rejected', result.text);
    assert.equal(value, 'xb:xbxc');
  });
}

test('NATLANG_EVAL_IDENTITY=0 copies each value on its own, as before', async () => {
  const before = process.env.NATLANG_EVAL_IDENTITY;
  process.env.NATLANG_EVAL_IDENTITY = '0';
  try {
    const { value } = await runEvals('node', ['const first = items[0]; const events = items.map(i => i);',
      'return JSON.stringify([first === items[0], events.indexOf(first)]);']);
    assert.equal(value, '[false,-1]');
  } finally { if (before === undefined) delete process.env.NATLANG_EVAL_IDENTITY; else process.env.NATLANG_EVAL_IDENTITY = before; }
});

test('evalDeclaredNames reads declarations, not names used as arguments or shorthand properties', () => {
  const names = evalDeclaredNames('const { a, b: [c] } = x; let d = 1; function e() {} class F {}\n' +
    'const g = h(1, grade, semanticMap); const o = { step, grade };\nimport { semanticFilter } from "@natlang/core/std";');
  assert.deepEqual([...names].sort(), ['F', 'a', 'c', 'd', 'e', 'g', 'o']);
});

test('a saved helper annotated with Untrusted<string> or Is<...> persists', () => {
  const compiled = compileScopeSnippet('const shout = (text: Untrusted<string>): Is<string, "loud"> => text.toUpperCase() as Is<string, "loud">;\nshout');
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  assert.deepEqual(compiled.persistentHelpers?.map(helper => helper.name), ['shout']);
});

test('a recursive type an eval declares types its locals and an nl literal result', async () => {
  const { value, results } = await runEvals('node', [
    'type Node = { id: string; kids: Node[] };\nconst local: Node = { id: "x", kids: [{ id: "y", kids: [] }] };\n' +
      'const built = await nl<Node>`Build the node.`();',
    'const again: Node = { id: "z", kids: [local, built] };\nreturn JSON.stringify(again);',
  ], { answer: session => session.applyAsync('eval', { code: 'return { id: "n", kids: [{ id: "m", kids: [] }] };', finish: true }) });
  for (const result of results) assert.notEqual(result.kind, 'error', result.text);
  assert.equal(JSON.parse(value).kids[1].kids[0].id, 'm');
});

test('types.ts interfaces of properties are record types, recursive and extending ones included', async () => {
  assert.deepEqual(readTypeAliases('/** A prerequisite tree. */\nexport interface Tree {\n  /** The story. */\n  storyId: string;\n  needs: Tree[];\n}\n' +
    'interface Named { name: string }\ninterface Person extends Named { age?: number }\ninterface Live { run(): void }\ninterface Box<T> { value: T }'), {
    Tree: '/** A prerequisite tree. */ {\n  /** The story. */\n  storyId: string;\n  needs: Tree[];\n}',
    Named: '{ name: string }',
    Person: 'Named & { age?: number }',
  });
  const { value } = await runEvals('node', ['const t = await slice("S1");\nreturn t.needs[0].storyId;'], {
    args: '  items: string[]', input: '["S1"]',
    files: { 'types.ts': 'export interface Tree { storyId: string; needs: Tree[] }',
      'owner/slice.nl': '---\nargs:\n  storyId: string\nreturns: Tree\n---\nBuild the tree of storyId.' },
    answer: session => session.applyAsync('eval', { code: 'return { storyId: "S1", needs: [{ storyId: "S2", needs: [] }] };', finish: true }) });
  assert.equal(value, 'S2');
});

test('an argument or function named by a JavaScript reserved word is refused with the rename', () => {
  const compile = files => core.compileVirtualProject({ root: '/reserved', files }, core, { platform: core.nodePlatform() });
  const argument = compile({ 'judge.nl': '---\nargs:\n  case: string\nreturns: boolean\n---\nJudge case.' });
  assert.equal(argument.ok, false);
  assert.match(JSON.stringify(argument.diagnostics), /\\"case\\" is a reserved word in JavaScript, so it cannot be an argument name; rename it, for example to \\"caseItem\\"/);
  const named = compile({ 'default.nl': '---\nargs:\n  text: string\nreturns: string\n---\nEcho text.' });
  assert.equal(named.ok, false);
  assert.match(JSON.stringify(named.diagnostics), /reserved word in JavaScript, so it cannot be a callable name/);
});

test('eval helpers may call each other while each re-entry gets smaller; a growing path names the shrinking walk', async () => {
  const { value, results } = await runEvals('node', [
    'const f = (n: number): number => n > 0 ? g(n - 1) : 0;\nconst g = (n: number): number => f(n) + 1;\nconst mutual = f(3);\n' +
      'const ids = ["a", "b", "c"]; const next = (id: string) => ids[(ids.indexOf(id) + 1) % 3];',
    'const grow = (id: string, path: string[]): string[] => path.includes(id) ? path : grow(next(id), [...path, id]);\ngrow("a", [])',
    'const walk = (id: string, unvisited: string[]): string[] => unvisited.includes(id) ? ' +
      '[id, ...walk(next(id), unvisited.filter(x => x !== id))] : [];\nreturn `${mutual}:${walk("a", ids).join("")}`;',
  ]);
  assert.match(results[1].text, /to walk a graph, pass the nodes not yet visited and recurse on fewer of them/);
  assert.equal(value, '3:abc');
});
