import test from 'node:test';
import assert from 'node:assert/strict';

test('browser bundle runs typed crisp and model programs without Node builtins', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try {
    globalThis.process = undefined;
    api = await import('../dist/browser/natlang.js');
  } finally { globalThis.process = nodeProcess; }
  const shared = { count: 2 };
  const host = new api.BrowserNatlangHost({ host: shared });
  try {
    const crisp = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', code: 'host.count += 3; return host.count;' } } } });
    assert.equal(crisp.outcome.kind, 'done');
    assert.equal(crisp.value, 5);
    assert.equal(shared.count, 5);
    assert.equal(crisp.trace[0].engine_contracts['typescript-host'].authority, 'shared-browser-host');
    let turns = 0;
    const natural = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', instructions: 'Return the current count.' } } },
    modelTurn: () => ++turns === 1 ? { calls: [['eval', { code: 'host.count' }]],
      completion_tokens: 1 } : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 } });
    assert.equal(natural.outcome.kind, 'done');
    assert.equal(natural.value, 5);
    assert(natural.trace.some(event => event.kind === 'action'));
  } finally { host.close(); }
});

test('browser retained eval preserves variables between tool calls', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const environment = new api.TypeScriptEnvironment({ mode: 'retained' });
  try {
    assert.equal(environment.execute({ code: 'var tally = 2; tally', scope: {}, body: false,
      path: 'eval', effectful: false }).result, 2);
    assert.equal(environment.execute({ code: 'tally += 3', scope: {}, body: false,
      path: 'eval', effectful: false }).result, 5);
  } finally { environment.close(); }
});

test('browser loads file source, lexical types, companions, and inputs from virtual files', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const files = {
    'tasks/add.ts': '/*---\nargs:\n  x: Num\nreturns: Result\n---*/\nreturn args.x + 2;',
    'tasks/types.ts': 'type Result = Num;',
    'tasks/add/label.nl': '---\nreturns: Text\n---\nDescribe the result.',
  };
  const loaded = api.loadFunctionFiles('tasks/add', files);
  assert.deepEqual(loaded.types.Result, { kind: 'prim', name: 'Num' });
  assert(loaded.codebase.label);
  const host = new api.BrowserNatlangHost();
  try {
    const result = await host.run({ source: { kind: 'files', root: 'tasks/add', files },
      inputs: { x: 5 } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 7);
  } finally { host.close(); }
});

test('browser virtual projects expose non-source files through the shared file-tree contract', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const files = {
    'main.nl': '---\nargs:\n  files: Dict<File>\ntypes:\n  File: \'{ kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num }\'\nreturns: Text\n---\nRead the project note and return it.',
    'notes/context.md': 'browser project context',
  };
  let turn = 0, observed = '';
  const host = new api.BrowserNatlangHost();
  try {
    const result = await host.run({ source: { kind: 'files', root: 'main.nl', files },
      inputs: { files: api.textFileTree(files) }, modelTurn: request => {
      observed = JSON.stringify(request);
      turn++;
      if (turn === 1) return { calls: [['eval', { code: 'files["notes/context.md"].text' }]] };
      if (turn === 2) return { calls: [['mark_lines', { start: 1 }]] };
      return { calls: [] };
    } });
    assert.equal(result.value, 'browser project context');
    assert.match(observed, /browser project context/);
  } finally { host.close(); }
});

test('browser accepts a large lexical codebase and more than sixteen locals', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const files = { 'main.nl': '---\nreturns: Num\n---\nBuild many intermediate values.' };
  for (let i = 0; i < 24; i++) files[`main/helper_${i}.nl`] = '---\nreturns: Num\n---\nReturn one.';
  const loaded = api.loadFunctionFiles('main.nl', files);
  assert.equal(Object.keys(loaded.codebase).length, 24);
  const host = new api.BrowserNatlangHost();
  let turns = 0;
  try {
    const result = await host.run({ source: { kind: 'files', root: 'main.nl', files },
      modelTurn: () => ++turns === 1 ? { calls: [
        ...Array.from({ length: 24 }, (_, i) => ['write', { path: `let/value_${i}`, type: 'Num', value: i }]),
        ['write', { path: 'return', type: 'Num', value: 24 }],
      ] } : { calls: [], text: 'done' } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 24);
  } finally { host.close(); }
});

test('browser source workspace invokes a checked child with browser eval', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const workspace = new api.NativeSourceWorkspace({ double: { args: { item: 'Num' },
    returns: 'Num', code: 'return args.item * 2;', engine: 'typescript-host' } }, 'double');
  const child = await workspace.invoke('double', { item: 4 });
  assert.equal(child.outcome, 'done');
  assert.equal(child.value, 8);
});
