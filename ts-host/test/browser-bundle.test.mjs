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
      type: '() => number', code: 'host.count += 3; return host.count;' } } } });
    assert.equal(crisp.outcome.kind, 'done');
    assert.equal(crisp.value, 5);
    assert.equal(shared.count, 5);
    assert.equal(crisp.trace[0].engine_contracts['typescript-host'].authority, 'shared-browser-host');
    let turns = 0;
    const natural = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: '() => number', instructions: 'Return the current count.' } } },
    modelTurn: () => ++turns === 1 ? { calls: [['eval', { code: 'host.count' }]],
      completion_tokens: 1 } : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 } });
    assert.equal(natural.outcome.kind, 'done');
    assert.equal(natural.value, 5);
    assert(natural.trace.some(event => event.kind === 'action'));
  } finally { host.close(); }
});

test('browser model can probe a failed eval snapshot and repair the function', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const host = new api.BrowserNatlangHost();
  const script = [
    ['eval', { code: 'result = value.missing.deep' }],
    ['eval', { code: 'debug.kind' }],
    ['eval', { code: 'result = value + 1' }],
    ['mark_lines', { start: 1 }],
  ];
  let turns = 0;
  try {
    const result = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: '(value: number) => number', instructions: 'Return the next number.', args: { value: 4 } } } },
    validationFeedback: 'caller', modelTurn: request => {
      if (turns === 1) assert.match(request.messages[0].content, /immutable debug/);
      return { calls: [script[turns++]], completion_tokens: 1 };
    } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 5);
    assert.equal(turns, 4);
  } finally { host.close(); }
});

test('browser natural functions call companion TypeScript subfunctions without await', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const host = new api.BrowserNatlangHost();
  const files = {
    'main.nl': '---\nargs:\n  value: number\nreturns: number\n---\nReturn double(value).\n',
    'main/double.ts': 'export default function double(value: number): number { return value * 2; }',
  };
  let turns = 0;
  try {
    const result = await host.run({ source: { kind: 'files', root: 'main.nl', files },
      inputs: { value: 7 }, modelTurn: request => {
        if (!turns++) {
          assert.match(String(request.messages.find(message => message.role === 'user')?.content),
            /double \[TypeScript\] \(value: number\): number/);
          return { calls: [['eval', { code: 'const result = double(value);' }]], completion_tokens: 1 };
        }
        return { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 14);
  } finally { host.close(); }
});

test('browser eval can assign the listed function result directly', async () => {
  const api = await import('../dist/browser/natlang.js');
  const host = new api.BrowserNatlangHost();
  let turns = 0;
  try {
    const outcome = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: '(value: number) => number', instructions: 'Double the value.', args: { value: 4 },
    } } }, modelTurn: () => ++turns === 1
      ? { calls: [['eval', { code: 'result = value * 2;' }]], completion_tokens: 1 }
      : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 } });
    assert.equal(outcome.outcome.kind, 'done');
    assert.equal(outcome.value, 8);
  } finally { host.close(); }
});

test('the agent sees referenced type shapes and Promise-returning TypeScript signatures', async () => {
  const api = await import('../dist/browser/natlang.js');
  const host = new api.BrowserNatlangHost();
  const files = {
    'main.nl': '---\nargs:\n  value: number\nreturns: State\n---\nReturn the prepared state.\n',
    'types.ts': 'export type State = { count: number, label: string };',
    'main/prepare.ts': 'export default function prepare(value: number): Promise<State> { return Promise.resolve({ count: value, label: "ok" }); }',
  };
  let turns = 0;
  try {
    const result = await host.run({ source: { kind: 'files', root: 'main.nl', files },
      inputs: { value: 3 }, modelTurn: request => {
        if (!turns++) {
          const opening = String(request.messages.find(message => message.role === 'user')?.content);
          assert.match(opening, /prepare \[TypeScript\] \(value: number\): Promise<State>/);
          assert.match(opening, /type State = \{ count: number, label: string \};/);
          return { calls: [['eval', { code: 'const result = await prepare(value);' }]], completion_tokens: 1 };
        }
        return { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.deepEqual(result.value, { count: 3, label: 'ok' });
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
    'tasks/add.ts': 'export default function add(x: number): Result { label(x + 2); return x + 2; }',
    'tasks/types.ts': 'export type Result = number;',
    'tasks/add/label.ts': 'export default function label(result: number): string { return String(result); }',
  };
  const loaded = api.loadFunctionFiles('tasks/add.ts', files);
  assert.deepEqual(loaded.types.Result, { kind: 'prim', name: 'number' });
  assert(loaded.codebase.label);
  const host = new api.BrowserNatlangHost();
  try {
    const result = await host.run({ source: { kind: 'files', root: 'tasks/add.ts', files },
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
    'main.ts': 'export type File = { kind: "text", text: string, bytes: number } | { kind: "binary", bytes: number };\nexport default function main(files: Record<string, File>): string { return files["notes/context.md"].text; }',
    'notes/context.md': 'browser project context',
  };
  const host = new api.BrowserNatlangHost();
  try {
    const result = await host.run({ source: { kind: 'files', root: 'main.ts', files },
      inputs: { files: api.textFileTree(files) } });
    assert.equal(result.value, 'browser project context');
  } finally { host.close(); }
});

test('browser runs ordinary TypeScript with more than sixteen locals', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const declarations = Array.from({ length: 24 }, (_, i) => `const value_${i} = ${i};`).join('\n');
  const files = { 'main.ts': `export default function main(): number {\n${declarations}\nreturn value_23 + 1;\n}` };
  const loaded = api.loadFunctionFiles('main.ts', files);
  assert.equal(Object.keys(loaded.codebase).length, 0);
  const host = new api.BrowserNatlangHost();
  try {
    const result = await host.run({ source: { kind: 'files', root: 'main.ts', files } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 24);
  } finally { host.close(); }
});

test('browser source workspace invokes a checked child with browser eval', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const workspace = new api.NativeSourceWorkspace({ double: { args: { item: 'number' },
    returns: 'number', code: 'return item * 2;', engine: 'typescript-host' } }, 'double');
  const child = await workspace.invoke('double', { item: 4 });
  assert.equal(child.outcome, 'done');
  assert.equal(child.value, 8);
});
