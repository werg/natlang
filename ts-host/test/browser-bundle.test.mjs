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

test('browser natural functions call synchronous TypeScript imports without await', async () => {
  const nodeProcess = globalThis.process;
  let api;
  try { globalThis.process = undefined; api = await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = nodeProcess; }
  const host = new api.BrowserNatlangHost();
  const files = {
    'main.nl': 'import double from "./double.ts";\n---\nargs:\n  value: number\nreturns: number\n---\nReturn double(value).\n',
    'double.ts': 'export default function double(value: number): number { return value * 2; }',
  };
  let turns = 0;
  try {
    const result = await host.run({ source: { kind: 'files', root: 'main.nl', files },
      inputs: { value: 7 }, modelTurn: request => {
        if (!turns++) {
          assert.match(String(request.messages.find(message => message.role === 'user')?.content),
            /double \[TypeScript\] \(value: number\): number/);
          return { calls: [['eval', { code: 'double(value)' }]], completion_tokens: 1 };
        }
        return { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 };
      } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 14);
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
    'tasks/add.ts': 'import type { Result } from "../types";\nimport label from "./add/label";\nexport default function add(x: number): Result { label(x + 2); return x + 2; }',
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
