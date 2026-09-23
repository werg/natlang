import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { scriptedModel } from './support/natlang.mjs';

/** The browser bundle, loaded with Node's `process` hidden so any Node dependency fails. */
async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

test('the browser bundle imports no Node built-ins', () => {
  const bundle = readFileSync(new URL('../dist/browser/natlang.js', import.meta.url), 'utf8');
  assert.doesNotMatch(bundle, /\bfrom\s*["']node:|require\(["']node:|import\(["']node:/);
});

test('browser tasks keep their own context across awaits in compiled code', async () => {
  const browser = await api();
  const model = scriptedModel(opening => `return "${/Label (\w+)/.exec(opening)?.[1] ?? '?'}".toUpperCase() + " " + label`);
  const project = browser.compileVirtualProject({ files: { 'main.ts': `import { nl } from '@natlang/browser';
export async function main(label: string, delay: number): Promise<string> {
  await new Promise(resolve => setTimeout(resolve, delay));
  const first: string = await nl\`Label \${label} for the input.\`(label);
  await new Promise(resolve => setTimeout(resolve, delay));
  return first;
}
` } }, browser);
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const runtime = browser.createNatlangRuntime({ model: model.driver });
  const tasks = { alpha: [], beta: [] };
  const [a, b] = await Promise.all(['alpha', 'beta'].map((label, index) =>
    runtime.run(() => project.require('main.ts').main(label, 10 - index * 5), { name: label, trace: trace => tasks[label].push(trace.taskId) })));
  assert.equal(a, 'ALPHA alpha'); assert.equal(b, 'BETA beta');
  assert.ok(tasks.alpha.length === 1 && tasks.alpha[0].startsWith('alpha-'));
  assert.ok(tasks.beta.length === 1 && tasks.beta[0].startsWith('beta-'));
});

test('a browser model sees a failed eval and repairs the function', async () => {
  const browser = await api();
  const script = [['eval', { code: 'return value.missing.deep' }], ['eval', { code: 'value' }],
    ['eval', { code: 'return value + 1' }]];
  let turns = 0;
  const runtime = browser.createNatlangRuntime({ model: { driver: request => {
    if (turns === 1) assert.match(request.messages.at(-1).content, /Nothing else from this eval was kept/);
    return turns < script.length ? { calls: [script[turns++]], completion_tokens: 1 } : (turns++, { text: 'done', completion_tokens: 1 });
  } } });
  const project = browser.newPlaygroundProject('Next', 'next.nl', { 'next.nl': '---\nargs:\n  value: number\nreturns: number\n---\nReturn the next number.\n' }, { value: 4 }, 5);
  const run = await browser.runPlaygroundProject(runtime, project);
  assert.equal(run.outcome.kind, 'done', run.outcome.detail); assert.equal(run.value, 5); assert.equal(turns, 4);
});

test('browser named functions use callable folders from virtual files, with types and listings', async () => {
  const browser = await api();
  const model = scriptedModel(opening => {
    assert.match(opening, /declare function is_short\(text: string\): boolean; {2}\/\/ TypeScript/);
    assert.match(opening, /type Summary = \{ text: string, short: boolean \}/);
    return 'return { text: text.trim(), short: is_short(text) }';
  });
  const project = browser.newPlaygroundProject('Summary', 'summarize.nl', {
    'types.ts': 'export type Summary = { text: string, short: boolean };\n',
    'summarize.nl': '---\nargs:\n  text: string\nreturns: Summary\n---\nSummarize text.\n',
    'summarize/is_short.ts': 'export default function is_short(text: string): boolean { return text.length < 20; }\n',
  }, { text: ' brief ' }, { text: 'brief', short: true });
  const run = await browser.runPlaygroundProject(browser.createNatlangRuntime({ model: model.driver }), project);
  assert.equal(run.correct, true, run.outcome.detail);
});

test('playground validates edits, pins sources, navigates traces, and admits captured outcomes', async () => {
  const browser = await api();
  assert.equal(browser.validProjectPath('../escape.ts'), false);
  const project = browser.newPlaygroundProject('Add', 'add.nl', { 'add.nl': '---\nargs:\n  a: number\nreturns: number\n---\nAdd two to a.\n' }, { a: 5 }, 7);
  assert.deepEqual(browser.validatePlaygroundProject(project), []);
  const invalid = browser.editPlaygroundProject(project, { files: { 'add.nl': 'no frontmatter' } });
  assert.notEqual(invalid.revision, project.revision);
  assert.match(browser.validatePlaygroundProject(invalid)[0].message, /frontmatter/);
  const run = await browser.runPlaygroundProject(browser.createNatlangRuntime({ model: scriptedModel(() => 'return a + 2').driver }), project);
  assert.equal(run.value, 7); assert.equal(run.correct, true);
  const first = browser.traceFrame(run.trace, 0), final = browser.traceFrame(run.trace, run.trace.length - 1);
  assert.equal(first.event.kind, 'manifest'); assert.equal(final.state.phase, 'final');
  assert.equal(browser.admitPlaygroundRun(run, { outcome: 'done', value: 7, effects: [] }).admitted, true);
  assert.throws(() => browser.admitPlaygroundRun(run, { outcome: 'done', value: 8 }), /does not match/);
  const tsProject = browser.newPlaygroundProject('Main', 'main.ts', { 'main.ts':
    "import { nl } from '@natlang/browser';\nexport async function main(input: { a: number }): Promise<number> { return await nl<number>`Double a.`(input.a); }\n" }, { a: 4 }, 8);
  const tsRun = await browser.runPlaygroundProject(browser.createNatlangRuntime({ model: scriptedModel(() => 'return input * 2').driver }), tsProject,
    { runtimeNamespace: browser });
  assert.equal(tsRun.correct, true, tsRun.outcome.detail);
});

test('the DOM renderer uses text nodes, emits typed events, and rejects executable markup', async () => {
  const { BrowserDomRenderer } = await api();
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.listeners = new Map(); }
    appendChild(child) { this.children.push(child); }
    replaceChildren(...children) { this.children = children; }
    addEventListener(name, listener) { this.listeners.set(name, listener); }
    setAttribute(name, value) { this[name] = value; }
    emit(name) { this.listeners.get(name)?.(); }
  }
  const prior = globalThis.document;
  globalThis.document = { createElement: tag => new Element(tag) };
  const root = new Element('root'), events = [];
  const renderer = new BrowserDomRenderer(root, event => { events.push(event); });
  try {
    renderer.render({ tag: 'main', children: [{ tag: 'p', text: '<script>alert(1)</script>' },
      { tag: 'input', id: 'entry', label: 'Entry', value: '' }, { tag: 'button', text: 'Add', action: { kind: 'command', from: 'entry' } }] });
    const [paragraph, input, button] = root.children[0].children;
    assert.equal(paragraph.textContent, '<script>alert(1)</script>');
    input.value = 'Buy milk'; button.emit('click');
    assert.deepEqual(events.map(event => [event.kind, event.value]), [['command', 'Buy milk']]);
    assert.throws(() => renderer.render({ tag: 'script', text: 'alert(1)' }), /unsupported/);
    assert.throws(() => renderer.render({ tag: 'button', action: { kind: 'command', from: 'missing' } }), /unknown input/);
  } finally { renderer.close(); globalThis.document = prior; }
});

test('a browser event loop drives a view from natural-language reductions', async () => {
  const browser = await api();
  const project = browser.compileVirtualProject({ files: { 'app.ts': `import { nl } from '@natlang/browser';
export type Todo = { items: string[] };
export async function reduce(state: Todo, event: { id: string, kind: string, value?: string }): Promise<Todo> {
  if (event.kind !== 'add' || !event.value) return state;
  const item: string = await nl\`Normalize the todo item in event.\`(event);
  return { items: [...state.items, item] };
}
export function view(state: Todo): { tag: string, text: string } { return { tag: 'p', text: state.items.join(', ') }; }
` } }, browser);
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const app = project.require('app.ts');
  const runtime = browser.createNatlangRuntime({ model: scriptedModel(() => 'return event.value.trim().toLowerCase()').driver });
  const loop = new browser.EventLoop({ initialState: { items: [] }, reduce: app.reduce, view: app.view,
    step: fn => runtime.run(fn) });
  await loop.start();
  await loop.dispatch({ id: 'e1', kind: 'add', value: '  Milk ' });
  await loop.dispatch({ id: 'e2', kind: 'add', value: 'Eggs' });
  assert.deepEqual(loop.view, { tag: 'p', text: 'milk, eggs' });
});
