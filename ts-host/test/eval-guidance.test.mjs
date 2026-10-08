import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { BUILT_INS_LINE, READ_CODE_DESCRIPTION } from '../dist/native/agent.js';
import { SEMANTIC_RESULT_PROMPT } from '../dist/native/prompt.js';

/** Run a call of greet(name) with scripted root calls; returns the opening, each tool result, and the value. */
async function script(calls, returns = 'string', runtimeOptions = {}) {
  const results = [];
  let opening = '', step = 0;
  const model = async request => {
    if (!opening) opening = String(request.messages[1].content); else results.push(String(request.messages.at(-1).content));
    const call = calls[step++];
    return call ? { calls: [call] } : { calls: [['return_result', { status: 'failed', reason: 'The scripted test has no more calls.' }]] };
  };
  const runtime = createNatlangRuntime({ ...runtimeOptions, model, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'root.nl': `---\nargs: { name: string }\nreturns: ${returns}\n---\nGreet name.\n` }, 'root.nl');
  let value;
  try { value = await runtime.run(() => fn('Ada')); } catch {}
  return { opening, results, value };
}

test('the opening names the built-ins and where their documentation is', async () => {
  const { opening } = await script([['return_result', { status: 'success', value: 'hi' }]]);
  assert.ok(opening.includes(BUILT_INS_LINE));
});

test('shared result guidance shows computed Neuralese text return and static marker bodies', () => {
  assert.match(SEMANTIC_RESULT_PROMPT, /At a direct parameter declared Neuralese<string>, an ordinary string argument is written through the configured writer/);
  assert.match(SEMANTIC_RESULT_PROMPT, /When this call's declared result is exactly Neuralese<string>/);
  assert.match(SEMANTIC_RESULT_PROMPT, /marker body is literal payload written before eval runs/);
  assert.match(SEMANTIC_RESULT_PROMPT, /for computed Neuralese<string> output, return the computed string/);
});

test('read_code shows the authentic built-in Neuralese type declaration', async () => {
  const { BUILT_IN_DOCS } = await import('../dist/native/runtime.js');
  const inspected = await script([['eval', { code: 'read_code("Neuralese")' }],
    ['return_result', { status: 'success', value: 'done' }]]);
  const declaration = JSON.parse(inspected.results[0]);
  assert.match(declaration, /Built-in compile-time type declaration/);
  assert.match(declaration, /type Neuralese<T, D extends string = DefaultDialect>/);
  assert.match(declaration, /interface NeuraleseValue<T, D extends string>/);
  assert.match(declaration, /no semantic fields to inspect/);
  assert.equal(declaration, BUILT_IN_DOCS.Neuralese);

  const prompt = (await import('../dist/model/text-neuralese-emulation.js')).TEXT_NEURALESE_EMULATION_PROMPT;
  assert.match(prompt, /Use read_code\("Neuralese"\) to inspect its declaration/);
  assert.doesNotMatch(prompt, /do not redefine it or use read_code\("Neuralese"\)/);
});

test('read_code documents the injected decide helper instead of asking for program source', async () => {
  const { BUILT_IN_DOCS } = await import('../dist/native/runtime.js');
  const inspected = await script([['eval', { code: 'read_code("decide")' }],
    ['return_result', { status: 'success', value: 'done' }]]);
  const docs = JSON.parse(inspected.results[0]);
  assert.equal(docs, BUILT_IN_DOCS.decide);
  assert.match(docs, /decide\(fn, \.\.\.args\)/);
  assert.match(docs, /first argument must be a callable function in scope/);
  assert.match(docs, /one probability-1 answer with scored=false/);
  assert.match(READ_CODE_DESCRIPTION, /eval built-ins \(nl, iterateOn, transcript, decide\)/);
});

test("return_result in eval with the tool's shape is read as the tool's request", async () => {
  const { results } = await script([['eval', { code: 'return_result({ status: "success", value: "Hello, " + name })' }]]);
  assert.match(results[0], /Staged "Hello, Ada" as the result/);
  const blocked = await script([['eval', { code: 'return_result({ status: "blocked", reason: "No greeting style is given." })' }]]);
  assert.deepEqual(blocked.results, [], 'blocked ends the call');
});

test('a value of the return type shaped like a request stays a value', async () => {
  const { results } = await script([['eval', { code: 'return_result({ status: "success", value: 1 })' }]],
    '"{ status: string, value: number }"');
  assert.match(results[0], /Staged \{ status: "success", value: 1 \} as the result/);
});

test('reaching for what the scope lacks is answered with what it has', async () => {
  const { results } = await script([['eval', { code: 'const fs = require("fs");\nfs' }], ['eval', { code: 'missingThing + 1' }]]);
  for (const text of results) assert.match(text, /This call's eval scope has name, the built-ins nl, iterateOn and transcript/);
});

test('scope feedback names declared captures separately from properties of the input record', async () => {
  const { session: open } = await import('./support/natlang.mjs');
  const { session } = open({ type: '(input: { context: string, evidence: string, current: string, constraint: string }) => string',
    instructions: 'Revise the note.' });
  session.lam.captures = Object.fromEntries(['context', 'evidence'].map(name => [name,
    { name, type: 'string', mutable: false, get: () => name }]));
  const guide = session.scopeGuide();
  assert.match(guide, /has input, context, evidence,/);
  assert.doesNotMatch(guide, /\bcurrent\b|\bconstraint\b/,
    'record fields are not standalone lexical names unless passed as arguments or captures');
});

test("redeclaring an input says it already holds the caller's value", async () => {
  const { results } = await script([['eval', { code: 'const name = "Test";\nname' }]]);
  assert.match(results[0], /name is this call's input and already holds the caller's value; use it directly/);
});

test('Node\'s own modules cannot be imported in eval; they are not packages', async () => {
  const { results } = await script([['eval', { code: "import * as fs from 'fs';\nfs.readdirSync('/')" }],
    ['eval', { code: "const cp = await import('child_process');\ncp" }]]);
  for (const text of results) assert.match(text, /is part of Node, not a package: eval code works with this call's scope/);
  assert.ok(results.every(text => !/\bbin\b/.test(text)), 'no /bin entry came back');
});

test('a local declared again takes its new type, and a local that held null takes what is assigned', async () => {
  const { results } = await script([['eval', { code: 'const m = "x".match(/y/);\nm' }],
    ['eval', { code: 'const m = "xy".match(/y/);\nm && m[0]' }],
    ['eval', { code: 'let n = null;\nn' }], ['eval', { code: 'n = 5;\nn' }]]);
  assert.ok(results.every(text => !/type-mismatch/.test(text)), results.join('\n---\n'));
});

test('read_code accepts its name or native-tool argument shape in eval and remains available as a tool', async () => {
  const { results } = await script([['eval', { code: 'read_code("nl")' }], ['read_page', { id: 'transcript', page: 1 }]]);
  assert.match(results[0], /nl: create a natural-language function inside eval code/);
  assert.match(results[1], /transcript is in eval's scope: search it in eval with transcript\.search/);

  const objectRead = await script([['eval', { code: 'read_code({ name: "nl" })' }]]);
  assert.match(objectRead.results[0], /nl: create a natural-language function inside eval code/);

  const malformed = await script([['eval', { code: 'read_code({ name: 7 })' }]]);
  assert.match(malformed.results[0], /read_code: type-mismatch, expected a name string or exactly \{ name: string \}, got object/);

  const toolRead = await script([['read_code', { name: 'nl' }], ['return_result', { status: 'success', value: 'done' }]]);
  assert.match(toolRead.results[0], /nl: create a natural-language function inside eval code/);
});

test('read_code discovers only currently offered native tool schemas and visible host service types', async () => {
  assert.match(READ_CODE_DESCRIPTION, /exact schema and instructions of a native tool currently offered/);
  assert.match(READ_CODE_DESCRIPTION, /metadata, not program function source or service implementation/);
  assert.match(READ_CODE_DESCRIPTION, /Standard JavaScript built-ins and methods are used directly/);
  const toolInspection = await script([
    ['eval', { code: 'read_code("return_result")' }],
    ['return_result', { status: 'success', value: 'done' }],
  ]);
  assert.match(toolInspection.results[0], /Native tool definition \(not program function source\): return_result/);
  assert.match(toolInspection.results[0], /Finish the call/);
  assert.match(toolInspection.results[0], /\\"enum\\": \[\\n\s+\\"success\\",\\n\s+\\"blocked\\",\\n\s+\\"failed\\"/);
  assert.match(toolInspection.results[0], /value/);
  assert.match(toolInspection.results[0], /Invoke this as a separate tool action/);

  const nativeToolInspection = await script([
    ['read_code', { name: 'return_result' }],
    ['return_result', { status: 'success', value: 'done' }],
  ]);
  assert.match(nativeToolInspection.results[0], /Native tool definition \(not program function source\): return_result/);
  assert.match(nativeToolInspection.results[0], /Argument schema \(JSON Schema\)/);

  let getterCalls = 0;
  const neuralese = {
    dialect: 'private-dialect-value', width: 384,
    bodies: { read: 'private learned body text' },
  };
  Object.defineProperty(neuralese, 'privateKey', { enumerable: true, get() {
    getterCalls++;
    return 'must not be read';
  } });
  const serviceInspection = await script([
    ['eval', { code: 'read_code("neuralese")' }],
    ['return_result', { status: 'success', value: 'done' }],
  ], 'string', { services: { neuralese } });
  assert.match(serviceInspection.results[0], /Host service object declaration \(type only, not implementation or field values\)/);
  assert.match(serviceInspection.results[0], /readonly \\"dialect\\": string/);
  assert.match(serviceInspection.results[0], /readonly \\"width\\": number/);
  assert.match(serviceInspection.results[0], /readonly \\"bodies\\": Readonly/);
  assert.match(serviceInspection.results[0], /readonly \\"privateKey\\": unknown/);
  assert.doesNotMatch(serviceInspection.results[0], /private-dialect-value|private learned body text|must not be read/);
  assert.equal(getterCalls, 0);

  const hiddenTool = await script([
    ['eval', { code: 'read_code("write_file")' }],
    ['return_result', { status: 'success', value: 'done' }],
  ]);
  assert.match(hiddenTool.results[0], /write_file is a native tool name but is not offered in this call/);
  assert.doesNotMatch(hiddenTool.results[0], /Argument schema|Create or replace a text file/);

  const scopedService = await script([
    ['eval', { code: 'read_code("neuralese")' }],
    ['return_result', { status: 'success', value: 'done' }],
  ], 'string', { services: { neuralese }, serviceScopes: { neuralese: ['other.nl'] } });
  assert.match(scopedService.results[0], /no-such-function/);
  assert.doesNotMatch(scopedService.results[0], /Host service object declaration|privateKey|private learned body text/);
});

test('an arrow that returns an uncalled nl is named as such', async () => {
  const { results } = await script([['eval', { code: 'const xs = ["a", "b"];\nconst ys = xs.map(x => nl`Is x a vowel? ${x}`);\nys.length' }]]);
  assert.match(results[0], /This arrow returns the `nl` function itself, never called/);
});


test('iteration documentation separates child draft input, fixed captures and parent progress', async () => {
  const { BUILT_IN_DOCS } = await import('../dist/native/runtime.js');
  assert.match(BUILT_IN_DOCS.iterateOn, /call that child with state\.draft/);
  assert.match(BUILT_IN_DOCS.iterateOn, /not the capture object or the outer progress state/);
  assert.match(BUILT_IN_DOCS.iterateOn, /Capture the applicable output contract/);
  assert.match(BUILT_IN_DOCS.iterateOn, /do not reconstruct later state from an outer initial draft/);
  assert.match(BUILT_IN_DOCS.iterateOn, /nl\.with accepts one type argument .*or two as <CaptureRecord, Result>/);
  assert.match(BUILT_IN_DOCS.iterateOn, /const nextDraft = await nl\.with<Draft>/);
  assert.match(BUILT_IN_DOCS.iterateOn, /\(state\.draft\)/);
  assert.match(BUILT_IN_DOCS.iterateOn, /The child returns Draft; the TypeScript step returns Progress/);
});
