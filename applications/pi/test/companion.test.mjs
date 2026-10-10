/**
 * Wiring of the companion (scripted executor: wiring evidence, not model quality). pi-durable's Harness runs its own
 * built-in tasks with the faux provider; the companion extension is installed. A tool round starts a background
 * `pi.companion` task, which learns the file the agent read and writes a briefing; the next request shows it in the
 * `companion` section.
 *
 * Run: node --test applications/pi/test/companion.test.mjs (builds the app into .natlang/test-build-companion first).
 */
import assert from 'node:assert/strict';
import { test, before } from 'node:test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const hostDist = fileURLToPath(new URL('../../../ts-host/dist/', import.meta.url));
const outDir = join(root, '.natlang', 'test-build-companion');
let m;

before(async () => {
  const natlang = await import(pathToFileURL(join(hostDist, 'index.js')).href);
  const runtime = pathToFileURL(join(hostDist, 'index.js'));
  const result = natlang.buildProject({ project: root, outDir, runtimeModule: { url: runtime.href, path: fileURLToPath(runtime),
    types: join(hostDist, 'index.d.ts'), specifiers: ['@natlang/node'] } });
  if (!result.ok) throw new Error(natlang.formatDiagnostics(result.diagnostics));
  const load = path => import(pathToFileURL(join(outDir, path)).href);
  const { scriptedModel } = await import(pathToFileURL(join(hostDist, '../test/support/natlang.mjs')).href);
  m = { natlang, scriptedModel, durable: await load('vendor/durable/src/index.js'), companion: await load('extensions/companion/index.js'),
    memory: await load('vendor/durable/src/storage/memory.js'), nodeEnv: await load('vendor/durable/src/env/node.js'),
    ai: await import('@earendil-works/pi-ai'),
    chord: await import('@earendil-works/chord/context') };
});

/** Eval code per companion function, recognized by its instructions. */
const CODE = [
  ['You accompany a coding agent', `
    const path = observation.touched[0];
    const file = await companion.file(path);
    if (file && !file.known) await companion.remember(path, await summarize(path, file.text));
    return { focus: 'Read ' + path, facts: [path + ': ' + file.text.trim()], warnings: [], suggestions: ['Run the tests'] };`],
  ['is the content of the workspace file', `return { purpose: 'A greeting', symbols: [], notes: [] };`],
];

test('a tool round starts the companion, which learns the read file and briefs the next request', async () => {
  const calls = [];
  const scripted = m.scriptedModel(opening => {
    const found = CODE.find(([marker]) => opening.includes(marker));
    calls.push(found ? found[0] : 'unknown');
    return found ? found[1] : null;
  });
  const natlang = m.natlang.createNatlangRuntime({ model: scripted.driver });
  const cwd = mkdtempSync(join(tmpdir(), 'pi-companion-'));
  writeFileSync(join(cwd, 'a.txt'), 'hello\n');
  const faux = m.ai.fauxProvider();
  const models = m.ai.createModels();
  models.setProvider(faux.provider);
  const context = m.chord.BACKGROUND_CONTEXT;
  const registry = m.durable.createRegistry();
  registry.install({ name: 'tools', tools: [{ name: 'read', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } } },
    execute: async () => ({ content: [{ type: 'text', text: 'hello' }] }) }] });
  let harness;
  const reports = [];
  registry.install(m.companion.companion(natlang, { harness: () => harness, onReport: error => reports.push(error) }));
  faux.setResponses([
    m.ai.fauxAssistantMessage([m.ai.fauxToolCall('read', { path: 'a.txt' }, { id: 'c1' })], { stopReason: 'toolUse' }),
    m.ai.fauxAssistantMessage('done'),
    m.ai.fauxAssistantMessage('again'),
  ]);
  // The companion reads the workspace through the conversation's execution environment.
  harness = await m.durable.Harness.open(new m.memory.MemoryStorage(), { models, registry, onReport: error => reports.push(error),
    env: ({ cwd: dir = cwd }) => new m.nodeEnv.NodeExecutionEnv({ cwd: dir }) }, context);
  const conversation = await harness.root(context, { agent: { model: { provider: 'faux', modelId: 'faux-1' }, cwd } });
  harness.resume();
  assert.equal((await (await conversation.submit({ type: 'input', content: 'read a.txt' }, context)).wait(context)).status, 'done');

  // The companion runs in the background; wait until its task ended.
  let companionTasks = [];
  for (let i = 0; i < 200; i++) {
    companionTasks = (await harness.commit(tx => tx.scanTasks({ conversationId: conversation.id, kind: 'pi.companion' }, 10), context)).items;
    if (companionTasks.length && companionTasks.every(task => task.state.status === 'terminal')) break;
    await new Promise(done => setTimeout(done, 25));
  }
  assert.deepEqual(companionTasks.map(task => [task.background, task.state.status, task.state.outcome?.status]), [[true, 'terminal', 'completed']], JSON.stringify(reports.map(String)));
  assert.deepEqual(calls, ['You accompany a coding agent', 'is the content of the workspace file']);
  const files = await harness.snapshot(m.companion.CompanionFiles, context);
  assert.equal(files.files['a.txt'].purpose, 'A greeting');
  const doc = await harness.snapshot(m.companion.CompanionDoc, conversation.id, context);
  assert.equal(doc.briefing.focus, 'Read a.txt');

  // The next request carries the briefing as the companion section.
  assert.equal((await (await conversation.submit({ type: 'input', content: 'and now?' }, context)).wait(context)).status, 'done');
  const entries = (await conversation.entries({}, 50, undefined, context)).items;
  const shown = entries.flatMap(entry => entry.model ?? []).filter(message => message.role === 'system').map(message => message.sections?.companion).filter(Boolean);
  assert.equal(shown.length, 1);
  assert.match(shown[0], /^<companion>\nNotes from your harness companion[\s\S]*Focus: Read a\.txt\nKnown:\n- a\.txt: hello\nConsider:\n- Run the tests\n<\/companion>$/);
  await harness.close(context);
});

test('a long tool output reaches the agent shaped, and recall returns it whole', async () => {
  const scripted = m.scriptedModel(opening => opening.includes('You accompany a coding agent')
    ? `return { focus: 'x', facts: [], warnings: [], suggestions: [] };` : null);
  const natlang = m.natlang.createNatlangRuntime({ model: scripted.driver });
  const faux = m.ai.fauxProvider();
  const models = m.ai.createModels();
  models.setProvider(faux.provider);
  const context = m.chord.BACKGROUND_CONTEXT;
  const registry = m.durable.createRegistry();
  const long = Array.from({ length: 2000 }, (_, i) => `line ${i}`).join('\n');
  registry.install({ name: 'tools', tools: [{ name: 'bash', description: 'Run', parameters: { type: 'object', properties: { command: { type: 'string' } } },
    execute: async () => ({ content: [{ type: 'text', text: long }] }) }] });
  let harness;
  const reports = [];
  registry.install(m.companion.companion(natlang, { harness: () => harness, onReport: error => reports.push(String(error)) }));
  faux.setResponses([
    m.ai.fauxAssistantMessage([m.ai.fauxToolCall('bash', { command: 'seq' }, { id: 'c1' })], { stopReason: 'toolUse' }),
    m.ai.fauxAssistantMessage([m.ai.fauxToolCall('recall', { handle: 'c1' }, { id: 'c2' })], { stopReason: 'toolUse' }),
    m.ai.fauxAssistantMessage('done'),
  ]);
  harness = await m.durable.Harness.open(new m.memory.MemoryStorage(), { models, registry }, context);
  const conversation = await harness.root(context, { agent: { model: { provider: 'faux', modelId: 'faux-1' } } });
  harness.resume();
  assert.equal((await (await conversation.submit({ type: 'input', content: 'count' }, context)).wait(context)).status, 'done');
  const results = (await conversation.entries({ order: 'ascending' }, 50, undefined, context)).items.flatMap(entry => entry.model ?? []).filter(message => message.role === 'toolResult');
  const [shaped, recalled] = results.map(message => message.content[0].text);
  assert.ok(shaped.length < 5_000, `shaped is ${shaped.length} characters; ${reports.join("; ")}`);
  assert.match(shaped, /^line 0\n[\s\S]*recall\("c1"\) returns the full output[\s\S]*line 1999$/);
  assert.equal(recalled, long);
  await harness.close(context);
});

test('with nl shaping, the companion keeps the lines shape.nl chose, once per call', async () => {
  const shapes = [];
  const scripted = m.scriptedModel(opening => {
    if (opening.includes('You accompany a coding agent')) return `return { focus: 'x', facts: [], warnings: [], suggestions: [] };`;
    if (!opening.includes('too long to show whole')) return null;
    shapes.push(opening);
    return `const lines = output.split("\\n"); const at = lines.findIndex(line => line.startsWith("FAIL")) + 1;
      return { keep: [{ from: at, to: at + 1 }, { from: lines.length, to: lines.length }], gist: "1 failing test of 2000" };`;
  });
  const natlang = m.natlang.createNatlangRuntime({ model: scripted.driver });
  const faux = m.ai.fauxProvider();
  const models = m.ai.createModels();
  models.setProvider(faux.provider);
  const context = m.chord.BACKGROUND_CONTEXT;
  const registry = m.durable.createRegistry();
  const lines = Array.from({ length: 2000 }, (_, i) => `ok ${i}`);
  lines[1200] = 'FAIL parser handles empty input';
  lines[1201] = '  expected [] but got undefined';
  lines[1999] = '1999 passed, 1 failed';
  registry.install({ name: 'tools', tools: [{ name: 'bash', description: 'Run', parameters: { type: 'object', properties: { command: { type: 'string' } } },
    execute: async () => ({ content: [{ type: 'text', text: lines.join('\n') }] }) }] });
  let harness;
  const reports = [];
  registry.install(m.companion.companion(natlang, { harness: () => harness, shaping: 'nl', onReport: error => reports.push(String(error)) }));
  faux.setResponses([
    m.ai.fauxAssistantMessage([m.ai.fauxToolCall('bash', { command: 'npm test' }, { id: 'c1' })], { stopReason: 'toolUse' }),
    m.ai.fauxAssistantMessage('done'),
  ]);
  harness = await m.durable.Harness.open(new m.memory.MemoryStorage(), { models, registry }, context);
  const conversation = await harness.root(context, { agent: { model: { provider: 'faux', modelId: 'faux-1' } } });
  harness.resume();
  assert.equal((await (await conversation.submit({ type: 'input', content: 'test' }, context)).wait(context)).status, 'done');
  const [shown] = (await conversation.entries({ order: 'ascending' }, 50, undefined, context)).items.flatMap(entry => entry.model ?? [])
    .filter(message => message.role === 'toolResult').map(message => message.content[0].text);
  assert.equal(shown, '[The companion shortened this output of 2000 lines: 1 failing test of 2000 recall("c1") returns it whole.]\n' +
    '[… lines 1–1200 …]\nFAIL parser handles empty input\n  expected [] but got undefined\n[… lines 1203–1999 …]\n1999 passed, 1 failed',
    reports.join('; '));
  assert.equal(shapes.length, 1);
  assert.match(shapes[0], /call: string = "bash \{\\"command\\":\\"npm test\\"\}"/);
  await harness.close(context);
});

test('a shape is clamped, merged and held to its budget whatever it asks for', () => {
  const text = ['a', 'bb', 'ccc', 'dddd', 'eeeee'].join('\n');
  assert.equal(m.companion.renderShape(text, 'h', { keep: [{ from: 4, to: 99 }, { from: 0, to: 1 }, { from: 2, to: 2 }], gist: 'g.' }),
    '[The companion shortened this output of 5 lines: g. recall("h") returns it whole.]\na\nbb\n[… lines 3–3 …]\ndddd\neeeee');
  assert.equal(m.companion.renderShape(text, 'h', { keep: [{ from: 1, to: 5 }], gist: 'g.' }, 6),
    '[The companion shortened this output of 5 lines: g. recall("h") returns it whole.]\na\nbb\n[… lines 3–5 …]');
  assert.equal(m.companion.renderShape(text, 'h', { keep: [], gist: 'g.' }),
    '[The companion shortened this output of 5 lines: g. recall("h") returns it whole.]\n[… lines 1–5 …]');
});
