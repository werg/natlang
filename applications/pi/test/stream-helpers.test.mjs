/**
 * Helpers on the live streams (plans/STREAMING.md §2 and §3), with scripted models: wiring evidence, not model quality.
 *
 * §2: the agent's model is reached through the natlang transport (host/natlang-provider.ts) with a scripted driver that
 * streams deltas and holds the end of its turn until the companion has acted, so each test shows the companion working
 * before the turn ends: research on what the reasoning names, a reset discarding it, a tool call prepared as soon as
 * its arguments are complete. §3: edit deltas change the draft document and produce offers without touching the
 * transcript; taking an offer adds it to the draft; sending is the only thing that enters the transcript.
 *
 * Run: node --test applications/pi/test/stream-helpers.test.mjs (builds the app into .natlang/test-build-stream first).
 */
import assert from 'node:assert/strict';
import { test, before } from 'node:test';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const hostDist = fileURLToPath(new URL('../../../ts-host/dist/', import.meta.url));
const outDir = join(root, '.natlang', 'test-build-stream');
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
    draft: await load('extensions/companion/draft.js'), drafts: await load('host/drafts.js'), views: await load('host/views.js'),
    provider: await load('host/natlang-provider.js'), memory: await load('vendor/durable/src/storage/memory.js'),
    nodeEnv: await load('vendor/durable/src/env/node.js'), ai: await import('@earendil-works/pi-ai'),
    chord: await import('@earendil-works/chord/context') };
});

/** The executor: eval code per companion function, recognized by its instructions. */
const EXECUTOR = [
  ['is the content of the workspace file', `return { purpose: 'A greeting for ' + path, symbols: [], notes: [] };`],
  ['You accompany a coding agent', `return { focus: 'x', facts: [], warnings: [], suggestions: [] };`],
  ['is a piece of what a coding agent is writing', `return { files: written.includes('greeting') ? ['a.txt'] : [], symbols: [] };`],
  ['is the message a user is typing', `const found = await companion.search('greet');
    return [{ kind: 'question', text: 'Which greeting?', insert: 'Greeting: ___' }, { kind: 'context', text: found[0] ?? 'none' }];`],
];
const executor = () => m.natlang.createNatlangRuntime({ model: m.scriptedModel(opening => EXECUTOR.find(([marker]) => opening.includes(marker))?.[1] ?? null).driver });

/** pi-ai models whose provider "agent" is the natlang transport over `driver` (a scripted model-turn driver). */
function agentModels(driver) {
  const models = m.ai.createModels();
  models.setProvider(m.ai.createProvider({ id: 'agent', name: 'Agent', baseUrl: 'http://agent.invalid/v1',
    auth: { apiKey: { name: 'agent key', resolve: async () => ({ auth: {} }) } },
    models: [{ id: 'm', name: 'm', api: 'natlang-model-turn', provider: 'agent', baseUrl: 'http://agent.invalid/v1', input: ['text'],
      reasoning: true, reader: { kind: 'text' }, contextWindow: 65536, maxTokens: 1000, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } }],
    api: m.provider.natlangApi(() => driver) }));
  return models;
}

/** A promise that settles when `predicate` holds for a speculation event, or fails after `ms`. */
function waiter() {
  const waits = [];
  const seen = [];
  return {
    seen,
    onSpeculation(event) { seen.push(event); for (const wait of [...waits]) if (wait.predicate(event)) { waits.splice(waits.indexOf(wait), 1); wait.resolve(event); } },
    when(predicate, ms = 5000) {
      const found = seen.find(predicate);
      if (found) return Promise.resolve(found);
      return new Promise((resolve, reject) => {
        const wait = { predicate, resolve };
        waits.push(wait);
        setTimeout(() => reject(new Error(`no speculation event matched within ${ms} ms: ${JSON.stringify(seen)}`)), ms);
      });
    },
  };
}

async function until(read, explain = () => '', ms = 5000) {
  const end = Date.now() + ms;
  for (;;) {
    const value = await read();
    if (value) return value;
    if (Date.now() > end) throw new Error(`timed out: ${explain()}`);
    await new Promise(done => setTimeout(done, 20));
  }
}

/** A harness with the companion over a workspace holding `files`, its agent the scripted natlang driver. */
async function setup(driver, files, companionOptions = {}) {
  const context = m.chord.BACKGROUND_CONTEXT;
  const cwd = mkdtempSync(join(tmpdir(), 'pi-stream-'));
  for (const [name, text] of Object.entries(files)) writeFileSync(join(cwd, name), text);
  const registry = m.durable.createRegistry();
  registry.install({ name: 'tools', tools: [{ name: 'read', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } } },
    execute: async () => ({ content: [{ type: 'text', text: 'hello' }] }) }] });
  const watch = waiter();
  const reports = [];
  let harness;
  registry.install(m.companion.companion(executor(), { harness: () => harness, onReport: error => reports.push(String(error)),
    onSpeculation: watch.onSpeculation, ...companionOptions }));
  harness = await m.durable.Harness.open(new m.memory.MemoryStorage(), { models: agentModels(driver), registry,
    onReport: error => reports.push(String(error)), env: ({ cwd: dir = cwd }) => new m.nodeEnv.NodeExecutionEnv({ cwd: dir }) }, context);
  const conversation = await harness.root(context, { agent: { model: { provider: 'agent', modelId: 'm' }, cwd, thinkingLevel: 'high' } });
  harness.resume();
  const ask = async text => assert.equal((await (await conversation.submit({ type: 'input', content: text }, context)).wait(context)).status, 'done',
    JSON.stringify({ reports, seen: watch.seen }));
  return { harness, conversation, context, cwd, watch, reports, ask };
}

const REASONING = 'The greeting lives in a.txt, so the next step is to read a.txt and check what the greeting says there now. ';

test('the companion reads the streamed reasoning and starts research before the turn ends', async () => {
  let watch, endedEarly = true;
  const driver = async (_request, _signal, options) => {
    options.onDelta({ type: 'reasoning', text: REASONING });
    // The turn ends only once the companion has started looking at a.txt.
    await watch.when(event => event.type === 'start' && event.job === 'file:a.txt');
    endedEarly = false;
    return { text: 'It says hello.', reasoning: REASONING };
  };
  const s = await setup(driver, { 'a.txt': 'hello\n' });
  watch = s.watch;
  await s.ask('What does the greeting say?');
  assert.equal(endedEarly, false);
  // The reply kept the reasoning that named a.txt: what was learned is committed for the next request.
  const doc = await until(async () => (await s.harness.snapshot(m.companion.CompanionDoc, s.conversation.id, s.context))?.research, () => JSON.stringify({ reports: s.reports, seen: s.watch.seen }));
  assert.deepEqual(doc.notes, ['a.txt: A greeting for a.txt']);
  assert.equal((await s.harness.snapshot(m.companion.CompanionFiles, s.context)).files['a.txt'].purpose, 'A greeting for a.txt');
  assert.equal(s.watch.seen.filter(event => event.type === 'discard').length, 0);
  await s.harness.close(s.context);
});

test('the natural-language hint policy finds what the reasoning refers to without naming it', async () => {
  let watch;
  const reasoning = 'I should look at the greeting before I answer, since the question is about what it says right now. ';
  const driver = async (_request, _signal, options) => {
    options.onDelta({ type: 'reasoning', text: reasoning });
    options.onDelta({ type: 'text', text: 'Checking.' });
    await watch.when(event => event.type === 'start' && event.job === 'file:a.txt');
    return { text: 'Checking.', reasoning };
  };
  const s = await setup(driver, { 'a.txt': 'hello\n' }, { hints: 'nl' });
  watch = s.watch;
  await s.ask('What does the greeting say?');
  const doc = await until(async () => (await s.harness.snapshot(m.companion.CompanionDoc, s.conversation.id, s.context))?.research, () => JSON.stringify({ reports: s.reports, seen: s.watch.seen }));
  assert.deepEqual(doc.notes, ['a.txt: A greeting for a.txt']);
  await s.harness.close(s.context);
});

test('a reset discards the abandoned attempt\'s speculation; only what the final reply keeps is committed', async () => {
  let watch;
  const abandoned = 'First I will open b.txt to see what the old greeting was before I change anything at all here. ';
  const kept = 'The answer is in a.txt, which holds the greeting the user asked about, so I point there for them. ';
  const driver = async (_request, _signal, options) => {
    options.onDelta({ type: 'text', text: abandoned });
    await watch.when(event => event.type === 'start' && event.job === 'file:b.txt');
    // The driver sends the request again (a malformed-call retry, for example): the parts so far are abandoned.
    options.onDelta({ type: 'reset' });
    options.onDelta({ type: 'text', text: kept });
    return { text: kept };
  };
  const s = await setup(driver, { 'a.txt': 'hello\n', 'b.txt': 'old\n' });
  watch = s.watch;
  await s.ask('Where is the greeting?');
  const discard = await s.watch.when(event => event.type === 'discard');
  assert.equal(discard.reason, 'reset');
  assert.deepEqual(discard.jobs, ['file:b.txt']);
  const doc = await until(async () => (await s.harness.snapshot(m.companion.CompanionDoc, s.conversation.id, s.context))?.research, () => JSON.stringify({ reports: s.reports, seen: s.watch.seen }));
  assert.deepEqual(doc.notes, ['a.txt: A greeting for a.txt']);
  // Whatever the abandoned attempt learned about b.txt was never committed.
  const files = (await s.harness.snapshot(m.companion.CompanionFiles, s.context)).files;
  assert.equal(files['b.txt'], undefined);
  assert.ok(files['a.txt']);
  await s.harness.close(s.context);
});

test('a tool call is prepared as soon as its arguments are complete, before the turn ends', async () => {
  let watch, turns = 0, preparedEarly = false;
  const driver = async (_request, _signal, options) => {
    if (++turns > 1) return { text: 'It says hello.' };
    options.onDelta({ type: 'text', text: 'Reading it.' });
    options.onDelta({ type: 'tool_call', index: 0, id: 'c1', name: 'read', arguments: '{"path":' });
    options.onDelta({ type: 'tool_call', index: 0, arguments: '"a.txt"}' });
    // The call is still open (it ends with the turn), but its arguments are a whole object.
    await watch.when(event => event.type === 'prepare' && event.callId === 'c1');
    await watch.when(event => event.type === 'start' && event.job === 'file:a.txt');
    preparedEarly = true;
    return { text: 'Reading it.', calls: [['read', { path: 'a.txt' }]], raw_calls: [{ id: 'c1' }] };
  };
  const s = await setup(driver, { 'a.txt': 'hello\n' });
  watch = s.watch;
  await s.ask('What does a.txt say?');
  assert.equal(preparedEarly, true);
  const prepared = s.watch.seen.find(event => event.type === 'prepare');
  assert.equal(prepared.intent, m.views.viewIntent({ content: [{ type: 'text', text: 'Reading it.' }] }, { name: 'read', arguments: { path: 'a.txt' } }));
  // The final message confirmed the call: the intent captured while streaming is the call's recorded intent.
  assert.equal((await s.harness.snapshot(m.views.ViewIntents, s.conversation.id, 'c1', s.context)).intent, prepared.intent);
  // The read call's file is learned as knowledge; the agent reads the file itself, so it is no research note.
  await until(async () => (await s.harness.snapshot(m.companion.CompanionFiles, s.context))?.files['a.txt'],
    () => JSON.stringify({ reports: s.reports, seen: s.watch.seen }));
  assert.equal((await s.harness.snapshot(m.companion.CompanionDoc, s.conversation.id, s.context))?.research, undefined);
  await s.harness.close(s.context);
});

test('draft deltas update the draft and produce offers without touching the transcript; sending is the only entry', async () => {
  const context = m.chord.BACKGROUND_CONTEXT;
  const cwd = mkdtempSync(join(tmpdir(), 'pi-draft-'));
  writeFileSync(join(cwd, 'lib.js'), "function greet() { return 'hi'; }\n");
  const faux = m.ai.fauxProvider();
  const models = m.ai.createModels();
  models.setProvider(faux.provider);
  faux.setResponses([m.ai.fauxAssistantMessage('ok')]);
  const registry = m.durable.createRegistry();
  const env = ({ cwd: dir = cwd }) => new m.nodeEnv.NodeExecutionEnv({ cwd: dir });
  const harness = await m.durable.Harness.open(new m.memory.MemoryStorage(), { models, registry, env }, context);
  const conversation = await harness.root(context, { agent: { model: { provider: 'faux', modelId: 'faux-1' }, cwd } });
  harness.resume();
  const reports = [];
  const changes = [];
  const natlang = executor();
  const drafts = m.drafts.openDrafts(harness, { env, debounceMs: 5, onReport: error => reports.push(String(error)), onChange: id => changes.push(id),
    helpers: [m.draft.companionDraftHelper(natlang, { offers: 'crisp' })] });
  const entries = async () => (await conversation.entries({}, 50, undefined, context)).items;
  const before = (await entries()).length;

  // The user types: deltas, as a text field reports them.
  assert.equal(await drafts.edit(conversation.id, { text: 'Make `greet` louder' }), 1);
  const typed = 'Make `greet` louder';
  assert.equal(await drafts.edit(conversation.id, [{ from: typed.length, to: typed.length, insert: ' and fix missing.ts' }]), 2);
  await assert.rejects(drafts.edit(conversation.id, { from: 5, to: 999, insert: 'x' }), /draft delta 0 replaces 5..999/);
  await drafts.idle(conversation.id);
  let state = await drafts.read(conversation.id);
  assert.equal(state.text, 'Make `greet` louder and fix missing.ts');
  assert.equal(state.offersVersion, 2);
  assert.deepEqual(state.offers.map(offer => [offer.kind, offer.text]), [['warning', 'missing.ts does not exist in the workspace.'],
    ['context', "`greet` is defined at lib.js:1:function greet() { return 'hi'; }"]]);
  assert.deepEqual(reports, []);
  // Typing and offers are documents, never entries.
  assert.equal((await entries()).length, before);
  assert.ok(changes.length >= 3);

  // Taking an offer adds it to the draft; ignoring one sends nothing of it.
  const context_ = state.offers.find(offer => offer.kind === 'context');
  await drafts.take(conversation.id, context_.id);
  state = await drafts.read(conversation.id);
  assert.equal(state.text, "Make `greet` louder and fix missing.ts\n\nRelevant: `greet` is defined at lib.js:1:function greet() { return 'hi'; }");
  await drafts.idle(conversation.id);

  // Sending submits the draft as input and clears it with its offers.
  const submission = await drafts.send(conversation.id);
  assert.equal((await submission.wait(context)).status, 'done');
  const users = (await entries()).flatMap(entry => entry.model ?? []).filter(message => message.role === 'user');
  assert.equal(users.length, 1);
  assert.equal(users[0].content, state.text);
  state = await drafts.read(conversation.id);
  assert.equal(state.text, '');
  assert.deepEqual(state.offers, []);
  await assert.rejects(drafts.send(conversation.id), /draft is empty/);

  // The natural-language offer policy, through the same mechanism.
  const nl = m.drafts.openDrafts(harness, { env, debounceMs: 5, helpers: [m.draft.companionDraftHelper(natlang, { offers: 'nl' })] });
  await nl.edit(conversation.id, { text: 'Change the greeting' });
  await nl.idle(conversation.id);
  state = await nl.read(conversation.id);
  assert.deepEqual(state.offers.map(offer => [offer.kind, offer.text, offer.helper]), [['question', 'Which greeting?', 'companion'],
    ['context', "lib.js:1:function greet() { return 'hi'; }", 'companion']]);
  nl.close();
  drafts.close();
  await harness.close(context);
});
