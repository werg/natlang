/**
 * pi's browser entry (browser.ts): it builds under the browser policy (ts-host/scripts/build-application-browser.mjs,
 * no `node:*` module in its graph, the Node host's modules left out), and the built bundle, loaded here with the
 * browser runtime (ts-host's dist/browser/natlang.js) and Node's `process` hidden, runs a coding task end to end on a
 * virtual folder: pi-durable's harness with pi's coding tools (their natural-language functions run by a scripted
 * executor), the agent model reached through natlang's transport (a local fake server), the session in sqlite-wasm,
 * and the folder environment's shell. No browser is needed; a page for one is test/browser/index.html.
 *
 * Run: node --test applications/pi/test/browser-bundle.test.mjs (after ts-host's build:node and build:browser).
 */
import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { join } from 'node:path';
import sqlite3InitModule from '@sqlite.org/sqlite-wasm';
import { buildApplicationBrowser } from '../../../ts-host/scripts/build-application-browser.mjs';
import { requireFreshBrowserBuild } from '../../../ts-host/scripts/browser-build-freshness.mjs';
import { scriptedModel } from '../../../ts-host/test/support/natlang.mjs';
import { fakeOpfs } from '../../../ts-host/test/fixtures/fake-opfs.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const runtimeUrl = new URL('../../../ts-host/dist/browser/natlang.js', import.meta.url).href;
const outdir = join(root, '.natlang', 'test-browser-bundle');
let built, browser, pi, server, endpoint;
/** The agent model's replies, in order, and the requests it was sent. */
let replies = [];
const seen = [];

/** Import with Node's `process` hidden, so a module that needs Node fails here as it would in a page. */
async function asInBrowser(url) {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import(url); }
  finally { globalThis.process = process; }
}

before(async () => {
  requireFreshBrowserBuild('applications/pi/test/browser-bundle.test.mjs');
  // Built against the runtime's file URL here; a page names `@natlang/browser` in its import map instead.
  built = await buildApplicationBrowser({ app: root, outdir, runtime: runtimeUrl });
  browser = await asInBrowser(runtimeUrl);
  pi = await asInBrowser(pathToFileURL(built.entry).href);
  server = createServer((request, response) => {
    let data = '';
    request.on('data', chunk => { data += chunk; });
    request.on('end', () => {
      seen.push({ path: request.url, body: data ? JSON.parse(data) : undefined });
      const send = (status, body) => { response.writeHead(status, { 'content-type': 'application/json' }); response.end(JSON.stringify(body)); };
      if (request.url !== '/v1/chat/completions') return send(404, { error: { message: 'not found' } });
      const message = replies.shift() ?? { content: 'out of replies' };
      send(200, { id: `r${seen.length}`, choices: [{ finish_reason: message.tool_calls ? 'tool_calls' : 'stop', message }],
        usage: { prompt_tokens: 10, completion_tokens: 5 } });
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  endpoint = `http://127.0.0.1:${server.address().port}`;
});

after(() => server?.close());

test('the browser entry builds under the browser policy, without the Node host', () => {
  const inputs = Object.keys(built.result.metafile.inputs);
  assert.ok(inputs.some(path => path.endsWith('browser.js')), 'the entry is in the graph');
  assert.ok(!inputs.some(path => path.startsWith('node:') || path.includes('browser-node-forbidden')), inputs.filter(path => path.startsWith('node:')).join(', '));
  for (const nodeOnly of ['host/node.js', 'host/resources.js', 'env/node.js', 'env/node-watch.js', 'sqlite/node.js', 'main.js'])
    assert.ok(!inputs.some(path => path.endsWith(nodeOnly)), `${nodeOnly} stays out of the browser graph`);
  const entry = readFileSync(built.entry, 'utf8');
  assert.doesNotMatch(entry, /\bfrom\s*["']node:|require\(["']node:|import\(["']node:/);
  assert.ok(entry.includes(JSON.stringify(runtimeUrl)), 'the runtime stays a module of its own');
  for (const name of ['openBrowserPi', 'openBrowserSession', 'FolderExecutionEnv', 'codingRegistry', 'agentModels'])
    assert.equal(typeof pi[name], 'function', name);
  assert.equal(typeof pi.Harness.open, 'function');
});

/** The eval code a small model writes for each coding tool's function (as in tools-wiring.test.mjs). */
const TOOLS = [
  ['Write args.content to the file args.path', `
    const p = await env.toolPath(args.path); if (p.error) throw new Error(p.error.message);
    const held = await env.lock(p.value); if (held.error) throw new Error(held.error.message);
    const w = await env.writeFile(p.value, args.content);
    env.unlock(held.value);
    if (w.error) throw new Error(w.error.message);
    return { content: [{ type: 'text', text: 'Successfully wrote to ' + args.path }] };`],
  // The companion looks at the workspace through the folder environment: a file, the listing, a search.
  ['You accompany a coding agent', `
    const file = await companion.file('hello.txt');
    const listed = await companion.list('.');
    const found = await companion.search('hi from');
    return { focus: 'checked', facts: [listed.join(','), found.join('|'), file ? file.hash : 'none'], warnings: [], suggestions: [] };`],
  ['Run args.command.', `
    const run = await env.runShell(args.command, args.timeout);
    if (run.error) throw new Error(run.error.message);
    if (run.value.exitCode !== 0) throw new Error('Command exited with code ' + run.value.exitCode);
    return {};`],
];
const call = (id, name, args) => ({ id, type: 'function', function: { name, arguments: JSON.stringify(args) } });

test('a coding task runs end to end on a virtual folder with the browser modules', async () => {
  seen.length = 0;
  replies = [
    { content: 'Writing it.', tool_calls: [call('c1', 'write', { path: 'hello.txt', content: 'hi from the browser\n' })] },
    { content: '', tool_calls: [call('c2', 'bash', { command: 'cat hello.txt && ls' })] },
    { content: 'hello.txt says hi.' },
  ];
  const executor = scriptedModel(opening => TOOLS.find(([marker]) => opening.includes(marker))?.[1] ?? null);
  const natlang = browser.createNatlangRuntime({ model: executor.driver });
  const folder = new pi.Folder({ 'README.md': '# demo\n' });
  const sqlite3 = await sqlite3InitModule();
  const reports = [];
  const { storage, kind } = await pi.openBrowserSession(new sqlite3.oo1.DB(':memory:'));
  assert.equal(kind, 'sqlite');
  const { models, ref } = pi.agentModels({ endpoint, modelId: 'fake', transport: 'natlang' });
  const envs = new pi.ExecutionEnvs(pi.WORKSPACE, cwd => new pi.FolderExecutionEnv({ folder, cwd }));
  const registry = pi.codingRegistry(natlang, { prompt: { cwd: pi.WORKSPACE, packageDir: '/pi' }, subagent: false });
  let harness;
  registry.install(pi.companion(natlang, { harness: () => harness, onReport: error => reports.push(String(error)) }));
  const context = pi.BACKGROUND_CONTEXT;
  harness = await pi.Harness.open(storage, { models, registry, env: envs.env, onReport: error => reports.push(String(error)) }, context);
  const result = await pi.runPiTask(harness, { model: ref, cwd: pi.WORKSPACE }, 'Write hello.txt and show it', context);
  assert.equal(result.status, 'done', `${result.reason}; ${reports.join('; ')}`);
  assert.equal(result.answer, 'hello.txt says hi.');
  assert.equal(await folder.readText('hello.txt'), 'hi from the browser\n');

  // The agent saw the shell's output over the same folder, and pi's prompt names the virtual working directory.
  const chats = seen.filter(item => item.path === '/v1/chat/completions');
  assert.equal(chats.length, 3);
  const tool = chats[2].body.messages.filter(message => message.role === 'tool').map(message => message.content);
  assert.equal(tool[0], 'Successfully wrote to hello.txt');
  assert.match(tool[1], /^hi from the browser\nREADME\.md\s+hello\.txt\s*$/);
  assert.match(chats[0].body.messages[0].content, /<cwd>\n\/workspace\n<\/cwd>/);
  assert.deepEqual(chats[0].body.tools.map(item => item.function.name).sort(), ['bash', 'edit', 'read', 'recall', 'write']);

  // The companion's briefing, from the folder environment's file, listing and grep.
  const conversation = await harness.root(context);
  let briefing;
  for (let i = 0; i < 200 && !briefing; i++) {
    briefing = (await harness.snapshot(pi.CompanionDoc, conversation.id, context))?.briefing;
    if (!briefing) await new Promise(done => setTimeout(done, 25));
  }
  const hash = createHash('sha256').update('hi from the browser\n').digest('hex').slice(0, 16);
  assert.deepEqual(briefing?.facts, ['README.md,hello.txt', 'hello.txt:1:hi from the browser', hash], reports.join('; '));
  await harness.close(context);
  await envs.cleanup(context);
});

test('the session falls back to memory where OPFS cannot hold a database; blocks go to the OPFS block store', async () => {
  const reports = [];
  const { kind } = await pi.openBrowserSession('pi-session', error => reports.push(error.message));
  assert.equal(kind, 'memory');
  assert.match(reports[0], /dedicated worker/);
  // A Neuralese agent's blocks: the browser entry archives them in OPFS (here a stand-in), and the runtime keeps them.
  const opfs = fakeOpfs();
  seen.length = 0;
  const info = createServer((request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ dialects: ['d1'], width: 8, dtype: 'f32', max_block_length: 32 }));
  });
  await new Promise(resolve => info.listen(0, '127.0.0.1', resolve));
  try {
    const opened = await pi.openBrowserPi({ natlang: browser.createNatlangRuntime({ model: scriptedModel(() => null).driver }),
      agent: { endpoint: `http://127.0.0.1:${info.address().port}`, model: 'fake', reader: 'd1' },
      folder: new pi.Folder({}), session: new (await sqlite3InitModule()).oo1.DB(':memory:'), blocks: opfs.storage });
    assert.equal(opened.session, 'sqlite');
    assert.ok(opened.natlang.options.neuralese.store, 'the runtime has the block archive');
    assert.ok(opfs.root.directories.has('natlang-neuralese-blocks'), 'the archive is in OPFS');
    await opened.close();
  } finally { info.close(); }
});
