import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createRequire } from 'node:module';
import { createInterface } from 'node:readline';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { build } from 'esbuild';
import { buildTarget } from '../scripts/vendor-build.mjs';
import { scenarios } from './support/core-scenarios.mjs';

// The self-contained workerd bundle (TypeScript bundled in, scripts/vendor-build.mjs target `workerd`) runs the same
// scripted-agent scenarios as the core in real workerd with `nodejs_compat`. The consumer here bundles only `yaml` and
// `@noble/hashes`, which that bundle leaves external, and defines nothing: the bundle carries its own TypeScript setup.
const root = fileURLToPath(new URL('..', import.meta.url));
const workerd = createRequire(import.meta.url)('workerd').default;

test('the self-contained workerd bundle runs the scripted-agent scenarios', { timeout: 300_000 }, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'natlang-workerd-bundled-'));
  let server, stopping = false;
  try {
    await buildTarget('workerd', join(directory, 'bundle'));
    await writeFile(join(directory, 'entry.mjs'), `import * as core from './bundle/natlang-workerd.js';
import { scenarios } from ${JSON.stringify(join(root, 'test/support/core-scenarios.mjs'))};
export default { async fetch(request, env) {
  const realm = core.constructorRealm((body, name, ...parameters) => env.UNSAFE_EVAL.newFunction(body, name, ...parameters));
  const results = {};
  for (const [name, run] of Object.entries(scenarios(core, realm))) {
    try { await run(); results[name] = 'ok'; } catch (error) { results[name] = String(error?.stack ?? error); }
  }
  return Response.json(results);
} };
`);
    await build({ entryPoints: [join(directory, 'entry.mjs')], bundle: true, format: 'esm', platform: 'neutral', target: 'es2022',
      mainFields: ['module', 'main'], conditions: ['workerd', 'worker', 'browser'], outfile: join(directory, 'worker.mjs'), absWorkingDir: root,
      logLevel: 'silent', external: ['node:*'], nodePaths: [join(root, 'node_modules')] });
    await writeFile(join(directory, 'workerd.capnp'), `using Workerd = import "/workerd/workerd.capnp";
const config :Workerd.Config = (
  services = [ (name = "main", worker = .worker) ],
  sockets = [ (name = "http", address = "127.0.0.1:0", http = (), service = "main") ]
);
const worker :Workerd.Worker = (
  modules = [ (name = "worker", esModule = embed "worker.mjs") ],
  compatibilityDate = "2026-01-01",
  compatibilityFlags = ["nodejs_compat"],
  bindings = [ (name = "UNSAFE_EVAL", unsafeEval = void) ],
);
`);
    server = spawn(workerd, ['serve', '--experimental', '--control-fd=3', join(directory, 'workerd.capnp')], { stdio: ['ignore', 'pipe', 'pipe', 'pipe'] });
    let log = '';
    server.stdout.on('data', chunk => { log += chunk; });
    server.stderr.on('data', chunk => { log += chunk; });
    const exited = new Promise((_, reject) => server.once('exit', code => { if (!stopping) reject(new Error(`workerd exited (${code}):\n${log}`)); }));
    const listening = (async () => {
      for await (const line of createInterface({ input: server.stdio[3] })) {
        const message = JSON.parse(line);
        if (message.event === 'listen' && message.socket === 'http') return message.port;
      }
      throw new Error('workerd closed its control channel before listening');
    })();
    const port = await Promise.race([listening, exited]);
    const response = await Promise.race([fetch(`http://127.0.0.1:${port}/`), exited]);
    const results = await response.json();
    const expected = Object.keys(scenarios({ workerdPlatform() {} }, {}));
    assert.deepEqual(Object.keys(results).sort(), expected.sort());
    for (const name of expected) assert.equal(results[name], 'ok', `${name}: ${results[name]}\n${log}`);
  } finally {
    stopping = true;
    if (server && server.exitCode === null) { server.kill('SIGTERM'); await once(server, 'exit'); }
    await rm(directory, { recursive: true, force: true });
  }
});
