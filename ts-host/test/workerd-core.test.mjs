import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { builtinModules, createRequire } from 'node:module';
import { createInterface } from 'node:readline';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { build } from 'esbuild';
import { scenarios } from './support/core-scenarios.mjs';

// The core's scripted-agent scenarios inside real workerd with `nodejs_compat`: the vendored bundle (dist/core, built by
// `npm run build:core`), TypeScript, yaml and @noble/hashes bundled around it as a consumer's build would, and a realm over
// the UNSAFE_EVAL binding. The scenarios are the ones core.test.mjs runs under Node.
const root = fileURLToPath(new URL('..', import.meta.url));
// The `workerd` package exports the path of its binary (its bin script would not pass the control descriptor on).
const workerd = createRequire(import.meta.url)('workerd').default;

/**
 * What a consumer's build does with the CommonJS packages the core leaves external (TypeScript, yaml): their
 * `require("process")` becomes the module workerd provides under `nodejs_compat`; built-ins workerd lacks (TypeScript only
 * reaches for them when it takes the host for Node) become empty modules.
 */
const COMPAT = new Set(['process', 'path', 'os', 'util', 'events', 'buffer', 'crypto', 'fs', 'url', 'stream', 'zlib', 'assert', 'string_decoder']);
function nodeCompat() {
  return { name: 'node-compat', setup(build) {
    build.onResolve({ filter: /^[a-z_]+$/ }, args => builtinModules.includes(args.path) ? { path: args.path, namespace: 'node-compat' } : undefined);
    build.onLoad({ filter: /.*/, namespace: 'node-compat' }, args => ({ loader: 'js', contents: COMPAT.has(args.path) ?
      `import * as all from 'node:${args.path}'; module.exports = all.default ?? all;` : 'module.exports = {};' }));
  } };
}

test('the core runs the scripted-agent scenarios in workerd', { timeout: 300_000 }, async () => {
  const directory = await mkdtemp(join(tmpdir(), 'natlang-workerd-'));
  let server, stopping = false;
  try {
    await build({ entryPoints: [join(root, 'test/workerd/worker.mjs')], bundle: true, format: 'esm', platform: 'neutral', target: 'es2022',
      mainFields: ['module', 'main'], conditions: ['workerd', 'worker', 'browser'], outfile: join(directory, 'worker.mjs'), absWorkingDir: root,
      logLevel: 'silent', external: ['node:*'], plugins: [nodeCompat()],
      // TypeScript takes `process` and `require` for a Node host and reaches for `fs`; workerd's `process` is not one.
      define: { 'process.browser': 'true' } });
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
    // workerd reports the port it listens on as a JSON line on the control descriptor (fd 3).
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
