/** The Node binding of the host (llama/host.ts): child_process, the platform's natlang directories, global fetch and a stream for output. */
import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:net';
import { homedir, platform } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultNatlangCacheDirectory, defaultNatlangStateDirectory } from '../package/directories.js';
import type { LlamaDirectories, LlamaHost, LlamaLog, LlamaProcess, LlamaProcesses } from './host.js';

export function defaultNatlangRuntimeDirectory(environment: NodeJS.ProcessEnv = process.env): string {
  if (environment.NATLANG_RUNTIME_HOME) return resolve(environment.NATLANG_RUNTIME_HOME);
  if (platform() === 'win32') return join(environment.LOCALAPPDATA ?? homedir(), 'natlang', 'runtimes');
  if (platform() === 'darwin') return join(homedir(), 'Library', 'Application Support', 'natlang', 'runtimes');
  return join(environment.XDG_DATA_HOME ?? join(homedir(), '.local', 'share'), 'natlang', 'runtimes');
}

export const nodeLlamaProcesses: LlamaProcesses = {
  spawn(command, args, options) {
    const child = spawn(command, [...args], { stdio: ['ignore', 'ignore', 'pipe'], ...(options.env ? { env: options.env } : {}) });
    let failed = false;
    child.stderr?.on('data', chunk => options.onStderr?.(String(chunk)));
    const exited = new Promise<void>(settle => {
      child.once('exit', () => settle());
      child.once('error', error => { failed = true; options.onError?.(error); settle(); });
    });
    return { pid: child.pid, exited, kill: signal => { child.kill(signal); },
      hasExited: () => failed || child.exitCode !== null || child.signalCode !== null } satisfies LlamaProcess;
  },
  run(command, args, options = {}) {
    const result = spawnSync(command, [...args], { encoding: 'utf8', ...(options.cwd ? { cwd: options.cwd } : {}),
      ...(options.env ? { env: options.env } : {}), ...(options.timeoutMs ? { timeout: options.timeoutMs } : {}),
      ...(options.output === 'host' ? { stdio: ['ignore', 'inherit', 'inherit'] as const } : {}) });
    return { status: result.status, stdout: result.stdout ?? '', stderr: result.stderr ?? '',
      ...(result.error ? { error: result.error } : {}) };
  },
  async freePort() {
    return await new Promise<number>((resolvePort, reject) => {
      const server = createServer();
      server.once('error', reject);
      server.listen(0, '127.0.0.1', () => {
        const address = server.address();
        if (!address || typeof address === 'string') { server.close(); reject(new Error('could not allocate a local model port')); return; }
        server.close(error => error ? reject(error) : resolvePort(address.port));
      });
    });
  },
};

/** The CLI's directories: the platform's natlang cache, state and runtime roots, and the checkout's or package's model files. */
export function nodeLlamaDirectories(environment: NodeJS.ProcessEnv = process.env): LlamaDirectories {
  const cache = defaultNatlangCacheDirectory(environment);
  return { runtime: defaultNatlangRuntimeDirectory(environment), models: resolve(cache, 'models'), modelHashes: resolve(cache, 'model-hashes'),
    warmPrefixes: resolve(cache, 'warm-prefixes'), state: defaultNatlangStateDirectory(environment),
    checkoutModels: resolve('models'),
    // A checkout's templates (the working directory's, then the repository's) and the package's own copies.
    templates: [resolve('models', 'templates'), fileURLToPath(new URL('../../../models/templates', import.meta.url)),
      fileURLToPath(new URL('../../model-assets', import.meta.url))] };
}

/** The host the natlang CLI passes: Node processes and directories, global `fetch`, and `log` (default stderr) for notices. */
export function nodeLlamaHost(environment: NodeJS.ProcessEnv = process.env, log: LlamaLog = process.stderr): LlamaHost {
  return { environment, directories: nodeLlamaDirectories(environment), processes: nodeLlamaProcesses, fetch: (input, init) => fetch(input, init), log };
}
