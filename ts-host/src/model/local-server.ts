import { createReadStream, createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from 'node:fs';
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:net';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { defaultNatlangCacheDirectory } from '../package/store.js';
import { DEFAULT_MODEL_RELEASE } from '../model-default.js';
import { openAICompatibleModelTurn, type OpenAICompatibleOptions } from './openai-compatible.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export const DEFAULT_LOCAL_MODEL = DEFAULT_MODEL_RELEASE;

export type ModelProfile = { endpoint?: string; model?: string; apiKeyEnv?: string;
  headers?: Record<string, string>; request?: Record<string, unknown> };
export type ManagedModelStatus = { source: 'external' | 'managed-local'; endpoint: string | null;
  model: string; executable: string | null; modelPath: string | null; running: boolean };
export type ManagedModelSession = { turn(request: ModelTurnRequest): Promise<ModelTurn>;
  status(): ManagedModelStatus; close(): Promise<void> };

function executable(command: string): string | null {
  if (command.includes('/') || command.includes('\\')) return existsSync(resolve(command)) ? resolve(command) : null;
  const probe = spawnSync(process.platform === 'win32' ? 'where.exe' : 'which', [command], { encoding: 'utf8' });
  return probe.status === 0 ? probe.stdout.trim().split(/\r?\n/, 1)[0] || null : null;
}

export function localModelPrerequisites(environment: NodeJS.ProcessEnv = process.env) {
  const command = environment.NATLANG_LLAMA_SERVER ?? 'llama-server';
  const found = executable(command);
  const explicitModel = environment.NATLANG_MODEL_PATH ? resolve(environment.NATLANG_MODEL_PATH) : null;
  const checkoutModel = resolve('models', DEFAULT_LOCAL_MODEL.file);
  const cachedModel = join(defaultNatlangCacheDirectory(environment), 'models',
    `${DEFAULT_LOCAL_MODEL.sha256.slice(0, 12)}-${DEFAULT_LOCAL_MODEL.file}`);
  const modelPath = [explicitModel, checkoutModel, cachedModel].find((value): value is string => Boolean(value && existsSync(value))) ?? null;
  const template = templateCandidates(environment).find(existsSync) ?? null;
  const modelAvailable = Boolean(modelPath || DEFAULT_LOCAL_MODEL.downloadUrl);
  return { executable: found, available: Boolean(found && modelAvailable && template), command,
    model: DEFAULT_LOCAL_MODEL.id, modelPath, template, downloadable: Boolean(DEFAULT_LOCAL_MODEL.downloadUrl) };
}

async function fileHash(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

async function verifiedDefault(path: string): Promise<boolean> {
  if (!existsSync(path) || statSync(path).size !== DEFAULT_LOCAL_MODEL.bytes) return false;
  return await fileHash(path) === DEFAULT_LOCAL_MODEL.sha256;
}

async function ensureModel(environment: NodeJS.ProcessEnv, error: NodeJS.WritableStream): Promise<string> {
  if (environment.NATLANG_MODEL_PATH) {
    const path = resolve(environment.NATLANG_MODEL_PATH);
    if (!existsSync(path)) throw new Error(`NATLANG_MODEL_PATH does not exist: ${path}`);
    return path;
  }
  const checkout = resolve('models', DEFAULT_LOCAL_MODEL.file);
  if (await verifiedDefault(checkout)) return checkout;
  const directory = join(defaultNatlangCacheDirectory(environment), 'models');
  const destination = join(directory, `${DEFAULT_LOCAL_MODEL.sha256.slice(0, 12)}-${DEFAULT_LOCAL_MODEL.file}`);
  if (await verifiedDefault(destination)) return destination;
  mkdirSync(directory, { recursive: true });
  const temporary = `${destination}.partial-${process.pid}`;
  if (!DEFAULT_LOCAL_MODEL.downloadUrl) throw new Error(`default model ${DEFAULT_LOCAL_MODEL.id} is not installed locally and has no distribution URL`);
  error.write(`natlang: downloading default model ${DEFAULT_LOCAL_MODEL.id} (${Math.ceil(DEFAULT_LOCAL_MODEL.bytes / 1_000_000)} MB)\n`);
  try {
    const response = await fetch(DEFAULT_LOCAL_MODEL.downloadUrl, { redirect: 'follow' });
    if (!response.ok || !response.body) throw new Error(`model download failed: HTTP ${response.status}`);
    await pipeline(Readable.from(response.body as AsyncIterable<Uint8Array>),
      createWriteStream(temporary, { flags: 'wx' }));
    if (!await verifiedDefault(temporary)) throw new Error('downloaded default model failed its size or SHA-256 check');
    try { renameSync(temporary, destination); }
    catch (failure) { if (!await verifiedDefault(destination)) throw failure; }
    return destination;
  } finally { rmSync(temporary, { force: true }); }
}

function templateCandidates(environment: NodeJS.ProcessEnv): string[] {
  return [environment.NATLANG_TEMPLATE && resolve(environment.NATLANG_TEMPLATE),
    resolve('models', 'templates', DEFAULT_LOCAL_MODEL.template),
    fileURLToPath(new URL('../../model-assets/default.jinja', import.meta.url))]
    .filter((value): value is string => Boolean(value));
}

function templatePath(environment: NodeJS.ProcessEnv): string {
  const selected = templateCandidates(environment).find(existsSync);
  if (!selected) throw new Error(`missing the template for default model ${DEFAULT_LOCAL_MODEL.id}`);
  return selected;
}

async function freePort(): Promise<number> {
  return await new Promise((resolvePort, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') { server.close(); reject(new Error('could not allocate a local model port')); return; }
      server.close(error => error ? reject(error) : resolvePort(address.port));
    });
  });
}

async function stopChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise<void>(resolveExit => child.once('exit', () => resolveExit()));
  child.kill('SIGTERM');
  let timer: NodeJS.Timeout | undefined;
  await Promise.race([exited, new Promise<void>(resolveTimeout => { timer = setTimeout(resolveTimeout, 5000); })]);
  if (timer) clearTimeout(timer);
  if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
}

export function createManagedModelSession(profile: ModelProfile,
  environment: NodeJS.ProcessEnv = process.env, error: NodeJS.WritableStream = process.stderr): ManagedModelSession {
  const external = profile.endpoint ? { endpoint: profile.endpoint,
    model: profile.model ?? DEFAULT_LOCAL_MODEL.id } : null;
  if (!profile.endpoint && profile.model)
    throw new Error('a configured model ID needs an endpoint; omit both to use the managed local default');
  let child: ChildProcess | null = null, local: OpenAICompatibleOptions | null = null;
  let starting: Promise<OpenAICompatibleOptions> | null = null, recentError = '';
  const prerequisites = localModelPrerequisites(environment);

  const start = async (): Promise<OpenAICompatibleOptions> => {
    if (external) return { ...profile, endpoint: external.endpoint, model: external.model,
      apiKey: environment[profile.apiKeyEnv ?? 'NATLANG_API_KEY'], toolAliases: { call: 'call_function' } };
    if (local) return local;
    if (starting) return starting;
    starting = (async () => {
      if (!prerequisites.executable) throw new Error(`managed model server needs llama-server; install llama.cpp or set NATLANG_LLAMA_SERVER`);
      const modelPath = await ensureModel(environment, error), port = await freePort();
      const endpoint = `http://127.0.0.1:${port}`;
      const args = ['-m', modelPath, '--host', '127.0.0.1', '--port', String(port), '--parallel', '1',
        '-c', String(DEFAULT_LOCAL_MODEL.contextTokens), '-ngl', '99', '--cache-ram', '256', '--no-webui',
        '--jinja', '--chat-template-file', templatePath(environment)];
      error.write(`natlang: starting managed model ${basename(modelPath)}\n`);
      child = spawn(prerequisites.executable, args, { stdio: ['ignore', 'ignore', 'pipe'] });
      child.stderr?.on('data', chunk => { recentError = (recentError + String(chunk)).slice(-16000); });
      child.once('error', failure => { recentError = `${recentError}\n${failure.message}`.slice(-16000); });
      const deadline = Date.now() + Number(environment.NATLANG_MODEL_START_TIMEOUT_MS ?? 120000);
      while (Date.now() < deadline) {
        if (child.exitCode !== null || child.signalCode !== null)
          throw new Error(`managed model server exited during startup\n${recentError.trim()}`);
        try { const response = await fetch(endpoint + '/health'); if (response.ok) {
          local = { endpoint, model: environment.NATLANG_MODEL ?? DEFAULT_LOCAL_MODEL.id,
            toolAliases: { call: 'call_function' } }; return local;
        } } catch { /* server is still loading */ }
        await new Promise(resolveWait => setTimeout(resolveWait, 250));
      }
      await stopChild(child);
      throw new Error(`managed model server did not become ready within the startup timeout\n${recentError.trim()}`);
    })();
    try { return await starting; }
    catch (failure) { if (child) await stopChild(child); child = null; throw failure; }
    finally { starting = null; }
  };

  const terminate = (signal: NodeJS.Signals) => {
    child?.kill(signal);
    process.removeListener(signal, terminate);
    process.kill(process.pid, signal);
  };
  process.once('SIGINT', terminate); process.once('SIGTERM', terminate);
  const onExit = () => { if (child && child.exitCode === null && child.signalCode === null) child.kill('SIGTERM'); };
  process.once('exit', onExit);
  return {
    async turn(request) { const options = await start(); return openAICompatibleModelTurn(options)(request); },
    status() { return external ? { source: 'external', endpoint: external.endpoint, model: external.model,
      executable: null, modelPath: null, running: false } : { source: 'managed-local', endpoint: local?.endpoint ?? null,
      model: environment.NATLANG_MODEL ?? DEFAULT_LOCAL_MODEL.id, executable: prerequisites.executable,
      modelPath: environment.NATLANG_MODEL_PATH ? resolve(environment.NATLANG_MODEL_PATH) : null, running: Boolean(child) }; },
    async close() { process.removeListener('SIGINT', terminate); process.removeListener('SIGTERM', terminate);
      process.removeListener('exit', onExit); if (child) await stopChild(child); child = null; local = null; },
  };
}
