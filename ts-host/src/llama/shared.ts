/**
 * Shared managed model servers: one llama-server per model file, build and setting on the machine, whichever natlang
 * process needs it first. A server is run by a small detached supervisor (llama/supervisor.ts), not by the process that
 * started it, so it outlives that process: every session that uses it holds a lease (a file named by its pid), and the
 * supervisor stops the server once no live lease has remained for the idle time (NATLANG_SHARED_SERVER_IDLE_S, default
 * 300 s) and its slots are idle. The first process to need a server takes the start lock and starts the supervisor; the
 * others wait for its record. NATLANG_SHARED_SERVER=off gives a session a private server that stops with it.
 *
 * Files under `<state>/model-servers/`: `<key>.json` the published server, `<key>.lock` the start lock (its starter's
 * pid), `<key>.failed` why the last start failed, `<key>.log` the server's output, `<key>.leases/<pid>-<n>` the leases.
 */
import { createHash } from 'node:crypto';
import { closeSync, existsSync, mkdirSync, openSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync, writeSync } from 'node:fs';
import { join } from 'node:path';
import type { LlamaHost } from './host.js';

export type SharedServer = { schema: 'natlang.shared-model-server/2'; key: string;
  /** The supervisor's pid: stopping it stops the server. */
  pid: number; serverPid: number; endpoint: string; modelSha256: string; runtimeBuild: number | null;
  /** The binary's `b<build>g<commit>` label (llama/models.ts runtimeLabel). */
  runtime: string | null; slots: number; contextPerSlot: number; model: string; startedAt: string;
  /** The server saves and restores warm prompt openings (it was started with their --slot-save-path). */
  warm: boolean;
  /** Stop after this many idle seconds without a lease. */
  idleSeconds: number };

/** What the supervisor runs and publishes (the record without its pids and start time). */
export type SupervisedServer = { command: string; args: readonly string[]; endpoint: string; directory: string; key: string;
  startTimeoutMs: number; record: Omit<SharedServer, 'schema' | 'key' | 'pid' | 'serverPid' | 'startedAt'> };

export const sharedServersDirectory = (host: LlamaHost) => join(host.directories.state, 'model-servers');
export const sharingEnabled = (environment: NodeJS.ProcessEnv) => environment.NATLANG_SHARED_SERVER !== 'off';
export const sharedIdleSeconds = (environment: NodeJS.ProcessEnv) => {
  const value = Number(environment.NATLANG_SHARED_SERVER_IDLE_S);
  return Number.isFinite(value) && value >= 0 && environment.NATLANG_SHARED_SERVER_IDLE_S !== '' ? value : 300;
};

/** Everything that makes two servers interchangeable. The slot count is the first starter's (planned from free memory)
 * unless the settings fix it, so it is part of the key only when configured. */
export function sharedServerKey(identity: Record<string, unknown>): string {
  return createHash('sha256').update(JSON.stringify(identity)).digest('hex').slice(0, 16);
}

export const alive = (pid: number) => {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM'; }
};

export const sharedPaths = (directory: string, key: string) => ({ record: join(directory, `${key}.json`), lock: join(directory, `${key}.lock`),
  failed: join(directory, `${key}.failed`), log: join(directory, `${key}.log`), leases: join(directory, `${key}.leases`) });

export function readSharedServerAt(path: string): SharedServer | undefined {
  try {
    const value = JSON.parse(readFileSync(path, 'utf8')) as SharedServer;
    return value.schema === 'natlang.shared-model-server/2' ? value : undefined;
  } catch { return undefined; }
}

export function writeSharedServer(directory: string, record: SharedServer): void {
  const path = sharedPaths(directory, record.key).record, temporary = `${path}.${process.pid}.tmp`;
  mkdirSync(directory, { recursive: true });
  writeFileSync(temporary, JSON.stringify(record, null, 2) + '\n'); renameSync(temporary, path);
}

/** Live leases of a server (a dead holder's lease is removed). */
export function liveLeases(directory: string, key: string): number {
  const leases = sharedPaths(directory, key).leases;
  let count = 0;
  for (const name of existsSync(leases) ? readdirSync(leases) : []) {
    const pid = Number(name.split('-')[0]);
    if (Number.isSafeInteger(pid) && pid > 0 && alive(pid)) count++;
    else rmSync(join(leases, name), { force: true });
  }
  return count;
}

/** The published servers whose supervisor is alive (a dead one's record is removed). */
export function listSharedServers(host: LlamaHost): SharedServer[] {
  const directory = sharedServersDirectory(host);
  if (!existsSync(directory)) return [];
  return readdirSync(directory).filter(name => /^[0-9a-f]{16}\.json$/.test(name)).flatMap(name => {
    const record = readSharedServerAt(join(directory, name));
    if (record && alive(record.pid)) return [record];
    rmSync(join(directory, name), { force: true });
    return [];
  });
}

/** A live published server for `key` that answers its health check. */
async function healthy(host: LlamaHost, directory: string, key: string): Promise<SharedServer | undefined> {
  const record = readSharedServerAt(sharedPaths(directory, key).record);
  if (!record || !alive(record.pid)) return undefined;
  try { return (await host.fetch(record.endpoint + '/health')).ok ? record : undefined; } catch { return undefined; }
}

let leaseCounter = 0;
/** Take a lease on `key`'s server; the returned function gives it back. Taken before the record is read, so the
 * supervisor never stops a server between a session finding it and leasing it. */
function takeLease(directory: string, key: string): () => void {
  const leases = sharedPaths(directory, key).leases;
  mkdirSync(leases, { recursive: true });
  const path = join(leases, `${process.pid}-${++leaseCounter}`);
  writeFileSync(path, '');
  let held = true;
  const release = () => { if (held) { held = false; rmSync(path, { force: true }); process.removeListener('exit', release); } };
  process.once('exit', release);
  return release;
}

/** The start lock: created exclusively, holding its taker's pid; a dead taker's lock is broken. */
function tryLock(path: string): boolean {
  for (let attempt = 0; attempt < 2; attempt++) {
    try { const fd = openSync(path, 'wx'); writeSync(fd, String(process.pid)); closeSync(fd); return true; }
    catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      const holder = Number(readFileSync(path, 'utf8').trim());
      if (Number.isSafeInteger(holder) && holder > 0 && alive(holder)) return false;
      rmSync(path, { force: true });
    }
  }
  return false;
}

export type SharedLease = { server: SharedServer; started: boolean; release(): void };

/**
 * Use the shared server for `key`: the published one, or, when none runs, start it (this process takes the start lock
 * and asks the host to supervise `spec()`, which may first download the model) or wait while another process starts
 * it. Resolves once it answers, holding a lease; null when none runs and this host cannot start one. Throws when the
 * start fails, with the server's output. The supervisor bounds the server's own startup (`startTimeoutMs`).
 */
export async function acquireSharedServer(host: LlamaHost, key: string, spec: () => Promise<SupervisedServer>,
  options: { signal?: AbortSignal } = {}): Promise<SharedLease | null> {
  const directory = sharedServersDirectory(host), paths = sharedPaths(directory, key);
  mkdirSync(directory, { recursive: true });
  const release = takeLease(directory, key);
  let locked = false;
  try {
    let supervisor: number | undefined;
    for (;;) {
      options.signal?.throwIfAborted();
      const server = await healthy(host, directory, key);
      if (server) return { server, started: supervisor !== undefined, release };
      if (supervisor === undefined) {
        if (!host.supervise) { release(); return null; }
        if (tryLock(paths.lock)) {
          locked = true;
          rmSync(paths.failed, { force: true });
          supervisor = host.supervise(await spec());
          locked = false; // the supervisor removes the lock once its server answers or fails
        }
      } else if (existsSync(paths.failed)) {
        throw new Error(readFileSync(paths.failed, 'utf8').trim() || 'managed model server failed to start');
      } else if (!alive(supervisor) && !existsSync(paths.record)) {
        rmSync(paths.lock, { force: true });
        let output = '';
        try { output = readFileSync(paths.log, 'utf8').slice(-4000); } catch { /* no log */ }
        throw new Error(`managed model server exited during startup\n${output.trim()}`);
      }
      await new Promise(resolveWait => setTimeout(resolveWait, 200));
    }
  } catch (error) {
    if (locked) rmSync(paths.lock, { force: true });
    release(); throw error;
  }
}

/** Stop a shared server now (its supervisor stops llama-server and removes its record). */
export function stopSharedServer(server: SharedServer): void {
  try { process.kill(server.pid, 'SIGTERM'); } catch { /* already gone */ }
}
