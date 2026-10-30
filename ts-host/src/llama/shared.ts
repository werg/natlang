/**
 * A managed llama.cpp server another natlang process keeps loaded (`natlang dashboard`'s "keep the model loaded"),
 * published in the state directory. A managed-local session whose model file and llama.cpp build match uses it instead
 * of starting its own server: no load time, the server's warm slots, and one copy of the model in memory however many
 * runs share it. NATLANG_SHARED_SERVER=off ignores it.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import type { LlamaHost } from './host.js';

export type SharedServer = { schema: 'natlang.shared-model-server/1'; pid: number; endpoint: string; modelSha256: string;
  runtimeBuild: number | null;
  /** The binary's `b<build>g<commit>` label (llama/models.ts runtimeLabel); sessions match on it. */
  runtime: string | null; slots: number; contextPerSlot: number; model: string; profile: string; startedAt: string;
  /** The server saves and restores warm prompt openings (it was started with their --slot-save-path). */
  warm: boolean };

export const sharedServerPath = (host: LlamaHost) => join(host.directories.state, 'model-server.json');

export function publishSharedServer(host: LlamaHost, server: Omit<SharedServer, 'schema'>): void {
  const path = sharedServerPath(host), temporary = `${path}.${process.pid}.tmp`;
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(temporary, JSON.stringify({ schema: 'natlang.shared-model-server/1', ...server }, null, 2) + '\n'); renameSync(temporary, path);
}

/** Remove the published server, if it is still the one `pid` published. */
export function withdrawSharedServer(host: LlamaHost, pid: number): void {
  const current = readSharedServer(host);
  if (current?.pid === pid) rmSync(sharedServerPath(host), { force: true });
}

export function readSharedServer(host: LlamaHost): SharedServer | undefined {
  const path = sharedServerPath(host);
  if (!existsSync(path)) return undefined;
  try {
    const value = JSON.parse(readFileSync(path, 'utf8')) as SharedServer;
    return value.schema === 'natlang.shared-model-server/1' ? value : undefined;
  } catch { return undefined; }
}

const alive = (pid: number) => { try { process.kill(pid, 0); return true; } catch (error) { return (error as NodeJS.ErrnoException).code === 'EPERM'; } };

/** The published server, when it serves exactly this model file on this llama.cpp binary and answers its health check. */
export async function findSharedServer(host: LlamaHost, modelSha256: string, runtime: string): Promise<SharedServer | undefined> {
  if (host.environment.NATLANG_SHARED_SERVER === 'off') return undefined;
  const server = readSharedServer(host);
  if (!server || server.pid === process.pid || server.modelSha256 !== modelSha256 || server.runtime !== runtime || !alive(server.pid)) return undefined;
  try {
    const response = await host.fetch(server.endpoint + '/health', { signal: AbortSignal.timeout(2000) });
    return response.ok ? server : undefined;
  } catch { return undefined; }
}
