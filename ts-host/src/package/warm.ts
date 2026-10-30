/**
 * The build lock of a compiled program, and the record of a warm prompt prefix.
 *
 * A compiled program carries what it runs with: the model it was built for (stock release or profile, request
 * settings, llama.cpp runtime), its active tuning release, its functions, and the result of an ahead-of-time tracing
 * run (`natlang.json` `warmup`, `natlang build --warm`). The warm run fills the managed server's saved prompt openings
 * (llama/warm.ts) and records which openings its program sends.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

export const BUILD_LOCK_SCHEMA = 'natlang.build-lock/1';

export type WarmPrefix = { sha256: string; chars: number; tokens: number; bytes: number; file: string; restoredSlots: number; created: boolean };
export type WarmCaseResult = { name: string; function: string; outcome: string; detail?: string; wall_ms: number;
  calls: number; tokens_in: number; tokens_out: number; call_id: string | null };
export type BuildLock = {
  schema: typeof BUILD_LOCK_SCHEMA;
  program: { name: string | null; version: string | null; root: string };
  builtAt: string; natlang: string;
  model: { profile: string; kind: string; model: string; executor: string;
    release: { id: string; sha256: string; file: string; quant: string; source: string; revision: string; template: string } | null;
    request: Record<string, unknown> | null; runtime: { version: string; build: number; label: string } | null };
  tuning: { release: string; base: string } | null;
  functions: { name: string; source: string; params: string[]; returns: string; description?: string }[];
  warm: { at: string; modelSha256: string | null; runtimeBuild: number | null;
    /** The llama.cpp binary's `b<build>g<commit>` label (llama/models.ts runtimeLabel). */
    runtime: string | null; prefixes: WarmPrefix[];
    cases: WarmCaseResult[]; wall_ms: number } | null;
};

export const buildLockPath = (buildDirectory: string) => join(buildDirectory, 'natlang.lock.json');

export function readBuildLock(buildDirectory: string): BuildLock | undefined {
  const path = buildLockPath(buildDirectory);
  if (!existsSync(path)) return undefined;
  try {
    const value = JSON.parse(readFileSync(path, 'utf8')) as BuildLock;
    return value.schema === BUILD_LOCK_SCHEMA ? value : undefined;
  } catch { return undefined; }
}

export function writeBuildLock(buildDirectory: string, lock: BuildLock): string {
  const path = buildLockPath(buildDirectory);
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.tmp`;
  writeFileSync(temporary, JSON.stringify(lock, null, 2) + '\n'); renameSync(temporary, path);
  return path;
}
