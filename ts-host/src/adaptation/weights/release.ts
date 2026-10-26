/**
 * Tuned releases on disk (plans/HIERARCHICAL_LORA_TUNING.md §6, §8). Node only.
 *
 *   <program>/.natlang/tune/releases/<digest>/natlang.adapters.json   immutable, content-addressed
 *   <program>/.natlang/tune/active.json                               the active pointer, replaced atomically
 *   <program>/natlang.adapters.json                                   a release carried by a package (pinned)
 *
 * Activation happens only after the backend acknowledged the exact artifacts and a normal call passed the readiness
 * probe; both receipts are part of the pointer. The previous release stays on disk for rollback. Discovery is
 * automatic: `run`, `call` and SDK runtimes read the active release at each task creation.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync, openSync, fsyncSync, closeSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { canonical } from '../identity.js';
import type { ProgramDescriptor } from '../types.js';
import type { NativeModelIdentity } from '../../native/neuralese-store.js';
import { parseManifest, validateManifest, type WeightManifest } from './manifest.js';
import { WeightBinding, type WeightStatus } from './binding.js';

export const ACTIVE_SCHEMA = 'natlang.tune-active/v1' as const;
export type ActiveRecord = {
  readonly schema: typeof ACTIVE_SCHEMA; readonly release: string; readonly previous: string | null;
  readonly activatedAt: string;
  /** What the serving backend reported after loading the exact artifacts. */
  readonly backend: { readonly kind: string; readonly loaded: readonly string[]; readonly identity: NativeModelIdentity };
  /** The normal-call readiness probe against a replayable test world. */
  readonly readiness: { readonly definition: string; readonly ok: true; readonly effectiveIdentity: string; readonly ms: number };
};

export const tuneDirectory = (root: string) => join(root, '.natlang', 'tune');
const releaseFile = (root: string, digest: string) => join(tuneDirectory(root), 'releases', digest, 'natlang.adapters.json');
const activeFile = (root: string) => join(tuneDirectory(root), 'active.json');
const jobLock = (root: string) => join(tuneDirectory(root), 'job.lock');

function atomicWrite(path: string, text: string): void {
  mkdirSync(dirname(path), { recursive: true });
  const temporary = `${path}.${process.pid}.${Date.now()}.tmp`;
  writeFileSync(temporary, text);
  const fd = openSync(temporary, 'r'); try { fsyncSync(fd); } finally { closeSync(fd); }
  renameSync(temporary, path);
}

/** Store a release immutably. Storing the same digest again is a no-op; different content under it is an error. */
export function storeRelease(root: string, manifest: WeightManifest): string {
  validateManifest(manifest);
  const path = releaseFile(root, manifest.digest), text = canonical(manifest) + '\n';
  if (existsSync(path)) {
    if (readFileSync(path, 'utf8') !== text) throw new Error(`release ${manifest.digest} exists with different content`);
    return path;
  }
  atomicWrite(path, text);
  return path;
}

export function readRelease(root: string, digest: string): WeightManifest {
  const manifest = parseManifest(readFileSync(releaseFile(root, digest), 'utf8'));
  if (manifest.digest !== digest) throw new Error(`release directory ${digest} holds ${manifest.digest}`);
  return manifest;
}

export function readActive(root: string): ActiveRecord | undefined {
  const path = activeFile(root);
  if (!existsSync(path)) return undefined;
  const record = JSON.parse(readFileSync(path, 'utf8')) as ActiveRecord;
  if (record.schema !== ACTIVE_SCHEMA || typeof record.release !== 'string') throw new Error(`${path} is not a ${ACTIVE_SCHEMA} record`);
  return record;
}

/**
 * Activate a stored release. `expectedCurrent` is compared before the swap, so an older build finishing late cannot
 * overwrite a newer activation (watch mode, concurrent tunes).
 */
export function activateRelease(root: string, digest: string, receipts: Pick<ActiveRecord, 'backend' | 'readiness'>,
  expectedCurrent: string | null): ActiveRecord {
  readRelease(root, digest);
  if (!receipts.readiness?.ok) throw new Error('a release is activated only after its readiness probe passed');
  const current = readActive(root)?.release ?? null;
  if (current !== expectedCurrent)
    throw new Error(`active release changed while this build ran (expected ${expectedCurrent ?? 'none'}, found ${current ?? 'none'}); rerun natlang tune`);
  const record: ActiveRecord = { schema: ACTIVE_SCHEMA, release: digest, previous: current, activatedAt: new Date().toISOString(), ...receipts };
  atomicWrite(activeFile(root), JSON.stringify(record, null, 2) + '\n');
  return record;
}

/** Point back at the previous coherent release. Nothing under an existing release ID is mutated. */
export function rollbackRelease(root: string): ActiveRecord {
  const active = readActive(root);
  if (!active?.previous) throw new Error('no previous release to roll back to');
  readRelease(root, active.previous);
  const record: ActiveRecord = { ...active, release: active.previous, previous: active.release, activatedAt: new Date().toISOString() };
  atomicWrite(activeFile(root), JSON.stringify(record, null, 2) + '\n');
  return record;
}

/** The release that ordinary execution of the program at `root` uses: the active one, else a packaged pin. */
export function currentManifest(root: string): WeightManifest | undefined {
  const active = readActive(root);
  if (active) return readRelease(root, active.release);
  const packaged = join(root, 'natlang.adapters.json');
  return existsSync(packaged) ? parseManifest(readFileSync(packaged, 'utf8')) : undefined;
}

export type TuneStatus = { status: 'untuned' | 'tuning' | WeightStatus; release?: string; reason?: string };

/** Author-facing status of a program, as `natlang tune --report` and diagnostics show it. */
export function tuneStatus(root: string, options: { program?: ProgramDescriptor; executor?: NativeModelIdentity } = {}): TuneStatus {
  const manifest = currentManifest(root);
  const tuning = existsSync(jobLock(root));
  if (!manifest) return tuning ? { status: 'tuning', reason: 'a tune job is running; calls use the base model until it activates' }
    : { status: 'untuned', reason: 'run `natlang tune` to specialize this program' };
  const binding = new WeightBinding(manifest, options);
  return { status: binding.status, release: manifest.digest, ...(binding.reason ? { reason: binding.reason } : {}) };
}

/**
 * Weight releases for a runtime, reread when the active pointer changes. Pass it as `weights` to
 * `createNatlangRuntime`: every new task pins the release active at its creation.
 */
export function activeWeights(root: string, options: Omit<ConstructorParameters<typeof WeightBinding>[1] & object, 'available'> = {}):
  () => readonly WeightBinding[] {
  let seen: string | undefined, current: readonly WeightBinding[] = [];
  const stamp = () => {
    const paths = [activeFile(root), join(root, 'natlang.adapters.json')];
    return paths.map(path => { try { const stat = statSync(path); return `${stat.mtimeMs}:${stat.size}:${stat.ino}`; } catch { return '-'; } }).join('|');
  };
  return () => {
    const next = stamp();
    if (next !== seen) {
      const manifest = currentManifest(root);
      const active = readActive(root);
      current = manifest ? [new WeightBinding(manifest, { ...options, ...(active ? { available: new Set(active.backend.loaded) } : {}) })] : [];
      seen = next;
    }
    return current;
  };
}

export { jobLock as tuneJobLock };
