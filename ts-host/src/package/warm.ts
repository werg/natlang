/**
 * The build lock and pre-warmed prompt prefixes of a compiled program (`natlang build [--warm]`).
 *
 * A compiled program carries what it runs with: the model it was built for (stock release or profile, request
 * settings, llama.cpp runtime), its active tuning release, its functions, and the result of an ahead-of-time tracing
 * run (`natlang.json` `warmup`). Every call of a natural-language function opens with the same long system prompt and
 * tool list; the warm run records which prefixes the program's calls send, and the managed llama.cpp server prefills
 * each one once and saves its slot state beside the build. `natlang run` restores those states into every slot before
 * the first call, so the first calls skip the prefill (minutes on a CPU, the bulk of a short call anywhere).
 *
 * Mechanism only; nothing here decides policy. States are bound to the exact model file and llama.cpp build and are
 * ignored, with a notice, when either differs.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

export const BUILD_LOCK_SCHEMA = 'natlang.build-lock/1';

export type WarmPrefix = { sha256: string; chars: number; tokens: number; requests: number; file: string; bytes: number };
export type WarmCaseResult = { name: string; function: string; outcome: string; detail?: string; wall_ms: number;
  calls: number; tokens_in: number; tokens_out: number; call_id: string | null };
export type BuildLock = {
  schema: typeof BUILD_LOCK_SCHEMA;
  program: { name: string | null; version: string | null; root: string };
  builtAt: string; natlang: string;
  model: { profile: string; kind: string; model: string; executor: string;
    release: { id: string; sha256: string; file: string; quant: string; source: string; revision: string; template: string } | null;
    request: Record<string, unknown> | null; runtime: { version: string; build: number } | null };
  tuning: { release: string; base: string } | null;
  functions: { name: string; source: string; params: string[]; returns: string; description?: string }[];
  warm: { at: string; modelSha256: string; runtimeBuild: number; directory: string; prefixes: WarmPrefix[];
    cases: WarmCaseResult[]; wall_ms: number } | null;
};

export const buildLockPath = (buildDirectory: string) => join(buildDirectory, 'natlang.lock.json');
export const warmDirectory = (buildDirectory: string) => join(buildDirectory, 'warm');

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

/** The warm states of a lock that the given model file and llama.cpp build can restore, as files in the build. */
export function usableWarmStates(buildDirectory: string, lock: BuildLock | undefined, modelSha256: string | undefined,
  runtimeBuild: number | null | undefined): { usable: { id: string; path: string }[]; reason?: string } {
  if (!lock?.warm?.prefixes.length) return { usable: [] };
  if (modelSha256 !== lock.warm.modelSha256) return { usable: [], reason: 'its warm prefixes were prefilled for another model file' };
  if (runtimeBuild !== lock.warm.runtimeBuild) return { usable: [], reason: `its warm prefixes were prefilled by llama.cpp build ${lock.warm.runtimeBuild}` };
  const directory = warmDirectory(buildDirectory);
  const usable = lock.warm.prefixes.map(prefix => ({ id: prefix.sha256, path: join(directory, prefix.file) }))
    .filter(item => existsSync(item.path));
  return usable.length ? { usable } : { usable, reason: 'its warm prefix files are missing' };
}

/** Distinct (system prompt, tools) openings of the requests a warm run sent, most frequent first. */
export function requestPrefixes(requests: readonly { messages: readonly unknown[]; tools?: readonly unknown[] }[]):
  { system: Record<string, unknown>; tools: unknown[]; requests: number; key: string }[] {
  const seen = new Map<string, { system: Record<string, unknown>; tools: unknown[]; requests: number; key: string }>();
  for (const request of requests) {
    const system = request.messages[0] as Record<string, unknown> | undefined;
    if (!system || system.role !== 'system' || typeof system.content !== 'string') continue;
    const tools = [...(request.tools ?? [])];
    const key = createHash('sha256').update(JSON.stringify([system.content, tools])).digest('hex');
    const entry = seen.get(key) ?? { system: { role: 'system', content: system.content }, tools, requests: 0, key };
    entry.requests++; seen.set(key, entry);
  }
  return [...seen.values()].sort((a, b) => b.requests - a.requests);
}

/**
 * The longest text every conversation with this opening starts with, cut just after a special token so that the rest
 * of any prompt tokenizes on from it (a hybrid model's recurrent state cannot be rolled back by a token).
 */
export function sharedPrefix(first: string, second: string): string {
  let length = 0;
  while (length < first.length && length < second.length && first[length] === second[length]) length++;
  const common = first.slice(0, length);
  const special = /<\|[^|<>\s]{1,40}\|>|<\/?[a-z_]{2,24}>/g;
  let cut = 0;
  for (const match of common.matchAll(special)) cut = match.index! + match[0].length;
  return common.slice(0, cut);
}

async function post(endpoint: string, path: string, body: unknown, signal?: AbortSignal): Promise<Record<string, unknown>> {
  const response = await fetch(endpoint + path, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body), ...(signal ? { signal } : {}) });
  const text = await response.text();
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status} ${text.slice(0, 400)}`);
  return text ? JSON.parse(text) as Record<string, unknown> : {};
}

/**
 * Prefill each opening once on slot 0 of a llama.cpp server started with `--slot-save-path directory`, save the slot
 * state as a file there, and report what was saved. Openings that render no shared prefix are skipped.
 */
export async function prefillPrefixes(endpoint: string, directory: string,
  openings: ReturnType<typeof requestPrefixes>, tag: string, signal?: AbortSignal): Promise<WarmPrefix[]> {
  const saved: WarmPrefix[] = [];
  for (const opening of openings) {
    const render = async (content: string) => String((await post(endpoint, '/apply-template',
      { messages: [opening.system, { role: 'user', content }], ...(opening.tools.length ? { tools: opening.tools } : {}) }, signal)).prompt ?? '');
    const prefix = sharedPrefix(await render('⁣A'), await render('⁣B'));
    if (prefix.length < 64) continue;
    const sha256 = createHash('sha256').update(prefix).digest('hex');
    await post(endpoint, '/slots/0?action=erase', {}, signal).catch(() => undefined);
    const completion = await post(endpoint, '/completion', { prompt: prefix, n_predict: 0, cache_prompt: true, id_slot: 0 }, signal);
    const tokens = Number((completion.timings as Record<string, unknown> | undefined)?.prompt_n ?? completion.tokens_evaluated ?? 0);
    const file = `${sha256.slice(0, 16)}-${tag}.slot`;
    await post(endpoint, '/slots/0?action=save', { filename: file }, signal);
    saved.push({ sha256, chars: prefix.length, tokens, requests: opening.requests, file, bytes: statSync(join(directory, file)).size });
  }
  return saved;
}

/** Restore saved states into every slot of a server started with `--slot-save-path`; returns how many slots took one. */
export async function restorePrefixes(endpoint: string, states: readonly { path: string }[], slots: number,
  log?: (line: string) => void): Promise<number> {
  let restored = 0;
  for (let slot = 0; slot < slots && states.length; slot++) {
    const state = states[slot % states.length]!;
    try { await post(endpoint, `/slots/${slot}?action=restore`, { filename: state.path.split(/[\\/]/).pop() }); restored++; }
    catch (error) { log?.(`natlang: could not restore a warm prefix into slot ${slot}: ${(error as Error).message.slice(0, 200)}`); break; }
  }
  return restored;
}
