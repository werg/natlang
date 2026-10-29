/**
 * The build lock of a compiled program and the warm prompt prefixes of the managed llama.cpp server.
 *
 * A compiled program carries what it runs with: the model it was built for (stock release or profile, request
 * settings, llama.cpp runtime), its active tuning release, its functions, and the result of an ahead-of-time tracing
 * run (`natlang.json` `warmup`, `natlang build --warm`).
 *
 * Every call of a natural-language function opens with the same long system prompt and tool list (about 3,400 LFM3
 * tokens). llama.cpp reuses a hybrid model's cached prefix only from a context checkpoint at least a sliding window
 * before the point where two prompts differ, and ordinary prompt processing makes none there, so without help every
 * call prefills that opening again (minutes on a CPU). The managed server therefore prefills each distinct opening once
 * on its own, saves the slot state (whose checkpoints sit early enough) in the machine cache, and restores it into
 * every slot: the first calls of later processes start warm, and so does every later call in a warm slot. The warm
 * run of `natlang build --warm` fills that cache ahead of time and records which openings its program sends.
 *
 * Mechanism only; nothing here decides policy. A state is keyed by the exact opening text, model file and build.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, utimesSync, writeFileSync } from 'node:fs';
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
    /** The llama.cpp binary's `b<build>g<commit>` label (model/local-server.ts runtimeLabel). */
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

export type Opening = { system: { role: 'system'; content: string }; tools: unknown[]; key: string };
/** A request's opening: its system prompt and tools, keyed by their exact content. */
export function openingOf(request: { messages: readonly unknown[]; tools?: readonly unknown[] }): Opening | undefined {
  const system = request.messages[0] as { role?: unknown; content?: unknown } | undefined;
  if (system?.role !== 'system' || typeof system.content !== 'string' || system.content.length < 256) return undefined;
  const tools = [...(request.tools ?? [])];
  return { system: { role: 'system', content: system.content }, tools,
    key: createHash('sha256').update(JSON.stringify([system.content, tools])).digest('hex') };
}

/**
 * The longest text every conversation with this opening starts with, cut just after a special token so that the rest
 * of any prompt tokenizes on from it.
 */
export function sharedPrefix(first: string, second: string): string {
  let length = 0;
  while (length < first.length && length < second.length && first[length] === second[length]) length++;
  const common = first.slice(0, length);
  let cut = 0;
  for (const match of common.matchAll(/<\|[^|<>\s]{1,40}\|>|<\/?[a-z_]{2,24}>/g)) cut = match.index! + match[0].length;
  return common.slice(0, cut);
}

async function post(endpoint: string, path: string, body: unknown): Promise<Record<string, unknown>> {
  const response = await fetch(endpoint + path, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
  const text = await response.text();
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status} ${text.slice(0, 300)}`);
  return text ? JSON.parse(text) as Record<string, unknown> : {};
}

/**
 * Make every slot of a llama.cpp server (started with `--slot-save-path directory`, all slots idle) start from this
 * opening: restore its saved state, or prefill it on slot 0 and save it first. `tag` names the model file and build.
 */
export async function warmOpening(endpoint: string, directory: string, opening: Opening, tag: string, slots: number): Promise<WarmPrefix | undefined> {
  const file = `${opening.key.slice(0, 16)}-${tag}.slot`, path = join(directory, file), meta = `${path}.json`;
  let created = false, info: { chars: number; tokens: number } | undefined;
  try { if (existsSync(path)) info = JSON.parse(readFileSync(meta, 'utf8')) as { chars: number; tokens: number }; } catch { info = undefined; }
  let first = 0;
  if (!info) {
    const render = async (content: string) => String((await post(endpoint, '/apply-template',
      { messages: [opening.system, { role: 'user', content }], ...(opening.tools.length ? { tools: opening.tools } : {}) })).prompt ?? '');
    const prefix = sharedPrefix(await render('⁣A'), await render('⁣B'));
    if (prefix.length < 256) return undefined;
    await post(endpoint, '/slots/0?action=erase', {}).catch(() => undefined);
    const completion = await post(endpoint, '/completion', { prompt: prefix, n_predict: 0, cache_prompt: true, id_slot: 0 });
    info = { chars: prefix.length, tokens: Number((completion.timings as Record<string, unknown> | undefined)?.prompt_n ?? completion.tokens_evaluated ?? 0) };
    await post(endpoint, '/slots/0?action=save', { filename: file });
    writeFileSync(meta, JSON.stringify(info));
    created = true; first = 1;
  } else { const now = new Date(); try { utimesSync(path, now, now); } catch { /* recency is only for pruning */ } }
  let restored = first;
  for (let slot = first; slot < slots; slot++) {
    try { await post(endpoint, `/slots/${slot}?action=restore`, { filename: file }); restored++; } catch { break; }
  }
  return { sha256: opening.key, chars: info.chars, tokens: info.tokens, bytes: statSync(path).size, file, restoredSlots: restored, created };
}

/** Keep the `keep` most recently used warm states of the directory. */
export function pruneWarmStates(directory: string, keep = 24): void {
  if (!existsSync(directory)) return;
  const states = readdirSync(directory).filter(name => name.endsWith('.slot'))
    .map(name => ({ name, at: statSync(join(directory, name)).mtimeMs })).sort((a, b) => b.at - a.at);
  for (const state of states.slice(keep)) { rmSync(join(directory, state.name), { force: true }); rmSync(join(directory, `${state.name}.json`), { force: true }); }
}
