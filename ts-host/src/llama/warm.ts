/**
 * The warm prompt prefixes of the managed llama.cpp server.
 *
 * Every call of a natural-language function opens with the same long system prompt and tool list (about 3,400 LFM3
 * tokens). llama.cpp reuses a hybrid model's cached prefix only from a context checkpoint at least a sliding window
 * before the point where two prompts differ, and ordinary prompt processing makes none there, so without help every
 * call prefills that opening again (minutes on a CPU). The managed server therefore prefills each distinct opening once
 * on its own, saves the slot state (whose checkpoints sit early enough) in the machine cache, and restores it into
 * every slot: the first calls of later processes start warm, and so does every later call in a warm slot. The warm
 * run of `natlang build --warm` fills that cache ahead of time and records which openings its program sends
 * (package/warm.ts).
 *
 * Openings differ mostly in their last tool: return_result, whose value schema is the function's return type, comes
 * last in the tool list, which the LFM3 template renders after the system prompt. So a new opening is warmed in two
 * levels: its base (the rendered opening without its last tool line, shared by every opening with the same system
 * prompt and other tools) is restored when saved, or prefilled and saved once; then only the rest, about 300
 * tokens of the `return_result` line, is prefilled (measured on a CPU: 3 s instead of 120 s for a new return type).
 *
 * Mechanism only; nothing here decides policy. A state is keyed by the exact opening text, model file and build.
 */
import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, rmSync, statSync, utimesSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { WarmPrefix } from '../package/warm.js';
import type { LlamaHost } from './host.js';

export type { WarmPrefix } from '../package/warm.js';

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

/** The rendered opening up to its last tool line (the shared base of openings that differ only there), if it has two
 * or more tools in a `<tools>` block. */
export function baseOf(prefix: string): string | undefined {
  const open = prefix.indexOf('<tools>\n'), close = prefix.lastIndexOf('\n</tools>');
  if (open < 0 || close < open) return undefined;
  const lastLine = prefix.lastIndexOf('\n', close - 1);
  return lastLine > open + '<tools>'.length ? prefix.slice(0, lastLine + 1) : undefined;
}

async function post(host: LlamaHost, endpoint: string, headers: Record<string, string> | undefined, path: string, body: unknown): Promise<Record<string, unknown>> {
  const response = await host.fetch(endpoint + path, { method: 'POST', headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
  const text = await response.text();
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status} ${text.slice(0, 300)}`);
  return text ? JSON.parse(text) as Record<string, unknown> : {};
}

/**
 * Make every slot of a llama.cpp server (started with `--slot-save-path directory`, all slots idle) start from this
 * opening: restore its saved state, or prefill it on slot 0 and save it first. `tag` names the model file and build.
 */
export async function warmOpening(host: LlamaHost, server: { endpoint: string; headers?: Record<string, string> }, directory: string,
  opening: Opening, tag: string, slots: number): Promise<WarmPrefix | undefined> {
  const { endpoint, headers } = server;
  const file = `${opening.key.slice(0, 16)}-${tag}.slot`, path = join(directory, file), meta = `${path}.json`;
  let created = false, info: { chars: number; tokens: number; prefilled?: number } | undefined;
  // Token IDs as the server's chat path makes them (the template carries its own BOS), so a base is an exact prefix.
  const tokenize = async (content: string) => (await post(host, endpoint, headers, '/tokenize',
    { content, add_special: false, parse_special: true })).tokens as number[];
  try { if (existsSync(path)) info = JSON.parse(readFileSync(meta, 'utf8')) as { chars: number; tokens: number }; } catch { info = undefined; }
  /** Put the opening's base into slot 0: restore its saved state, or prefill and save it. The base file's name, or undefined
   * when the opening has none; a failed restore leaves the slot empty and the opening is prefilled whole. */
  async function restoreOrSaveBase(text: string | undefined, tokens: number[]): Promise<string | undefined> {
    if (!text) return undefined;
    const own = await tokenize(text);
    let shared = 0;
    while (shared < own.length && shared < tokens.length && own[shared] === tokens[shared]) shared++;
    // The last token of the cut text may merge differently inside the whole opening: the base stops before it.
    const prefixTokens = tokens.slice(0, Math.max(0, shared - 1));
    if (prefixTokens.length < 256 || prefixTokens.length >= tokens.length) return undefined;
    const name = `${createHash('sha256').update(JSON.stringify(prefixTokens)).digest('hex').slice(0, 16)}-${tag}.base.slot`;
    if (existsSync(join(directory, name))) {
      try {
        await post(host, endpoint, headers, '/slots/0?action=restore', { filename: name });
        const now = new Date(); try { utimesSync(join(directory, name), now, now); } catch { /* recency is only for pruning */ }
        return name;
      } catch { await post(host, endpoint, headers, '/slots/0?action=erase', {}).catch(() => undefined); return undefined; }
    }
    await post(host, endpoint, headers, '/completion', { prompt: prefixTokens, n_predict: 0, cache_prompt: true, id_slot: 0 });
    await post(host, endpoint, headers, '/slots/0?action=save', { filename: name });
    return name;
  }
  let first = 0, base: string | undefined;
  if (!info) {
    const render = async (content: string) => String((await post(host, endpoint, headers, '/apply-template',
      { messages: [opening.system, { role: 'user', content }], ...(opening.tools.length ? { tools: opening.tools } : {}) })).prompt ?? '');
    const prefix = sharedPrefix(await render('⁣A'), await render('⁣B'));
    if (prefix.length < 256) return undefined;
    const tokens = await tokenize(prefix);
    await post(host, endpoint, headers, '/slots/0?action=erase', {}).catch(() => undefined);
    base = await restoreOrSaveBase(baseOf(prefix), tokens);
    const completion = await post(host, endpoint, headers, '/completion', { prompt: tokens, n_predict: 0, cache_prompt: true, id_slot: 0 });
    info = { chars: prefix.length, tokens: tokens.length,
      prefilled: Number((completion.timings as Record<string, unknown> | undefined)?.prompt_n ?? tokens.length) };
    await post(host, endpoint, headers, '/slots/0?action=save', { filename: file });
    writeFileSync(meta, JSON.stringify(info));
    created = true; first = 1;
  } else { const now = new Date(); try { utimesSync(path, now, now); } catch { /* recency is only for pruning */ } }
  let restored = first;
  for (let slot = first; slot < slots; slot++) {
    try { await post(host, endpoint, headers, `/slots/${slot}?action=restore`, { filename: file }); restored++; } catch { break; }
  }
  return { sha256: opening.key, chars: info.chars, tokens: info.tokens, bytes: statSync(path).size, file, restoredSlots: restored, created,
    ...(created && info.prefilled !== undefined ? { prefilled: info.prefilled } : {}), ...(base ? { base } : {}) };

}

/** Keep the `keep` most recently used warm states of the directory. */
export function pruneWarmStates(directory: string, keep = 24): void {
  if (!existsSync(directory)) return;
  const states = readdirSync(directory).filter(name => name.endsWith('.slot'))
    .map(name => ({ name, at: statSync(join(directory, name)).mtimeMs })).sort((a, b) => b.at - a.at);
  for (const state of states.slice(keep)) { rmSync(join(directory, state.name), { force: true }); rmSync(join(directory, `${state.name}.json`), { force: true }); }
}
