/**
 * Warm prompt openings of the in-page engine, kept across page reloads (docs/ENGINE_PARITY.md).
 *
 * The page does what the managed server does across restarts (llama/warm.ts): the first time a load serves a
 * function's opening (its system prompt and tools, model/warm-opening.ts), the opening is prefilled on its own in one
 * slot and that slot's state (KV cache, tokens and context checkpoints, the fork's slot save) is written to the
 * origin-private file system; a later load of the same model, engine build and settings restores it into a slot
 * instead of prefilling the opening again. The next request that starts with the opening continues from it.
 *
 * A state is found only under its exact key: the opening's SHA-256 and a tag naming everything that makes a slot state
 * restorable and equal (`warmTag`: model SHA-256, engine build, worker build, context size, slots, GPU or CPU, full
 * sliding-window cache, chat template). A stored file repeats its key and the SHA-256 of its state, so a truncated, corrupted or foreign file is
 * refused and removed. Storage is a convenience: every read, write and restore failure (no OPFS, private mode, quota,
 * a state the engine refuses) leaves the page on its cold path, silently.
 *
 * Mechanism only: which openings to warm is the page's request stream, nothing here decides policy.
 */
import { OPENING_PROBES, openingParts, sharedPrefix } from '../model/warm-opening.js';

/** The engine calls a warm opening needs (the vendored wllama with `capabilities.slotState`). */
export type WarmSlotEngine = {
  applyTemplate(options: Record<string, unknown>): Promise<string>;
  createCompletion(request: Record<string, unknown>): Promise<unknown>;
  saveSlot(idSlot: number): Promise<{ state: Uint8Array; n_saved: number }>;
  restoreSlot(idSlot: number, state: Uint8Array | Blob): Promise<{ n_restored: number }>;
  eraseSlot(idSlot: number): Promise<unknown>;
};

/** Where warm states live: named binary files. Every method may throw; callers treat that as "no state". */
export type WarmStateStore = {
  read(name: string): Promise<Blob | undefined>;
  write(name: string, data: Blob): Promise<void>;
  remove(name: string): Promise<void>;
  list(): Promise<{ name: string; lastModified: number }[]>;
};

/** One opening a load warmed: restored from storage, or prefilled (and saved when `saved`). */
export type PageWarmOpening = { sha256: string; restored: boolean; tokens: number; bytes: number; ms: number; slot: number; saved?: boolean };

const SCHEMA = 'natlang.page-warm-opening/1';
const MAGIC = 0x4d52574e; // "NWRM"
/** States kept per origin (each is the opening's KV cache, a few to tens of MB). */
export const WARM_STATES_KEPT = 6;

export const hexDigest = async (data: Uint8Array) =>
  [...new Uint8Array(await crypto.subtle.digest('SHA-256', data as Uint8Array<ArrayBuffer>))].map(byte => byte.toString(16).padStart(2, '0')).join('');
const textDigest = (text: string) => hexDigest(new TextEncoder().encode(text));

/** What makes a saved slot state restorable into this load and equal to a fresh prefill. */
export type WarmTagParts = { modelSha256: string; engineBuild: string; compat: boolean; contextTokens: number; slots: number;
  gpu: boolean; fullSlidingWindowCache: boolean; chatTemplate: string | null };
export const warmTag = (parts: WarmTagParts) => textDigest(JSON.stringify([SCHEMA, parts.modelSha256, parts.engineBuild, parts.compat,
  parts.contextTokens, parts.slots, parts.gpu, parts.fullSlidingWindowCache, parts.chatTemplate]));

/** The page's OPFS directory of warm states, or null where the page has no OPFS. */
export function opfsWarmStateStore(storage: { getDirectory?(): Promise<unknown> } | undefined =
  globalThis.navigator?.storage as unknown as { getDirectory?(): Promise<unknown> } | undefined): WarmStateStore | null {
  if (!storage?.getDirectory) return null;
  type Writable = { write(data: Blob): Promise<void>; close(): Promise<void>; abort(): Promise<void> };
  type Directory = { getFileHandle(name: string, options?: { create?: boolean }): Promise<{ getFile(): Promise<File>; createWritable(): Promise<Writable> }>;
    removeEntry(name: string): Promise<void>; keys(): AsyncIterable<string> };
  const directory = async () => (await (await storage.getDirectory!() as {
    getDirectoryHandle(name: string, options: { create: boolean }): Promise<Directory> }).getDirectoryHandle('natlang-warm-openings', { create: true }));
  return {
    async read(name) {
      try { return await (await (await directory()).getFileHandle(name)).getFile(); }
      catch (error) { if ((error as { name?: string }).name === 'NotFoundError') return undefined; throw error; }
    },
    async write(name, data) {
      // createWritable writes a swap file that replaces the target only on close: a reader never sees half a state.
      const writable = await (await (await directory()).getFileHandle(name, { create: true })).createWritable();
      try { await writable.write(data); await writable.close(); }
      catch (error) { await writable.abort().catch(() => undefined); throw error; }
    },
    async remove(name) { await (await directory()).removeEntry(name).catch(() => undefined); },
    async list() {
      const handle = await directory(), entries: { name: string; lastModified: number }[] = [];
      for await (const name of handle.keys()) {
        if (!name.endsWith('.state')) continue;
        try { entries.push({ name, lastModified: (await (await handle.getFileHandle(name)).getFile()).lastModified }); } catch { /* removed meanwhile */ }
      }
      return entries;
    },
  };
}

type Header = { schema: typeof SCHEMA; opening: string; tag: string; tokens: number; bytes: number; sha256: string };

async function encode(header: Header, state: Uint8Array): Promise<Blob> {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const prefix = new DataView(new ArrayBuffer(8));
  prefix.setUint32(0, MAGIC, true); prefix.setUint32(4, json.byteLength, true);
  return new Blob([prefix.buffer, json, state as Uint8Array<ArrayBuffer>]);
}

/** The state of a stored file, if it is exactly the state of this opening and tag. */
async function decode(file: Blob, opening: string, tag: string): Promise<{ state: Uint8Array; header: Header } | undefined> {
  if (file.size < 8) return undefined;
  const prefix = new DataView(await file.slice(0, 8).arrayBuffer());
  if (prefix.getUint32(0, true) !== MAGIC) return undefined;
  const length = prefix.getUint32(4, true);
  if (8 + length > file.size) return undefined;
  let header: Header;
  try { header = JSON.parse(new TextDecoder().decode(await file.slice(8, 8 + length).arrayBuffer())) as Header; } catch { return undefined; }
  if (header?.schema !== SCHEMA || header.opening !== opening || header.tag !== tag || header.bytes !== file.size - 8 - length) return undefined;
  const state = new Uint8Array(await file.slice(8 + length).arrayBuffer());
  return await hexDigest(state) === header.sha256 ? { state, header } : undefined;
}

/**
 * The warm openings of one loaded model: `before(request)` makes sure a request's opening was restored or prefilled
 * (once per opening and load) before the request is sent. Call it only while no other request runs on the engine.
 */
export function pageWarmOpenings(engine: WarmSlotEngine, store: WarmStateStore, tag: string, slots: number,
  onWarm?: (warm: PageWarmOpening) => void) {
  const seen = new Set<string>();
  let nextSlot = 0;
  /** Writes run behind the engine: a later request does not wait for storage. */
  let writing: Promise<void> = Promise.resolve();

  async function warm(opening: NonNullable<ReturnType<typeof openingParts>>): Promise<void> {
    const sha256 = await textDigest(opening.text), name = `${sha256.slice(0, 24)}-${tag.slice(0, 24)}.state`;
    const slot = nextSlot; nextSlot = (nextSlot + 1) % slots;
    const started = performance.now();
    let stored: Awaited<ReturnType<typeof decode>>;
    try {
      const file = await store.read(name);
      stored = file ? await decode(file, sha256, tag) : undefined;
      if (file && !stored) await store.remove(name);
    } catch { stored = undefined; }
    if (stored) {
      try {
        const restored = await engine.restoreSlot(slot, stored.state);
        onWarm?.({ sha256, restored: true, tokens: restored.n_restored, bytes: stored.state.byteLength, ms: Math.round(performance.now() - started), slot });
        return;
      } catch {
        // The engine refused it (a state that does not fit this context): drop it and prefill instead.
        await engine.eraseSlot(slot).catch(() => undefined);
        await store.remove(name).catch(() => undefined);
      }
    }
    const render = (content: string) => engine.applyTemplate({ messages: [opening.system, { role: 'user', content }],
      ...(opening.tools.length ? { tools: opening.tools } : {}) });
    const prefix = sharedPrefix(await render(OPENING_PROBES[0]), await render(OPENING_PROBES[1]));
    if (prefix.length < 256) return;
    await engine.eraseSlot(slot);
    await engine.createCompletion({ prompt: prefix, n_predict: 0, cache_prompt: true, id_slot: slot, stream: false });
    const saved = await engine.saveSlot(slot);
    const record = { sha256, restored: false, tokens: saved.n_saved, bytes: saved.state.byteLength, ms: Math.round(performance.now() - started), slot };
    const state = saved.state;
    writing = writing.then(async () => {
      try {
        await store.write(name, await encode({ schema: SCHEMA, opening: sha256, tag, tokens: saved.n_saved, bytes: state.byteLength,
          sha256: await hexDigest(state) }, state));
        const states = (await store.list()).sort((a, b) => b.lastModified - a.lastModified);
        for (const old of states.slice(WARM_STATES_KEPT)) if (old.name !== name) await store.remove(old.name);
        onWarm?.({ ...record, saved: true });
      } catch { onWarm?.({ ...record, saved: false }); }
    });
  }

  return {
    async before(request: { messages?: unknown; tools?: unknown }): Promise<void> {
      if (!Array.isArray(request.messages)) return;
      const opening = openingParts({ messages: request.messages, tools: Array.isArray(request.tools) ? request.tools : [] });
      if (!opening || seen.has(opening.text)) return;
      seen.add(opening.text);
      // A failed warm-up leaves the slot to the request's own prefill (the cold path).
      try { await warm(opening); } catch { /* cold start */ }
    },
    /** Resolves once every state saved so far is written (or failed to be). */
    flushed: () => writing,
  };
}
