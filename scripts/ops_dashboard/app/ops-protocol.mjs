// On-demand reads over the application session's live signals (browser and native share this file).
// A reader sends `ops/q` {id, q, input}; the native participant answers with ordered `ops/r` chunks {id, i, n, z} carrying
// base64(gzip(JSON)), or one {id, error}. Signals are capped at 16 KiB each (MAX_SIGNAL_BYTES), so payloads are chunked.
// `ops/tick` (latest mode) announces fresh exporter state so readers refetch only what changed.
// Signals are visible to every application reader: they carry nothing a reader may not see. Nothing here is retained:
// durable facts (edits, asks) go through the declared commands in domain.mjs.
export const TOPICS = Object.freeze({ query: 'ops/q', reply: 'ops/r', tick: 'ops/tick' });
export const CHUNK_CHARS = 12000;                 // base64 characters per reply signal (envelope stays < 16 KiB)
export const MAX_REPLY_CHUNKS = 160;              // ~1.4 MB compressed per reply; larger answers paginate
export const REPLY_TTL_MS = 30000;
/** The reference recordApplication keeps for an edit (its schema bounds strings at 128): the operation id, or its last
 * 128 characters for namespaced ids (native_<attachment>_<attempt hash>), which still contain the full attempt hash. */
export const editRef = operationId => String(operationId).length <= 128 ? String(operationId) : String(operationId).slice(-128);
export const validId = v => typeof v === 'string' && /^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$/.test(v);

const b64 = bytes => { let s = ''; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };
const unb64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
async function pipe(bytes, stream) { return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer()); }

/** JSON value -> base64 gzip chunks. */
export async function encodeReply(value) {
  const gz = await pipe(new TextEncoder().encode(JSON.stringify(value)), new CompressionStream('gzip'));
  const text = b64(gz), parts = [];
  for (let i = 0; i < text.length; i += CHUNK_CHARS) parts.push(text.slice(i, i + CHUNK_CHARS));
  return parts.length ? parts : [''];
}
export async function decodeReply(parts) {
  return JSON.parse(new TextDecoder().decode(await pipe(unb64(parts.join('')), new DecompressionStream('gzip'))));
}

/**
 * Reader side: query(name, input) -> Promise<value>. `signals` is the session's signal port ({send, subscribe}).
 * One subscription receives every reply; requests time out without retrying (the caller decides).
 */
export function createQueryClient(signals, { timeoutMs = 45000, onTick } = {}) {
  const waiting = new Map(), stop = new AbortController();
  (async () => {
    for await (const sig of signals.subscribe({ topics: [TOPICS.reply], mode: 'all', signal: stop.signal })) {
      const p = sig.payload, w = p && waiting.get(p.id);
      if (!w) continue;
      if (p.error) { waiting.delete(p.id); w.reject(Object.assign(Error(p.error), { code: p.code })); continue; }
      w.parts[p.i] = p.z; w.n = p.n; w.got += w.parts[p.i] === undefined ? 0 : 1;
      if (w.parts.filter(x => x !== undefined).length === p.n) {
        waiting.delete(p.id);
        decodeReply(w.parts).then(w.resolve, w.reject);
      }
    }
  })().catch(() => {});
  if (onTick) (async () => {
    for await (const sig of signals.subscribe({ topics: [TOPICS.tick], mode: 'latest', signal: stop.signal })) onTick(sig.payload, sig);
  })().catch(() => {});
  let n = 0;
  return {
    query(q, input = null, { signal } = {}) {
      const id = `${Date.now().toString(36)}-${(n++).toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => { waiting.delete(id); reject(Object.assign(Error(`No answer to ${q} within ${timeoutMs / 1000} s. Is the native ops participant running?`), { code: 'timeout' })); }, timeoutMs);
        const done = f => v => { clearTimeout(timer); f(v); };
        waiting.set(id, { parts: [], got: 0, resolve: done(resolve), reject: done(reject) });
        signal?.addEventListener('abort', () => { if (waiting.delete(id)) { clearTimeout(timer); reject(signal.reason); } }, { once: true });
        signals.send(TOPICS.query, { id, q, input }, { ttlMs: REPLY_TTL_MS }).catch(e => { if (waiting.delete(id)) { clearTimeout(timer); reject(e); } });
      });
    },
    close() { stop.abort(); for (const w of waiting.values()) w.reject(Error('closed')); waiting.clear(); },
  };
}
