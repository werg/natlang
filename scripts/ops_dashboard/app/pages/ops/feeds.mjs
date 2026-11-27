// Live data between the agent program (DGX) and the page, shared by both. Built on the starter's data feeds
// (shared/helpers/data-feed.mjs: serveDataFeeds on the agent side, useDataFeed on the page; signal-request.mjs underneath):
//   feed `snapshot`  the exporter's latest.json, pushed whenever it changes and pulled on connect (default query);
//   feed `data.*`    on-demand training-data reads (catalog, items, ...): `{ query: <input> }` pulls, never pushed or cached;
//   feed `machines`/`events`/`metrics`  exporter jsonl tails: `{ query: { offset } }`;
//   topic `ops/tick` one small signal every 10 s from the agent program (build stamp, inbox heartbeat); the page measures its
//                    clock against the tick's signing time;
//   topic `ops/t`    page telemetry, from the page to the agent program (logged on the DGX).
// Nothing here is retained in the journal: durable facts (edits, questions) go through the declared commands in domain.mjs.
// A reply is one JSON value of at most 4 MiB. Bigger answers are compressed first (`pack`); the helper splits them into signals.
export const TOPICS = Object.freeze({ tick: 'ops/tick', telemetry: 'ops/t' });
export const FEEDS = Object.freeze({ snapshot: 'snapshot', metrics: 'metrics', machines: 'machines', events: 'events' });
export const PACK_OVER_CHARS = 8000;                    // JSON longer than this is gzip+base64 packed
/** Signals are accepted up to this far ahead of the receiver's clock (Chronolog's default; this release does not make it
 * configurable per session). A page whose clock runs ahead of the DGX by more than this loses its signals: see README. */
export const SIGNAL_CLOCK_SKEW_MS = 2000;
export const validQuery = name => typeof name === 'string' && /^[A-Za-z0-9._-]{1,64}$/.test(name);

const b64 = bytes => { let s = ''; for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000)); return btoa(s); };
const unb64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
async function pipe(bytes, stream) { return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(stream)).arrayBuffer()); }

/** A JSON value -> itself when small, else `{ gz: base64(gzip(JSON)) }`. */
export async function pack(value) {
  const text = JSON.stringify(value);
  if (text.length <= PACK_OVER_CHARS) return value;
  return { gz: b64(await pipe(new TextEncoder().encode(text), new CompressionStream('gzip'))) };
}
export async function unpack(reply) {
  if (reply && typeof reply === 'object' && typeof reply.gz === 'string' && Object.keys(reply).length === 1)
    return JSON.parse(new TextDecoder().decode(await pipe(unb64(reply.gz), new DecompressionStream('gzip'))));
  return reply;
}
