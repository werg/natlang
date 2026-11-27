// The live data path end to end over an in-memory signal pair: the agent program's serveDataFeeds (snapshot pushes, on-demand
// data pulls) against the page's useDataFeed and createSignalRequests, with the real chunking, packing and request helpers.
import assert from 'node:assert/strict';
import { test } from 'node:test';

const AGENT = 'a'.repeat(64), PAGE = 'b'.repeat(64);

/** Two sessions whose signal ports deliver to each other (never to themselves), like a connected peer pair. */
function pair() {
  const make = (id, other) => {
    const subs = new Set(), statusListeners = new Set(), stop = new AbortController();
    const s = { identity: { publicKey: id }, signal: stop.signal, status: { connectedPeers: 1, connectionRevision: 1 }, subs,
      onStatus(fn) { statusListeners.add(fn); return () => statusListeners.delete(fn); },
      signals: {
        async send(topic, payload) {
          const frame = { topic, payload: structuredClone(payload), authorId: id, issuedAtMs: Date.now() };
          queueMicrotask(() => { for (const sub of other().subs) if (sub.topics.includes(topic)) sub.push(frame); });
          return frame;
        },
        subscribe({ topics, signal }) {
          const queue = [], waiting = [], sub = { topics, push: f => { const w = waiting.shift(); w ? w({ value: f, done: false }) : queue.push(f); } };
          subs.add(sub);
          signal?.addEventListener('abort', () => { subs.delete(sub); for (const w of waiting.splice(0)) w({ value: undefined, done: true }); }, { once: true });
          return { [Symbol.asyncIterator]() { return { next: () => queue.length ? Promise.resolve({ value: queue.shift(), done: false }) : signal?.aborted ? Promise.resolve({ done: true }) : new Promise(r => waiting.push(r)), return: async () => ({ done: true }) }; } };
        } } };
    return s;
  };
  const agent = make(AGENT, () => page), page = make(PAGE, () => agent);
  return { agent, page, close() { agent.subs.clear(); page.subs.clear(); } };
}

test('the page reads the pushed snapshot feed and pulls data queries from the agent program', async () => {
  const { serveDataFeeds, useDataFeed } = await import('../shared/helpers/data-feed.mjs');
  const { createSignalRequests } = await import('../shared/helpers/signal-request.mjs');
  const { feedSpecs } = await import('../pages/ops/agent.mjs');
  const { pack, unpack } = await import('../pages/ops/feeds.mjs');
  const { agent, page, close } = pair();
  let n = 0;
  const big = () => ({ v: 3, ts: 1_000 + n, rounds: Array.from({ length: 1500 }, (_, i) => ({ i, text: 'row ' + i + ' ' + n })) });
  const rpc = async (q, input) => q === 'ops.snapshot' ? big() : { q, input };
  const specs = feedSpecs(rpc);
  specs.snapshot.pollMs = 100;                                       // fast for the test; production polls every 10 s
  specs.snapshot.fetch = async () => pack(await rpc('ops.snapshot', null));
  const feeds = serveDataFeeds(agent, specs, { signal: agent.signal });
  const updates = [];
  const feed = useDataFeed(page, { peer: AGENT, feed: 'snapshot', onUpdate: state => updates.push(state) });
  const requests = createSignalRequests(page, { peer: AGENT, requestTopic: 'feed/pull', replyTopic: 'feed/reply' });
  try {
    const waitFor = async (pred, ms = 8000) => { const end = Date.now() + ms; while (!pred()) { if (Date.now() > end) throw Error('timeout'); await new Promise(r => setTimeout(r, 25)); } };
    await waitFor(() => updates.some(u => u.data));                  // the first pull: larger than one signal, so chunked and packed
    const first = await unpack(updates.findLast(u => u.data).data);
    assert.equal(first.v, 3);
    assert.equal(first.rounds.length, 1500);
    // a changed snapshot is pushed (pullRequired) and read again without the page asking
    n = 1;
    await waitFor(() => updates.some(u => u.data && u.revision > updates.find(x => x.data).revision), 10000);
    assert.equal((await unpack(updates.findLast(u => u.data).data)).ts, 1_001);
    // an on-demand data query: the reply carries the packed value of the data service
    const reply = await requests.request('data.items', { query: { dataset: 'd', offset: 3 }, force: true }, { timeoutMs: 8000 });
    assert.deepEqual(await unpack(reply.data), { q: 'data.items', input: { dataset: 'd', offset: 3 } });
    // an unknown feed fails with a named error, not a hang
    await assert.rejects(requests.request('data.nope', { query: {} }, { timeoutMs: 8000 }), e => e.code === 'FEED_NOT_FOUND');
  } finally { feed.close(); requests.close(); feeds.close(); close(); }
});
