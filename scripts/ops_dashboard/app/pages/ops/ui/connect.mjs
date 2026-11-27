// Browser side of the natlang ops application session: attaching to this browser's session owner (the Worker, worker.mjs),
// installing the Tendriller runtime bindings, and exposing one `source` to the views:
//   query(name, input)   on-demand read answered by the agent program on the DGX (feed pull, feeds.mjs)
//   snapshot             the exporter snapshot as a live data feed: {data, fetchedAtMs, stale, error, loading}
//   command(name, input) a declared command with a retained operation id (pending ops survive reloads)
//   edits / inbox        live views of the retained data (domain.mjs); inbox = {asks, answers} derived from the agent-request pipe
//   presence             the inbox worker's liveness card
//   ticks / clock        the agent program's 10 s tick, and this device's clock minus the DGX's
// Admission of a new browser is the starter's (shared/connection.mjs): the agent program on the DGX admits the signed-in owner.
// The preview server (preview_server.py) supplies the same interface over HTTP for layout checks with real data.
import { openBrowserOwner } from '@tendriller/core/browser-owner';
import { createLiveView, createApplicationBackend } from '@chronolog/peer-session';
import { browserProfileId } from '../../../shared/helpers/browser-profile.mjs';
import { createSignalRequests } from '../../../shared/helpers/signal-request.mjs';
import { useDataFeed } from '../../../shared/helpers/data-feed.mjs';
import { readAgentFeed, watchAgentPresence } from '../../../shared/helpers/agent-requests.mjs';
import { reloadOnOwnerReplaced } from '../../../shared/connection.mjs';
import { definition, deriveEdits, ASK_KINDS } from '../domain.mjs';
import { FEEDS, TOPICS, unpack } from '../feeds.mjs';
import { clockSkew } from './model.mjs';

/* global __OPS_BUILD__ */
export const BUILD = typeof __OPS_BUILD__ === 'object' ? __OPS_BUILD__ : { build: 'dev', commit: '' };
export const INBOX_CONSUMER = 'ops-inbox';

const PENDING_KEY = 'ops.pending-operations';
const readPending = () => { try { return JSON.parse(localStorage.getItem(PENDING_KEY)) || []; } catch { return []; } };
const writePending = list => localStorage.setItem(PENDING_KEY, JSON.stringify(list.slice(-200)));

/** A small observable: {get(), set(v), on(fn)}. */
export function cell(value) {
  const fns = new Set();
  return { get: () => value, set(v) { value = v; for (const f of fns) f(v); }, on(f) { fns.add(f); f(value); return () => fns.delete(f); } };
}

/** The page's `ask` as the starter's agent-request command: the question is a chat message to the page's primary agent whose
 * `data` carries the kind and subject; the inbox worker answers with a receipt and the supervising agent with a reply. */
export function toCommand(name, input) {
  if (name !== 'ask') return [name, input];
  return ['agentMessage', { channel: 'chat', to: [], text: input.text, data: JSON.stringify({ kind: input.kind, subject: input.subject }), expectReply: input.kind !== 'ack_alert' }];
}

/** {asks, answers} from the agent feed: a question is a message that is not by the agent and answers nothing; an answer is the
 * agent's message with `inReplyTo`. Each question carries the receipts the inbox worker recorded. */
export function inboxOf(feed, agentKey) {
  const asks = [], answers = [];
  for (const m of feed?.messages || []) {
    if (m.author === agentKey) { if (m.inReplyTo) answers.push({ ask: m.inReplyTo.operationId, text: m.text, author: m.author, operationId: m.operationId, atMs: m.atMs }); continue; }
    if (m.inReplyTo) continue;
    let d = {}; try { d = JSON.parse(m.data || '{}'); } catch { /* a plain chat message */ }
    if (!ASK_KINDS.includes(d.kind) || !d.subject?.kind || !d.subject?.id) continue;
    asks.push({ operationId: m.operationId, author: m.author, kind: d.kind, subject: d.subject, text: m.text, atMs: m.atMs,
      receipts: (feed.receipts?.[`${m.author}:${m.operationId}`] || []).map(r => ({ state: r.state, atMs: r.atMs })) });
  }
  return { asks, answers };
}

/** Commands keep their operation id in localStorage until the journal shows them applied; retry reuses the same id. */
function commandTracker(session, onChange) {
  let list = readPending();
  const save = () => { writePending(list); onChange(list); };
  const watch = async item => {
    try {
      const op = await session.operations.get(item.id);
      const st = op ? await op.materialized({ signal: AbortSignal.timeout(120000) }) : null;
      item.state = st ? (st.outcome === 'accepted' ? 'applied' : st.outcome) : 'uncertain'; item.outcome = st?.outcome; item.rejection = st?.rejectionCode;
    } catch (e) { item.state = 'uncertain'; item.error = e.message; }
    if (item.state === 'applied') list = list.filter(x => x !== item);
    save();
  };
  for (const item of list) void watch(item);
  return {
    list: () => list,
    async submit(name, input) {
      const id = crypto.randomUUID(), item = { id, name, input, state: 'saving', ts: Date.now() };
      list.push(item); save();
      try {
        const [command, payload] = toCommand(name, input);
        const op = await session.command(command, payload, { operationId: id });
        item.state = 'saved'; save();
        const st = await op.materialized({ signal: AbortSignal.timeout(60000) }).catch(e => ({ outcome: 'uncertain', error: e.message }));
        item.state = st.outcome === 'accepted' ? 'applied' : st.outcome; item.outcome = st.outcome; item.rejection = st.rejectionCode;
        if (item.state === 'applied') list = list.filter(x => x !== item); else void watch(item);
        save();
        return { operationId: id, ...st };
      } catch (e) { if (item.state === 'saving') { item.state = 'uncertain'; item.error = e.message; save(); } throw Object.assign(e, { operationId: id }); }
    },
    async retry(id) {
      const item = list.find(x => x.id === id);
      if (!item) return;
      const saved = await session.operations.get(id);
      if (saved) await session.operations.retry(id); else { const [c, p] = toCommand(item.name, item.input); await session.command(c, p, { operationId: id }); }
      item.state = 'saved'; save(); void watch(item);
    },
    dismiss(id) { list = list.filter(x => x !== id && x.id !== id); save(); },
  };
}

/** Page telemetry for the operators, never shown to the person: what this page sent, received and drew goes to the agent
 * program as `ops/t` signals every 10 s (counts per kind plus the latest events), logged on the DGX. */
export function createTelemetry(signals, { extra = () => ({}) } = {}) {
  const counts = {}, events = [];
  let seq = 0;
  const COUNT_ONLY = new Set(['query_ok']);        // frequent and benign: counted, not listed
  const event = e => { counts[e.ev] = (counts[e.ev] || 0) + 1; if (!COUNT_ONLY.has(e.ev) && events.length < 200) events.push({ ...e, at: Date.now() }); };
  const flush = async () => {
    if (!events.length && !Object.keys(counts).length) return;
    const batch = events.splice(0, 40);
    const make = () => ({ seq: seq + 1, build: BUILD, publication: window.tendril?.identity?.publication?.slice?.(0, 12) ?? null, version: window.tendril?.identity?.version ?? null,
      visible: document.visibilityState, counts: { ...counts }, events: batch, ...extra() });
    let payload = make();
    while (JSON.stringify(payload).length > 11_000 && batch.length > 1) { events.unshift(batch.pop()); payload = make(); }     // one signal is at most 16 KiB signed
    seq++;
    try { await signals.send(TOPICS.telemetry, payload, { ttlMs: 30000 }); }
    catch { events.unshift(...batch.slice(0, 200 - events.length)); }
  };
  const timer = setInterval(() => void flush(), 10000);
  return { event, count: k => { counts[k] = (counts[k] || 0) + 1; }, flush, close: () => clearInterval(timer) };
}

/**
 * Connection controller for the starter's connection component (shared/connection.mjs createOwnerConnection): `configuration` is
 * the published configuration.json; `connect({signal})` attaches to the owner and resolves with the source.
 */
export async function loadConfiguration() {
  return fetch(new URL('./configuration.json', document.baseURI)).then(r => r.json());
}

export async function connect(configuration, { signal } = {}) {
  const profileId = browserProfileId(configuration.applicationId);
  const attachment = await openBrowserOwner({ workerUrl: new URL('./worker.mjs', document.baseURI), profileId,
    applicationId: configuration.applicationId, definitionId: definition.id, definitionVersion: definition.version, signal, timeoutMs: 30000,
    context: { viewId: sessionStorage.getItem('ops.view') || (() => { const v = crypto.randomUUID(); sessionStorage.setItem('ops.view', v); return v; })() },
    // The browser's device owner moved (another tab took it over, or it restarted): reconnect when this tab is next in the foreground.
    onOwnerReplaced: reloadOnOwnerReplaced });
  try { return await mount(attachment, configuration); } catch (e) { await attachment.detach().catch(() => {}); throw e; }
}

async function mount(attachment, configuration) {
  const session = attachment.session, agentKey = configuration.agentKey;
  if (!window.tendril) throw Error('Open the published Tendriller page to install its runtime binding.');
  const edits = cell(null), inbox = cell(null), pending = cell([]), ticks = cell(null), clock = cell(null);
  const snapshot = cell({ data: null, fetchedAtMs: null, stale: false, error: null, loading: true });
  const presence = cell({ status: 'unknown' });
  const tele = createTelemetry(session.signals, { extra: () => ({ clock_skew_ms: clock.get() }) });
  const report = e => tele.event(e);
  const ev = createLiveView(session, deriveEdits, { onError: e => report({ ev: 'edits_view_error', error: String(e.message).slice(0, 160) }) });
  const iv = createLiveView(session, readAgentFeed, { onError: e => report({ ev: 'inbox_view_error', error: String(e.message).slice(0, 160) }) });
  ev.subscribe(s => edits.set(s.value)); iv.subscribe(s => inbox.set(inboxOf(s.value, agentKey)));
  await Promise.allSettled([ev.ready({ timeoutMs: 30000 }), iv.ready({ timeoutMs: 30000 })]);
  // The runtime binding installs once per page: only after the session is usable, so a failed attempt can retry.
  window.tendril.application.install(createApplicationBackend(session));
  window.tendril.signals.install(session.signals);
  report({ ev: 'connected', ms_since_load: Math.round(performance.now()), shared_worker: typeof SharedWorker === 'function', ua: navigator.userAgent.slice(0, 120) });
  // The snapshot feed: pushed when the exporter output changes, pulled right after connecting and on every new peer link.
  const feed = useDataFeed(session, { peer: agentKey, feed: FEEDS.snapshot, staleAfterMs: 90_000, onUpdate: async state => {
    if (state.error) report({ ev: 'feed_error', error: String(state.error).slice(0, 160) });
    const data = state.data ? await unpack(state.data).catch(e => { report({ ev: 'unpack_failed', error: String(e.message).slice(0, 160) }); return null; }) : null;
    snapshot.set({ data, fetchedAtMs: state.fetchedAtMs, stale: state.stale, error: state.error, loading: state.loading, revision: state.revision });
    if (data) tele.count('snapshot');
  } });
  const rpc = createSignalRequests(session, { peer: agentKey, requestTopic: 'feed/pull', replyTopic: 'feed/reply' });
  const query = async (q, input, { signal } = {}) => {
    const t0 = Date.now();
    try { const r = await rpc.request(q, { query: input ?? {}, force: true }, { signal, timeoutMs: 40000 }); const v = await unpack(r.data); report({ ev: 'query_ok', q, ms: Date.now() - t0 }); return v; }
    catch (e) { report({ ev: 'query_failed', q, code: e.code || e.name, error: String(e.message || '').slice(0, 160) }); throw e; }
  };
  // The agent program's tick: build stamp for pinned tabs, the supervising agent's inbox heartbeat; its signing time gives the clock skew.
  const abort = new AbortController();
  void (async () => {
    for await (const sig of session.signals.subscribe({ topics: [TOPICS.tick], mode: 'latest', includeLatest: true, signal: abort.signal })) {
      if (sig.authorId !== agentKey) continue;
      tele.count('tick'); const k = clockSkew(sig.issuedAtMs, Date.now()); if (k != null) clock.set(k); ticks.set(sig.payload);
    }
  })().catch(() => {});
  const watcher = watchAgentPresence(session, { peer: agentKey, consumerId: INBOX_CONSUMER, requestTopic: 'chat', staleAfterMs: 45_000, onChange: v => presence.set(v) });
  const tracker = commandTracker(session, list => pending.set([...list]));
  pending.set([...tracker.list()]);
  window.addEventListener('pagehide', event => { tele.close(); abort.abort(); feed.close(); rpc.close(); watcher.close(); ev.close(); iv.close();
    if (event.persisted) void attachment.suspend?.().catch(() => {}); else void attachment.detach().catch(() => {}); }, { once: true });
  setTimeout(() => void tele.flush(), 3000);
  return { kind: 'session', session, report, deviceId: session.deviceId ?? session.identity?.publicKey, agentKey,
    query, refresh: () => feed.refresh(), command: (n, i) => tracker.submit(n, i), clock,
    retry: id => tracker.retry(id), dismiss: id => tracker.dismiss(id), edits, inbox, pending, ticks, snapshot, presence };
}

/** Preview source: the same interface over the local preview server (no session; commands are kept in its memory only). */
export function previewSource() {
  const post = async (path, body) => {
    const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    const v = await r.json();
    if (!r.ok) throw Object.assign(Error(v.error || `HTTP ${r.status}`), { code: v.code });
    return v;
  };
  const edits = cell({ edits: [], applications: [] }), inbox = cell({ asks: [], answers: [] }), pending = cell([]), ticks = cell(null);
  const snapshot = cell({ data: null, fetchedAtMs: null, stale: false, error: null, loading: true }), presence = cell({ status: 'unknown' });
  const pull = async () => { const s = await post('/__preview/state', {}); edits.set(s.edits); inbox.set(s.inbox); };
  const read = async () => { try { const data = await post('/__preview/query', { q: 'ops.snapshot', input: null }); snapshot.set({ data, fetchedAtMs: Date.now(), stale: false, error: null, loading: false }); } catch (e) { snapshot.set({ ...snapshot.get(), error: e.message, loading: false }); } };
  void pull(); void read();
  setInterval(() => void read(), 10000);
  return { kind: 'preview', deviceId: 'preview', agentKey: 'preview-agent',
    query: (q, input) => post('/__preview/query', { q, input }), refresh: read,
    async command(name, input) { const r = await post('/__preview/command', { name, input }); await pull(); return r; },
    retry: async () => {}, dismiss: () => {}, edits, inbox, pending, ticks, snapshot, presence };
}
