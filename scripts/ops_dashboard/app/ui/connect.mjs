// Browser side of the natlang ops application session: pairing this browser's device, attaching to the owning Worker,
// installing the Tendriller application binding, and exposing one `source` to the views:
//   query(name, input)        on-demand read answered by the native participant (ops-protocol.mjs)
//   command(name, input)      a declared command with a retained operation id (pending ops survive reloads)
//   edits / inbox             live views of the retained queries
// The preview server (preview_server.py) supplies the same interface over HTTP for layout checks with real data.
import { openBrowserOwner } from '@tendriller/core/collaboration/browser';
import { createApplicationBackend } from '@chronolog/peer-session/tendriller';
import { createLiveView } from '../helpers/live-view.mjs';
import { createBrowserPairing } from '../pairing-browser.mjs';
import { createQueryClient, TOPICS } from '../ops-protocol.mjs';

/* global __OPS_BUILD__ */
export const BUILD = typeof __OPS_BUILD__ === 'object' ? __OPS_BUILD__ : { build: 'dev', commit: '' };

/** Page telemetry for the operators, never shown to the person: what this page sent, received and drew goes to the native
 * participant as `ops/t` signals every 10 s (counts per kind plus the latest events), logged on the DGX. */
export function createTelemetry(signals) {
  const counts = {}, events = [];
  let seq = 0;
  const COUNT_ONLY = new Set(['reply_unmatched', 'query_ok']);    // frequent and benign: counted, not listed
  const event = e => { counts[e.ev] = (counts[e.ev] || 0) + 1; if (!COUNT_ONLY.has(e.ev) && events.length < 200) events.push({ ...e, at: Date.now() }); };
  const flush = async () => {
    if (!events.length && !Object.keys(counts).length) return;
    const batch = events.splice(0, 40);
    const payload = { seq: ++seq, build: BUILD, publication: window.tendril?.identity?.publication?.slice?.(0, 12) ?? null, version: window.tendril?.identity?.version ?? null,
      visible: document.visibilityState, counts: { ...counts }, events: batch };
    try { await signals.send(TOPICS.telemetry, payload, { ttlMs: 30000 }); }
    catch { events.unshift(...batch.slice(0, 200 - events.length)); }
  };
  const timer = setInterval(() => void flush(), 10000);
  return { event, count: k => { counts[k] = (counts[k] || 0) + 1; }, flush, close: () => clearInterval(timer) };
}
import { definition } from '../domain.mjs';

const PENDING_KEY = 'ops.pending-operations';
const readPending = () => { try { return JSON.parse(localStorage.getItem(PENDING_KEY)) || []; } catch { return []; } };
const writePending = list => localStorage.setItem(PENDING_KEY, JSON.stringify(list.slice(-200)));

/** A small observable: {get(), set(v), on(fn)}. */
export function cell(value) {
  const fns = new Set();
  return { get: () => value, set(v) { value = v; for (const f of fns) f(v); }, on(f) { fns.add(f); f(value); return () => fns.delete(f); } };
}

/** Commands keep their operation id in localStorage until the journal shows them applied; retry reuses the same id. */
function commandTracker(session, onChange) {
  let list = readPending();
  const save = () => { writePending(list); onChange(list); };
  const watch = async item => {
    try {
      const st = await session.operations.wait(item.id, { signal: AbortSignal.timeout(120000) });
      item.state = st.status; item.outcome = st.outcome; item.rejection = st.rejectionCode;
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
      try { await session.command(name, input, { operationId: id }); item.state = 'saved'; save(); }
      catch (e) { item.state = 'uncertain'; item.error = e.message; save(); throw Object.assign(e, { operationId: id }); }
      const op = await session.operations.wait(id, { signal: AbortSignal.timeout(60000) }).catch(e => ({ status: 'uncertain', error: e.message }));
      item.state = op.status; item.outcome = op.outcome; item.rejection = op.rejectionCode;
      if (op.status === 'applied') list = list.filter(x => x !== item); else void watch(item);
      save();
      return { operationId: id, ...op };
    },
    async retry(id) {
      const item = list.find(x => x.id === id);
      if (!item) return;
      const saved = await session.operations.get(id);
      if (saved) await session.retry(id); else await session.command(item.name, item.input, { operationId: id });
      item.state = 'saved'; save(); void watch(item);
    },
    dismiss(id) { list = list.filter(x => x.id !== id); save(); },
  };
}

/**
 * Connection panel controller. `ui` = {status(text, level), pairing(status|null)}; returns {request, refresh, connect}.
 * connect() resolves with the source once a verified application query succeeded.
 */
export async function createConnection(ui) {
  const configuration = await fetch(new URL('./configuration.json', document.baseURI)).then(r => r.json());
  const profileId = `natlang-ops-${configuration.applicationId}`;
  const onPublishedOrigin = location.origin === new URL(configuration.page_url).origin;
  const pairing = createBrowserPairing({ configuration, profileId, workerUrl: new URL('./pairing-worker.mjs', document.baseURI) });
  const show = status => ui.pairing(status);
  return {
    configuration, onPublishedOrigin, retained: () => pairing.read(),
    async request(newReview = false) { show(await pairing.request({ newReview, signal: AbortSignal.timeout(35000) })); },
    async refresh() { show(await pairing.refresh({ signal: AbortSignal.timeout(15000) })); },
    async connect() {
      const identity = window.tendril?.identity;
      const attachment = await openBrowserOwner({ workerUrl: new URL('./worker.mjs', document.baseURI), profileId,
        applicationId: configuration.applicationId, definitionId: definition.id, definitionVersion: definition.version, timeoutMs: 30000,
        context: { viewId: sessionStorage.getItem('ops.view') || (() => { const v = crypto.randomUUID(); sessionStorage.setItem('ops.view', v); return v; })(),
          ...(identity?.documentPath ? { documentPath: identity.documentPath } : {}), ...(identity?.publication ? { publicationId: identity.publication } : {}) },
        // The browser's device owner moved (another tab took it over, or it restarted): reconnect by reloading, at most once a minute.
        onOwnerReplaced() { const last = Number(sessionStorage.getItem('ops.reloaded') || 0); if (Date.now() - last > 60000) { sessionStorage.setItem('ops.reloaded', String(Date.now())); location.reload(); } else ui.status('The connection moved to another tab of this page. Reload to reconnect here.', 'warn'); },
      });
      try { return await mount(attachment); } catch (e) { await attachment.detach().catch(() => {}); throw e; }
    },
  };
  async function mount(attachment) {
      const session = attachment.session;
      if (!window.tendril) throw Error('Open the published Tendriller page to install its runtime binding.');
      window.tendril.application.install(createApplicationBackend({ id: session.applicationId, session }));
      window.tendril.signals.install(session.signals);
      session.participation.onStatus(s => ui.status(`Session ${s.phase}${s.error ? ` (${s.error})` : ''}`, s.phase === 'connected' || s.phase === 'ready' ? 'ok' : 'warn'));
      const edits = cell(null), inbox = cell(null), pending = cell([]), ticks = cell(null);
      const mk = (query, target) => createLiveView(session, { query, input: null, onError: e => ui.status(`${query}: ${e.message}`, 'warn'),
        onRecovery: n => ui.status(`Selected history changed: ${n.abandoned.length} operations excluded (unknown loss: ${n.unknownLoss}).`, 'crit') });
      const ev = mk('edits'), iv = mk('inbox');
      ev.subscribe(s => edits.set(s.value)); iv.subscribe(s => inbox.set(s.value));
      await ev.refresh(); await iv.refresh();
      const tele = createTelemetry(session.signals);
      tele.event({ ev: 'connected', ms_since_load: Math.round(performance.now()), shared_worker: typeof SharedWorker === 'function', ua: navigator.userAgent.slice(0, 120) });
      const client = createQueryClient(session.signals, { onTick: t => { tele.count('tick'); ticks.set(t); }, onEvent: e => tele.event(e) });
      const tracker = commandTracker(session, list => pending.set([...list]));
      pending.set([...tracker.list()]);
      window.addEventListener('pagehide', () => { tele.close(); client.close(); ev.close(); iv.close(); void attachment.detach().catch(() => {}); }, { once: true });
      setTimeout(() => void tele.flush(), 3000);
      return { kind: 'session', report: e => tele.event(e), deviceId: session.deviceId, nativeDeviceId: configuration.nativeDeviceId,
        query: (q, input, opts) => client.query(q, input, opts), command: (n, i) => tracker.submit(n, i),
        retry: id => tracker.retry(id), dismiss: id => tracker.dismiss(id), edits, inbox, pending, ticks };
  }
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
  const pull = async () => { const s = await post('/__preview/state', {}); edits.set(s.edits); inbox.set(s.inbox); };
  void pull();
  setInterval(async () => { try { ticks.set(await post('/__preview/query', { q: 'ops.tick', input: null })); } catch {} }, 10000);
  void post('/__preview/query', { q: 'ops.tick', input: null }).then(t => ticks.set(t)).catch(() => {});
  return { kind: 'preview', deviceId: 'preview', nativeDeviceId: 'preview-native',
    query: (q, input) => post('/__preview/query', { q, input }),
    async command(name, input) { const r = await post('/__preview/command', { name, input }); await pull(); return r; },
    retry: async () => {}, dismiss: () => {}, edits, inbox, pending, ticks };
}
