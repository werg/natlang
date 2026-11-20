// Page entry: connect this browser's enrolled device to the natlang ops application session, then start the dashboard.
// The local layout preview (preview_server.py) answers /__preview/* and is used instead, labelled as a preview.
//
// First connection is a short guided flow that shows only the step that applies to this browser, with one primary action:
//   new browser       -> Request access (creates this browser's device key and a pairing request)
//   pending review    -> Approve on tendriller.dev (exact review link), polled every 5 s and on return to the tab
//   approved          -> progress while the native participant on the DGX adds this device
//   committed         -> connect automatically
//   denied / expired  -> request another review
import { start, showGate } from './app.mjs';
import { createConnection, previewSource } from './connect.mjs';
import { h } from './charts.mjs';

const preview = await fetch('/__preview/ping', { cache: 'no-store' }).then(r => r.ok).catch(() => false);
if (preview) {
  start(previewSource());
} else {
  const KEY = 'ops.connected-once';
  const ui = { status: () => {}, pairing: () => {} };
  const st = { busy: false, error: '', pairing: null, status: '', phase: 'start' };
  let conn = null;
  try { conn = await createConnection(ui); } catch (e) { st.error = `This page could not load its configuration: ${e.message}`; }
  const NOT_ENROLLED = /DEVICE_NOT_FOUND|DEVICE_NOT_ENROLLED|NOT_ENROLLED|UNKNOWN_DEVICE/;
  const plain = e => {
    const c = `${e?.code || ''} ${e?.message || ''}`;
    if (NOT_ENROLLED.test(c)) return null;           // handled as the "register this browser" step
    if (/pairing_account_required|ACCOUNT_REQUIRED|sign in/i.test(c)) return 'Sign in to Tendriller first (the account menu on this page), then try again.';
    if (/TIMEOUT|timed out|AbortError/i.test(c)) return 'The DGX did not answer in time. Check that the ops participant is running, then try again; the same request is reused.';
    if (/LOCK_REQUIRED|STORAGE_REQUIRED/.test(c)) return 'This browser blocks the storage the page needs (private mode or storage disabled). Open the page in a normal tab.';
    return e?.message || String(e);
  };
  const steps = (cur) => h('ol', { class: 'gsteps', 'aria-label': 'Progress' }, ['Request access', 'Approve on tendriller.dev', 'Added by the DGX', 'Connected'].map((t, i) => h('li', { class: i < cur ? 'done' : i === cur ? 'cur' : '' }, t)));
  const draw = () => {
    const p = st.pairing, state = p?.state;
    let cur = 0, body;
    if (!conn) body = [h('p', {}, 'The page could not start.')];
    else if (!conn.onPublishedOrigin) body = [h('p', {}, 'Open the published page to connect; this address cannot register a browser.')];
    else if (st.phase === 'connecting') { cur = 3; body = [h('p', { class: 'row' }, h('i', { class: 'spinner' }), ' Connecting this browser to the ops session…')]; }
    else if (st.phase === 'connected') { cur = 3; body = [h('p', {}, 'Connected.')]; }
    else if (state === 'committed') { cur = 3; body = [h('p', {}, 'This browser is registered.'), h('button', { class: 'btn primary big', disabled: st.busy, onclick: connect }, 'Open the dashboard')]; }
    else if (['pending', 'approved', 'committing', 'stale'].includes(state) && !(state === 'stale' && st.tries > 6)) {
      // The ops participant enrolls a request made from the owner's own signed-in page by itself (owner enrollment);
      // the service-origin review stays available as the fallback for any other case.
      cur = 1;
      const waited = (Date.now() - (st.since || Date.now())) / 1000;
      body = [h('p', { class: 'row' }, h('i', { class: 'spinner' }), h('span', {}, h('b', {}, 'Adding this browser… '), 'The ops participant on the DGX registers it for your account. This takes a few seconds; the dashboard opens by itself.')),
        waited > 45 && p.reviewUrl ? h('div', { class: 'small' }, 'Still waiting? ', h('a', { href: p.reviewUrl, target: '_blank', rel: 'noopener noreferrer' }, 'Approve it on tendriller.dev'), ' (one tap, then come back).') : null];
    } else if (['denied', 'expired', 'stale'].includes(state)) {
      cur = 1; body = [h('p', {}, state === 'denied' ? 'This request was declined on tendriller.dev.' : 'This request expired before it was approved.'), h('button', { class: 'btn primary big', disabled: st.busy, onclick: () => act(() => conn.request(true)) }, 'Request a new approval')];
    } else {
      body = [h('p', {}, st.phase === 'not-enrolled' ? h('b', {}, 'This browser isn\'t registered yet. ') : null, 'Each browser gets its own signed device key, approved once by you on tendriller.dev. The data then comes live from the DGX.'),
        h('button', { class: 'btn primary big', disabled: st.busy, onclick: () => act(() => conn.request(false)) }, st.busy ? 'Requesting…' : p ? 'Continue this request' : 'Request access for this browser')];
    }
    const card = h('div', { class: 'card gate' }, h('h2', {}, 'Connect this browser to natlang ops'), steps(cur), h('div', { class: 'gbody' }, body),
      st.error ? h('div', { class: 'alert crit', role: 'alert' }, st.error) : null,
      h('details', { class: 'small muted' }, h('summary', {}, 'What happens'), h('p', {}, 'This page is a Tendriller application. A device key is created inside this browser and never leaves it. You approve the exact key on tendriller.dev; the ops participant on the DGX then adds it to the application, and this browser can read the live data and sign edits.'),
        p ? h('p', {}, 'Request state: ', h('span', { class: 'mono' }, `${p.state}`), p.nextAction ? ` · ${p.nextAction}` : '') : null,
        h('p', {}, h('button', { class: 'linkbtn', onclick: connect }, 'This browser was approved before: connect directly'))));
    showGate(card);
  };
  ui.pairing = s => { if (s && s.state !== st.pairing?.state) st.since = Date.now(); st.pairing = s; draw(); if (s?.state === 'committed' && st.phase !== 'connecting') void connect(); };
  // While a request is open, try the device connection now and then: owner enrollment does not move the host's review state.
  st.tries = 0;
  async function quietConnect() {
    if (st.phase === 'connecting' || st.trying) return;
    st.trying = true; st.tries++;
    try { const src = await conn.connect(); localStorage.setItem(KEY, '1'); start(src); st.phase = 'connected'; }
    catch (e) { if (plain(e)) { /* a real failure surfaces on the next explicit attempt */ } }
    st.trying = false;
  }
  async function act(fn) { st.busy = true; st.error = ''; draw(); try { await fn(); } catch (e) { const m = plain(e); if (m) st.error = m; else st.phase = 'not-enrolled'; } st.busy = false; draw(); }
  async function connect() {
    st.phase = 'connecting'; st.error = ''; draw();
    try {
      const src = await conn.connect();
      localStorage.setItem(KEY, '1');
      start(src);
    } catch (e) {
      st.phase = 'start';
      const m = plain(e);
      if (m) st.error = `Could not connect: ${m}`;
      else { localStorage.removeItem(KEY); st.phase = 'not-enrolled'; }
      draw();
    }
  }
  // keep a pending or approved request moving without clicks
  setInterval(() => {
    const s = st.pairing?.state;
    if (st.busy || st.phase === 'connecting' || st.phase === 'connected' || !['pending', 'approved', 'committing', 'stale'].includes(s)) return;
    void conn.refresh().catch(() => {});
    void quietConnect();
  }, 6000);
  window.addEventListener('focus', () => { if (st.pairing && st.pairing.state !== 'committed' && st.phase !== 'connecting') void conn.refresh().catch(() => {}); });
  draw();
  let retained = null;
  try { retained = conn?.retained(); } catch (e) { st.error = plain(e); }
  if (conn?.onPublishedOrigin) {
    if (localStorage.getItem(KEY)) void connect();
    else if (retained?.pairingId) void act(() => conn.refresh());
  }
}
