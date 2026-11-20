// Page entry: connect this browser to the natlang ops application session, then start the dashboard.
// The local layout preview (preview_server.py) answers /__preview/* and is used instead, labelled as a preview.
//
// Opening the page is the whole flow. One "Connecting…" state, no buttons:
//   1. attach to this browser's device owner (shared by every tab of the page) and open the session;
//   2. if this browser is not registered yet, request access automatically: the ops participant on the DGX registers a
//      request made from the owner's own signed-in page by itself (owner enrollment), then step 1 succeeds;
//   3. every other failure is said in plain words and retried automatically with backoff.
import { start, showGate } from './app.mjs';
import { createConnection, previewSource } from './connect.mjs';
import { h } from './charts.mjs';

const preview = await fetch('/__preview/ping', { cache: 'no-store' }).then(r => r.ok).catch(() => false);
if (preview) {
  start(previewSource());
} else {
  const st = { msg: 'Connecting…', note: '', code: '', review: null, since: Date.now() };
  const draw = () => showGate(h('div', { class: 'card gate' },
    h('div', { class: 'gconnect' }, h('i', { class: 'spinner big' }), h('div', {}, h('div', { class: 'gmsg', role: 'status' }, st.msg), st.note ? h('div', { class: 'small muted' }, st.note) : null)),
    st.review ? h('p', { class: 'small' }, 'Taking longer than usual. You can also ', h('a', { href: st.review, target: '_blank', rel: 'noopener noreferrer' }, 'approve this browser on tendriller.dev'), '; this page continues by itself.') : null,
    st.code ? h('details', { class: 'small muted' }, h('summary', {}, 'Technical details'), h('span', { class: 'mono' }, st.code)) : null));
  draw();
  const codeOf = e => String(e?.code || e?.name || '') + (e?.message ? `: ${e.message}` : '');
  const is = (e, re) => re.test(`${e?.code || ''} ${e?.message || ''}`);
  const NOT_ENROLLED = /DEVICE_NOT_FOUND|DEVICE_NOT_ENROLLED|NOT_ENROLLED|UNKNOWN_DEVICE|not enrolled/i;
  const BUSY = /OWNER_BUSY|OWNER_UNREACHABLE|ENDPOINT_STALE|OWNER_REPLACED|LockManager|lock/i;
  const plain = e => is(e, /pairing_account_required|ACCOUNT_REQUIRED|unauthorized|401/i) ? ['Your Tendriller sign-in has expired.', 'Reload the page to sign in again.']
    : is(e, /TIMEOUT|timed out|AbortError|Failed to fetch|NetworkError|network/i) ? ['The connection is slow or offline. Retrying…', '']
    : is(e, /STORAGE_REQUIRED|LOCK_REQUIRED|IndexedDB|QuotaExceeded/i) ? ['This browser blocks the storage the dashboard needs.', 'Open the page in a normal (not private) tab.']
    : is(e, /PAIRING_DEVICE_KNOWN/) ? ['Finishing the registration of this browser…', '']
    : ['Could not connect yet. Retrying…', ''];
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  let conn;
  for (let delay = 1000; !conn; delay = Math.min(delay * 2, 30000)) {
    try { conn = await createConnection({ status: () => {}, pairing: () => {} }); }
    catch (e) { [st.msg, st.note] = plain(e); st.code = codeOf(e); draw(); await sleep(delay); }
  }
  if (!conn.onPublishedOrigin) { st.msg = 'Open the published page address to connect.'; st.note = ''; draw(); }
  else {
    // One tab at a time requests access; every tab keeps trying to attach, so the first registered connection wins everywhere.
    let requested = false, delay = 1500;
    async function ensureRequest() {
      const retained = conn.retained();
      if (retained?.pairingId) {
        const s = await conn.refresh().catch(() => null);
        if (s?.reviewUrl) st.reviewUrl = s.reviewUrl;
        if (s && ['expired', 'denied', 'stale'].includes(s.state)) { const n = await conn.request(true).catch(() => null); if (n?.reviewUrl) st.reviewUrl = n.reviewUrl; }
        return;
      }
      if (!requested) { requested = true; const s = await conn.request(false); if (s?.reviewUrl) st.reviewUrl = s.reviewUrl; }
    }
    for (;;) {
      try {
        const src = await conn.connect();
        start(src);
        break;
      } catch (e) {
        st.code = codeOf(e);
        if (is(e, NOT_ENROLLED)) {
          st.msg = 'Adding this browser to your dashboard…'; st.note = 'This happens once per browser and takes a few seconds.'; delay = 3000;
          draw();
          try { await ensureRequest(); } catch (e2) { [st.msg, st.note] = plain(e2); st.code = codeOf(e2); }
          if (Date.now() - st.since > 60000 && st.reviewUrl) st.review = st.reviewUrl;
        } else if (is(e, BUSY)) { st.msg = 'Connecting… (another tab of this page is starting the connection)'; st.note = ''; delay = 2000; }
        else { [st.msg, st.note] = plain(e); delay = Math.min(delay * 2, 30000); }
        draw();
        await sleep(delay);
      }
    }
  }
}
