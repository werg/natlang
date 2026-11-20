// Page entry: connect this browser's enrolled device to the natlang ops application session, then start the dashboard.
// The local layout preview (preview_server.py) answers /__preview/* and is used instead, labelled as a preview.
import { start, showGate } from './app.mjs';
import { createConnection, previewSource } from './connect.mjs';
import { h } from './charts.mjs';

const preview = await fetch('/__preview/ping', { cache: 'no-store' }).then(r => r.ok).catch(() => false);
if (preview) {
  document.getElementById('notices').textContent = 'Layout preview with real data from the local data service. Edits stay in the preview server\'s memory and are never applied.';
  start(previewSource());
} else {
  const ui = { status: () => {}, pairing: () => {} };
  const conn = await createConnection(ui);
  const st = { busy: false, error: '', pairing: null, status: '' };
  const KEY = 'ops.connected-once';
  const draw = () => {
    const committed = st.pairing?.state === 'committed';
    const card = h('div', { class: 'card gate' }, h('h2', {}, 'Connect this browser to natlang ops'),
      h('p', {}, 'This page is a Tendriller application. Its data comes from the native ops participant on the DGX through this browser\'s own signed device; edits you make are signed operations that the participant applies to the training-data edit layers.'),
      !conn.onPublishedOrigin ? h('div', { class: 'notice' }, 'Open the published page to connect; this origin cannot enroll a device.') : null,
      h('ol', { class: 'steps' },
        h('li', { class: committed || localStorage.getItem(KEY) ? 'done' : '' }, h('b', {}, 'Request access for this browser. '), 'Creates this browser\'s device key and asks the native participant to stage an exact proposal. ',
          h('button', { class: 'btn', disabled: st.busy || !conn.onPublishedOrigin, onclick: () => act(() => conn.request(false)) }, st.pairing ? 'Resume this request' : 'Request browser access')),
        h('li', { class: committed ? 'done' : '' }, h('b', {}, 'Review it on tendriller.dev. '), 'Compare the fingerprint, then approve this device. ',
          st.pairing?.reviewUrl && !committed ? h('a', { class: 'btn primary', href: st.pairing.reviewUrl, target: '_blank', rel: 'noopener noreferrer' }, 'Review this exact request') : null,
          st.pairing ? h('button', { class: 'btn', disabled: st.busy, onclick: () => act(() => conn.refresh()) }, 'Check status') : null,
          st.pairing && ['expired', 'denied', 'stale'].includes(st.pairing.state) ? h('button', { class: 'btn', onclick: () => act(() => conn.request(true)) }, 'Request another review') : null,
          st.pairing ? h('div', { class: 'small muted' }, `${st.pairing.state}: ${st.pairing.nextAction}`, h('br'), 'Device key ', h('span', { class: 'mono' }, st.pairing.challenge?.publicKey)) : null),
        h('li', {}, h('b', {}, 'Connect. '), h('button', { class: 'btn primary', disabled: st.busy || !conn.onPublishedOrigin, onclick: connect }, 'Connect existing device'))),
      st.status ? h('p', { class: 'small', role: 'status' }, st.status) : null, st.error ? h('div', { class: 'alert crit' }, st.error) : null);
    showGate(card);
  };
  ui.pairing = s => { st.pairing = s; draw(); };
  ui.status = (text) => { document.getElementById('notices').textContent = text; };
  async function act(fn) { st.busy = true; st.error = ''; draw(); try { await fn(); } catch (e) { st.error = e.message; } st.busy = false; draw(); }
  async function connect() {
    st.busy = true; st.error = ''; st.status = 'Connecting: opening the device, verifying authority, reading the edits view…'; draw();
    try {
      const src = await conn.connect();
      localStorage.setItem(KEY, '1');
      start(src);
    } catch (e) {
      st.busy = false; st.status = '';
      st.error = e.code === 'DEVICE_NOT_ENROLLED' ? 'This browser is not enrolled yet: request access and approve it first.' : `Could not connect: ${e.message}`;
      draw();
    }
  }
  window.addEventListener('focus', () => { if (st.pairing && st.pairing.state !== 'committed') void act(() => conn.refresh()); });
  draw();
  if (conn.retained()?.pairingId) void act(() => conn.refresh());
  if (localStorage.getItem(KEY) && conn.onPublishedOrigin) void connect();
}
