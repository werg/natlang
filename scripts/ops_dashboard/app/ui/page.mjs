// Page entry: connect this browser to the natlang ops application session, then start the dashboard.
// The local layout preview (preview_server.py) answers /__preview/* and is used instead, labelled as a preview.
//
// Opening the page is the whole flow (Tendriller's connection component, ./connection.mjs):
//   - a registered browser connects at once, in any tab;
//   - a new browser creates its own key and a pairing request; the ops participant on the DGX prepares it, and the page
//     opens tendriller.dev once, where the owner taps Approve; tendriller.dev returns here and the page connects;
//   - every failure is said in plain words (describeConnectionError) and retried.
import { start, showGate } from './app.mjs';
import { createConnection, previewSource } from './connect.mjs';
import { createOwnerConnection } from './connection.mjs';
import { mountConnectionView } from './connect-view.mjs';

const preview = await fetch('/__preview/ping', { cache: 'no-store' }).then(r => r.ok).catch(() => false);
if (preview) {
  start(previewSource());
} else {
  const holder = document.createElement('div'), view = mountConnectionView(holder, { hideWhenConnected: false });
  view.render({ phase: 'connecting', title: 'Connecting…', detail: '' });
  showGate(holder);
  const conn = await createConnection({ status: () => {}, pairing: () => {} });
  let source;
  const connection = createOwnerConnection({ configuration: conn.configuration, pairing: conn.pairing, appName: 'natlang ops',
    async connect() { source = await conn.connect(); },
    onState(state) { if (state.phase === 'connected') start(source); else { view.render(state); showGate(holder); } } });
  void connection.start();
}
