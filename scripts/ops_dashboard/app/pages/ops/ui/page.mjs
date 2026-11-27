// Page entry: connect this browser to the natlang ops application session, then start the dashboard.
// The local layout preview (preview_server.py) answers /__preview/* and is used instead, labelled as a preview.
//
// Opening the page is the whole flow (the starter's connection component, shared/connection.mjs):
//   - a registered browser connects at once, in any tab;
//   - a new browser signs in to tendriller.dev (the owner's account), creates its device key, and the agent program on the DGX
//     admits it (shared/helpers/devices-view.mjs lists the devices); the page says so while it waits;
//   - every failure is said in plain words (describeConnectionError) and retried.
import { start, showGate } from './app.mjs';
import { connect, loadConfiguration, previewSource } from './connect.mjs';
import { createOwnerConnection } from '../../../shared/connection.mjs';
import { mountConnectionView } from '../../../shared/connect-view.mjs';

const preview = await fetch('/__preview/ping', { cache: 'no-store' }).then(r => r.ok).catch(() => false);
if (preview) {
  start(previewSource());
} else {
  const holder = document.createElement('div'), view = mountConnectionView(holder, { hideWhenConnected: false });
  view.render({ phase: 'opening', title: 'Connecting…', detail: '' });
  showGate(holder);
  const configuration = await loadConfiguration();
  let source;
  const connection = createOwnerConnection({ configuration, appName: 'natlang ops',
    async connect({ signal } = {}) { source = await connect(configuration, { signal }); return { session: source.session, source }; },
    onState(state) { if (state.phase === 'connected' && source) start(source); else { view.render(state); showGate(holder); } } });
  void connection.start();
}
