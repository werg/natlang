// natlang dashboard, standalone: the rail, the hash router and the HTTP implementations of DashboardApi and RunHost, mounting
// the pages of pages.js. All state comes from the local API (src/dashboard/server.ts). A host embedding the dashboard
// does the same with its own api, run host and navigation.
import { availablePages, httpDashboardApi, httpRunHost } from './pages.js';
import { createH, fill, messageSummary } from './common.js';

const token = document.querySelector('meta[name="natlang-token"]').content;
const h = createH(document);
const api = httpDashboardApi({ token }), runs = httpRunHost(api);
const available = availablePages({ api, runs });
const root = document.getElementById('main'), nav = document.getElementById('nav'), railModel = document.getElementById('rail-model');

let toastTimer;
function toast(text, error = false) {
  const element = document.getElementById('toast');
  element.textContent = messageSummary(text); element.className = error ? 'error' : ''; element.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { element.hidden = true; }, error ? 7000 : 3500);
}

async function refreshRail() {
  try {
    const status = await api.status.get();
    const model = status.model;
    railModel.textContent = model.error ? `Model profile problem: ${model.error}` :
      `${model.profile} · ${model.release?.label ?? model.model}${model.provider ? ` (${model.provider})` : ''}`;
    const count = document.getElementById('running-count');
    if (count) { count.hidden = !status.jobs.running; count.textContent = status.jobs.running; }
  } catch (error) { railModel.textContent = error.message; }
}

fill(nav, available.map(page => h('a', { href: `#${page.id}`, 'data-view': page.id }, page.title,
  page.id === 'runs' ? h('span', { class: 'count', id: 'running-count', hidden: true }) : null)));

// Each route mounts into its own element: a page still loading when the route changes writes into a detached one.
let dispose = () => {};
function route() {
  dispose(); dispose = () => {};
  const [name, ...rest] = (location.hash.slice(1) || 'overview').split('-');
  const page = available.find(item => item.id === name) ?? available[0];
  for (const link of nav.querySelectorAll('a')) link.classList.toggle('active', link.dataset.view === page.id);
  const target = h('div');
  fill(root, target);
  dispose = page.mount(target, { api, runs, navigate: go, toast, refreshStatus: refreshRail, arg: rest.join('-') });
}
window.addEventListener('hashchange', route);
const go = hash => { if (location.hash === '#' + hash) route(); else location.hash = hash; };

refreshRail(); setInterval(refreshRail, 10000);
route();
