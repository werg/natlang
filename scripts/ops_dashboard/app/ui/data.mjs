// Training data: catalog, dataset browser, task editor, trajectory / SFT row / chain viewers, step selection, compare, edits.
// Routes live under #data/... (see route()). Everything is fetched on demand through ctx.src.query (native data service);
// edits are declared commands (proposeEdit) applied by the native participant to the errata / exclusion / annotation layers.
import { h } from './charts.mjs';
import { fmt, ago, dayClock, signed } from './model.mjs';
import { editRef } from '../ops-protocol.mjs';

const enc = encodeURIComponent, dec = decodeURIComponent;
export const link = {
  home: () => '#data',
  ds: (id, o = {}) => `#data/ds/${enc(id)}${qs(o)}`,
  item: (ds, id, o = {}) => `#data/item/${enc(ds)}/${enc(id)}${qs(o)}`,
  step: (ds, step) => `#data/step/${enc(ds)}/${step}`,
  chain: (ds, step, i) => `#data/chain/${enc(ds)}/${step}/${i}`,
  task: id => `#data/task/${enc(id)}`,
  edits: () => '#data/edits',
  search: q => `#data/search${qs({ q })}`,
  compare: (a, b) => `#data/compare${qs({ a, b })}`,
  run: runId => `#data/ds/${enc('grpo:' + String(runId).replace(/\./g, '/'))}`,
};
function qs(o) { const e = Object.entries(o).filter(([, v]) => v != null && v !== ''); return e.length ? '?' + e.map(([k, v]) => `${enc(k)}=${enc(v)}`).join('&') : ''; }
function parse(route) {
  const [path, query = ''] = String(route || '').split('?');
  const params = Object.fromEntries(query.split('&').filter(Boolean).map(kv => { const [k, v = ''] = kv.split('='); return [dec(k), dec(v)]; }));
  return { parts: path.split('/').filter(Boolean).map(dec), params };
}
const bytes = b => b == null ? '—' : b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${(b / 1e6).toFixed(1)} MB` : b >= 1e3 ? `${(b / 1e3).toFixed(0)} kB` : `${b} B`;
const count = n => n == null ? '—' : Number(n).toLocaleString('en-US');
const a = (href, ...kids) => h('a', { href }, ...kids);
const chip = (text, cls = '', title) => h('span', { class: `chip ${cls}`, title }, text);
const mono = t => h('span', { class: 'mono' }, t);
const pre = (t, cls = '') => h('pre', { class: `code ${cls}` }, t);
const json = v => JSON.stringify(v, null, 2);

/* ---------- fetch cache: one in-flight request per key; results kept for the session ---------- */
const cache = new Map();
function load(ctx, key, q, input, { ttl = 120000 } = {}) {
  const hit = cache.get(key);
  if (hit && (hit.state === 'loading' || Date.now() - hit.ts < ttl)) return hit;
  const ent = { state: 'loading', ts: Date.now(), value: hit?.value };
  cache.set(key, ent);
  ctx.src.query(q, input).then(v => { ent.state = 'ok'; ent.value = v; ent.ts = Date.now(); ctx.render(); },
    e => { ent.state = 'error'; ent.error = e.message; ent.ts = Date.now(); ctx.render(); });
  return ent;
}
const refresh = key => cache.delete(key);
function pending(ent, what) {
  if (ent.state === 'error') return h('div', { class: 'alert crit' }, h('span', { class: 'ic' }, '✖'), h('div', { class: 'what' }, `Could not load ${what}: ${ent.error}`));
  if (ent.value === undefined) return h('div', { class: 'loading' }, h('i', { class: 'spinner' }), `Loading ${what}…`);
  return null;
}

/* ---------- edits state (retained query joined with native application records) ---------- */
export function editsIndex(ctx) {
  const st = ctx.src.edits.get() || { edits: [], applications: [] };
  const native = ctx.src.nativeDeviceId;
  const appl = new Map();
  for (const x of st.applications) if (x.author === native || ctx.src.kind === 'preview') appl.set(x.edit, x);
  const byItem = new Map(), all = [];
  for (const e of st.edits) {
    const rec = { ...e, application: appl.get(editRef(e.operationId)) || null };
    all.push(rec);
    const k = `${e.target.dataset}|${e.target.id}`;
    if (!byItem.has(k)) byItem.set(k, []);
    byItem.get(k).push(rec);
  }
  for (const p of ctx.src.pending.get() || []) if (p.name === 'proposeEdit' && !all.some(e => e.operationId === p.id)) {
    const rec = { ...p.input, operationId: p.id, author: ctx.src.deviceId, application: null, local: p.state };
    all.push(rec);
    const k = `${p.input.target.dataset}|${p.input.target.id}`;
    if (!byItem.has(k)) byItem.set(k, []);
    byItem.get(k).push(rec);
  }
  return { all, byItem, status: (ds, id) => itemStatus(byItem.get(`${ds}|${id}`) || []) };
}
function itemStatus(list) {
  const s = { flagged: false, excluded: false, notes: 0, corrections: 0, pending: 0 };
  for (const e of list) {
    if (e.application?.status === 'refused' || e.application?.status === 'failed') continue;
    if (!e.application) s.pending++;
    if (e.action === 'flag') s.flagged = true; else if (e.action === 'unflag') s.flagged = false;
    else if (e.action === 'exclude') s.excluded = true; else if (e.action === 'include') s.excluded = false;
    else if (e.action === 'annotate') s.notes++; else if (e.action === 'correct') s.corrections++;
  }
  return s;
}
function statusChips(s) {
  return [s.excluded ? chip('⊘ excluded', 'crit', 'Excluded from training by an edit') : null, s.flagged ? chip('⚑ flagged', 'warn') : null,
    s.corrections ? chip(`✎ ${s.corrections} correction${s.corrections > 1 ? 's' : ''}`, 'evidence') : null, s.notes ? chip(`🗨 ${s.notes}`, '', 'annotations') : null,
    s.pending ? chip(`${s.pending} pending`, 'open', 'Submitted; waiting for the native participant to apply it') : null];
}

/* ---------- entry ---------- */
export function dataView(ctx) {
  const { parts, params } = parse(ctx.route);
  const v = h('div', { class: 'dataview' });
  v.append(toolbar(ctx, parts, params));
  try {
    const [kind, ...rest] = parts;
    const body = !kind ? catalog(ctx, params)
      : kind === 'ds' ? datasetPage(ctx, rest[0], params)
        : kind === 'item' ? itemPage(ctx, rest[0], rest.slice(1).join('/'), params)
          : kind === 'step' ? stepPage(ctx, rest[0], Number(rest[1]))
            : kind === 'chain' ? chainPage(ctx, rest[0], Number(rest[1]), Number(rest[2]))
              : kind === 'task' ? taskRefsPage(ctx, rest.join('/'))
                : kind === 'edits' ? editsPage(ctx, params)
                  : kind === 'search' ? searchPage(ctx, params.q || '')
                    : kind === 'compare' ? comparePage(ctx, params)
                      : h('div', { class: 'card muted' }, 'Unknown data view.');
    v.append(body);
  } catch (e) { v.append(h('div', { class: 'alert crit' }, `This view failed: ${e.message}`)); console.error(e); }
  return v;
}

function toolbar(ctx, parts, params) {
  const ed = editsIndex(ctx), open = ed.all.filter(e => !e.application).length;
  const input = h('input', { type: 'search', placeholder: 'Search datasets, task ids, families…', value: parts[0] === 'search' ? params.q || '' : '', 'aria-label': 'Search training data',
    onkeydown: e => { if (e.key === 'Enter' && e.target.value.trim()) location.hash = link.search(e.target.value.trim()); } });
  const crumbs = h('nav', { class: 'crumbs', 'aria-label': 'Breadcrumb' }, a(link.home(), 'Training data'));
  const cat = cache.get('catalog')?.value;
  const dsTitle = id => cat?.datasets.find(d => d.id === id)?.title || id;
  if (['ds', 'item', 'step', 'chain'].includes(parts[0]) && parts[1]) crumbs.append(' / ', a(link.ds(parts[1]), dsTitle(parts[1])));
  if (parts[0] === 'item') crumbs.append(' / ', h('span', {}, parts.slice(2).join('/')));
  if (parts[0] === 'step' || parts[0] === 'chain') crumbs.append(' / ', a(link.step(parts[1], parts[2]), `step ${parts[2]}`));
  if (parts[0] === 'chain') crumbs.append(' / ', h('span', {}, `chain ${parts[3]}`));
  if (parts[0] === 'task') crumbs.append(' / ', h('span', {}, `task ${parts.slice(1).join('/')}`));
  if (parts[0] === 'edits') crumbs.append(' / ', h('span', {}, 'Edits'));
  return h('div', { class: 'toolbar datatools' }, crumbs, h('span', { style: 'flex:1' }), input,
    h('a', { class: 'btn', href: link.edits() }, `Edits${ed.all.length ? ` (${ed.all.length}${open ? `, ${open} pending` : ''})` : ''}`));
}

/* ---------- catalog ---------- */
const KIND_LABEL = { 'rl-tasks': 'RL tasks', held: 'held', run: 'GRPO run', 'teacher-demos': 'teacher demos', trajectories: 'episodes', 'sft-rows': 'SFT rows', chains: 'chains', corpus: 'corpus', index: 'index' };
function catalog(ctx, params) {
  const ent = load(ctx, 'catalog', 'data.catalog', null, { ttl: 300000 });
  const p = pending(ent, 'the training-data catalog (the first build counts rows of large pools and can take a minute)');
  if (p) return p;
  const all = ent.value.datasets, ed = editsIndex(ctx);
  const kinds = [...new Set(all.map(d => d.kind))];
  const kf = params.kind || '', tf = (params.f || '').toLowerCase();
  const shown = all.filter(d => (!kf || d.kind === kf) && (!tf || `${d.id} ${d.title} ${d.path}`.toLowerCase().includes(tf)) && (params.consumed !== '1' || (d.consumers || []).length));
  const groups = new Map();
  for (const d of shown) { if (!groups.has(d.group)) groups.set(d.group, []); groups.get(d.group).push(d); }
  const total = all.reduce((s, d) => s + (d.rows || 0), 0), unreg = all.filter(d => !d.provenance_brief?.registered && d.kind !== 'run' && d.group !== 'Registries').length;
  const v = h('div', {});
  v.append(h('div', { class: 'grid g4' },
    statCard(count(all.length), 'datasets in the catalog'), statCard(count(total), 'rows / tasks / episodes counted'),
    statCard(count(all.filter(d => d.protected).length), 'protected (held-out, view only)'),
    statCard(count(unreg), 'not in the corpus registry', 'Found through code constants, run directories or the catalog hand layer: see the discoverability findings.')));
  const nav = (o) => link.home() + qs({ kind: kf, f: params.f, consumed: params.consumed, ...o });
  v.append(h('div', { class: 'toolbar', style: 'margin-top:12px' },
    h('span', { class: 'seg-ctl', role: 'group', 'aria-label': 'Kind' }, h('button', { 'aria-pressed': !kf, onclick: () => { location.hash = nav({ kind: '' }); } }, 'All'),
      kinds.map(k => h('button', { 'aria-pressed': kf === k, onclick: () => { location.hash = nav({ kind: k }); } }, KIND_LABEL[k] || k))),
    h('input', { type: 'search', placeholder: 'Filter by name or path', value: params.f || '', onchange: e => { location.hash = nav({ f: e.target.value }); } }),
    h('label', { class: 'small' }, h('input', { type: 'checkbox', checked: params.consumed === '1', onchange: e => { location.hash = nav({ consumed: e.target.checked ? '1' : '' }); } }), ' only data a run consumed'),
    h('span', { class: 'muted small' }, `catalog built ${ago(ent.value.generated, Date.now())}`), h('button', { class: 'btn sm', onclick: () => { refresh('catalog'); ctx.render(); } }, 'Rebuild')));
  for (const [g, list] of groups) {
    const rows = list.map(d => {
      const st = ed.all.filter(e => e.target.dataset === d.id).length;
      const pb = d.provenance_brief || {};
      return h('tr', { class: 'clickrow', onclick: e => { if (!e.target.closest('a')) location.hash = link.ds(d.id); } },
        h('td', {}, a(link.ds(d.id), d.title), d.protected ? chip('🔒 protected', 'crit') : null, d.missing ? chip('missing', 'warn') : null,
          h('div', { class: 'small muted mono ellip', title: d.path }, d.path)),
        h('td', {}, chip(KIND_LABEL[d.kind] || d.kind)),
        h('td', { class: 'n', title: d.rows_estimated ? 'estimated from a 4 MB sample (file on the HDD)' : '' }, d.rows_estimated ? '~' + count(d.rows) : count(d.rows)), h('td', { class: 'n' }, bytes(d.size_bytes)),
        h('td', { class: 'small' }, pb.registered ? chip('registry', 'ok') : chip('unregistered', 'warn', 'No entry in training/neuralese_corpora.json'), pb.code_ref ? h('div', { class: 'muted mono' }, pb.code_ref) : null,
          pb.producer ? h('div', { class: 'muted' }, pb.producer) : null, pb.license ? h('div', { class: 'muted ellip', title: pb.license }, `license: ${String(pb.license).slice(0, 60)}`) : null),
        h('td', { class: 'small' }, consumersBrief(d)),
        h('td', { class: 'small' }, Object.entries(d.relations || {}).map(([k, x]) => h('div', {}, h('span', { class: 'muted' }, k.replace(/_/g, ' ') + ': '), String(x)))),
        h('td', { class: 'n' }, st ? chip(`${st} edit${st > 1 ? 's' : ''}`, 'evidence') : ''));
    });
    v.append(h('div', { class: 'sect' }, h('h2', {}, g), h('span', { class: 'muted' }, `${list.length} dataset${list.length > 1 ? 's' : ''}`)),
      h('div', { class: 'card flush' }, h('div', { class: 'scroll', style: 'max-height:none' }, h('table', { class: 'cat' },
        h('thead', {}, h('tr', {}, ['dataset', 'kind', 'rows', 'size', 'provenance', 'consumed by', 'relations', ''].map((c, i) => h('th', { class: i === 2 || i === 3 ? 'n' : '' }, c)))), h('tbody', {}, rows)))));
  }
  v.append(h('p', { class: 'muted small' }, 'Sources: ', mono('grpo.py'), ' constants (POOLS, MEMETIC_TASKS, HELD, ERRATA, SD_TEACHER_SOURCES), run directories under ops/, ', mono('training/neuralese_corpora.json'),
    ', and the explicit hand layer ', mono('scripts/ops_dashboard/catalog_sources.json'), ' for what no registry records. Row counts are line counts; zstd files are counted when indexed.'));
  return v;
}
function consumersBrief(d) {
  const by = new Map();
  for (const c of d.consumers || []) { if (!by.has(c.how)) by.set(c.how, []); by.get(c.how).push(c.run); }
  if (!by.size) return d.consumers_note ? h('div', { class: 'muted' }, d.consumers_note) : h('span', { class: 'muted' }, '—');
  return [...by].map(([how, runs]) => {
    const sorted = [...new Set(runs)].sort().reverse();
    return h('div', {}, h('span', { class: 'muted' }, `${how}: `), sorted.slice(0, 3).map((r, i) => [i ? ', ' : '', r.includes(':') && !r.startsWith('sft-run') ? a(link.ds(r), r.replace(/^[a-z-]+:/, '')) : r]),
      sorted.length > 3 ? h('span', { class: 'muted' }, ` +${sorted.length - 3}`) : null);
  });
}
const statCard = (value, label, title) => h('div', { class: 'card', title }, h('div', { class: 'stat' }, h('b', {}, value), h('span', {}, label)));

/* ---------- dataset page ---------- */
function datasetPage(ctx, id, params) {
  const ent = load(ctx, `ds:${id}`, 'data.dataset', { ds: id });
  const p = pending(ent, id);
  if (p) return p;
  const ds = ent.value, v = h('div', {});
  v.append(datasetHeader(ctx, ds));
  const parts = ds.parts || [];
  if (!parts.length) { v.append(h('div', { class: 'card muted' }, 'This dataset has no row reader yet; its files and provenance are listed above.')); return v; }
  const part = params.part || parts[0].id;
  if (parts.length > 1) v.append(h('div', { class: 'tabs2', role: 'tablist' }, parts.map(pt => h('a', { role: 'tab', class: 'tab2', 'aria-selected': pt.id === part, href: link.ds(id, { part: pt.id }) }, pt.title))));
  v.append(itemsTable(ctx, ds, part, params));
  return v;
}

function kvCols(pairs) { const el = kv(pairs); el.classList.add('cols3'); return el; }
function kv(pairs) {
  return h('dl', { class: 'kvgrid' }, pairs.filter(([, x]) => x != null && x !== '' && !(Array.isArray(x) && !x.length)).map(([k, x]) => [h('dt', {}, k), h('dd', {}, x instanceof Node ? x : typeof x === 'object' ? pre(json(x), 'small') : String(x))]));
}

function datasetHeader(ctx, ds) {
  const prov = ds.provenance || {};
  const card = h('div', { class: 'card dshead' }, h('header', {}, h('div', {}, h('h2', {}, ds.title, ' ', ds.protected ? chip('🔒 protected · view only', 'crit') : null),
    h('div', { class: 'mono small muted' }, ds.path, ' ', h('button', { class: 'btn sm', onclick: () => navigator.clipboard?.writeText(ds.path) }, 'Copy path'))), chip(KIND_LABEL[ds.kind] || ds.kind)));
  if (ds.note) card.append(h('div', { class: 'notice' }, ds.note));
  const cols = h('div', { class: 'grid g3' });
  cols.append(h('div', {}, h('h3', {}, 'What'), kv([['format', ds.format], ['rows', count(ds.rows)], ['size', bytes(ds.size_bytes)], ['changed', ds.mtime ? dayClock(ds.mtime * 1000) : null],
    ['run code', ds.code ? `${ds.code.git_commit?.slice(0, 10)}${ds.code.dirty ? ' (dirty)' : ''}` : null], ['rollouts', ds.rollouts]])));
  const reg = prov.registry;
  cols.append(h('div', {}, h('h3', {}, 'Provenance'), kv([['registry', reg ? `${reg.id} · ${reg.admission || ''}` : 'not registered'], ['license', reg?.license_provenance || reg?.license],
    ['code', prov.code_ref], ['producer', prov.producer], ['records', (prov.records || []).join(', ')], ['receipt', prov.receipt], ['pool', prov.pool], ['eligible', prov.eligible]])));
  cols.append(h('div', {}, h('h3', {}, 'Consumed by'), (ds.consumers || []).length ? h('ul', { class: 'plain' }, ds.consumers.map(c => h('li', {}, c.run.startsWith('grpo:') || c.run.startsWith('teacher-gen:') || c.run.startsWith('sft:') ? a(link.ds(c.run), c.run) : c.run,
    h('div', { class: 'small muted' }, c.how, c.amount ? ` · ${c.amount}` : '')))) : h('div', { class: 'muted small' }, ds.consumers_note || 'No run records reading it.'),
    Object.keys(ds.relations || {}).length ? [h('h3', { style: 'margin-top:8px' }, 'Relations'), kv(Object.entries(ds.relations).map(([k, x]) => [k.replace(/_/g, ' '), Array.isArray(x) ? x.join(', ') : x]))] : null));
  card.append(cols);
  if (ds.mix) card.append(mixView(ds.mix));
  if (ds.held_manifest) card.append(h('details', {}, h('summary', {}, 'Held manifest'), pre(json(ds.held_manifest))));
  if (ds.args) card.append(h('details', {}, h('summary', {}, 'Launch arguments (selection settings)'), kv(Object.entries(ds.args).filter(([k]) => /group|band|explore|stale|reward|pool|exclude|tail|lr|slots|task|variant|memetic|budget|replay|sd-/.test(k)).map(([k, x]) => [`--${k}`, String(x)]))));
  return card;
}

function mixView(mix) {
  const parts = Object.entries(mix.parts || {});
  const colors = ['var(--s1)', 'var(--s2)', 'var(--s3)', 'var(--s4)', 'var(--s5)', 'var(--s7)'];
  const box = h('div', { style: 'margin-top:10px' }, h('h3', {}, 'Mix: loss share by source'));
  if (parts.length) {
    const bar = h('div', { class: 'seg', style: 'height:22px' });
    parts.forEach(([k, x], i) => bar.append(h('i', { style: `flex:0 0 ${(100 * (x.loss_share || 0)).toFixed(2)}%;background:${colors[i % colors.length]}`, title: `${k}: ${(100 * (x.loss_share || 0)).toFixed(1)}%` })));
    box.append(bar, h('table', { class: 'data' }, h('thead', {}, h('tr', {}, ['source', 'chains', 'forward tokens', 'weighted target tokens', 'loss share', 'forward share'].map(c => h('th', {}, c)))),
      h('tbody', {}, parts.map(([k, x], i) => h('tr', {}, h('td', {}, h('span', { class: 'swatch', style: `background:${colors[i % colors.length]}` }), k), h('td', { class: 'n' }, count(x.chains)), h('td', { class: 'n' }, count(x.forward_tokens)),
        h('td', { class: 'n' }, count(x.weighted_target_tokens)), h('td', { class: 'n' }, x.loss_share != null ? `${(100 * x.loss_share).toFixed(1)}%` : '—'), h('td', { class: 'n' }, x.forward_share != null ? `${(100 * x.forward_share).toFixed(1)}%` : '—'))))));
  }
  if (mix.funnel) box.append(h('details', {}, h('summary', {}, 'Selection funnel (why rows were admitted or dropped)'), funnelTable(mix.funnel)));
  if (mix.aux) box.append(h('details', {}, h('summary', {}, 'Auxiliary inputs'), pre(json(mix.aux))));
  if (mix.inputs) box.append(h('details', {}, h('summary', {}, 'Inputs'), pre(json(mix.inputs))));
  if (mix.policy) box.append(h('details', {}, h('summary', {}, 'Build policy'), pre(json(mix.policy))));
  return box;
}
function funnelTable(f) {
  const srcs = Object.keys(f), keys = [...new Set(srcs.flatMap(s => Object.keys(f[s] || {})))];
  return h('div', { class: 'scroll' }, h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, 'stage'), srcs.map(s => h('th', { class: 'n' }, s)))),
    h('tbody', {}, keys.map(k => h('tr', {}, h('th', {}, k.replace(/_/g, ' ')), srcs.map(s => h('td', { class: 'n' }, typeof f[s]?.[k] === 'object' ? JSON.stringify(f[s][k]).slice(0, 40) : count(f[s]?.[k]))))))));
}

const COLS = {
  task: [['id', 'task'], ['family', 'family'], ['level', 'lvl'], ['constructs', 'constructs'], ['world', 'world files'], ['oracle', 'oracle']],
  sft_row: [['id', 'row'], ['source', 'source'], ['family', 'family'], ['call', 'call'], ['tool', 'action'], ['target_tokens', 'target tok'], ['loss_from', 'zero-weight lead']],
  demo: [['id', 'episode'], ['family', 'family'], ['teacher', 'teacher'], ['calls', 'calls'], ['base_rank', 'base rank']],
  manifest: [['id', 'task'], ['family', 'family'], ['teacher', 'teacher'], ['episodes', 'episodes'], ['successes', 'successes'], ['outcome', 'first outcome'], ['base_rank', 'base rank']],
  steps: [['step', 'step'], ['groups', 'groups'], ['groups_with_spread', 'with spread'], ['episodes', 'episodes'], ['reward_mean', 'reward'], ['dropped_stale', 'stale dropped'], ['unscored', 'unscored'], ['truncated', 'truncated'], ['budget_stopped', 'budget-stopped'], ['tail_masked', 'tail masked'], ['second_close', '2nd close'], ['flags', 'sentinels'], ['chains_kept', 'batch kept']],
  groups: [['group', 'group'], ['task_id', 'task'], ['family', 'family'], ['n', 'n'], ['mean', 'mean reward'], ['spread', 'spread'], ['versions', 'actor versions'], ['outcomes', 'outcomes']],
  held: [['id', 'eval'], ['delta_pts', 'Δ pts vs base'], ['p', 'p'], ['success_rate', 'pass rate'], ['ran', 'ran'], ['complete', 'complete']],
  samples: [['task_id', 'task'], ['family', 'family'], ['sample', 's'], ['outcome', 'outcome'], ['reward', 'reward'], ['model_calls', 'calls'], ['completion_tokens', 'tokens']],
  entries: [['id', 'entry'], ['path', 'path'], ['owner', 'owner'], ['admission', 'admission'], ['training_admission', 'training'], ['superseded_by', 'superseded by']],
  errata: [['id', 'task'], ['family', 'family'], ['reason', 'reason'], ['source', 'source']],
  raw: [['id', 'id'], ['family', 'family'], ['source', 'source'], ['keys', 'keys']],
};
function cell(k, x) {
  if (x == null) return '—';
  if (k === 'outcomes' && typeof x === 'object') return Object.entries(x).map(([o, n]) => `${o.replace('task_', '')} ${n}`).join(', ');
  if (k === 'outcome') return h('span', { class: `chip ${x === 'task_success' ? 'ok' : 'crit'}` }, String(x).replace('task_', '').replace(/_/g, ' '));
  if (typeof x === 'boolean') return x ? '✔' : '';
  if (Array.isArray(x)) return x.length ? x.join(', ') : '';
  if (k === 'reward_mean' || k === 'mean' || k === 'success_rate' || k === 'tail_masked') return fmt(x, 3);
  if (k === 'delta_pts') return signed(x, 1);
  if (typeof x === 'number') return count(x);
  return String(x);
}
function itemsTable(ctx, ds, part, params) {
  const kind = ds.reader === 'grpo-run' ? part : ds.reader === 'held-eval' ? 'samples' : ds.reader === 'registry' ? 'entries' : (ds.part_kind || {})[part] || ds.item_kind || 'raw';
  const filter = Object.fromEntries(Object.entries(params).filter(([k]) => k.startsWith('f.')).map(([k, x]) => [k.slice(2), x]));
  const input = { ds: ds.id, part, q: params.q || '', filter, sort: params.sort || '', offset: Number(params.o || 0), limit: 50 };
  const key = `items:${JSON.stringify(input)}`;
  const ent = load(ctx, key, 'data.items', input);
  const nav = o => link.ds(ds.id, { part, q: params.q, sort: params.sort, ...Object.fromEntries(Object.entries(filter).map(([k, x]) => [`f.${k}`, x])), ...o });
  const box = h('div', { class: 'grid', style: 'grid-template-columns:minmax(0,220px) minmax(0,1fr);gap:14px;margin-top:10px' });
  const search = h('input', { type: 'search', placeholder: kind === 'steps' ? 'Filter steps…' : 'Search inside this dataset (raw text)…', value: params.q || '', style: 'width:100%',
    onkeydown: e => { if (e.key === 'Enter') location.hash = nav({ q: e.target.value, o: '' }); } });
  const side = h('aside', { class: 'card facets' }, h('h3', {}, 'Filter'), search);
  const main = h('div', { class: 'card flush' });
  box.append(side, main);
  const p = pending(ent, 'items');
  if (p) { main.append(p); return box; }
  const r = ent.value, ed = editsIndex(ctx);
  for (const [fk, vals] of Object.entries(r.facets || {})) {
    if (!vals.length) continue;
    side.append(h('h4', {}, fk), h('div', { class: 'facet' }, vals.slice(0, 25).map(([val, n]) => h('a', { class: `facetrow${filter[fk] === String(val) ? ' on' : ''}`, href: nav({ [`f.${fk}`]: filter[fk] === String(val) ? '' : val, o: '' }) }, h('span', { class: 'ellip' }, String(val)), h('span', { class: 'muted' }, count(n))))));
  }
  if (Object.values(filter).some(Boolean)) side.append(h('a', { class: 'btn sm', href: nav(Object.fromEntries(Object.keys(filter).map(k => [`f.${k}`, '']))) }, 'Clear filters'));
  const cols = COLS[kind] || COLS.raw;
  const itemHref = row => kind === 'steps' ? link.step(ds.id, row.step) : link.item(ds.id, row.id ?? row._loc, { part });
  main.append(h('div', { class: 'tablehead' }, h('b', {}, `${count(r.total)} ${kind === 'steps' ? 'steps' : 'items'}`), r.searched ? h('span', { class: 'muted' }, ` matching “${params.q}”`) : null,
    h('span', { style: 'flex:1' }), pager(r, nav)));
  const tbody = h('tbody', {}, r.rows.map(row => {
    const st = ed.status(ds.id, String(row.id ?? row._loc));
    return h('tr', { class: 'clickrow', onclick: e => { if (!e.target.closest('a,button')) location.hash = itemHref(row); } },
      cols.map(([k], i) => h('td', { class: typeof row[k] === 'number' ? 'n' : i === 0 ? 'mono' : '' }, i === 0 ? [a(itemHref(row), cell(k, row[k] ?? row._loc)), ' ', ...statusChips(st)] : cell(k, row[k]))));
  }));
  main.append(h('div', { class: 'scroll', style: 'max-height:none' }, h('table', { class: 'items' }, h('thead', {}, h('tr', {}, cols.map(([k, label]) => h('th', {}, h('a', { href: nav({ sort: params.sort === k ? `-${k}` : k, o: '' }) }, label, params.sort === k ? ' ▲' : params.sort === `-${k}` ? ' ▼' : ''))))), tbody)));
  if (!r.rows.length) main.append(h('div', { class: 'empty' }, 'No items match.'));
  main.append(h('div', { class: 'tablehead' }, h('span', { style: 'flex:1' }), pager(r, nav)));
  return box;
}
function pager(r, nav) {
  const lim = 50, off = r.offset || 0;
  return h('span', { class: 'row' }, h('a', { class: 'btn sm', href: nav({ o: Math.max(0, off - lim) || '' }), 'aria-disabled': off === 0 }, '‹ Prev'),
    h('span', { class: 'muted small' }, r.total ? `${count(off + 1)}–${count(Math.min(off + lim, r.total))} of ${count(r.total)}` : ''),
    h('a', { class: 'btn sm', href: nav({ o: off + lim }), 'aria-disabled': off + lim >= r.total }, 'Next ›'));
}

/* ---------- item page ---------- */
function itemPage(ctx, dsId, id, params) {
  const input = { ds: dsId, id, part: params.part };
  const ent = load(ctx, `item:${JSON.stringify(input)}`, 'data.item', input);
  const p = pending(ent, id);
  if (p) return p;
  const it = ent.value, protectedDs = (cache.get(`ds:${dsId}`)?.value || cache.get('catalog')?.value?.datasets.find(d => d.id === dsId) || {}).protected;
  const target = { kind: targetKind(it, dsId, protectedDs), dataset: dsId, id: String(it.id ?? id), ref: String(it.ref || '').slice(0, 1024), label: String(it.task_id || it.family || '').slice(0, 300) };
  const body = it.kind === 'task' ? taskEditor(ctx, it, target)
    : it.kind === 'trajectory' ? trajectoryView(ctx, it, { dsId, id })
      : it.kind === 'sft_row' ? sftRowView(ctx, it)
        : it.kind === 'group' ? groupView(ctx, it, dsId)
          : it.kind === 'step' ? stepBody(ctx, it, dsId)
            : it.kind === 'teacher_task' ? teacherTaskView(ctx, it)
              : it.kind === 'registry_entry' ? registryView(it)
                : h('div', { class: 'card' }, pre(json(it.value ?? it)));
  if (['group', 'step'].includes(it.kind)) return body;
  return h('div', { class: 'itemlayout' }, h('div', { class: 'itemmain' }, body), h('aside', { class: 'itemside' }, editPanel(ctx, target, it)));
}
function targetKind(it, dsId, prot) {
  if (prot || it.held && it.kind === 'task') return it.kind === 'trajectory' ? 'held_episode' : 'held_task';
  return { task: 'task', trajectory: dsId.startsWith('grpo:') ? 'episode' : 'teacher_demo', sft_row: 'sft_row', teacher_task: 'task', registry_entry: 'corpus' }[it.kind] || 'other';
}

/* ---------- edit panel ---------- */
const ACTIONS = [['flag', '⚑ Flag', 'Mark for review; nothing changes in training.'], ['exclude', '⊘ Exclude from training', 'Adds the id to the exclusion layer (RL tasks: ops/pool-errata/ops-dashboard.jsonl, read at the next GRPO start or resume).'],
  ['annotate', '🗨 Annotate', 'A note shown wherever this item appears.'], ['correct', '✎ Correct', 'Propose corrected content for one field; recorded in the corrections layer for re-qualification, never written into the source file.']];
const UNDO = { flag: 'unflag', exclude: 'include' };
function editPanel(ctx, target, it) {
  const ed = editsIndex(ctx), list = ed.byItem.get(`${target.dataset}|${target.id}`) || [], st = itemStatus(list);
  const box = h('div', { class: 'card editpanel' }, h('h3', {}, 'Review and edit'));
  if (target.kind.startsWith('held_')) {
    box.append(h('div', { class: 'notice' }, '🔒 Held-out data is view-only. Edits to held tasks or held-eval episodes are refused by the application (AGENTS.md held-data separation); change the held set only through a new held manifest.'));
  } else {
    box.append(h('div', { class: 'row' }, ...statusChips(st), !st.flagged && !st.excluded && !st.notes && !st.corrections ? h('span', { class: 'muted small' }, 'No edits yet.') : null));
    const form = editForm(ctx, target, it, st);
    box.append(form);
  }
  if (list.length) box.append(h('h4', {}, 'History'), h('ol', { class: 'history' }, [...list].reverse().map(e => editRow(e, ctx))));
  box.append(h('div', { class: 'small muted', style: 'margin-top:8px' }, 'Target: ', mono(`${target.kind} · ${target.dataset} · ${target.id}`)));
  return box;
}
function editRow(e, ctx) {
  const ap = e.application;
  const state = ap ? { applied: chip('✔ applied', 'ok'), refused: chip('refused', 'crit'), failed: chip('failed', 'crit'), superseded: chip('superseded') }[ap.status] || chip(ap.status)
    : chip(e.local ? `local: ${e.local}` : 'waiting for native', 'open', 'The native participant applies edits to the data layers and records the result.');
  return h('li', {}, h('div', { class: 'row' }, h('b', {}, e.action), state, h('span', { class: 'muted small mono' }, (e.author || '').slice(0, 8))),
    e.text ? h('div', {}, e.text) : null, e.field ? h('div', { class: 'small muted' }, `field: ${e.field}`) : null,
    e.correction?.length ? h('details', {}, h('summary', {}, 'correction'), pre(e.correction.join(''))) : null,
    ap?.detail ? h('div', { class: 'small muted' }, ap.detail) : null, ap?.files ? h('div', { class: 'small mono muted' }, ap.files) : null,
    !ap && e.local === 'uncertain' ? h('button', { class: 'btn sm', onclick: () => ctx.src.retry(e.operationId) }, 'Retry same operation') : null);
}
const FIELDS = { task: [['oracle.expected', 'Oracle: expected value'], ['program.inputs', 'Inputs'], ['program.file', 'A program file'], ['world.folder_files', 'World file'], ['other', 'Other (describe)']],
  episode: [['outcome', 'Outcome / reward'], ['call', 'One call'], ['other', 'Other']], teacher_demo: [['call', 'One call'], ['outcome', 'Outcome'], ['other', 'Other']],
  sft_row: [['target', 'Target reply'], ['weights', 'Trained / masked parts'], ['other', 'Other']], other: [['other', 'Other']] };
function currentValue(it, field, sub) {
  if (field === 'oracle.expected') return json(it.oracle?.expected);
  if (field === 'program.inputs') return json(it.program?.inputs);
  if (field === 'program.file') return it.program?.files?.[sub] ?? '';
  if (field === 'world.folder_files') return it.world?.folder_files?.[sub] ?? '';
  if (field === 'target') return json(it.target);
  if (field === 'call') return json(it.calls?.[Number(sub) || 0]?.target);
  return '';
}
function editForm(ctx, target, it, st) {
  const S = ctx.ui;   // per-view form state survives re-renders
  const key = `${target.dataset}|${target.id}`;
  const f = S.forms[key] ||= { action: 'flag', text: '', field: (FIELDS[target.kind] || FIELDS.other)[0][0], sub: '', correction: null, status: '' };
  const form = h('form', { class: 'editform', onsubmit: async e => {
    e.preventDefault();
    if (['flag', 'exclude', 'correct'].includes(f.action) && !f.text.trim()) { f.status = 'Write a reason first.'; ctx.render(); return; }
    const input = { target, action: f.action, text: f.text.trim().slice(0, 4000) };
    if (f.action === 'correct') {
      const c = f.correction ?? currentValue(it, f.field, f.sub);
      if (f.field.endsWith('expected') || f.field.endsWith('inputs')) { try { JSON.parse(c); } catch { f.status = 'The correction is not valid JSON.'; ctx.render(); return; } }
      input.correction = c.match(/[\s\S]{1,16000}/g) || [''];
      input.correctionFormat = f.field.endsWith('expected') || f.field.endsWith('inputs') ? 'json' : 'text';
      input.field = `${f.field}${f.sub ? ':' + f.sub : ''}`.slice(0, 200);
    }
    if (it.digest) input.basedOn = it.digest;
    f.status = 'Submitting…'; ctx.render();
    try {
      const r = await ctx.src.command('proposeEdit', input);
      f.status = r.status === 'applied' ? (r.outcome === 'accepted' ? 'Recorded. The native participant applies it to the data layers.' : `Rejected: ${r.rejection || r.outcome}`) : 'Saved on this device; waiting for the native participant.';
      if (r.outcome === 'accepted' || r.status !== 'applied') Object.assign(f, { text: '', correction: null, action: 'flag' });
    } catch (err) { f.status = `Not submitted: ${err.message}`; }
    ctx.render();
  } });
  const actions = ACTIONS.map(([act, label]) => {
    const undo = (act === 'flag' && st.flagged) || (act === 'exclude' && st.excluded);
    const val = undo ? UNDO[act] : act;
    return h('label', { class: `actopt${f.action === val ? ' on' : ''}` }, h('input', { type: 'radio', name: `act-${key}`, value: val, checked: f.action === val, onchange: () => { f.action = val; ctx.render(); } }),
      undo ? (act === 'flag' ? '⚐ Remove flag' : '↺ Include again') : label);
  });
  form.append(h('div', { class: 'actopts' }, actions), h('div', { class: 'small muted' }, (ACTIONS.find(x => x[0] === f.action) || ['', '', f.action === 'include' ? 'Removes the id from the exclusion layer.' : 'Clears the flag.'])[2]));
  if (f.action === 'correct') {
    const fields = FIELDS[target.kind] || FIELDS.other;
    form.append(h('label', {}, 'Field ', h('select', { onchange: e => { f.field = e.target.value; f.correction = null; f.sub = ''; ctx.render(); } }, fields.map(([v2, l]) => h('option', { value: v2, selected: f.field === v2 }, l)))));
    const subs = f.field === 'program.file' ? Object.keys(it.program?.files || {}) : f.field === 'world.folder_files' ? Object.keys(it.world?.folder_files || {}) : f.field === 'call' ? (it.calls || []).map(c => String(c.i)) : null;
    if (subs) { if (!f.sub) f.sub = subs[0] || ''; form.append(h('select', { onchange: e => { f.sub = e.target.value; f.correction = null; ctx.render(); } }, subs.map(s => h('option', { value: s, selected: f.sub === s }, f.field === 'call' ? `call ${s}` : s)))); }
    const cur = currentValue(it, f.field, f.sub);
    const live = h('div', {});
    const showDiff = () => live.replaceChildren(...(f.correction != null && f.correction !== cur ? [h('div', { class: 'small muted' }, 'Change against the current value:'), diffView(cur, f.correction)] : []));
    const ta = h('textarea', { rows: 10, class: 'mono', spellcheck: 'false', oninput: e => { f.correction = e.target.value; showDiff(); } });
    ta.value = f.correction ?? cur;
    form.append(h('div', { class: 'small muted' }, 'Corrected content (prefilled with the current value):'), ta, live);
    showDiff();
  }
  const reason = h('textarea', { rows: 3, placeholder: f.action === 'annotate' ? 'Your note' : 'Reason (required for flag, exclude and correct)', maxlength: 4000, oninput: e => { f.text = e.target.value; } });
  reason.value = f.text;
  form.append(reason, h('div', { class: 'row' }, h('button', { class: 'btn primary', type: 'submit' }, 'Submit edit'), h('span', { class: 'small muted', role: 'status' }, f.status)));
  return form;
}
function diffView(a0, b0) {
  const A = String(a0).split('\n'), B = String(b0).split('\n'), out = h('div', { class: 'diff mono' });
  let i = 0, j = 0;
  while (i < A.length || j < B.length) {
    if (i < A.length && j < B.length && A[i] === B[j]) { out.append(h('div', { class: 'd0' }, '  ' + A[i])); i++; j++; continue; }
    const nb = B.indexOf(A[i], j), na = A.indexOf(B[j], i);
    if (i < A.length && (nb < 0 || (na >= 0 && na - i <= nb - j))) out.append(h('div', { class: 'd-' }, '− ' + A[i++]));
    else out.append(h('div', { class: 'd+' }, '+ ' + B[j++]));
  }
  return out;
}

/* ---------- task editor ---------- */
function taskEditor(ctx, t, target) {
  const v = h('div', { class: 'task' });
  v.append(h('div', { class: 'card' }, h('header', {}, h('div', {}, h('h2', { class: 'mono' }, t.id), h('div', { class: 'row', style: 'margin-top:4px' },
    chip(t.family || '—'), t.level != null ? chip(`level ${t.level}`) : null, t.split ? chip(t.split) : null, t.held ? chip('🔒 held-out', 'crit') : null,
    (t.constructs || []).map(c => chip(c, 'evidence')))), h('div', { class: 'row' }, a(link.task(t.id), 'Everywhere this task appears →'))),
    t.held ? h('div', { class: 'notice' }, 'This task id or its split group is in the held-out exclusion set: training never sees it, and it is view-only here.') : null,
    whereView(t.where)));
  const sel = ctx.ui.taskTab[t.id] || 'program';
  const tabs = [['program', `Program (${Object.keys(t.program.files || {}).length} files)`], ['world', `World (${Object.keys(t.world.folder_files || {}).length} files)`], ['oracle', 'Oracle'], ['qualification', 'Qualification'], ['curriculum', 'Curriculum'], ['provenance', 'Provenance']];
  v.append(h('div', { class: 'tabs2', role: 'tablist' }, tabs.map(([k, l]) => h('button', { role: 'tab', class: 'tab2', 'aria-selected': sel === k, onclick: () => { ctx.ui.taskTab[t.id] = k; ctx.render(); } }, l))));
  const pane = h('div', { class: 'card' });
  const corr = (editsIndex(ctx).byItem.get(`${target.dataset}|${target.id}`) || []).filter(e => e.action === 'correct' && !['refused', 'failed'].includes(e.application?.status));
  const fieldOf = { program: ['program.file', 'program.inputs'], world: ['world.folder_files'], oracle: ['oracle.expected'] }[sel] || [];
  for (const e of corr.filter(e => fieldOf.some(f => (e.field || '').startsWith(f)))) {
    const [field, sub] = String(e.field).split(':');
    pane.append(h('div', { class: 'proposal' }, h('div', { class: 'row' }, chip('✎ proposed correction', 'evidence'), h('b', { class: 'mono' }, e.field), h('span', { class: 'muted small' }, e.application ? `${e.application.status} · ${e.application.detail || ''}` : 'waiting for the native participant')),
      h('div', { class: 'small' }, e.text), diffView(currentValue(t, field, sub), (e.correction || []).join(''))));
  }
  if (sel === 'program') pane.append(fileBrowser(ctx, `p:${t.id}`, t.program.files, t.program.root), h('h3', { style: 'margin-top:12px' }, 'Inputs'), pre(json(t.program.inputs)));
  else if (sel === 'world') {
    pane.append(kv([['kind', t.world.kind], ['spec', t.world.spec]]));
    if (Object.keys(t.world.folder_files || {}).length) pane.append(fileBrowser(ctx, `w:${t.id}`, t.world.folder_files));
    if (t.world.evidence && Object.values(t.world.evidence).some(x => x?.length)) pane.append(h('h3', {}, 'Evidence'), pre(json(t.world.evidence)));
  } else if (sel === 'oracle') {
    pane.append(kv([['kind', t.oracle.kind], ['depth', t.oracle.depth], ['files oracle', t.oracle.files_oracle ? json(t.oracle.files_oracle) : null]]), h('h3', {}, 'Expected value'), pre(json(t.oracle.expected)));
    const ef = t.oracle.expected_files || {}, wf = t.world.folder_files || {};
    if (Object.keys(ef).length) {
      pane.append(h('h3', {}, 'Expected files (diff against the starting world)'));
      for (const [name, text] of Object.entries(ef)) pane.append(h('details', { open: Object.keys(ef).length <= 4 }, h('summary', {}, mono(name), name in wf ? (wf[name] === text ? chip('unchanged') : chip('changed', 'warn')) : chip('new', 'ok')),
        name in wf && wf[name] !== text ? diffView(wf[name], text) : pre(text)));
      const removed = Object.keys(wf).filter(n => !(n in ef));
      if (removed.length) pane.append(h('div', { class: 'small' }, h('b', {}, 'Moved or removed: '), removed.map(n => mono(n + ' '))));
    }
  } else if (sel === 'qualification') {
    const q = t.qualification;
    pane.append(kv([['basis', q.basis], ['gold sources', (q.gold_sources || []).join(', ')]]));
    if (q.wrong_answers?.length) pane.append(h('h3', {}, `Wrong answers the oracle must reject (${q.wrong_answers.length})`), pre(json(q.wrong_answers)));
    if (q.alias_answers?.length) pane.append(h('h3', {}, 'Alias answers the oracle must accept'), pre(json(q.alias_answers)));
    if (q.wrong_code?.length) pane.append(h('h3', {}, `Wrong code (negative replays, ${q.wrong_code.length})`), ...q.wrong_code.map(c => pre(typeof c === 'string' ? c : json(c))));
    if (q.reference) pane.append(h('h3', {}, 'Reference solution'), referenceView(q.reference));
    if (q.smoke) pane.append(h('h3', {}, 'Smoke'), pre(json(q.smoke)));
    if (!q.wrong_answers?.length && !q.wrong_code?.length && !q.reference) pane.append(h('div', { class: 'muted' }, 'This row carries no qualification checks; its pool may keep them in side files (qualification.json, verification/).'));
  } else if (sel === 'curriculum') pane.append(kv(Object.entries(t.curriculum)));
  else pane.append(kv(Object.entries(t.provenance)));
  v.append(pane);
  return v;
}
function whereView(w) {
  if (!w) return null;
  return h('div', { class: 'row small', style: 'margin-top:8px' }, (w.pools || []).map(p => a(link.ds(p.ds), chip(`pool ${p.ds.replace(/^pool:/, '')}`))),
    w.base_rate ? chip(`base pass rate ${w.base_rate.successes}/${w.base_rate.scored}`, '', 'ops/exp1-rft/base-pass-rates.json') : null,
    (w.errata || []).map(e => chip(`errata: ${e.file}`, 'crit', e.reason || '')));
}
function referenceView(ref) {
  const out = h('div', {});
  for (const [inv, steps] of Object.entries(ref || {})) for (const s of steps || []) {
    const [tool, args] = Array.isArray(s) ? s : [s.tool, s];
    out.append(h('div', { class: 'small muted' }, `${inv} · ${tool}${args?.finish ? ' · finish' : ''}`), pre(args?.code ?? json(args)));
  }
  return out;
}
function fileBrowser(ctx, key, files, root) {
  const names = Object.keys(files || {});
  if (!names.length) return h('div', { class: 'muted' }, 'No files.');
  const sel = ctx.ui.file[key] && names.includes(ctx.ui.file[key]) ? ctx.ui.file[key] : (root && names.includes(root) ? root : names[0]);
  const tree = h('ul', { class: 'filetree' }, names.sort().map(n => h('li', {}, h('button', { class: n === sel ? 'on' : '', onclick: () => { ctx.ui.file[key] = n; ctx.render(); } },
    h('span', { style: `padding-left:${(n.split('/').length - 1) * 12}px` }, n.split('/').pop()), n === root ? chip('root') : null))));
  return h('div', { class: 'filebrowser' }, tree, h('div', { class: 'filepane' }, h('div', { class: 'mono small muted' }, sel), nlSource(files[sel])));
}
/** natlang source: YAML front matter, numbered steps, `code` spans. */
function nlSource(text) {
  const t = String(text ?? ''), m = t.match(/^---\n([\s\S]*?)\n---\n?/);
  const box = h('div', { class: 'nlsrc' });
  if (m) box.append(h('pre', { class: 'code front' }, m[1]));
  const body = m ? t.slice(m[0].length) : t;
  const p = h('pre', { class: 'code body' });
  for (const line of body.split('\n')) {
    const num = line.match(/^(\s*\d+\.)(.*)$/);
    const row = h('span', { class: num ? 'step' : '' });
    const rest = num ? num[2] : line;
    if (num) row.append(h('b', {}, num[1]));
    rest.split(/(`[^`]+`)/).forEach(seg => row.append(seg.startsWith('`') && seg.endsWith('`') && seg.length > 1 ? h('code', {}, seg.slice(1, -1)) : seg));
    p.append(row, '\n');
  }
  box.append(p);
  return box;
}

/* ---------- trajectory viewer ---------- */
function trajectoryView(ctx, t, { dsId, id }) {
  const opts = ctx.ui.traj;
  const v = h('div', { class: 'traj' });
  const tok = t.calls.reduce((s, c) => s + (c.usage?.completion_tokens || c.sampling?.tokens || 0), 0);
  v.append(h('div', { class: 'card' }, h('header', {}, h('div', {}, h('h2', {}, t.group_label ? `Episode ${t.group_label} · sample ${t.sample}` : t.id),
    h('div', { class: 'row', style: 'margin-top:4px' }, t.task_id ? a(link.task(t.task_id), chip(t.task_id, 'mono')) : null, t.family ? chip(t.family) : null,
      t.outcome ? h('span', { class: `chip ${t.outcome === 'task_success' || t.accepted ? 'ok' : 'crit'}` }, String(t.outcome).replace(/_/g, ' ')) : null, t.held ? chip('🔒 held-out', 'crit') : null)),
    t.group_label ? h('a', { class: 'btn sm', href: link.item(dsId, t.group_label, { part: 'groups' }) }, 'Whole group and advantages') : null),
  h('div', { class: 'grid g4 mini4' }, statBox(t.reward != null ? fmt(t.reward) : '—', 'reward'), statBox(t.graded != null ? fmt(t.graded) : '—', 'graded'),
    statBox(t.advantage != null ? signed(t.advantage, 2) : '—', t.spread === false ? 'advantage: group without spread, trains nothing' : 'advantage (reconstructed)'), statBox(`${t.calls.length} · ${count(tok)}`, 'calls · generated tokens')),
  t.checks ? h('div', { class: 'row small', style: 'margin-top:6px' }, Object.entries(t.checks).map(([k, x]) => chip(`${x ? '✔' : '✖'} ${k.replace(/_/g, ' ')}`, x ? 'ok' : 'crit'))) : null,
  (t.unknown || []).length ? h('details', { class: 'small' }, h('summary', {}, 'What these files cannot tell'), h('ul', {}, t.unknown.map(u => h('li', {}, u)))) : null));
  const invs = [...new Set(t.calls.map(c => c.invocation))];
  const fn = inv => t.calls.find(c => c.invocation === inv)?.function || '';
  const tools = c => (c.target.tool_calls || []).map(x => x.name).join(', ') || (c.decision ? 'decision' : c.target.content ? 'reply' : '—');
  const rail = h('ol', { class: 'callrail' }, t.calls.map(c => h('li', {}, h('a', { href: `#call-${c.i}`, onclick: e => { e.preventDefault(); document.getElementById(`call-${c.i}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); } },
    h('span', { class: 'ci' }, c.i), h('span', { class: 'inv', style: `background:var(--s${(invs.indexOf(c.invocation) % 8) + 1})` }), h('span', { class: 'ellip' }, tools(c)),
    c.neuralese ? h('span', { class: 'nz', title: 'Neuralese span in context' }, 'N') : null, c.sampling?.tail_tokens ? h('span', { class: 'muted small' }, ` ·${c.sampling.tail_tokens} tail`) : null))));
  const ctl = h('div', { class: 'toolbar small' }, ...[['reasoning', 'Reasoning'], ['context', 'Context'], ['results', 'Tool results']].map(([k, l]) =>
    h('label', {}, h('input', { type: 'checkbox', checked: opts[k], onchange: e => { opts[k] = e.target.checked; ctx.render(); } }), ` ${l}`)),
  invs.length > 1 ? h('span', { class: 'muted' }, `${invs.length} invocations: `, invs.slice(0, 8).map((inv, k) => h('span', { class: 'invkey' }, h('i', { style: `background:var(--s${(k % 8) + 1})` }), fn(inv) || inv.slice(-8)))) : null);
  const calls = h('div', { class: 'calls' }, t.calls.map(c => callView(ctx, c, t, { dsId, id, invs })));
  v.append(h('div', { class: 'trajgrid' }, h('nav', { class: 'card railcard', 'aria-label': 'Calls' }, h('h3', {}, 'Calls'), rail), h('div', {}, ctl, calls)));
  if (t.graph?.length) v.append(h('details', { class: 'card' }, h('summary', {}, `Execution graph (${t.graph.length} invocations)`), h('table', { class: 'data' }, h('thead', {}, h('tr', {}, ['definition', 'signature', 'context', 'events', 'phases'].map(x => h('th', {}, x)))),
    h('tbody', {}, t.graph.map(n => h('tr', {}, h('td', { class: 'mono' }, n.definition || '—'), h('td', { class: 'mono small' }, n.signature || ''), h('td', {}, n.context || ''), h('td', { class: 'n' }, n.events), h('td', {}, (n.phases || []).join(', '))))))));
  return v;
}
const statBox = (val, label) => h('div', { class: 'stat' }, h('b', {}, val), h('span', {}, label));
function callView(ctx, c, t, { dsId, id, invs }) {
  const o = ctx.ui.traj;
  const box = h('section', { class: 'call', id: `call-${c.i}` });
  box.append(h('header', {}, h('span', { class: 'inv', style: `background:var(--s${(invs.indexOf(c.invocation) % 8) + 1})` }), h('b', {}, `Call ${c.i}`),
    h('span', { class: 'muted small' }, ` ${c.function && c.function !== 'unknown' ? c.function : ''} ${c.role ? `· ${c.role}` : ''} ${c.context_shared ? `· continues ${c.context_shared} messages` : ''}`),
    h('span', { style: 'flex:1' }), c.sampling ? h('span', { class: 'small muted' }, `${count(c.sampling.tokens)} tok · logp̄ ${fmt(c.sampling.logp_mean, 3)} · ${c.sampling.finish || ''}`) : null,
    c.sampling && dsId.startsWith('grpo:') ? h('button', { class: 'btn sm', onclick: () => { ctx.ui.tokens[`${id}:${c.i}`] = !ctx.ui.tokens[`${id}:${c.i}`]; ctx.render(); } }, ctx.ui.tokens[`${id}:${c.i}`] ? 'Hide tokens' : 'Tokens & weights') : null));
  if (ctx.ui.tokens[`${id}:${c.i}`]) box.append(tokenPanel(ctx, dsId, id, c.i));
  if (o.context) for (const m of c.context) {
    if (m.role === 'system' && c.i > 0) continue;
    if (m.role === 'tool' && !o.results) { box.append(h('div', { class: 'msg tool collapsed muted small' }, `tool result · ${m.content.length.toLocaleString()} chars (hidden)`)); continue; }
    box.append(messageView(m));
  }
  box.append(h('div', { class: 'target' }, h('div', { class: 'tlabel' }, 'Model output', t.role === 'teacher' ? ' (teacher)' : t.advantage != null ? ` · trained with A = ${signed(t.advantage, 2)}` : ''),
    o.reasoning && c.target.reasoning ? h('details', { class: 'reasoning', open: c.target.reasoning.length < 1500 }, h('summary', {}, `Reasoning · ${c.target.reasoning.length.toLocaleString()} chars`), h('div', { class: 'rtext' }, c.target.reasoning)) : null,
    c.target.content ? h('div', { class: 'content' }, c.target.content) : null,
    (c.target.tool_calls || []).map(toolCallView), c.decision ? h('div', { class: 'decision' }, h('b', {}, 'Decision '), pre(json(c.decision))) : null));
  return box;
}
function messageView(m) {
  if (m.role === 'system') return h('details', { class: 'msg system' }, h('summary', { class: 'mlabel' }, `system prompt · ${m.content.length.toLocaleString()} chars`), h('div', { class: 'pre' }, m.content));
  const cls = `msg ${m.role}${m.host_generated ? ' host' : ''}`;
  const label = m.role === 'tool' ? `tool result${m.tool_call_id ? ' · ' + m.tool_call_id.slice(-10) : ''}` : m.role === 'assistant' && m.host_generated ? 'assistant (host-generated)' : m.role;
  const long = (m.content || '').length > 2500;
  const body = h('div', { class: m.role === 'tool' ? 'mono pre' : 'pre' }, long ? m.content.slice(0, 2500) : m.content);
  const el = h('div', { class: cls }, h('div', { class: 'mlabel' }, label), m.reasoning ? h('details', { class: 'reasoning' }, h('summary', {}, 'reasoning'), h('div', { class: 'rtext' }, m.reasoning)) : null, body,
    (m.tool_calls || []).map(toolCallView));
  if (long) el.append(h('button', { class: 'btn sm', onclick: e => { body.textContent = m.content; e.target.remove(); } }, `Show all ${m.content.length.toLocaleString()} chars`));
  return el;
}
function toolCallView(tc) {
  const p = tc.parsed;
  const code = p && typeof p.code === 'string' ? p.code : null;
  return h('div', { class: 'toolcall' }, h('div', { class: 'tclabel' }, '→ ', h('b', { class: 'mono' }, tc.name || '?'), p?.finish ? chip('finish', 'ok') : null, p?.status ? chip(String(p.status)) : null),
    code ? pre(code, 'ts') : null, p ? (() => { const rest = Object.fromEntries(Object.entries(p).filter(([k]) => k !== 'code' && k !== 'finish')); return Object.keys(rest).length ? pre(json(rest), 'small') : null; })() : pre(tc.arguments || '', 'small'));
}
function tokenPanel(ctx, dsId, id, call) {
  const input = { ds: dsId, id, call };
  const ent = load(ctx, `tok:${JSON.stringify(input)}`, 'data.tokens', input);
  const p = pending(ent, 'tokens');
  if (p) return p;
  const t = ent.value, box = h('div', { class: 'tokens' });
  box.append(h('div', { class: 'legend small' }, h('span', { class: 'key' }, h('i', { class: 'tk-hi' }), 'likely (logp > −1)'), h('span', { class: 'key' }, h('i', { class: 'tk-mid' }), 'logp −1…−4'), h('span', { class: 'key' }, h('i', { class: 'tk-lo' }), 'logp < −4'),
    h('span', { class: 'key' }, h('i', { class: 'tk-mask' }), 'masked (weight 0): tail floor or second </think>'), h('span', { class: 'muted' }, `A = ${signed(t.advantage, 2)}`)));
  const line = h('div', { class: 'tokline mono' });
  t.pieces.forEach((pc, k) => {
    const lp = t.logp[k], masked = t.tail_masked[k] || t.second_close_masked[k];
    const cls = masked ? 'tk tk-mask' : lp > -1 ? 'tk tk-hi' : lp > -4 ? 'tk tk-mid' : 'tk tk-lo';
    line.append(h('span', { class: cls, title: `#${k} id ${t.ids[k]} logp ${fmt(lp, 3)}${t.tail_masked[k] ? ' · tail-floor masked' : ''}${t.second_close_masked[k] ? ' · second-close masked' : ''}${t.roles[k] !== 'ordinary' ? ' · ' + t.roles[k] : ''}` }, pc));
  });
  box.append(line, h('div', { class: 'small muted' }, t.weight_rule));
  return box;
}

/* ---------- SFT row ---------- */
function sftRowView(ctx, r) {
  const v = h('div', {});
  v.append(h('div', { class: 'card' }, h('header', {}, h('div', {}, h('h2', { class: 'mono' }, r.id), h('div', { class: 'row', style: 'margin-top:4px' }, chip(r.source || '?'), r.teacher ? chip(r.teacher) : null,
    r.task_id ? a(link.task(r.task_id), chip(r.task_id, 'mono')) : null, r.family ? chip(r.family) : null, r.held ? chip('🔒 held task', 'crit') : null))),
  h('div', { class: 'grid g4 mini4' }, statBox(count(r.prompt_tokens), 'prompt tokens (context, weight 0)'), statBox(count(r.target_tokens), 'target tokens'), statBox(count(r.loss_from), 'leading zero-weight target tokens'), statBox(`call ${r.call}`, r.role || '')),
  kv([['episode', episodeLink(r.episode, r.task_id)], ['run', r.run], ['invocation', r.invocation], ['function', r.function], ['too long', r.too_long ? 'yes (dropped)' : null], ['live', r.live]]),
  h('details', { class: 'small' }, h('summary', {}, 'What these files cannot tell'), h('ul', {}, (r.unknown || []).map(u => h('li', {}, u))))));
  const o = ctx.ui.traj;
  const ctl = h('div', { class: 'toolbar small' }, h('label', {}, h('input', { type: 'checkbox', checked: o.context, onchange: e => { o.context = e.target.checked; ctx.render(); } }), ' Context'),
    h('label', {}, h('input', { type: 'checkbox', checked: o.results, onchange: e => { o.results = e.target.checked; ctx.render(); } }), ' Tool results'));
  const tgt = h('section', { class: 'call' }, h('header', {}, h('b', {}, 'Target'), h('span', { class: 'muted small' }, ' what the learner is trained to produce for this call')));
  for (const p of r.parts) {
    const w = p.weight ? chip('trained', 'ok') : chip('weight 0', '', 'zero-weight at build time');
    if (p.part === 'reasoning') tgt.append(h('details', { class: `reasoning ${p.weight ? '' : 'zero'}`, open: true }, h('summary', {}, 'Reasoning ', w), h('div', { class: 'rtext' }, p.text)));
    else if (p.part === 'content') tgt.append(h('div', { class: `content ${p.weight ? '' : 'zero'}` }, w, ' ', p.text));
    else tgt.append(h('div', { class: 'row' }, w), toolCallView(p.call));
  }
  const call = h('section', { class: 'call' }, h('header', {}, h('b', {}, `Context · ${r.context.length} messages`), h('span', { class: 'muted small' }, ' (zero-weight)')));
  if (o.context) for (const m of r.context) call.append(m.role === 'tool' && !o.results ? h('div', { class: 'msg tool collapsed muted small' }, `tool result · ${m.content.length.toLocaleString()} chars (hidden)`) : messageView(m));
  v.append(tgt, ctl, call);
  return v;
}

/** Episode keys of SFT sources: teacher:<campaign>/<pool>:tNNNNNN:<sample> or rl:<grpo run>:gNNNNNN:<sample>. */
function episodeLink(key, taskId) {
  const m = String(key || '').match(/^(teacher|rl[^:]*):(.+):(t|g)(\d{6}):(\d+)$/);
  if (!m) return key;
  if (m[3] === 't') return a(link.item(`teacher-gen:${m[2]}`, taskId, { part: 'tasks' }), key);
  return a(link.item(`grpo:${m[2]}`, `g${m[4]}:${m[5]}`, { part: 'groups' }), key);
}

/* ---------- group (all samples of one task in a GRPO run) ---------- */
function groupView(ctx, g, dsId) {
  const max = Math.max(1e-9, ...g.episodes.map(e => Math.abs(e.advantage || 0)));
  const sel = ctx.ui.cmp;
  return h('div', {}, h('div', { class: 'card' }, h('header', {}, h('h2', {}, `Group ${g.id}`), g.task_id ? a(link.task(g.task_id), chip(g.task_id, 'mono')) : null),
    h('p', { class: 'small muted' }, `Advantages are (r − mean) / (std + 1e-6) over the group's ${g.reward_mode} rewards, as grpo.group_advantages computes them. A group without spread trains nothing. Select two episodes to compare.`),
    h('table', { class: 'items' }, h('thead', {}, h('tr', {}, ['', 'sample', 'outcome', 'reward', 'advantage', 'calls'].map(x => h('th', {}, x)))),
      h('tbody', {}, g.episodes.map(e => h('tr', {}, h('td', {}, h('input', { type: 'checkbox', checked: sel.includes(`${dsId}|${g.id}:${e.sample}|groups`), onchange: ev => { const k = `${dsId}|${g.id}:${e.sample}|groups`; ev.target.checked ? sel.push(k) : sel.splice(sel.indexOf(k), 1); while (sel.length > 2) sel.shift(); ctx.render(); } })),
        h('td', {}, a(link.item(dsId, `${g.id}:${e.sample}`, { part: 'groups' }), `sample ${e.sample}`)), h('td', {}, cell('outcome', e.outcome)), h('td', { class: 'n' }, fmt(e.reward)),
        h('td', {}, h('div', { class: 'advbar' }, h('i', { class: e.advantage >= 0 ? 'pos' : 'neg', style: `width:${(50 * Math.abs(e.advantage || 0) / max).toFixed(1)}%` }), h('span', {}, e.advantage == null ? '—' : signed(e.advantage, 2)))),
        h('td', { class: 'n' }, e.calls))))),
    sel.length === 2 ? h('div', { class: 'row', style: 'margin-top:8px' }, h('a', { class: 'btn primary', href: link.compare(sel[0], sel[1]) }, 'Compare the two selected episodes')) : null));
}

/* ---------- step selection ---------- */
function stepPage(ctx, dsId, step) {
  const input = { ds: dsId, id: step, part: 'steps' };
  const ent = load(ctx, `item:${JSON.stringify(input)}`, 'data.item', input);
  return pending(ent, `step ${step}`) || stepBody(ctx, ent.value, dsId);
}
function stepBody(ctx, s, dsId) {
  const sel = s.selection || {}, low = sel.lowering || {};
  const v = h('div', {});
  v.append(h('div', { class: 'card' }, h('header', {}, h('h2', {}, `Step ${s.id}: what reached the learner`), h('div', { class: 'row' }, h('a', { class: 'btn sm', href: link.step(dsId, s.id - 1) }, '‹ step ' + (s.id - 1)), h('a', { class: 'btn sm', href: link.step(dsId, s.id + 1) }, 'step ' + (s.id + 1) + ' ›'))),
    h('div', { class: 'grid g4 mini4' }, statBox(count(sel.groups), 'groups taken'), statBox(count(sel.groups_with_spread), 'with reward spread (carry advantage)'), statBox(count(sel.dropped_stale_groups), 'dropped as stale (> max staleness)'), statBox(fmt(sel.reward_mean), 'mean reward')),
    h('h3', { style: 'margin-top:10px' }, 'Lowering: episodes and calls turned into learner rows (drops, self-distillation, decisions)'), kvCols(Object.entries(low).filter(([, x]) => typeof x === 'number').map(([k, x]) => [k.replace(/_/g, ' '), count(x)])),
    h('h3', {}, 'Token masks'), kv([['tail logp floor', sel.tail_logp_floor], ['tail-masked fraction', sel.tail_masked_fraction], ['second-close masked', sel.second_close_masked], ['loss tokens', count(sel.loss_tokens)], ['sequences', count(sel.sequences)]]),
    ['staleness', 'dynamic_sampling', 'memetic_groups', 'replay_corpora', 'sd', 'teacher', 'reasoning_rl'].filter(k => sel[k] && Object.keys(sel[k]).length).map(k => h('details', {}, h('summary', {}, k.replace(/_/g, ' ')), pre(json(sel[k])))),
    h('details', { class: 'small' }, h('summary', {}, 'What these files cannot tell'), h('ul', {}, (s.unknown || []).map(u => h('li', {}, u))))));
  if (s.by_role && Object.keys(s.by_role).length) v.append(h('div', { class: 'card', style: 'margin-top:12px' }, h('h3', {}, 'Trained chains by role'),
    h('div', { class: 'row' }, Object.entries(s.by_role).map(([r, x]) => chip(`${r}: ${x.chains} chains · ${count(x.trained_tokens)} trained tokens`)))));
  if (s.chains) v.append(h('div', { class: 'card flush', style: 'margin-top:12px' }, h('div', { class: 'tablehead' }, h('b', {}, `${s.chains.length} chains in the learner batch`)),
    h('div', { class: 'scroll' }, h('table', { class: 'items' }, h('thead', {}, h('tr', {}, ['chain', 'role', 'task', 'group / round', 'advantage', 'trained / total tokens', 'segments'].map(x => h('th', {}, x)))),
      h('tbody', {}, s.chains.map(c => h('tr', { class: 'clickrow', onclick: () => { location.hash = link.chain(dsId, s.id, c.i); } }, h('td', {}, a(link.chain(dsId, s.id, c.i), `#${c.i}`)), h('td', {}, chip(c.role || '?')),
        h('td', { class: 'mono small' }, c.task ? a(link.task(c.task), c.task) : '—'), h('td', { class: 'mono small' }, typeof c.round === 'number' ? a(link.item(dsId, `g${String(c.round).padStart(6, '0')}`, { part: 'groups' }), `g${String(c.round).padStart(6, '0')}`) : String(c.round ?? '')),
        h('td', { class: 'n' }, c.advantage == null ? '—' : signed(c.advantage, 2)), h('td', { class: 'n' }, `${count(c.trained_tokens)} / ${count(c.tokens)}`), h('td', { class: 'n' }, c.segments))))))));
  if (s.decisions?.length) v.append(h('details', { class: 'card', style: 'margin-top:12px' }, h('summary', {}, `${s.decisions.length} trained decisions`), pre(json(s.decisions.slice(0, 50)))));
  return v;
}
function chainPage(ctx, dsId, step, i) {
  const input = { ds: dsId, step, chain: i };
  const ent = load(ctx, `chain:${JSON.stringify(input)}`, 'data.chain', input);
  const p = pending(ent, `chain ${i}`);
  if (p) return p;
  const c = ent.value, v = h('div', { class: 'card' });
  v.append(h('header', {}, h('h2', {}, `Chain ${c.i} of step ${step}`), h('div', { class: 'row' }, chip(c.role || '?'), c.task ? a(link.task(c.task), chip(c.task, 'mono')) : null, c.advantage != null ? chip(`A ${signed(c.advantage, 2)}`) : null)),
    h('div', { class: 'legend small' }, h('span', { class: 'key' }, h('i', { class: 'tk-ctx' }), 'context (not trained)'), h('span', { class: 'key' }, h('i', { class: 'tk-hi' }), 'trained target'), h('span', { class: 'key' }, h('i', { class: 'tk-mask' }), 'target with stored weight 0'),
      h('span', { class: 'muted' }, `${count(c.trained_tokens)} trained of ${count(c.tokens)} tokens in ${c.segments} segments`)));
  const line = h('div', { class: 'tokline mono' });
  c.pieces.forEach((pc, k) => {
    const tr = c.trained[k];
    const cls = !tr ? 'tk tk-ctx' : tr[2] === 0 ? 'tk tk-mask' : 'tk tk-hi';
    line.append(h('span', { class: cls, title: tr ? `#${k} seg ${tr[0]} logp ${tr[1] ?? '—'} weight ${tr[2] ?? 1}` : `#${k} context` }, pc));
  });
  v.append(line, h('div', { class: 'small muted' }, (c.unknown || []).join(' ')));
  return v;
}

/* ---------- teacher task, registry ---------- */
function teacherTaskView(ctx, t) {
  const m = t.manifest || {};
  return h('div', {}, h('div', { class: 'card' }, h('header', {}, h('h2', { class: 'mono' }, t.id), a(link.task(t.id), 'Task and everywhere it appears →')),
    kv([['teacher', m.teacher?.model_id], ['base rank', m.base_rank], ['seconds', m.seconds], ['episodes', (m.episodes || []).map(e => `${e.sample}: ${e.outcome} (${e.turns} turns)`).join('; ')], ['directory', t.dir]])),
  ...t.episodes.map((e, k) => h('details', { class: 'card', open: k === 0 }, h('summary', {}, `Teacher episode ${k}: ${e.outcome}`), trajectoryView(ctx, { ...e, role: 'teacher' }, { dsId: 'teacher', id: `${t.id}:${k}` }))));
}
function registryView(r) {
  return h('div', { class: 'card' }, h('h2', { class: 'mono' }, r.entry.id), kv([['exists on disk', r.exists ? 'yes' : 'no'], ['resolves to', r.realpath]]), pre(json(r.entry)),
    r.manifest ? [h('h3', {}, `Manifest: ${r.manifest.files.length} files, ${bytes(r.manifest.bytes)}`), h('table', { class: 'data' }, h('tbody', {}, r.manifest.files.map(f => h('tr', {}, h('td', { class: 'mono small' }, f.path), h('td', { class: 'n' }, bytes(f.bytes)), h('td', { class: 'n' }, count(f.lines)), h('td', { class: 'mono small muted' }, (f.sha256 || '').slice(0, 12))))))] : h('div', { class: 'muted' }, 'No manifest.'));
}

/* ---------- task across datasets ---------- */
function taskRefsPage(ctx, taskId) {
  const ent = load(ctx, `refs:${taskId}`, 'data.task_refs', { task_id: taskId });
  const p = pending(ent, `where ${taskId} appears (scans run groups and teacher manifests)`);
  if (p) return p;
  const w = ent.value;
  const v = h('div', { class: 'card' }, h('h2', { class: 'mono' }, taskId), w.held ? h('div', { class: 'notice' }, '🔒 Held-out task: view only.') : null);
  v.append(h('h3', {}, 'Task pools'), (w.pools || []).length ? h('ul', { class: 'plain' }, w.pools.map(p2 => h('li', {}, a(link.item(p2.ds, taskId), `${p2.ds} → open the task editor`)))) : h('div', { class: 'muted' }, 'In no current pool.'),
    w.base_rate ? h('div', {}, `Base pass rate: ${w.base_rate.successes}/${w.base_rate.scored}`) : null,
    (w.errata || []).length ? [h('h3', {}, 'Errata'), pre(json(w.errata))] : null,
    h('h3', {}, `GRPO groups (${(w.groups || []).length})`), (w.groups || []).length ? h('table', { class: 'data' }, h('tbody', {}, w.groups.map(g => h('tr', {}, h('td', {}, a(link.ds(g.ds), g.ds)), h('td', {}, a(link.item(g.ds, g.group, { part: 'groups' }), g.group)), h('td', { class: 'n' }, `mean ${fmt(g.mean)}`), h('td', {}, g.spread ? 'spread' : 'no spread'))))) : h('div', { class: 'muted' }, 'No GRPO run sampled it.'),
    h('h3', {}, 'Teacher campaigns'), (w.teacher || []).length ? h('ul', { class: 'plain' }, w.teacher.map(x => h('li', {}, a(link.item(x.ds, taskId, { part: 'tasks' }), x.ds), ` · ${x.teacher || ''} · ${x.successes}/${x.episodes} successes`))) : h('div', { class: 'muted' }, 'No teacher attempts.'),
    h('h3', {}, 'SFT sets'), (w.sft || []).length ? h('ul', { class: 'plain' }, w.sft.map(x => h('li', {}, a(link.ds(x.ds, { part: 'rows', q: taskId }), x.ds), ` · ${x.rows} rows`))) : h('div', { class: 'muted' }, 'In no SFT set.'));
  return v;
}

/* ---------- edits page ---------- */
function editsPage(ctx, params) {
  const ed = editsIndex(ctx), f = params.state || '';
  const rows = ed.all.filter(e => !f || (f === 'pending' ? !e.application : e.application?.status === f)).reverse();
  return h('div', {}, h('div', { class: 'notice' }, 'Every edit is a signed operation in the application journal. The native participant on the DGX applies it to an edit layer (', mono('ops/data-edits/'), ', and for RL task exclusions ', mono('ops/pool-errata/ops-dashboard.jsonl'), ') and records the outcome; live training files are never modified. Exclusions take effect when a run next starts or resumes.'),
    h('div', { class: 'toolbar' }, h('span', { class: 'seg-ctl' }, [['', 'All'], ['pending', 'Pending'], ['applied', 'Applied'], ['refused', 'Refused']].map(([k, l]) => h('button', { 'aria-pressed': f === k, onclick: () => { location.hash = `#data/edits${qs({ state: k })}`; } }, l)))),
    h('div', { class: 'card flush' }, rows.length ? h('table', { class: 'items' }, h('thead', {}, h('tr', {}, ['item', 'action', 'reason / note', 'status', 'by'].map(x => h('th', {}, x)))),
      h('tbody', {}, rows.map(e => h('tr', {}, h('td', {}, a(link.item(e.target.dataset, e.target.id), mono(e.target.id)), h('div', { class: 'small muted' }, `${e.target.kind} · ${e.target.dataset}`)),
        h('td', {}, chip(e.action)), h('td', {}, e.text, e.field ? h('div', { class: 'small muted' }, `field ${e.field}`) : null),
        h('td', {}, e.application ? [chip(e.application.status, e.application.status === 'applied' ? 'ok' : 'crit'), h('div', { class: 'small muted' }, e.application.detail)] : chip(e.local ? `local: ${e.local}` : 'waiting for native', 'open')),
        h('td', { class: 'mono small' }, (e.author || '').slice(0, 8)))))) : h('div', { class: 'empty' }, 'No edits yet. Open any task, row or episode and use the Review and edit panel.')));
}

/* ---------- search and compare ---------- */
function searchPage(ctx, q) {
  const ent = load(ctx, `search:${q}`, 'data.search', { q });
  const p = pending(ent, `results for “${q}”`);
  if (p) return p;
  const r = ent.value;
  return h('div', {}, h('div', { class: 'sect' }, h('h2', {}, `Datasets (${r.datasets.length})`)), r.datasets.length ? h('ul', { class: 'card plain' }, r.datasets.map(d => h('li', {}, a(link.ds(d.id), d.title), ' ', chip(KIND_LABEL[d.kind] || d.kind)))) : h('div', { class: 'muted' }, 'No dataset matches.'),
    h('div', { class: 'sect' }, h('h2', {}, `Tasks (${r.tasks.length}${r.tasks.length >= 200 ? '+' : ''})`)),
    r.tasks.length ? h('div', { class: 'card flush' }, h('table', { class: 'items' }, h('tbody', {}, r.tasks.map(t => h('tr', {}, h('td', { class: 'mono' }, a(link.item(t.ds, t.id), t.id)), h('td', {}, t.family), h('td', {}, a(link.ds(t.ds), t.ds), t.protected ? chip('🔒') : null), h('td', {}, a(link.task(t.id), 'everywhere →'))))))) : h('div', { class: 'muted' }, 'No task id or family matches.'));
}
function comparePage(ctx, params) {
  const sides = [params.a, params.b].filter(Boolean).map(s => { const [ds, id, part] = s.split('|'); return { ds, id, part }; });
  if (sides.length < 2) return h('div', { class: 'card muted' }, 'Select two items to compare (for example two episodes of one group).');
  const ents = sides.map(s => load(ctx, `item:${JSON.stringify({ ds: s.ds, id: s.id, part: s.part })}`, 'data.item', { ds: s.ds, id: s.id, part: s.part }));
  const p = ents.map((e, k) => pending(e, sides[k].id)).find(Boolean);
  if (p) return p;
  const [A, B] = ents.map(e => e.value);
  const head = x => h('div', { class: 'card' }, h('b', {}, x.id), ' ', x.outcome ? cell('outcome', x.outcome) : null, x.advantage != null ? chip(`A ${signed(x.advantage, 2)}`) : null, x.calls ? h('div', { class: 'small muted' }, `${x.calls.length} calls`) : null);
  const v = h('div', {}, h('div', { class: 'grid g2' }, head(A), head(B)));
  if (A.calls && B.calls) {
    const n = Math.max(A.calls.length, B.calls.length);
    const o = { ...ctx.ui.traj };
    for (let i = 0; i < n; i++) {
      const row = h('div', { class: 'grid g2 cmprow' });
      for (const x of [A, B]) {
        const c = x.calls[i];
        row.append(c ? h('div', { class: 'call' }, h('header', {}, h('b', {}, `Call ${i}`), h('span', { class: 'muted small' }, ` ${(c.target.tool_calls || []).map(t => t.name).join(', ')}`)),
          o.reasoning && c.target.reasoning ? h('details', { class: 'reasoning' }, h('summary', {}, 'reasoning'), h('div', { class: 'rtext' }, c.target.reasoning)) : null,
          c.target.content ? h('div', { class: 'content' }, c.target.content) : null, (c.target.tool_calls || []).map(toolCallView),
          c.context.filter(m => m.role === 'tool').slice(0, 1).map(messageView)) : h('div', { class: 'muted' }, '—'));
      }
      v.append(row);
    }
  } else v.append(h('div', { class: 'grid g2' }, pre(json(A)), pre(json(B))));
  return v;
}
