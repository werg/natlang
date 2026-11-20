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
const a = (href, ...kids) => kids[0] && typeof kids[0] === 'object' && !(kids[0] instanceof Node) && !Array.isArray(kids[0]) ? h('a', { href, ...kids[0] }, ...kids.slice(1)) : h('a', { href }, ...kids);
const chip = (text, cls = '', title) => h('span', { class: `chip ${cls}`, title }, text);
const mono = t => h('span', { class: 'mono' }, t);
const pre = (t, cls = '') => h('pre', { class: `code ${cls}` }, t);
const json = v => JSON.stringify(v, null, 2);
const store = (k, v) => { try { if (v === undefined) return JSON.parse(localStorage.getItem(k)); localStorage.setItem(k, JSON.stringify(v)); } catch { return null; } };

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
const KIND_LABEL = { 'rl-tasks': 'RL tasks', held: 'Held-out', run: 'GRPO runs', 'teacher-demos': 'Teacher demos', trajectories: 'Episodes', 'sft-rows': 'SFT rows', chains: 'Chains', corpus: 'Sources', index: 'Indexes' };
const KIND_ONE = { 'rl-tasks': 'RL tasks', held: 'held-out', run: 'GRPO run', 'teacher-demos': 'teacher demos', trajectories: 'episodes', 'sft-rows': 'SFT rows', chains: 'chains', corpus: 'source', index: 'index' };
const CONS_CAT = how => /SD_TEACHER|sd-teacher/.test(how) ? 'Self-distillation' : /^GRPO/.test(how) ? 'GRPO' : /^teacher/.test(how) ? 'Teacher-gen' : /^SFT/.test(how) ? 'SFT' : /errata/.test(how) ? 'Errata' : how;
/** grpo:grpo-v4/run2 -> ['grpo-v4', 'run2']; sft:sft-mix-20261120-v1 -> ['sft-mix-20261120-v1', '']. */
function runFamily(run) { const id = String(run).replace(/^[a-z-]+:/, ''); const k = id.indexOf('/'); return k > 0 ? [id.slice(0, k), id.slice(k + 1)] : [id, '']; }
/** ['run1','run2','run3'] -> 'run1–3'; otherwise a short join. */
function compactMembers(ms) {
  if (!ms.length) return '';
  const m = ms.map(x => x.match(/^(.*?)(\d+)$/));
  if (m.every(Boolean) && new Set(m.map(x => x[1])).size === 1) {
    const ns = m.map(x => Number(x[2])).sort((a, b) => a - b);
    if (ns.length > 1 && ns.every((n, i) => !i || n === ns[i - 1] + 1)) return `${m[0][1]}${ns[0]}–${ns.at(-1)}`;
    return `${m[0][1]}${ns.join('·')}`;
  }
  return ms.length > 2 ? `×${ms.length}` : ms.join(' · ');
}
/** References that exclude a dataset (held manifests, errata) are not uses: they are listed apart as "excluded by". */
const isExclusion = c => /held_manifest|errata|exclu/i.test(c.how || '');
export function usage(consumers) {
  const cats = new Map();
  for (const c of consumers || []) {
    const k = CONS_CAT(c.how); if (!cats.has(k)) cats.set(k, new Map());
    const [fam, mem] = runFamily(c.run), m = cats.get(k);
    if (!m.has(fam)) m.set(fam, { members: new Set(), runs: [] });
    m.get(fam).members.add(mem); m.get(fam).runs.push(c);
  }
  return [...cats].map(([cat, fams]) => ({ cat, fams: [...fams].map(([fam, x]) => ({ fam, members: [...x.members].filter(Boolean).sort(), runs: x.runs })).sort((a, b) => b.fam.localeCompare(a.fam)) }));
}
const isDs = run => /^(grpo|teacher-gen|sft|sft-mix|pool):/.test(run);
/** One line: "GRPO grpo-v4 run1–3, grpo-v1 run1–2 · Teacher-gen 3 campaigns". */
function usageLine(consumers, { max = 2, prefix = null } = {}) {
  const u = usage(consumers);
  if (!u.length) return null;
  const all = (consumers || []).map(c => `${c.run} (${c.how})`).join('\n');
  return h('span', { class: 'usage', title: all }, prefix, u.map(({ cat, fams }, i) => [i ? h('span', { class: 'sep' }, ' · ') : null, h('span', { class: 'ucat' }, cat), ' ',
    fams.slice(0, max).map((f, j) => { const last = f.runs.at(-1).run; const label = `${f.fam}${f.members.length ? ' ' + compactMembers(f.members) : ''}`; return [j ? ', ' : '', isDs(last) ? a(link.ds(last), label) : label]; }),
    fams.length > max ? h('span', { class: 'muted' }, ` +${fams.length - max}`) : null]));
}
function catalog(ctx, params) {
  const ent = load(ctx, 'catalog', 'data.catalog', null, { ttl: 300000 });
  const p = pending(ent, 'the training-data catalog (the first build stats every pool and can take a minute)');
  if (p) return p;
  const all = ent.value.datasets, ed = editsIndex(ctx), ui = ctx.ui;
  ui.catOpen ||= {};
  const st = { kind: params.kind || '', f: params.f || '', used: params.used || '' };
  const kinds = [...new Set(all.map(d => d.kind))];
  const v = h('div', { class: 'catalog' });
  const total = all.reduce((s, d) => s + (d.rows || 0), 0), unreg = all.filter(d => !d.provenance_brief?.registered && d.kind !== 'run' && d.kind !== 'held').length;
  v.append(h('div', { class: 'cathead' }, h('div', {}, h('h1', {}, 'Training data'), h('p', { class: 'muted' }, `${count(all.length)} datasets · ${count(total)} rows, tasks and episodes · `,
    h('span', { title: 'Datasets the corpus registry does not record: found through trainer constants, run directories or the hand list' }, `${unreg} not in the registry`), ` · catalog built ${ago(ent.value.generated, Date.now())} `,
    h('button', { class: 'linkbtn', onclick: () => { refresh('catalog'); ctx.render(); } }, 'rebuild'))),
  h('div', { class: 'row' }, h('a', { class: 'btn', href: link.edits() }, 'Review queue'), h('a', { class: 'btn', href: link.ds('corpus-registry') }, 'Corpus registry'))));
  const side = h('aside', { class: 'card catside' });
  const main = h('div', { class: 'catmain' });
  const filterBox = h('input', { type: 'search', placeholder: 'Filter by name, path, family of use…', value: st.f, 'aria-label': 'Filter datasets', 'data-key': 'filter',
    oninput: e => { st.f = e.target.value; history.replaceState(null, '', link.home() + qs(st)); paint(); } });
  const paint = () => {
    const tf = st.f.toLowerCase();
    const shown = all.filter(d => (!st.kind || d.kind === st.kind) && (!st.used || (d.consumers || []).length) &&
      (!tf || `${d.id} ${d.title} ${d.path} ${d.what} ${(d.consumers || []).map(c => c.run).join(' ')}`.toLowerCase().includes(tf)));
    const groups = new Map();
    for (const d of shown) { if (!groups.has(d.group)) groups.set(d.group, []); groups.get(d.group).push(d); }
    const out = [];
    if (!shown.length) out.push(h('div', { class: 'card empty' }, h('div', {}, h('b', {}, 'No dataset matches.'), h('div', { class: 'small' }, 'Clear the filter or search task ids with the box at the top.'))));
    for (const [g, list] of groups) {
      list.sort((x, y) => (y.mtime || 0) - (x.mtime || 0));
      const lim = ui.catOpen[g] || tf ? list.length : 8;
      out.push(h('section', { class: 'catgroup', id: `g-${g.replace(/\W+/g, '-')}` }, h('div', { class: 'sect' }, h('h2', {}, g), h('span', { class: 'muted' }, `${list.length}`)),
        h('div', { class: 'card flush dslist', role: 'list' }, h('div', { class: 'dsrow dshead', 'aria-hidden': 'true' }, h('span', {}, 'dataset'), h('span', { class: 'n' }, 'rows'), h('span', { class: 'n' }, 'size'), h('span', { class: 'n' }, 'changed'), h('span', {}, 'used by')),
          list.slice(0, lim).map(d => dsRow(d, ed)),
          list.length > lim ? h('button', { class: 'dsmore', onclick: () => { ui.catOpen[g] = true; paint(); } }, `Show all ${list.length}`) : null)));
    }
    main.replaceChildren(...out);
    kindNav.querySelectorAll('[data-kind]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.kind === st.kind)));
    groupNav.replaceChildren(...[...groups].map(([g, list]) => h('a', { href: `#g-${g.replace(/\W+/g, '-')}`, class: 'facetrow', onclick: e => { e.preventDefault(); document.getElementById(`g-${g.replace(/\W+/g, '-')}`)?.scrollIntoView({ block: 'start' }); } }, h('span', { class: 'ellip' }, g), h('span', { class: 'muted' }, list.length))));
  };
  const setKind = k => { st.kind = k; history.replaceState(null, '', link.home() + qs(st)); paint(); };
  const kindNav = h('div', { class: 'facet' }, h('button', { class: 'facetrow', 'data-kind': '', onclick: () => setKind('') }, h('span', {}, 'All kinds'), h('span', { class: 'muted' }, all.length)),
    kinds.map(k => h('button', { class: 'facetrow', 'data-kind': k, onclick: () => setKind(st.kind === k ? '' : k) }, h('span', {}, KIND_LABEL[k] || k), h('span', { class: 'muted' }, all.filter(d => d.kind === k).length))));
  const groupNav = h('div', { class: 'facet' });
  side.append(filterBox, h('label', { class: 'small row', style: 'margin-top:8px' }, h('input', { type: 'checkbox', checked: !!st.used, onchange: e => { st.used = e.target.checked ? '1' : ''; history.replaceState(null, '', link.home() + qs(st)); paint(); } }), 'Only data a run used'),
    h('h4', {}, 'Kind'), kindNav, h('h4', {}, 'Jump to'), groupNav,
    h('p', { class: 'small muted', style: 'margin-top:12px' }, 'Found through the trainer\'s constants, run directories, the corpus registry and a hand list (', mono('catalog_sources.json'), '). Keys: ', h('kbd', {}, '/'), ' search, ', h('kbd', {}, 'j'), h('kbd', {}, 'k'), ' move, ', h('kbd', {}, 'Enter'), ' open.'));
  v.append(h('div', { class: 'catgrid' }, side, main));
  paint();
  return v;
}
function dsRow(d, ed) {
  const n = ed.all.filter(e => e.target.dataset === d.id).length;
  return h('div', { class: 'dsrow kbrow', role: 'listitem', 'data-href': link.ds(d.id), onclick: e => { if (!e.target.closest('a,button')) location.hash = link.ds(d.id); } },
    h('div', { class: 'dsname' }, h('div', { class: 'row' }, a(link.ds(d.id), d.title), d.protected ? chip('🔒 view only', 'crit', 'Held-out: never trained on; edits are refused') : null,
      d.missing ? chip('missing on disk', 'warn') : null, n ? chip(`${n} edit${n > 1 ? 's' : ''}`, 'evidence') : null),
    h('div', { class: 'small muted ellip', title: `${d.path}\nfound via ${d.found_via}` }, d.what || d.format || '', h('span', { class: 'via' }, ` · ${d.found_via}`))),
    h('span', { class: 'n', title: d.rows_estimated ? 'estimated from a 4 MB sample (large file on the HDD)' : '' }, d.rows == null ? '—' : (d.rows_estimated ? '~' : '') + count(d.rows)),
    h('span', { class: 'n muted' }, bytes(d.size_bytes)), h('span', { class: 'n muted', title: d.mtime ? dayClock(d.mtime * 1000) : '' }, d.mtime ? ago(d.mtime * 1000, Date.now()) : '—'),
    h('span', { class: 'small ellip' }, usedCell(d)));
}
function usedCell(d) {
  const uses = (d.consumers || []).filter(c => !isExclusion(c)), excl = (d.consumers || []).filter(isExclusion);
  const pend = (d.pending || []).map(p => p.run);
  return [usageLine(uses) || (pend.length || excl.length ? null : h('span', { class: 'muted' }, '—')),
    pend.length ? h('span', { class: 'pend', title: (d.pending || []).map(p => `${p.run} runs on ${p.pin || 'its pin'}, started before this pool was listed`).join('\n') }, `${uses.length ? ' · ' : ''}pending: enters ${pend.join(', ')} at its next resume on a newer pin`) : null,
    excl.length ? h('span', { class: 'excl' }, uses.length || pend.length ? ' · ' : '', usageLine(excl, { prefix: 'excluded by ' })) : null];
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
  return h('dl', { class: 'kvgrid' }, pairs.filter(([, x]) => x != null && x !== '' && !(Array.isArray(x) && !x.length)).map(([k, x]) => [h('dt', {}, k), h('dd', {}, valueView(x))]));
}
/** Plain values as text, short lists inline, objects as nested key-value lists (never a JSON dump unless deep). */
function valueView(x, depth = 0) {
  if (x instanceof Node) return x;
  if (Array.isArray(x)) return x.every(y => y == null || typeof y !== 'object') ? x.join(', ') : depth > 1 ? pre(json(x), 'small') : h('ul', { class: 'plain small' }, x.map(y => h('li', {}, valueView(y, depth + 1))));
  if (x && typeof x === 'object') return depth > 1 ? pre(json(x), 'small') : h('dl', { class: 'kvgrid nested' }, Object.entries(x).filter(([, y]) => y != null && y !== '').map(([k, y]) => [h('dt', {}, k.replace(/_/g, ' ')), h('dd', {}, valueView(y, depth + 1))]));
  if (typeof x === 'number') return count(x);
  return String(x);
}
/** Text that may be long: the first `n` characters with a "show all" toggle. */
function clampText(text, n = 400, cls = '') {
  const t = String(text ?? '');
  if (t.length <= n) return h('div', { class: cls }, t);
  const body = h('div', { class: cls }, t.slice(0, n) + '…');
  return h('div', {}, body, h('button', { class: 'linkbtn', onclick: e => { body.textContent = t; e.target.remove(); } }, `show all ${t.length.toLocaleString()} characters`));
}

function datasetHeader(ctx, ds) {
  const prov = ds.provenance || {}, reg = prov.registry;
  const card = h('div', { class: 'card dshead' });
  card.append(h('div', { class: 'dstitle' }, h('div', {}, h('div', { class: 'row' }, h('h1', {}, ds.title), chip(KIND_ONE[ds.kind] || ds.kind), ds.protected ? chip('🔒 held-out · view only', 'crit') : null),
    h('p', { class: 'lede' }, ds.what || ''),
    h('div', { class: 'facts' }, ds.rows != null ? h('span', {}, h('b', {}, (ds.rows_estimated ? '~' : '') + count(ds.rows)), ' rows') : null, ds.size_bytes != null ? h('span', {}, h('b', {}, bytes(ds.size_bytes))) : null,
      ds.mtime ? h('span', { title: dayClock(ds.mtime * 1000) }, 'changed ', h('b', {}, ago(ds.mtime * 1000, Date.now()))) : null, ds.started ? h('span', {}, 'started ', h('b', {}, dayClock(Date.parse(ds.started) || ds.started * 1000))) : null,
      ds.code ? h('span', { title: 'code commit the run recorded' }, 'code ', h('b', { class: 'mono' }, `${ds.code.git_commit?.slice(0, 10)}${ds.code.dirty ? ' (dirty)' : ''}`)) : null)),
  h('div', { class: 'pathbox' }, h('span', { class: 'mono small ellip', title: ds.path }, ds.path), h('button', { class: 'btn sm', onclick: e => { navigator.clipboard?.writeText(ds.path); e.target.textContent = 'Copied'; } }, 'Copy path'))));
  if (ds.note) card.append(h('div', { class: 'notice' }, ds.note));
  const used = usage((ds.consumers || []).filter(c => !isExclusion(c))), excluded = (ds.consumers || []).filter(isExclusion);
  const usedBox = h('div', {}, h('h3', {}, 'Used by'), used.length ? used.map(({ cat, fams }) => h('div', { class: 'usegroup' }, h('div', { class: 'ucat' }, cat),
    h('ul', { class: 'plain' }, fams.flatMap(f => f.runs).map(c => h('li', {}, isDs(c.run) ? a(link.ds(c.run), c.run.replace(/^[a-z-]+:/, '')) : c.run, c.amount ? h('span', { class: 'small muted' }, ` · ${c.amount}`) : null)))))
    : h('div', { class: 'muted small' }, ds.consumers_note || 'No run, build or campaign record trains on it.'),
    (ds.pending || []).map(p => h('div', { class: 'pend small' }, `Pending: enters ${p.run} at its next resume on a newer pin (it runs on ${p.pin || 'an older pin'}, started before this pool was listed).`)),
    excluded.length ? h('div', { class: 'usegroup' }, h('div', { class: 'ucat' }, 'Excluded by (these runs and builds keep it out of training)'),
      h('ul', { class: 'plain' }, excluded.map(c => h('li', {}, isDs(c.run) ? a(link.ds(c.run), c.run.replace(/^[a-z-]+:/, '')) : c.run, h('span', { class: 'small muted' }, ` · ${c.how}`))))) : null);
  const fromBox = h('div', {}, h('h3', {}, 'Where it comes from'), kv([['found via', ds.found_via], ['format', ds.format], ['producer', prov.producer || prov.writer], ['trainer constant', prov.code_ref],
    ['registry', reg ? a(link.item('corpus-registry', reg.id, { part: 'entries' }), `${reg.id}${reg.admission ? ' · ' + reg.admission : ''}`) : 'not registered'],
    ['license', reg?.license_provenance || reg?.license ? clampText(reg.license_provenance || reg.license, 220) : null], ['side records', (prov.records || []).join(', ')],
    ['receipt', prov.receipt], ['eligible tasks', prov.eligible]]));
  card.append(h('div', { class: 'grid g2', style: 'margin-top:12px' }, usedBox, fromBox));
  const rel = Object.entries(ds.relations || {});
  if (rel.length) card.append(h('div', { class: 'row small', style: 'margin-top:8px' }, h('b', {}, 'Related: '), rel.map(([k, x]) => [h('span', { class: 'muted' }, k.replace(/_/g, ' ')), ' ', (Array.isArray(x) ? x : [x]).map(y => a(link.ds(y), y))])));
  if (ds.mix) card.append(mixView(ds.mix));
  if (ds.held_manifest) card.append(h('details', {}, h('summary', {}, 'Held manifest'), valueView(ds.held_manifest)));
  if (ds.reader === 'grpo-run') card.append(trainedOn(ctx, ds));
  return card;
}

/** A GRPO run's inputs: task pools, memetic tasks, replay corpora, self-distillation sources, held set and errata, linked to their datasets. */
function trainedOn(ctx, ds) {
  const cat = load(ctx, 'catalog', 'data.catalog', null, { ttl: 300000 }).value?.datasets || [];
  const byPath = p => cat.find(d => d.path === p || d.path === p?.replace(/\/cases\.jsonl$/, '') || (d.path && p && d.path.replace(/\/[^/]+$/, '') === p.replace(/\/[^/]+$/, '') && /cases\.jsonl$/.test(d.path)));
  const S = ds.settings || {}, pool = (ds.provenance || {}).pool || {};
  const rows = [];
  const add = (role, p, extra) => { const d = byPath(p); rows.push(h('tr', {}, h('td', {}, chip(role)), h('td', {}, d ? a(link.ds(d.id), d.title) : h('span', { class: 'mono small' }, p)), h('td', { class: 'n' }, d?.rows != null ? count(d.rows) : ''), h('td', { class: 'small muted' }, extra || ''))); };
  for (const p of (ds.provenance?.pools || [])) add('task pool', p);
  if (S.memetic_tasks) add('memetic tasks', S.memetic_tasks, S.memetic_share != null ? `${Math.round(100 * S.memetic_share)}% of groups` : '');
  for (const r of S.replay_corpus || []) { const m = String(r).match(/^([^=]+)=([^,]+)(.*)$/); if (m) add(`replay: ${m[1]}`, m[2], m[3].replace(/^,/, '').replace(/,/g, ' · ')); }
  for (const p of [].concat(S.sd_teacher_source || [])) add('self-distillation', p, S.sd_teacher_weight != null ? `weight ${S.sd_teacher_weight}` : '');
  if (S.held_manifest) add('held out (excluded)', S.held_manifest, 'evaluation only');
  for (const p of ds.errata_files || []) add('errata (excluded)', p);
  const summary = pool.eligible != null ? `${count(pool.eligible)} eligible tasks of ${count(pool.union_rows)} pool rows; ${count(pool.excluded_ids)} excluded as held-out or their siblings, ${count(pool.excluded_errata_ids)} by errata${pool.excluded_family_regex && pool.excluded_family_regex !== '(?!)' ? `, families matching ${pool.excluded_family_regex}` : ''}.` : '';
  const sel = S.pass_band ? `Selection: ${Math.round(100 * (1 - (S.explore ?? 0.4)))}% of groups from tasks whose latest pass rate is in ${S.pass_band.join('–')}, ${Math.round(100 * (S.explore ?? 0.4))}% exploring unmeasured tasks; ${S.group || 8} samples per group, ${S.min_groups}–${S.max_groups} groups a step.` : '';
  return h('div', { class: 'trainedon' }, h('h3', {}, 'What this run trained on'), summary ? h('p', { class: 'small' }, summary) : null, sel ? h('p', { class: 'small muted' }, sel) : null,
    h('table', { class: 'data' }, h('tbody', {}, rows)),
    h('div', { class: 'row small', style: 'margin-top:6px' }, h('a', { class: 'btn sm', href: link.ds(ds.id, { part: 'groups' }) }, 'Every sampled group →'), h('a', { class: 'btn sm', href: link.ds(ds.id, { part: 'steps' }) }, 'What each step trained →'),
      h('a', { class: 'btn sm', href: link.ds(ds.id, { part: 'held' }) }, 'Held evals →')),
    ds.args ? h('details', { class: 'small' }, h('summary', {}, 'All launch arguments'), kvCols(Object.entries(ds.args).map(([k, x]) => [`--${k}`, String(x)]))) : null);
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
  if (mix.aux) box.append(h('details', {}, h('summary', {}, 'Auxiliary inputs'), valueView(mix.aux)));
  if (mix.inputs) box.append(h('details', {}, h('summary', {}, 'Inputs'), valueView(mix.inputs)));
  if (mix.policy) box.append(h('details', {}, h('summary', {}, 'Build policy'), valueView(mix.policy)));
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
  heldtasks: [['task_id', 'task'], ['family', 'family'], ['base_samples', 'start policy'], ['policy_samples', 'this checkpoint'], ['change_pts', 'change'], ['direction', '']],
  entries: [['id', 'entry'], ['path', 'path'], ['owner', 'owner'], ['admission', 'admission'], ['training_admission', 'training'], ['superseded_by', 'superseded by']],
  errata: [['id', 'task'], ['family', 'family'], ['reason', 'reason'], ['source', 'source']],
  raw: [['id', 'id'], ['family', 'family'], ['source', 'source'], ['keys', 'keys']],
};
const OUT_CLS = o => o === 'task_success' ? 'ok' : o === 'not run' || o === 'unresolved' ? 'none' : 'bad';
function cell(k, x, row, ds) {
  if (k === 'base_samples' || k === 'policy_samples') {
    const part = k === 'base_samples' ? 'base' : 'samples';
    return h('span', { class: 'dots' }, (x || []).map(([sm, o]) => o === 'not run' ? h('i', { class: 'dot none', title: `sample ${sm}: not run` }) :
      h('a', { class: `dot ${OUT_CLS(o)}`, href: link.item(ds.id, `${row.task_id}#${sm}`, { part }), title: `sample ${sm}: ${String(o).replace(/_/g, ' ')} (open episode)` })),
      h('span', { class: 'muted small' }, ` ${row[k === 'base_samples' ? 'base' : 'policy'] || ''}`));
  }
  if (k === 'direction') return x === 'regressed' ? chip('regressed', 'crit') : x === 'improved' ? chip('improved', 'ok') : h('span', { class: 'muted small' }, x);
  if (k === 'change_pts') return x == null ? '—' : h('b', { class: x < 0 ? 'neg' : x > 0 ? 'pos' : '' }, `${signed(x, 0)} pts`);
  if (k === 'flags' && Array.isArray(x)) return x.map(f => chip(String(f).replace(/_shift$|_spike$/, m => m === '_spike' ? ' spike' : ''), 'warn', f));
  if (k === 'spread') return x ? '✔' : h('span', { class: 'muted small', title: 'all samples got the same reward: the group trains nothing' }, 'none');
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
  const kind = ds.reader === 'grpo-run' ? part : ds.reader === 'held-eval' ? (part === 'tasks' ? 'heldtasks' : 'samples') : ds.reader === 'registry' ? 'entries' : (ds.part_kind || {})[part] || ds.item_kind || 'raw';
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
  const cols0 = COLS[kind] || COLS.raw;
  // columns with no value on this page are noise: drop them (the first column always stays)
  const cols = cols0.filter(([k], i) => !i || r.rows.some(row => row[k] != null && row[k] !== '' && !(Array.isArray(row[k]) && !row[k].length)));
  const heldPair = row => { const b = (row.base_samples || []).find(x => x[1] === 'task_success') || row.base_samples?.[0], p2 = (row.policy_samples || []).find(x => x[1] !== 'task_success' && x[1] !== 'not run') || row.policy_samples?.[0];
    return b && p2 ? link.compare(`${ds.id}|${row.task_id}#${b[0]}|base`, `${ds.id}|${row.task_id}#${p2[0]}|samples`) : link.task(row.task_id); };
  const itemHref = row => kind === 'steps' ? link.step(ds.id, row.step) : kind === 'held' ? link.ds(`held-eval:${ds.run}:${row.id}`) : kind === 'heldtasks' ? heldPair(row) : link.item(ds.id, row.id ?? row._loc, { part });
  main.append(h('div', { class: 'tablehead' }, h('b', {}, `${count(r.total)} ${kind === 'steps' ? 'steps' : 'items'}`), r.searched ? h('span', { class: 'muted' }, ` matching “${params.q}”`) : null,
    h('span', { style: 'flex:1' }), pager(r, nav)));
  const tbody = h('tbody', {}, r.rows.map(row => {
    const st = ed.status(ds.id, String(row.id ?? row._loc));
    return h('tr', { class: 'clickrow kbrow', 'data-href': itemHref(row), onclick: e => { if (!e.target.closest('a,button')) location.hash = itemHref(row); } },
      cols.map(([k], i) => h('td', { class: typeof row[k] === 'number' ? 'n' : i === 0 ? 'idcell' : '' }, i === 0 ? [a(itemHref(row), { class: 'mono' }, cell(k, row[k] ?? row._loc, row, ds)), ' ', ...statusChips(st)] : cell(k, row[k], row, ds))));
  }));
  main.append(h('div', { class: 'scroll', style: 'max-height:none' }, h('table', { class: 'items' }, h('thead', {}, h('tr', {}, cols.map(([k, label]) => h('th', {}, h('a', { href: nav({ sort: params.sort === k ? `-${k}` : k, o: '' }) }, label, params.sort === k ? ' ▲' : params.sort === `-${k}` ? ' ▼' : ''))))), tbody)));
  if (!r.rows.length) main.append(h('div', { class: 'empty' }, h('div', {}, h('b', {}, 'No items match.'), Object.values(filter).some(Boolean) || params.q ? h('div', {}, h('a', { href: nav({ q: '', ...Object.fromEntries(Object.keys(filter).map(k => [`f.${k}`, ''])) }) }, 'Clear the search and filters')) : null)));
  if (kind === 'heldtasks') main.prepend(h('div', { class: 'notice', style: 'margin:10px' }, 'Each dot is one held episode (green pass, red fail, grey not run): the start policy on the left, this checkpoint on the right. Click a dot to read the episode, or a row to compare a passing start-policy episode with a failing one from this checkpoint.'));
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
  const body = it.kind === 'task' ? taskEditor(ctx, it, target, params)
    : it.kind === 'trajectory' ? trajectoryView(ctx, it, { dsId, id, part: params.part })
      : it.kind === 'sft_row' ? sftRowView(ctx, it)
        : it.kind === 'group' ? groupView(ctx, it, dsId)
          : it.kind === 'step' ? stepBody(ctx, it, dsId)
            : it.kind === 'teacher_task' ? teacherTaskView(ctx, it)
              : it.kind === 'registry_entry' ? registryView(it)
                : h('div', { class: 'card' }, pre(json(it.value ?? it)));
  if (['group', 'step'].includes(it.kind)) return body;
  const st = editsIndex(ctx).status(target.dataset, target.id);
  const banner = st.excluded ? h('div', { class: 'excluded-banner' }, h('b', {}, '⊘ Excluded from training. '), target.kind === 'task' ? 'GRPO and teacher-gen skip this task from their next start or resume (ops/pool-errata/ops-dashboard.jsonl).' : 'Recorded in the exclusion layer; no SFT builder reads it yet.')
    : st.flagged ? h('div', { class: 'flag-banner' }, h('b', {}, '⚑ Flagged for review.'), ' Nothing changes in training until it is excluded or corrected.') : null;
  // episodes are wide: their review panel starts folded; tasks and rows show it open (remembered per kind)
  const ek = it.kind === 'trajectory' ? 'trajectory' : 'other';
  ctx.ui.editOpen ||= store('ops.editOpen') || {};
  const open = ctx.ui.editOpen[ek] ?? ek !== 'trajectory';
  const toggle = () => { ctx.ui.editOpen[ek] = !open; store('ops.editOpen', ctx.ui.editOpen); ctx.render(); };
  const side = open ? h('aside', { class: 'itemside' }, h('button', { class: 'linkbtn fold', onclick: toggle, title: 'Fold the review panel' }, 'Fold ›'), editPanel(ctx, target, it))
    : h('aside', { class: 'itemside folded' }, h('button', { class: 'btn foldbtn', onclick: toggle, title: 'Flag, exclude, annotate or correct this item' }, '‹ Review', h('span', { class: 'row' }, ...statusChips(st))));
  return h('div', {}, banner, h('div', { class: `itemlayout${open ? '' : ' wide'}` }, h('div', { class: 'itemmain' }, body), side));
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
function taskEditor(ctx, t, target, params = {}) {
  const v = h('div', { class: 'task' });
  v.append(h('div', { class: 'card' }, h('header', {}, h('div', {}, h('h1', { class: 'mono wrap' }, t.id), h('div', { class: 'row', style: 'margin-top:4px' },
    chip(t.family || '—'), t.level != null ? chip(`level ${t.level}`) : null, t.split ? chip(t.split) : null, t.held ? chip('🔒 held-out', 'crit') : null,
    (t.constructs || []).map(c => chip(c, 'evidence')))), h('div', { class: 'row' }, h('a', { class: 'btn', href: link.task(t.id) }, 'Selection and every use →'))),
    t.held ? h('div', { class: 'notice' }, 'This task id or its split group is in the held-out exclusion set: training never sees it, and it is view-only here.') : null,
    whereView(t.where)));
  const sel = params.tab || 'program';
  const tabs = [['program', `Program (${Object.keys(t.program.files || {}).length} files)`], ['world', `World (${Object.keys(t.world.folder_files || {}).length} files)`], ['oracle', 'Oracle'], ['qualification', 'Qualification'], ['curriculum', 'Curriculum'], ['provenance', 'Provenance']];
  v.append(h('div', { class: 'tabs2', role: 'tablist' }, tabs.map(([k, l]) => h('a', { role: 'tab', class: 'tab2', 'aria-selected': sel === k, href: link.item(target.dataset, t.id, { part: params.part, tab: k === 'program' ? '' : k }) }, l))));
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
/** Attach each tool result to the call that asked for it (results arrive in a later call's context), and find the task instruction. */
function prepTrajectory(t) {
  if (t._prep) return t._prep;
  const results = new Map(), consumed = new Set();
  t.calls.forEach((c, i) => {
    const ids = (c.target.tool_calls || []).map(x => x.id);
    if (!ids.length) return;
    const next = t.calls.slice(i + 1).find(n => n.invocation === c.invocation) || t.calls[i + 1];
    const later = t.calls.slice(i + 1);
    ids.forEach((id, k) => {
      let hit = null;
      for (const n of later) { const m = n.context.find(m2 => m2.role === 'tool' && id && m2.tool_call_id === id); if (m) { hit = m; break; } }
      if (!hit && next) hit = next.context.filter(m2 => m2.role === 'tool' && !consumed.has(m2))[k] || null;
      if (hit) { results.set(`${i}:${k}`, hit); consumed.add(hit); }
    });
  });
  const first = t.calls[0]?.context || [];
  const instruction = first.find(m => m.role === 'user');
  const system = first.find(m => m.role === 'system');
  t._prep = { results, consumed, instruction, system };
  return t._prep;
}
const callActions = c => {
  const names = (c.target.tool_calls || []).map(x => x.name + (x.parsed?.finish ? ' ✓' : '')), out = [];
  for (const n of names) { const last = out.at(-1); if (last && last[0] === n) last[1]++; else out.push([n, 1]); }
  return out.map(([n, k]) => k > 1 ? `${n} ×${k}` : n).join(', ') || (c.decision ? 'decision' : c.target.content ? 'reply' : '—');
};
function trajectoryView(ctx, t, { dsId, id, part }) {
  const opts = ctx.ui.traj, P = prepTrajectory(t);
  const v = h('div', { class: 'traj' });
  const tok = t.calls.reduce((s, c) => s + (c.usage?.completion_tokens || c.sampling?.tokens || 0), 0);
  const ok = t.outcome === 'task_success' || t.accepted;
  const isGrpo = dsId.startsWith('grpo:'), isHeld = dsId.startsWith('held-eval:');
  const title = t.group_label ? `Episode ${t.group_label} · sample ${t.sample}` : isHeld ? `Held episode · sample ${t.sample}` : t.role === 'teacher' ? 'Teacher episode' : 'Episode';
  // neighbours: other samples of the group (GRPO) or the other side of a held pair
  let nav = null;
  if (isGrpo && t.group_label) {
    const g = load(ctx, `item:${JSON.stringify({ ds: dsId, id: t.group_label, part: 'groups' })}`, 'data.item', { ds: dsId, id: t.group_label, part: 'groups' }).value;
    if (g) nav = h('div', { class: 'samplenav', role: 'group', 'aria-label': 'Samples of this group' }, h('span', { class: 'muted small' }, 'Samples '), g.episodes.map(e => h('a', { class: `sbtn ${OUT_CLS(e.outcome)}${String(e.sample) === String(t.sample) ? ' on' : ''}`, href: link.item(dsId, `${t.group_label}:${e.sample}`, { part: 'groups' }),
      title: `sample ${e.sample}: ${String(e.outcome).replace(/_/g, ' ')}, A ${e.advantage == null ? '—' : signed(e.advantage, 2)}`, 'data-nav': String(e.sample) === String(t.sample) ? 'cur' : '' }, e.sample)),
      (() => { const other = g.episodes.find(e => (e.outcome === 'task_success') !== ok); return other ? h('a', { class: 'btn sm', href: link.compare(`${dsId}|${t.group_label}:${t.sample}|groups`, `${dsId}|${t.group_label}:${other.sample}|groups`) }, `Compare with a ${ok ? 'failure' : 'success'}`) : null; })());
  } else if (isHeld) {
    const other = part === 'base' ? 'samples' : 'base';
    nav = h('div', { class: 'samplenav' }, h('a', { class: 'btn sm', href: link.item(dsId, `${t.task_id}#${t.sample}`, { part: other }) }, part === 'base' ? 'Same task on the checkpoint →' : 'Same task on the start policy →'),
      h('a', { class: 'btn sm', href: link.compare(`${dsId}|${t.task_id}#${t.sample}|base`, `${dsId}|${t.task_id}#${t.sample}|samples`) }, 'Compare side by side'));
  }
  v.append(h('div', { class: 'card ephead' }, h('div', { class: 'eptitle' }, h('div', {}, h('div', { class: 'row' }, h('h1', {}, title), t.outcome ? h('span', { class: `chip big ${ok ? 'ok' : 'crit'}` }, String(t.outcome).replace(/_/g, ' ')) : null,
    t.held ? chip('🔒 held-out', 'crit') : null, t.side ? chip(t.side) : null),
  h('div', { class: 'row small', style: 'margin-top:4px' }, t.task_id ? a(link.task(t.task_id), { class: 'mono', title: 'Selection and every use of this task' }, t.task_id) : null, t.family ? h('span', { class: 'muted' }, `· ${t.family}`) : null)),
  t.group_label ? h('a', { class: 'btn sm', href: link.item(dsId, t.group_label, { part: 'groups' }) }, 'Whole group') : null),
  h('div', { class: 'facts' }, t.reward != null ? h('span', {}, 'reward ', h('b', {}, fmt(t.reward))) : null, t.graded != null ? h('span', {}, 'graded ', h('b', {}, fmt(t.graded))) : null,
    t.advantage != null ? h('span', { title: 'Group-normalized advantage (r − mean) / (std + 1e-6), reconstructed from the group rewards' }, 'advantage ', h('b', { class: t.advantage > 0 ? 'pos' : t.advantage < 0 ? 'neg' : '' }, signed(t.advantage, 2)), t.spread === false ? h('span', { class: 'muted' }, ' (no spread: trains nothing)') : null) : null,
    h('span', {}, h('b', {}, t.calls.length), ' calls'), tok ? h('span', {}, h('b', {}, count(tok)), ' generated tokens') : null),
  nav,
  t.checks ? h('div', { class: 'row small', style: 'margin-top:6px' }, Object.entries(t.checks).map(([k, x]) => chip(`${x ? '✔' : '✖'} ${k.replace(/_/g, ' ')}`, x ? 'ok' : 'crit'))) : null,
  (t.rejection_reasons || []).length ? h('div', { class: 'small', style: 'margin-top:4px' }, h('b', {}, 'Rejected: '), t.rejection_reasons.join('; ')) : null,
  P.instruction ? h('details', { class: 'instruction', open: P.instruction.content.length < 900 }, h('summary', {}, h('b', {}, 'Task instruction'), h('span', { class: 'muted small' }, ` · ${P.instruction.content.length.toLocaleString()} characters`)), h('div', { class: 'pre' }, P.instruction.content)) : null,
  (t.unknown || []).length ? h('details', { class: 'small' }, h('summary', {}, 'What these files cannot tell'), h('ul', {}, t.unknown.map(u => h('li', {}, u)))) : null));
  const invs = [...new Set(t.calls.map(c => c.invocation))];
  const fnOf = c => c.function && c.function !== 'unknown' ? c.function : '';
  const rail = h('ol', { class: 'callrail' }, t.calls.map(c => h('li', {}, h('a', { href: `#call-${c.i}`, 'data-call': c.i, onclick: e => { e.preventDefault(); jumpCall(c.i); } },
    h('span', { class: 'ci' }, c.i), h('span', { class: 'inv', style: `background:var(--s${(invs.indexOf(c.invocation) % 8) + 1})` }),
    h('span', { class: 'rtx' }, h('span', { class: 'ract' }, callActions(c)), fnOf(c) ? h('span', { class: 'rfn' }, fnOf(c)) : null),
    c.neuralese ? h('span', { class: 'nz', title: 'Neuralese span in context' }, 'N') : null, c.sampling?.tail_tokens ? h('span', { class: 'tailn', title: 'sampled tokens below the tail floor (masked when A > 0)' }, c.sampling.tail_tokens) : null))));
  const ctl = h('div', { class: 'toolbar small trajctl' }, ...[['reasoning', 'Expand reasoning'], ['context', 'Show what each call saw'], ['results', 'Expand tool results']].map(([k, l]) =>
    h('label', {}, h('input', { type: 'checkbox', checked: opts[k], onchange: e => { opts[k] = e.target.checked; store('ops.traj', opts); ctx.render(); } }), ` ${l}`)),
  h('span', { class: 'muted' }, h('kbd', {}, 'j'), h('kbd', {}, 'k'), ' calls', isGrpo ? [' · ', h('kbd', {}, '['), h('kbd', {}, ']'), ' samples'] : null),
  invs.length > 1 ? h('span', { class: 'muted invkeys' }, `${invs.length} invocations: `, invs.slice(0, 8).map((inv, k) => h('span', { class: 'invkey' }, h('i', { style: `background:var(--s${(k % 8) + 1})` }), fnOf(t.calls.find(c => c.invocation === inv)) || String(inv).slice(-8)))) : null);
  const calls = h('div', { class: 'calls' }, t.calls.map(c => callView(ctx, c, t, { dsId, id, invs, P })));
  v.append(h('div', { class: 'trajgrid' }, h('nav', { class: 'card railcard', 'aria-label': 'Calls' }, h('h3', {}, `${t.calls.length} calls`), rail), h('div', {}, ctl, calls)));
  if (t.graph?.length) v.append(h('details', { class: 'card', style: 'margin-top:12px' }, h('summary', {}, `Execution graph (${t.graph.length} invocations)`), h('table', { class: 'data' }, h('thead', {}, h('tr', {}, ['definition', 'signature', 'context', 'events', 'phases'].map(x => h('th', {}, x)))),
    h('tbody', {}, t.graph.map(n => h('tr', {}, h('td', { class: 'mono' }, n.definition || '—'), h('td', { class: 'mono small' }, n.signature || ''), h('td', {}, n.context || ''), h('td', { class: 'n' }, n.events), h('td', {}, (n.phases || []).join(', '))))))));
  if (t.value !== undefined && t.value !== null) v.append(h('details', { class: 'card', style: 'margin-top:12px' }, h('summary', {}, 'Returned value'), pre(json(t.value))));
  setTimeout(() => { watchCalls(); if (ctx.ui.focusCall != null) { jumpCall(ctx.ui.focusCall, 'auto'); ctx.ui.focusCall = null; } }, 0);
  return v;
}
function jumpCall(i, behavior = 'smooth') { document.getElementById(`call-${i}`)?.scrollIntoView({ behavior, block: 'start' }); markCall(i); }
function markCall(i) { document.querySelectorAll('.callrail a').forEach(x => x.classList.toggle('current', x.dataset.call === String(i))); }
let callObserver = null;
function watchCalls() {
  callObserver?.disconnect();
  const secs = [...document.querySelectorAll('section.call[id^="call-"]')];
  if (!secs.length) return;
  callObserver = new IntersectionObserver(es => { const top = es.filter(e => e.isIntersecting).sort((x, y) => x.boundingClientRect.top - y.boundingClientRect.top)[0]; if (top) markCall(top.target.id.slice(5)); }, { rootMargin: '-20% 0px -70% 0px' });
  secs.forEach(s => callObserver.observe(s));
}
const statBox = (val, label) => h('div', { class: 'stat' }, h('b', {}, val), h('span', {}, label));
function callView(ctx, c, t, { dsId, id, invs, P }) {
  const o = ctx.ui.traj;
  const box = h('section', { class: 'call', id: `call-${c.i}`, tabindex: '-1' });
  const tk = `${id}:${c.i}`;
  box.append(h('header', {}, h('span', { class: 'inv', style: `background:var(--s${(invs.indexOf(c.invocation) % 8) + 1})` }), h('b', {}, `Call ${c.i}`),
    h('span', { class: 'muted small' }, [c.function && c.function !== 'unknown' ? c.function : null, c.role, c.context_shared ? `continues ${c.context_shared} messages` : null].filter(Boolean).join(' · ')),
    h('span', { style: 'flex:1' }), c.sampling ? h('span', { class: 'small muted', title: 'sampled tokens · mean log-probability · finish reason' }, `${count(c.sampling.tokens)} tok · logp̄ ${fmt(c.sampling.logp_mean, 2)}${c.sampling.finish ? ' · ' + c.sampling.finish : ''}`) : null,
    c.sampling && dsId.startsWith('grpo:') ? h('button', { class: `btn sm${ctx.ui.tokens[tk] ? ' on' : ''}`, 'aria-pressed': !!ctx.ui.tokens[tk], onclick: () => { ctx.ui.tokens[tk] = !ctx.ui.tokens[tk]; ctx.render(); } }, 'Tokens & weights') : null));
  if (ctx.ui.tokens[tk]) box.append(tokenPanel(ctx, dsId, id, c.i));
  const out = h('div', { class: 'target' }, h('div', { class: 'tlabel' }, t.role === 'teacher' ? 'Teacher output' : 'Model output', t.advantage != null && dsId.startsWith('grpo:') ? h('span', { class: 'muted' }, ` · trained with A = ${signed(t.advantage, 2)}`) : null));
  if (c.target.reasoning) out.append(h('details', { class: 'reasoning', open: o.reasoning || c.target.reasoning.length < 600 }, h('summary', {}, `Reasoning · ${c.target.reasoning.length.toLocaleString()} chars`, h('span', { class: 'rprev' }, ' ' + c.target.reasoning.slice(0, 140).replace(/\s+/g, ' '))), h('div', { class: 'rtext' }, c.target.reasoning)));
  if (c.target.content) out.append(h('div', { class: 'content' }, c.target.content));
  (c.target.tool_calls || []).forEach((tc, k) => out.append(toolCallView(tc, P.results.get(`${c.i}:${k}`), o.results)));
  if (c.decision) out.append(h('div', { class: 'decision' }, h('b', {}, 'Decision '), valueView(c.decision)));
  if (!c.target.reasoning && !c.target.content && !(c.target.tool_calls || []).length && !c.decision) out.append(h('div', { class: 'muted small' }, 'Empty output.'));
  box.append(out);
  const fresh = c.context.filter(m => !P.consumed.has(m) && !(c.i === 0 && m === P.instruction));
  if (fresh.length) {
    const sum = fresh.map(m => `${m.role === 'tool' ? 'tool result' : m.role} ${m.content.length > 999 ? (m.content.length / 1000).toFixed(1) + 'k' : m.content.length}`).join(' · ');
    const saw = h('details', { class: 'saw', open: o.context && fresh.every(m => m.role !== 'system') }, h('summary', {}, h('span', { class: 'tlabel' }, `What it saw before this call`), h('span', { class: 'muted small' }, ` ${fresh.length} new message${fresh.length > 1 ? 's' : ''}: ${sum}`)));
    for (const m of fresh) { if (m.role === 'system' && c.i > 0) continue; saw.append(messageView(m)); }
    box.append(saw);
  }
  return box;
}
function messageView(m) {
  if (m.role === 'system') return h('details', { class: 'msg system' }, h('summary', { class: 'mlabel' }, `system prompt · ${m.content.length.toLocaleString()} chars`), h('div', { class: 'pre' }, m.content));
  const cls = `msg ${m.role}${m.host_generated ? ' host' : ''}`;
  const label = m.role === 'tool' ? `tool result${m.name ? ' · ' + m.name : ''}` : m.role === 'assistant' && m.host_generated ? 'assistant (host-generated)' : m.role;
  return h('div', { class: cls }, h('div', { class: 'mlabel' }, label), m.reasoning ? h('details', { class: 'reasoning' }, h('summary', {}, 'reasoning'), h('div', { class: 'rtext' }, m.reasoning)) : null,
    clampText(m.content, 2500, m.role === 'tool' ? 'mono pre' : 'pre'), (m.tool_calls || []).map(tc => toolCallView(tc)));
}
function toolCallView(tc, result, expand) {
  const p = tc.parsed;
  const code = p && typeof p.code === 'string' ? p.code : null;
  const rest = p ? Object.fromEntries(Object.entries(p).filter(([k]) => k !== 'code' && k !== 'finish')) : null;
  const box = h('div', { class: 'toolcall' }, h('div', { class: 'tclabel' }, h('b', { class: 'mono' }, tc.name || '?'), p?.finish ? chip('finish', 'ok', 'This call ends the invocation') : null, p?.status ? chip(String(p.status)) : null),
    code ? pre(code, 'ts') : null, rest && Object.keys(rest).length ? h('div', { class: 'small' }, valueView(rest)) : !p ? pre(tc.arguments || '', 'small') : null);
  if (result) {
    const txt = result.content || '';
    const err = /^(error|Error|TypeError|ReferenceError|SyntaxError)\b|"error"\s*:/.test(txt.slice(0, 200));
    box.append(h('details', { class: `result${err ? ' err' : ''}`, open: expand || txt.length < 400 }, h('summary', {}, h('span', { class: 'tlabel' }, err ? 'Result (error)' : 'Result'), h('span', { class: 'muted small' }, ` ${txt.length.toLocaleString()} chars `), h('span', { class: 'rprev mono' }, txt.slice(0, 120).replace(/\s+/g, ' '))),
      clampText(txt, 4000, 'mono pre')));
  }
  return box;
}
function tokenPanel(ctx, dsId, id, call) {
  const input = { ds: dsId, id, call };
  const ent = load(ctx, `tok:${JSON.stringify(input)}`, 'data.tokens', input);
  const p = pending(ent, 'sampled tokens');
  if (p) return p;
  const t = ent.value, box = h('div', { class: 'tokens' });
  const n = t.pieces.length, tail = t.tail_masked.filter(Boolean).length, sec = t.second_close_masked.filter(Boolean).length;
  box.append(h('div', { class: 'toksum' }, h('span', {}, h('b', {}, count(n)), ' sampled tokens'), h('span', {}, h('b', {}, count(n - t.tail_masked.filter((x, k) => x || t.second_close_masked[k]).length)), ' trained'),
    h('span', {}, h('b', {}, tail), ' masked by the tail floor'), h('span', {}, h('b', {}, sec), ' masked after a second </think>'),
    h('span', {}, 'weight per token ', h('b', {}, `A·min(ρ, 2), A = ${signed(t.advantage, 2)}`))),
  h('div', { class: 'legend small' }, h('span', { class: 'key' }, h('i', { class: 'tk-hi' }), 'likely (logp > −1)'), h('span', { class: 'key' }, h('i', { class: 'tk-mid' }), 'logp −1 to −4'), h('span', { class: 'key' }, h('i', { class: 'tk-lo' }), 'unlikely (logp < −4)'),
    h('span', { class: 'key' }, h('i', { class: 'tk-mask' }), 'masked: weight 0'), h('span', { class: 'muted' }, 'Hover a token for its id and log-probability.')));
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
  const sel = ctx.ui.cmp, key = e => `${dsId}|${g.id}:${e.sample}|groups`;
  const wins = g.episodes.filter(e => e.outcome === 'task_success'), losses = g.episodes.filter(e => e.outcome !== 'task_success');
  const best = [...wins].sort((x, y) => (y.advantage ?? 0) - (x.advantage ?? 0))[0], worst = [...losses].sort((x, y) => (x.advantage ?? 0) - (y.advantage ?? 0))[0];
  const spread = new Set(g.episodes.map(e => e.reward)).size > 1;
  return h('div', {}, h('div', { class: 'card' }, h('div', { class: 'eptitle' }, h('div', {}, h('h1', {}, `Group ${g.id}`),
    h('div', { class: 'row small', style: 'margin-top:4px' }, g.task_id ? a(link.task(g.task_id), { class: 'mono' }, g.task_id) : null)),
    h('div', { class: 'row' }, best && worst ? h('a', { class: 'btn primary', href: link.compare(key(best), key(worst)) }, 'Compare a success with a failure') : null)),
  h('div', { class: 'facts' }, h('span', {}, h('b', {}, `${wins.length}/${g.episodes.length}`), ' passed'), h('span', {}, 'mean reward ', h('b', {}, fmt(g.episodes.reduce((s, e) => s + (e.reward || 0), 0) / Math.max(1, g.episodes.length)))),
    h('span', {}, spread ? 'rewards differ: the group trains' : h('b', { class: 'neg' }, 'no reward spread: every advantage is 0, the group trains nothing')), h('span', { class: 'muted' }, `${g.reward_mode} rewards`)),
  h('table', { class: 'items', style: 'margin-top:10px' }, h('thead', {}, h('tr', {}, ['compare', 'sample', 'outcome', 'reward', h('span', { title: '(r − mean) / (std + 1e-6), as grpo.group_advantages computes it' }, 'advantage'), 'calls'].map(x => h('th', {}, x)))),
    h('tbody', {}, g.episodes.map(e => h('tr', { class: 'clickrow kbrow', 'data-href': link.item(dsId, `${g.id}:${e.sample}`, { part: 'groups' }), onclick: ev => { if (!ev.target.closest('a,input')) location.hash = link.item(dsId, `${g.id}:${e.sample}`, { part: 'groups' }); } },
      h('td', {}, h('input', { type: 'checkbox', 'aria-label': `select sample ${e.sample} to compare`, checked: sel.includes(key(e)), onchange: ev => { ev.target.checked ? sel.push(key(e)) : sel.splice(sel.indexOf(key(e)), 1); while (sel.length > 2) sel.shift(); ctx.render(); } })),
      h('td', {}, a(link.item(dsId, `${g.id}:${e.sample}`, { part: 'groups' }), `sample ${e.sample}`)), h('td', {}, cell('outcome', e.outcome)), h('td', { class: 'n' }, fmt(e.reward)),
      h('td', {}, h('div', { class: 'advbar' }, h('i', { class: e.advantage >= 0 ? 'pos' : 'neg', style: `width:${(50 * Math.abs(e.advantage || 0) / max).toFixed(1)}%` }), h('span', {}, e.advantage == null ? '—' : signed(e.advantage, 2)))),
      h('td', { class: 'n' }, e.calls))))),
  h('div', { class: 'row', style: 'margin-top:8px' }, sel.length === 2 ? h('a', { class: 'btn', href: link.compare(sel[0], sel[1]) }, 'Compare the two selected samples') : h('span', { class: 'muted small' }, 'Tick two samples to compare them side by side.'))));
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
  const last = s.last_step ?? null;
  v.append(h('div', { class: 'card' }, h('div', { class: 'eptitle' }, h('div', {}, h('h1', {}, `Step ${s.id}`), h('p', { class: 'lede' }, 'What reached the learner at this step: the groups taken, how episodes became learner rows, the token masks, and the trained chains.')),
    h('div', { class: 'row' }, s.id > 1 ? h('a', { class: 'btn sm', href: link.step(dsId, s.id - 1) }, `‹ step ${s.id - 1}`) : null, last == null || s.id < last ? h('a', { class: 'btn sm', href: link.step(dsId, s.id + 1) }, `step ${s.id + 1} ›`) : h('span', { class: 'muted small' }, 'latest step'),
      h('a', { class: 'btn sm', href: link.ds(dsId, { part: 'steps' }) }, 'All steps'))),
    (s.flags || []).length ? h('div', { class: 'row', style: 'margin:6px 0' }, h('b', { class: 'small' }, 'Sentinels fired: '), s.flags.map(f => chip(f.replace(/_/g, ' '), 'warn'))) : null,
    h('div', { class: 'grid g4 mini4' }, statBox(count(sel.groups), 'groups taken'), statBox(count(sel.groups_with_spread), 'with reward spread (carry advantage)'), statBox(count(sel.dropped_stale_groups), 'dropped as stale (> max staleness)'), statBox(fmt(sel.reward_mean), 'mean reward')),
    h('h3', { style: 'margin-top:10px' }, 'Lowering: episodes and calls turned into learner rows (drops, self-distillation, decisions)'), kvCols(Object.entries(low).filter(([, x]) => typeof x === 'number').map(([k, x]) => [k.replace(/_/g, ' '), count(x)])),
    h('h3', {}, 'Token masks'), kv([['tail logp floor', sel.tail_logp_floor], ['tail-masked fraction', sel.tail_masked_fraction], ['second-close masked', sel.second_close_masked], ['loss tokens', count(sel.loss_tokens)], ['sequences', count(sel.sequences)]]),
    ['staleness', 'dynamic_sampling', 'memetic_groups', 'replay_corpora', 'sd', 'teacher', 'reasoning_rl'].filter(k => sel[k] && Object.keys(sel[k]).length).map(k => h('details', {}, h('summary', {}, k.replace(/_/g, ' ')), pre(json(sel[k])))),
    h('details', { class: 'small' }, h('summary', {}, 'What these files cannot tell'), h('ul', {}, (s.unknown || []).map(u => h('li', {}, u))))));
  if (s.by_role && Object.keys(s.by_role).length) v.append(h('div', { class: 'card', style: 'margin-top:12px' }, h('h3', {}, 'Trained chains by role'),
    h('div', { class: 'row' }, Object.entries(s.by_role).map(([r, x]) => chip(`${r}: ${x.chains} chains · ${count(x.trained_tokens)} trained tokens`)))));
  if (s.chains) v.append(h('div', { class: 'card flush', style: 'margin-top:12px' }, h('div', { class: 'tablehead' }, h('b', {}, `${s.chains.length} chains in the learner batch`)),
    h('div', { class: 'scroll' }, h('table', { class: 'items' }, h('thead', {}, h('tr', {}, ['chain', 'role', 'task', 'group / round', 'advantage', 'trained / total tokens', 'segments'].map(x => h('th', {}, x)))),
      h('tbody', {}, s.chains.map(c => h('tr', { class: 'clickrow kbrow', 'data-href': link.chain(dsId, s.id, c.i), onclick: () => { location.hash = link.chain(dsId, s.id, c.i); } }, h('td', {}, a(link.chain(dsId, s.id, c.i), `#${c.i}`)), h('td', {}, chip(c.role || '?')),
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

/* ---------- one task: selection in every run, and every use ---------- */
const VERDICT = { drawn: ['drawn earlier', 'evidence'], band: ['band pick candidate', 'ok'], explore: ['exploration only', 'evidence'], out: ['outside the pass band', 'warn'], excluded: ['excluded', 'crit'], absent: ['not in its pools', ''] };
function taskRefsPage(ctx, taskId) {
  const ent = load(ctx, `refs:${taskId}`, 'data.task_refs', { task_id: taskId });
  const p = pending(ent, `where ${taskId} is used (reads every run's groups and every teacher manifest; the first time can take a minute)`);
  if (p) return p;
  const w = ent.value;
  const v = h('div', { class: 'taskrefs' });
  const firstPool = (w.pools || [])[0];
  v.append(h('div', { class: 'card' }, h('div', { class: 'eptitle' }, h('div', {}, h('h1', { class: 'mono wrap' }, taskId), h('div', { class: 'row small', style: 'margin-top:4px' }, w.family ? chip(w.family) : null, w.held ? chip('🔒 held-out · view only', 'crit') : null,
    w.base_rate ? chip(`start policy ${w.base_rate.successes}/${w.base_rate.scored} passed`, '', 'ops/exp1-rft/base-pass-rates.json') : null, (w.errata || []).map(e => chip(`errata: ${e.file}`, 'crit', e.reason || '')))),
  firstPool ? h('a', { class: 'btn primary', href: link.item(firstPool.ds, taskId, { part: 'tasks' }) }, 'Open the task') : null)));
  const runs = w.runs || [], relevant = runs.filter(r => r.state !== 'absent' || r.groups.length), other = runs.filter(r => !relevant.includes(r));
  v.append(h('div', { class: 'sect' }, h('h2', {}, 'Why each GRPO run did or did not draw it')),
    h('div', { class: 'card flush' }, relevant.length ? h('table', { class: 'items' }, h('thead', {}, h('tr', {}, ['run', 'verdict', 'groups sampled', 'why'].map(x => h('th', {}, x)))),
      h('tbody', {}, relevant.map(r => h('tr', {}, h('td', {}, a(link.ds(r.ds), r.run)), h('td', {}, chip(...VERDICT[r.state] || [r.state])),
        h('td', {}, r.groups.length ? h('span', { class: 'dots' }, r.groups.map(g => h('a', { class: `gpill ${g.pass === g.n ? 'ok' : g.pass ? 'mid' : 'bad'}`, href: link.item(r.ds, g.group, { part: 'groups' }), title: `${g.group}: ${g.pass}/${g.n} passed${g.spread ? '' : ', no spread'}` }, `${g.pass}/${g.n}`))) : h('span', { class: 'muted' }, 'none')),
        h('td', { class: 'small' }, r.why, r.later ? h('div', { class: 'warnline' }, r.later) : null))))) : h('div', { class: 'empty' }, 'No GRPO run has this task in its pools.')),
    other.length ? h('p', { class: 'small muted' }, `Not in the pools of ${other.map(r => r.run).join(', ')}.`) : null);
  const list = (title, items, row, empty) => [h('div', { class: 'sect' }, h('h2', {}, title), h('span', { class: 'muted' }, items.length || '')), items.length ? h('div', { class: 'card' }, h('ul', { class: 'plain' }, items.map(x => h('li', {}, row(x))))) : h('p', { class: 'muted small' }, empty)];
  v.append(...list('Task pools', w.pools || [], x => [a(link.item(x.ds, taskId, { part: 'tasks' }), x.ds.replace(/^pool:/, '')), h('span', { class: 'muted small mono' }, ` ${x.path}`)], 'In no current task pool.'),
    ...list('Teacher attempts', w.teacher || [], x => [a(link.item(x.ds, taskId, { part: 'tasks' }), x.ds.replace(/^teacher-gen:/, '')), h('span', { class: 'small muted' }, ` · ${x.teacher || ''} · ${x.successes}/${x.episodes} succeeded`)], 'No teacher campaign attempted it.'),
    ...list('SFT sets', w.sft || [], x => [a(link.ds(x.ds, { part: 'rows', q: taskId }), x.ds.replace(/^[a-z-]+:/, '')), h('span', { class: 'small muted' }, ` · ${x.rows} rows`)], 'In no built SFT set.'));
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
  if (sides.length < 2) return h('div', { class: 'card empty' }, h('div', {}, h('b', {}, 'Nothing to compare yet.'), h('div', { class: 'small' }, 'Open a GRPO group and tick two samples, or open a held evaluation and click a task row.')));
  const ents = sides.map(s => load(ctx, `item:${JSON.stringify({ ds: s.ds, id: s.id, part: s.part })}`, 'data.item', { ds: s.ds, id: s.id, part: s.part }));
  const p = ents.map((e, k) => pending(e, `episode ${sides[k].id}`)).find(Boolean);
  if (p) return p;
  const [A, B] = ents.map(e => e.value);
  if (!A.calls || !B.calls) return h('div', { class: 'grid g2' }, h('div', { class: 'card' }, valueView(A)), h('div', { class: 'card' }, valueView(B)));
  const label = (x, s) => s.part === 'base' ? 'Start policy' : s.ds.startsWith('held-eval:') ? `Checkpoint ${s.ds.split(':').pop()}` : `Sample ${x.sample ?? s.id}`;
  const head = (x, s) => {
    const ok = x.outcome === 'task_success' || x.accepted;
    const tok = x.calls.reduce((n, c) => n + (c.usage?.completion_tokens || c.sampling?.tokens || 0), 0);
    return h('div', { class: `card cmphead ${ok ? 'okside' : 'badside'}` }, h('div', { class: 'row' }, h('h2', {}, label(x, s)), h('span', { class: `chip big ${ok ? 'ok' : 'crit'}` }, String(x.outcome || '?').replace(/_/g, ' ')), h('span', { style: 'flex:1' }), h('a', { class: 'btn sm', href: link.item(s.ds, s.id, { part: s.part }) }, 'Open episode')),
      h('div', { class: 'facts' }, x.reward != null ? h('span', {}, 'reward ', h('b', {}, fmt(x.reward))) : null, x.advantage != null ? h('span', {}, 'A ', h('b', {}, signed(x.advantage, 2))) : null, h('span', {}, h('b', {}, x.calls.length), ' calls'), tok ? h('span', {}, h('b', {}, count(tok)), ' tokens') : null),
      (x.rejection_reasons || []).length ? h('div', { class: 'small' }, h('b', {}, 'Rejected: '), x.rejection_reasons.join('; ')) : null);
  };
  const PA = prepTrajectory(A), PB = prepTrajectory(B);
  const v = h('div', { class: 'compare' }, h('div', { class: 'row', style: 'margin-bottom:8px' }, A.task_id ? a(link.task(A.task_id), { class: 'mono small' }, A.task_id) : null, h('span', { style: 'flex:1' }),
    h('a', { class: 'btn sm', href: link.compare(params.b, params.a) }, '⇄ Swap sides')),
  h('div', { class: 'grid g2 sticky-cmp' }, head(A, sides[0]), head(B, sides[1])));
  if (PA.instruction) v.append(h('details', { class: 'card instruction' }, h('summary', {}, h('b', {}, 'Task instruction'), h('span', { class: 'muted small' }, PB.instruction && PB.instruction.content !== PA.instruction.content ? ' · the two sides saw different instructions' : ' · same on both sides')), h('div', { class: 'pre' }, PA.instruction.content)));
  const n = Math.max(A.calls.length, B.calls.length);
  const side = (x, P, i) => {
    const c = x.calls[i];
    if (!c) return h('div', { class: 'call muted empty' }, 'no call');
    const out = h('div', { class: 'call' }, h('header', {}, h('b', {}, `Call ${i}`), h('span', { class: 'muted small' }, ` ${c.function && c.function !== 'unknown' ? c.function : ''}`)));
    if (c.target.reasoning) out.append(h('details', { class: 'reasoning', open: ctx.ui.traj.reasoning }, h('summary', {}, `Reasoning · ${c.target.reasoning.length.toLocaleString()} chars`, h('span', { class: 'rprev' }, ' ' + c.target.reasoning.slice(0, 100).replace(/\s+/g, ' '))), h('div', { class: 'rtext' }, c.target.reasoning)));
    if (c.target.content) out.append(h('div', { class: 'content' }, c.target.content));
    (c.target.tool_calls || []).forEach((tc, k) => out.append(toolCallView(tc, P.results.get(`${i}:${k}`), ctx.ui.traj.results)));
    return out;
  };
  for (let i = 0; i < n; i++) {
    const same = A.calls[i] && B.calls[i] && callActions(A.calls[i]) === callActions(B.calls[i]) && (A.calls[i].target.tool_calls || []).map(t => t.arguments).join() === (B.calls[i].target.tool_calls || []).map(t => t.arguments).join();
    v.append(h('div', { class: `cmprow${same ? ' same' : ''}` }, h('div', { class: 'cmpidx' }, `${i}`, same ? h('span', { class: 'muted small', title: 'Both sides made the same tool calls with the same arguments' }, ' same') : null), h('div', { class: 'grid g2' }, side(A, PA, i), side(B, PB, i))));
  }
  return v;
}
