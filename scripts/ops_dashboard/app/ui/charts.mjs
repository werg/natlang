// Small DOM and SVG chart helpers for the ops dashboard. Page-owned code: plain SVG, no dependencies.
// Series colours are CSS variables (--s1 ... --s8, themed in style.css); text always wears text tokens, never the series colour.
import { num, fmt, ema } from './model.mjs';

export function h(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k.startsWith('aria-') && typeof v === 'boolean') { n.setAttribute(k, String(v)); continue; }   // aria states need "true"/"false"
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)) if (kid != null && kid !== false) n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  return n;
}
export const svg = (tag, attrs = {}, ...kids) => {
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) n.setAttribute(k, v);
  n.append(...kids);
  return n;
};

export const slotVar = slot => `var(--s${(slot % 8) + 1})`;

/** Round tick values covering [lo, hi]. */
export function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) { hi = lo + 1; }
  const raw = (hi - lo) / Math.max(1, n), p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : Number(v.toPrecision(12)));
  return out;
}
const TIME_STEPS = [60e3, 300e3, 900e3, 1800e3, 3600e3, 3 * 3600e3, 6 * 3600e3, 12 * 3600e3, 86400e3, 2 * 86400e3, 7 * 86400e3];
export function timeTicks(lo, hi, n = 5) {
  const off = -new Date(lo).getTimezoneOffset() * 60e3;
  const step = TIME_STEPS.find(s => (hi - lo) / s <= n) ?? TIME_STEPS.at(-1);
  const out = [];
  for (let v = Math.ceil((lo + off) / step) * step - off; v <= hi; v += step) out.push(v);
  return out;
}

const tipFmt = v => fmt(v, 4);

/**
 * Line chart with raw + smoothed lines, optional error whiskers (lo/hi), horizontal reference lines, vertical markers, gaps, a crosshair
 * tooltip that lists every series at the exact x, and keyboard stepping. Returns {el, set(opts)}; call set() with new data to redraw.
 * opts: {series:[{id,label,slot,points:[{x,y,lo?,hi?}],dashed?}], smooth (EMA alpha, 0 = raw), logy, yLabel, xLabel, xKind:'num'|'time',
 *        xFmt, yFmt, hlines:[{y,label}], vmarks:[{x,label}], gap (max x distance joined by a line), height, yMin, yMax, empty}
 */
export function lineChart(initial) {
  let o = {}, W = 600, hover = null;
  const root = h('div', { class: 'chart' });
  const legend = h('div', { class: 'legend' });
  const holder = h('div', { class: 'plot', tabindex: '0', role: 'img' });
  const tip = h('div', { class: 'tip', hidden: true });
  root.append(legend, holder, tip);

  const ro = new ResizeObserver(entries => { const w = Math.round(entries[0].contentRect.width); if (w && w !== W) { W = w; draw(); } });
  ro.observe(holder);

  function bounds() {
    let xlo = Infinity, xhi = -Infinity, ylo = Infinity, yhi = -Infinity;
    for (const s of o.series) for (const p of s.points) {
      if (!num(p.x)) continue;
      xlo = Math.min(xlo, p.x); xhi = Math.max(xhi, p.x);
      for (const v of [p.y, p.lo, p.hi]) if (num(v) && (!o.logy || v > 0)) { ylo = Math.min(ylo, v); yhi = Math.max(yhi, v); }
    }
    for (const l of o.hlines || []) if (num(l.y)) { ylo = Math.min(ylo, l.y); yhi = Math.max(yhi, l.y); }
    if (num(o.yMin)) ylo = Math.min(ylo, o.yMin);
    if (num(o.yMax)) yhi = Math.max(yhi, o.yMax);
    if (!(xhi >= xlo)) return null;
    if (xhi === xlo) { xlo -= 1; xhi += 1; }
    if (!(yhi >= ylo)) return null;
    if (yhi === ylo) { const d = Math.abs(ylo) * 0.1 || 1; ylo -= d; yhi += d; }
    return { xlo, xhi, ylo, yhi };
  }

  function draw() {
    holder.replaceChildren();
    const have = o.series.filter(s => s.points.length);
    legend.replaceChildren(...(o.series.length > 1 ? o.series.map(s => h('span', { class: 'key' }, h('i', { class: s.dashed ? 'dashed' : '', style: `background:${slotVar(s.slot)}` }), s.label)) : []));
    const b = bounds();
    if (!have.length || !b) { holder.append(h('div', { class: 'empty' }, o.empty || 'No data in the retained range yet.')); holder.setAttribute('aria-label', 'no data'); return; }
    const H = o.height || 220, w = Math.max(160, W), yfmt = o.yFmt || fmt, xfmt = o.xFmt || (v => String(Math.round(v)));
    const ly = v => Math.log10(v);
    const y0 = o.logy ? ly(b.ylo) : b.ylo, y1 = o.logy ? ly(b.yhi) : b.yhi, pad = (y1 - y0) * 0.06;
    const lo = y0 - pad, hi = y1 + pad;
    const yt = o.logy ? niceTicks(Math.ceil(lo), Math.floor(hi), 4).map(e => 10 ** e) : niceTicks(lo, hi, Math.max(3, Math.floor((H - 30) / 44)));
    const m = { l: Math.max(40, 12 + 6.3 * Math.max(3, ...yt.map(v => yfmt(v).length))), r: 12, t: 8, b: 22 }, pw = w - m.l - m.r, ph = H - m.t - m.b;
    const X = v => m.l + ((v - b.xlo) / (b.xhi - b.xlo)) * pw;
    const Y = v => m.t + (1 - ((o.logy ? ly(Math.max(v, 1e-12)) : v) - lo) / (hi - lo)) * ph;
    const root2 = svg('svg', { viewBox: `0 0 ${w} ${H}`, width: w, height: H, 'aria-hidden': 'true' });
    for (const t of yt) {
      if (Y(t) < m.t - 1 || Y(t) > m.t + ph + 1) continue;
      root2.append(svg('line', { class: 'grid', x1: m.l, x2: w - m.r, y1: Y(t), y2: Y(t) }), svg('text', { class: 'tick', x: m.l - 6, y: Y(t) + 3.5, 'text-anchor': 'end' }, yfmt(t)));
    }
    const xt = o.xKind === 'time' ? timeTicks(b.xlo, b.xhi, Math.max(2, Math.floor(pw / 90))) : niceTicks(b.xlo, b.xhi, Math.max(2, Math.floor(pw / 70)));
    for (const t of xt) root2.append(svg('text', { class: 'tick', x: X(t), y: H - 6, 'text-anchor': 'middle' }, xfmt(t)));
    root2.append(svg('line', { class: 'axis', x1: m.l, x2: w - m.r, y1: m.t + ph, y2: m.t + ph }));
    for (const l of o.hlines || []) {
      if (!num(l.y)) continue;
      root2.append(svg('line', { class: 'ref', x1: m.l, x2: w - m.r, y1: Y(l.y), y2: Y(l.y) }), svg('text', { class: 'reflabel', x: w - m.r - 2, y: Y(l.y) - 4, 'text-anchor': 'end' }, l.label));
    }
    for (const v of o.vmarks || []) {
      if (v.x < b.xlo || v.x > b.xhi) continue;
      root2.append(svg('line', { class: 'vmark', x1: X(v.x), x2: X(v.x), y1: m.t, y2: m.t + ph }), svg('text', { class: 'reflabel', x: X(v.x) + 3, y: m.t + 10 }, v.label));
    }
    const smooth = o.smooth || 0;
    for (const s of o.series) {
      const pts = s.points.filter(p => num(p.x));
      const sm = ema(pts, smooth);
      const color = `stroke:${slotVar(s.slot)}`, fillc = `stroke:${slotVar(s.slot)};fill:${slotVar(s.slot)}`;
      const path = key => {
        let d = '', prev = null;
        for (const p of sm) {
          const v = p[key];
          if (!num(v) || (o.logy && v <= 0)) { prev = null; continue; }
          const joined = prev && (!o.gap || p.x - prev.x <= o.gap);
          d += `${joined ? 'L' : 'M'}${X(p.x).toFixed(1)} ${Y(v).toFixed(1)}`;
          prev = p;
        }
        return d;
      };
      if (smooth) root2.append(svg('path', { class: 'raw', style: color, d: path('y') }));
      root2.append(svg('path', { class: `line${s.dashed ? ' dashed' : ''}`, style: color, d: path(smooth ? 's' : 'y') }));
      for (const p of sm) {
        if (num(p.lo) && num(p.hi)) root2.append(svg('line', { class: 'whisker', style: color, x1: X(p.x), x2: X(p.x), y1: Y(p.lo), y2: Y(p.hi) }));
      }
      if (pts.length <= 40 || s.dots) for (const p of sm) { const v = smooth ? p.s : p.y; if (num(v)) root2.append(svg('circle', { class: 'dot', style: fillc, cx: X(p.x), cy: Y(v), r: 3 })); }
      const last = sm.findLast(p => num(smooth ? p.s : p.y));
      if (last) root2.append(svg('circle', { class: 'enddot', style: fillc, cx: X(last.x), cy: Y(smooth ? last.s : last.y), r: 4 }));
      if (o.series.length === 1 && last) root2.append(svg('text', { class: 'endlabel', x: Math.min(X(last.x) + 7, w - m.r), y: Y(smooth ? last.s : last.y) - 8, 'text-anchor': X(last.x) > w - 60 ? 'end' : 'start' }, yfmt(smooth ? last.s : last.y)));
    }
    const cross = svg('line', { class: 'cross', y1: m.t, y2: m.t + ph, visibility: 'hidden' });
    const marks = svg('g', {});
    root2.append(cross, marks);
    holder.append(root2);
    holder.setAttribute('aria-label', `${o.yLabel || 'chart'}: ${have.map(s => `${s.label} latest ${yfmt(s.points.at(-1).y)}`).join('; ')}. Press Enter for the value table.`);

    const xs = [...new Set(have.flatMap(s => s.points.map(p => p.x)))].sort((a, c) => a - c);
    const show = x => {
      hover = x;
      if (x == null) { cross.setAttribute('visibility', 'hidden'); marks.replaceChildren(); tip.hidden = true; return; }
      cross.setAttribute('visibility', 'visible'); cross.setAttribute('x1', X(x)); cross.setAttribute('x2', X(x));
      marks.replaceChildren();
      tip.replaceChildren(h('div', { class: 'tip-x' }, (o.xLabel ? o.xLabel + ' ' : '') + xfmt(x)));
      for (const s of o.series) {
        const p = s.points.find(q => q.x === x);
        const v = p && (smooth ? sm_at(s, x) : p.y);
        if (p && num(p.y)) marks.append(svg('circle', { class: 'hoverdot', style: `stroke:${slotVar(s.slot)}`, cx: X(x), cy: Y(smooth && num(v) ? v : p.y), r: 5 }));
        const val = p && num(p.y) ? yfmt(p.y) + (num(p.lo) ? `  [${yfmt(p.lo)}, ${yfmt(p.hi)}]` : '') + (smooth && num(v) ? `  (smoothed ${yfmt(v)})` : '') : '—';
        tip.append(h('div', { class: 'tip-row' }, h('i', { style: `background:${slotVar(s.slot)}` }), h('b', {}, val), h('span', {}, s.label)));
      }
      tip.hidden = false;
      const left = X(x) + 14 + tip.offsetWidth > w ? X(x) - 14 - tip.offsetWidth : X(x) + 14;
      tip.style.left = `${Math.max(0, left)}px`; tip.style.top = `${m.t}px`;
    };
    const sm_at = (s, x) => ema(s.points.filter(p => num(p.x)), smooth).find(p => p.x === x)?.s;
    holder.onpointermove = e => {
      const r = holder.getBoundingClientRect(), px = e.clientX - r.left, v = b.xlo + ((px - m.l) / pw) * (b.xhi - b.xlo);
      let best = xs[0];
      for (const x of xs) if (Math.abs(x - v) < Math.abs(best - v)) best = x;
      show(best);
    };
    holder.onpointerleave = () => show(null);
    holder.onkeydown = e => {
      const i = xs.indexOf(hover);
      if (e.key === 'ArrowLeft') { show(xs[Math.max(0, (i < 0 ? xs.length : i) - 1)]); e.preventDefault(); }
      else if (e.key === 'ArrowRight') { show(xs[Math.min(xs.length - 1, i + 1)]); e.preventDefault(); }
      else if (e.key === 'Escape') show(null);
    };
    holder.onblur = () => show(null);
  }

  function set(next) { o = { series: [], ...next }; hover = null; draw(); }
  set(initial);
  return { el: root, set, destroy: () => ro.disconnect() };
}

/** Value table for a chart's series: the same numbers as the tooltip, reachable without hovering. */
export function valueTable(series, { xLabel = 'step', xFmt = v => v, yFmt = fmt, limit = 60 } = {}) {
  const xs = [...new Set(series.flatMap(s => s.points.map(p => p.x)))].sort((a, b) => b - a).slice(0, limit);
  const rows = xs.map(x => h('tr', {}, h('th', { scope: 'row' }, xFmt(x)), ...series.map(s => { const p = s.points.find(q => q.x === x); return h('td', {}, p && num(p.y) ? yFmt(p.y) : '—'); })));
  return h('div', { class: 'scroll' }, h('table', { class: 'data' }, h('thead', {}, h('tr', {}, h('th', {}, xLabel), ...series.map(s => h('th', {}, s.label)))), h('tbody', {}, rows)));
}

/** Tiny trend line with an end dot; `floor`/`ceil` draw a reference line (e.g. a limit). */
export function spark(values, { width = 140, height = 32, ref, slot = 0, label = '' } = {}) {
  const v = values.filter(num);
  if (v.length < 2) return h('span', { class: 'muted small' }, v.length ? 'one value so far' : 'no data yet');
  let lo = Math.min(...v, ref ?? Infinity), hi = Math.max(...v, ref ?? -Infinity);
  if (hi === lo) { hi += 1; lo -= 1; }
  const x = i => 3 + (i / (v.length - 1)) * (width - 6), y = val => height - 4 - ((val - lo) / (hi - lo)) * (height - 8);
  const el = svg('svg', { class: 'spark', viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `${label} trend over ${v.length} values, latest ${fmt(v.at(-1))}` });
  if (ref != null) el.append(svg('line', { class: 'ref', x1: 0, x2: width, y1: y(ref), y2: y(ref) }));
  el.append(svg('path', { class: 'line', style: `stroke:${slotVar(slot)}`, d: v.map((val, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)} ${y(val).toFixed(1)}`).join('') }),
    svg('circle', { class: 'enddot', style: `fill:${slotVar(slot)};stroke:${slotVar(slot)}`, cx: x(v.length - 1), cy: y(v.at(-1)), r: 3 }));
  return el;
}

/**
 * Horizontal segmented bar. parts: [{label, value, color?}], total sets the full width, marks: [{at, label}] vertical limit ticks.
 * Segments are separated by a surface-coloured gap; every segment has a tooltip via title and a text alternative.
 */
export function segBar(parts, { total, marks = [], height = 14, label } = {}) {
  const sum = total ?? parts.reduce((a, p) => a + (p.value || 0), 0);
  const el = h('div', { class: 'seg', style: `height:${height}px`, role: 'img', 'aria-label': label || parts.map(p => `${p.label} ${fmt(p.value)}`).join(', ') });
  for (const p of parts) if (p.value > 0 && sum > 0) el.append(h('i', { style: `flex:0 0 ${(100 * p.value / sum).toFixed(3)}%;background:${p.color || 'var(--muted-fill)'}`, title: `${p.label}: ${fmt(p.value)}` }));
  for (const mk of marks) if (num(mk.at) && sum > 0) el.append(h('span', { class: 'segmark', style: `left:${(100 * mk.at / sum).toFixed(2)}%`, title: mk.label }));
  return el;
}

/** Observed width of an element, redrawn on change. */
function responsive(root, draw, fallback = 340) {
  let W = fallback;
  const ro = new ResizeObserver(es => { const w = Math.round(es[0].contentRect.width); if (w && w !== W) { W = w; draw(W); } });
  ro.observe(root);
  queueMicrotask(() => draw(W));
  return ro;
}

/**
 * Stacked bars over time (episodes per hour by machine). bars: [{t, <key>: n}], keys: [{key, label, slot}]. Tap or hover a bar to read
 * its numbers in the line below the plot; the same numbers are in the aria label.
 */
export function barStack(bars, { keys, height = 150, unit = 'episodes', tFmt } = {}) {
  const root = h('div', { class: 'chart' }), holder = h('div', { class: 'plot', role: 'img' }), detail = h('div', { class: 'panel-note barnote' }, 'Tap a bar for its numbers.');
  const legend = h('div', { class: 'legend' }, keys.map(k => h('span', { class: 'key' }, h('i', { style: `background:${slotVar(k.slot)}` }), k.label)));
  root.append(legend, holder, detail);
  const total = b => keys.reduce((a, k) => a + (b[k.key] || 0), 0);
  const label = b => `${(tFmt || (t => new Date(t).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })))(b.t)}: ${keys.map(k => `${k.label} ${b[k.key] || 0}`).join(', ')}, total ${total(b)} ${unit}`;
  holder.setAttribute('aria-label', `${unit} per hour. ${bars.map(label).join('; ')}`);
  const draw = W => {
    holder.replaceChildren();
    const max = Math.max(1, ...bars.map(total)), ticks = niceTicks(0, max, 3), top = ticks.at(-1) || max;
    const m = { l: 34, r: 6, t: 6, b: 20 }, pw = W - m.l - m.r, ph = height - m.t - m.b, bw = pw / bars.length;
    const el = svg('svg', { viewBox: `0 0 ${W} ${height}`, width: W, height, 'aria-hidden': 'true' });
    const Y = v => m.t + (1 - v / top) * ph;
    for (const t of ticks) el.append(svg('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }), svg('text', { class: 'tick', x: m.l - 5, y: Y(t) + 3.5, 'text-anchor': 'end' }, fmt(t)));
    bars.forEach((b, i) => {
      let acc = 0;
      const x = m.l + i * bw + 1.5, w = Math.max(2, bw - 3);
      for (const k of keys) {
        const v = b[k.key] || 0; if (!v) continue;
        const y0 = Y(acc), y1 = Y(acc + v);
        el.append(svg('rect', { class: 'bar', x, width: w, y: y1, height: Math.max(1, y0 - y1 - 1), rx: 2, style: `fill:${slotVar(k.slot)}` }));
        acc += v;
      }
      const hit = svg('rect', { class: 'hit', x: m.l + i * bw, width: bw, y: m.t, height: ph + 4, fill: 'transparent' });
      const show = () => { detail.textContent = label(b); };
      hit.addEventListener('pointerenter', show); hit.addEventListener('click', show);
      el.append(hit);
      if (i % 3 === 0 || i === bars.length - 1) el.append(svg('text', { class: 'tick', x: m.l + i * bw + bw / 2, y: height - 5, 'text-anchor': 'middle' }, new Date(b.t).toLocaleTimeString([], { hour: '2-digit' }).replace(/\s?(AM|PM)/i, '')));
    });
    el.append(svg('line', { class: 'axis', x1: m.l, x2: W - m.r, y1: m.t + ph, y2: m.t + ph }));
    holder.append(el);
  };
  responsive(holder, draw);
  return root;
}

/**
 * Round timeline: one row per round, coloured stage segments on a shared time axis, a "now" line, estimated parts hatched with an outline.
 * rows: [{label, segments:[{key,from,to,est}]}], stages: {key:{label, slot}}.
 */
export function gantt(rows, { now, stages, rowH = 26 } = {}) {
  const root = h('div', { class: 'chart' }), holder = h('div', { class: 'plot', role: 'img' });
  root.append(h('div', { class: 'legend' }, Object.values(stages).map(s => h('span', { class: 'key' }, h('i', { style: `background:${slotVar(s.slot)};height:10px;width:10px;border-radius:3px` }), s.label))), holder);
  const all = rows.flatMap(r => r.segments), lo = Math.min(...all.map(s => s.from), now), hi = Math.max(...all.map(s => s.to), now) + 20 * 60e3;
  holder.setAttribute('aria-label', `Round timeline. ${rows.map(r => `${r.label}: ${r.segments.map(s => `${stages[s.key]?.label}${s.est ? ' (estimate)' : ''} ${new Date(s.from).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} to ${new Date(s.to).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`).join(', ')}`).join('; ')}`);
  const draw = W => {
    holder.replaceChildren();
    const m = { l: 44, r: 8, t: 4, b: 20 }, pw = W - m.l - m.r, H = m.t + m.b + rows.length * rowH;
    const X = t => m.l + ((t - lo) / (hi - lo)) * pw;
    const el = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, 'aria-hidden': 'true' });
    for (const t of timeTicks(lo, hi, Math.max(2, Math.floor(pw / 80)))) {
      el.append(svg('line', { class: 'grid', x1: X(t), x2: X(t), y1: m.t, y2: H - m.b }));
      const d = new Date(t), day = d.getHours() < 2 ? d.toLocaleDateString([], { month: 'short', day: 'numeric' }) : '';
      el.append(svg('text', { class: 'tick', x: X(t), y: H - 6, 'text-anchor': 'middle' }, day || d.toLocaleTimeString([], { hour: '2-digit' }).replace(/\s?(AM|PM)/i, 'h')));
    }
    rows.forEach((r, i) => {
      const y = m.t + i * rowH;
      el.append(svg('text', { class: 'tick rowlabel', x: m.l - 6, y: y + rowH / 2 + 3, 'text-anchor': 'end' }, r.label));
      for (const s of r.segments) {
        const x = X(s.from), w = Math.max(2, X(s.to) - x), c = slotVar(stages[s.key]?.slot ?? 0);
        el.append(svg('rect', { x, y: y + 5, width: w, height: rowH - 10, rx: 4, style: s.est ? `fill:${c};fill-opacity:.25;stroke:${c};stroke-width:1.5;stroke-dasharray:4 3` : `fill:${c}` }));
      }
    });
    el.append(svg('line', { class: 'nowline', x1: X(now), x2: X(now), y1: m.t, y2: H - m.b }), svg('text', { class: 'reflabel', x: X(now) > W - 40 ? X(now) - 3 : X(now) + 3, y: m.t + 8, 'text-anchor': X(now) > W - 40 ? 'end' : 'start' }, 'now'));
    holder.append(el);
  };
  responsive(holder, draw);
  return root;
}

/**
 * Held-out pass rates by policy, in the order they were measured. points: [{label, y, lo, hi, series, scored, ran}] with y, lo, hi in [0,1].
 * Dots of one measurement series are joined and share a colour: only points of one series are comparable. Whiskers are 95% intervals over tasks.
 */
export function ladder(points, { height = 190 } = {}) {
  const root = h('div', { class: 'chart' }), holder = h('div', { class: 'plot', role: 'img' });
  const seriesIds = [...new Set(points.map(p => p.series))], slotOf = s => seriesIds.indexOf(s);
  root.append(h('div', { class: 'legend' }, seriesIds.map(s => h('span', { class: 'key' }, h('i', { style: `background:${slotVar(slotOf(s))}` }), `series ${String(s).slice(0, 4)}`)),
    seriesIds.length > 1 ? h('span', { class: 'key muted' }, 'only dots of one colour are comparable') : null), holder);
  holder.setAttribute('aria-label', `Held-out pass rate by policy: ${points.map(p => `${p.label} ${(100 * p.y).toFixed(1)}%, series ${String(p.series).slice(0, 4)}`).join('; ')}`);
  const draw = W => {
    holder.replaceChildren();
    const lo = Math.min(...points.map(p => p.lo ?? p.y)), hi = Math.max(...points.map(p => p.hi ?? p.y));
    const ticks = niceTicks(Math.floor(lo * 20) / 20, Math.ceil(hi * 20) / 20, 4), y0 = ticks[0], y1 = ticks.at(-1);
    const m = { l: 38, r: 14, t: 18, b: 38 }, pw = W - m.l - m.r, ph = height - m.t - m.b;
    const X = i => m.l + (points.length === 1 ? pw / 2 : (i / (points.length - 1)) * pw), Y = v => m.t + (1 - (v - y0) / (y1 - y0)) * ph;
    const el = svg('svg', { viewBox: `0 0 ${W} ${height}`, width: W, height, 'aria-hidden': 'true' });
    for (const t of ticks) el.append(svg('line', { class: 'grid', x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }), svg('text', { class: 'tick', x: m.l - 5, y: Y(t) + 3.5, 'text-anchor': 'end' }, `${Math.round(100 * t)}%`));
    for (const s of seriesIds) {
      const idx = points.map((p, i) => [p, i]).filter(([p]) => p.series === s);
      el.append(svg('path', { class: 'line', style: `stroke:${slotVar(slotOf(s))}`, d: idx.map(([p, i], k) => `${k ? 'L' : 'M'}${X(i).toFixed(1)} ${Y(p.y).toFixed(1)}`).join('') }));
    }
    points.forEach((p, i) => {
      const c = slotVar(slotOf(p.series));
      if (num(p.lo) && num(p.hi)) el.append(svg('line', { class: 'whisker', style: `stroke:${c}`, x1: X(i), x2: X(i), y1: Y(p.lo), y2: Y(p.hi) }));
      el.append(svg('circle', { class: 'dot', style: `fill:${c};stroke:var(--surface)`, cx: X(i), cy: Y(p.y), r: 5 }));
      el.append(svg('text', { class: 'endlabel', x: X(i), y: Y(p.hi ?? p.y) - 5, 'text-anchor': 'middle' }, `${(100 * p.y).toFixed(1)}`));
      const words = String(p.label).split(' ');
      el.append(svg('text', { class: 'tick', x: X(i), y: height - 22, 'text-anchor': 'middle' }, words[0]), svg('text', { class: 'tick', x: X(i), y: height - 10, 'text-anchor': 'middle' }, words.slice(1).join(' ')));
    });
    holder.append(el);
  };
  responsive(holder, draw);
  return root;
}
