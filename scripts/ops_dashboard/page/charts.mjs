// Small DOM and SVG chart helpers for the ops dashboard. Page-owned code: plain SVG, no dependencies.
// Series colours are CSS variables (--s1 ... --s8, themed in style.css); text always wears text tokens, never the series colour.
import { num, fmt, ema } from './model.mjs';

export function h(tag, attrs, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
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
