// natlang crisp standard library v0.1 (SPEC 9.4). Pure helpers only.
const count = (xs, f) => (f ? xs.filter(f).length : xs.length);
const sum = (xs, f) => xs.reduce((a, x) => a + (f ? f(x) : x), 0);
const min = (xs, f) => xs.reduce((a, x) => { const v = f ? f(x) : x; return a === undefined || v < a ? v : a; }, undefined);
const max = (xs, f) => xs.reduce((a, x) => { const v = f ? f(x) : x; return a === undefined || v > a ? v : a; }, undefined);
const mean = (xs, f) => (xs.length ? sum(xs, f) / xs.length : undefined);
const _key = (f) => (typeof f === "function" ? f : (x) => x[f]);
const sortBy = (xs, f, desc) => { const k = f ? _key(f) : (x) => x; const out = [...xs].sort((a, b) => (k(a) < k(b) ? -1 : k(a) > k(b) ? 1 : 0)); return desc ? out.reverse() : out; };
const groupBy = (xs, f) => { const k = _key(f), out = Object.create(null); for (const x of xs) (out[k(x)] ||= []).push(x); return out; };
const countBy = (xs, f) => { const k = f ? _key(f) : (x) => x, out = Object.create(null); for (const x of xs) out[k(x)] = (out[k(x)] || 0) + 1; return out; };
const uniq = (xs) => [...new Set(xs)];
const uniqBy = (xs, f) => { const k = _key(f), seen = new Set(), out = []; for (const x of xs) { const v = k(x); if (!seen.has(v)) { seen.add(v); out.push(x); } } return out; };
const zip = (...ls) => ls[0].map((_, i) => ls.map((l) => l[i]));
const range = (a, b) => { const out = []; for (let i = a; i <= b; i++) out.push(i); return out; };
const chunk = (xs, n) => { const out = []; for (let i = 0; i < xs.length; i += n) out.push(xs.slice(i, i + n)); return out; };
const windows = (xs, n, step = 1) => { const out = []; for (let i = 0; i + n <= xs.length; i += step) out.push(xs.slice(i, i + n)); return out; };
const flatten = (xs) => xs.flat();
const take = (xs, n) => xs.slice(0, n);
const drop = (xs, n) => xs.slice(n);
const partition = (xs, f) => [xs.filter(f), xs.filter((x) => !f(x))];
const topK = (xs, k, f) => sortBy(xs, f, true).slice(0, k);
const indexBy = (xs, f) => { const k = _key(f), out = Object.create(null); for (const x of xs) out[k(x)] = x; return out; };
const splitLines = (s) => s.split(/\r?\n/).filter((l) => l.trim() !== "");
const splitOn = (s, sep) => s.split(sep);
const joinWith = (xs, sep) => xs.join(sep);
const trim = (s) => s.trim();
const lower = (s) => s.toLowerCase();
const upper = (s) => s.toUpperCase();
const contains = (s, sub) => s.includes(sub);
const startsWith = (s, p) => s.startsWith(p);
const matchAll = (s, re) => [...s.matchAll(new RegExp(re, "g"))].map((m) => m[0]);
const replaceAll = (s, a, b) => s.split(a).join(b);
const wordCount = (s) => (s.trim() === "" ? 0 : s.trim().split(/\s+/).length);
const parseNum = (s) => { const v = Number(String(s).replace(/[, ]/g, "")); return Number.isFinite(v) ? v : null; };
const parseDate = (s) => { const d = new Date(s); return isNaN(d) ? null : d.toISOString().slice(0, 10); };
const formatDate = (s) => new Date(s).toISOString().slice(0, 10);
const daysBetween = (a, b) => Math.round((new Date(b) - new Date(a)) / 86400000);
const hash = (x) => { const s = typeof x === "string" ? x : JSON.stringify(x); let h = 5381; for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) >>> 0; return h.toString(16); };
const enumOf = (xs) => xs.map((x) => (typeof x === "number" ? String(x) : JSON.stringify(String(x)))).join(" | ");

// Constructors for pending nodes returned from crisp lambdas (SPEC 9.3).
const lambda = (o) => ({ $lambda: o });
const map = (o) => ({ $map: o });
const fold = (o) => ({ $fold: o });
const iterate = (o) => ({ $iterate: o });

const __deepFreeze = (o) => { if (o && typeof o === "object") { Object.values(o).forEach(__deepFreeze); Object.freeze(o); } return o; };
globalThis.fx = new Proxy({}, {
  get: (_, cap) => new Proxy({}, {
    get: (_, fn) => (...args) => {
      const r = JSON.parse(__fx(String(cap), String(fn), JSON.stringify(args)));
      if (r && r.__error) throw new Error("NATLANG:" + r.__error);
      return r === null ? undefined : r.value;
    },
  }),
});
const __runExpr = (code) => { const v = (0, eval)('"use strict";' + code); return JSON.stringify(v === undefined ? null : v); };
const __runBody = (code) => { const v = new Function("self", "args", "fx", '"use strict";' + code)(globalThis.self, globalThis.self.args, globalThis.fx); return JSON.stringify(v === undefined ? null : v); };
