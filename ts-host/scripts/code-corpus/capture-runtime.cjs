const fs = require('node:fs');

function snapshot(value) {
  const seen = new WeakMap();
  const reasons = [];
  const reject = (reason, path) => { reasons.push({ path, reason }); return null; };
  const visit = (item, path) => {
    if (item === undefined) return reject('undefined', path);
    if (typeof item === 'number' && !Number.isFinite(item)) return reject(String(item), path);
    if (typeof item === 'function' || typeof item === 'symbol' || typeof item === 'bigint') return reject(typeof item, path);
    if (item === null || typeof item !== 'object') return item;
    if (seen.has(item)) return reject('cycle-or-alias', path);
    seen.set(item, path);
    if (item instanceof Date) return reject('Date', path);
    if (item instanceof Map) return reject('Map', path);
    if (item instanceof Set) return reject('Set', path);
    if (!Array.isArray(item) && Object.getPrototypeOf(item) !== Object.prototype && Object.getPrototypeOf(item) !== null) return reject('non-plain prototype', path);
    const descriptors = Object.getOwnPropertyDescriptors(item);
    if (Array.isArray(item)) {
      const out = [];
      for (let i = 0; i < item.length; i++) {
        const d = descriptors[i];
        if (!d) { reasons.push({ path: `${path}[${i}]`, reason: 'sparse array slot' }); out.push(null); }
        else if (!Object.hasOwn(d, 'value')) out.push(reject('accessor property', `${path}[${i}]`));
        else out.push(visit(d.value, `${path}[${i}]`));
      }
      if (Object.keys(descriptors).some((k) => k !== 'length' && !/^\d+$/.test(k))) reasons.push({ path, reason: 'array extra properties' });
      return out;
    }
    const out = {};
    for (const key of Object.keys(descriptors).sort()) {
      const descriptor = descriptors[key];
      if (!descriptor.enumerable) continue;
      if (!Object.hasOwn(descriptor, 'value')) { out[key] = reject('accessor property', `${path}.${key}`); continue; }
      out[key] = visit(descriptor.value, `${path}.${key}`);
    }
    return out;
  };
  try { return { value: visit(value, '$'), portable: reasons.length === 0, reasons }; }
  catch (error) { return { value: null, portable: false, reasons: [{ path: '$', reason: `snapshot failed: ${String(error?.name ?? 'Error')}` }] }; }
}

const records = [];
const active = new Map();
const runtime = {
  records,
  enter(id, args) {
    const captured = snapshot(args);
    const item = { ...id, args: captured.value, outcome: 'pending', portable: captured.portable, reasons: captured.reasons.map((r) => ({ ...r, path: `args${r.path.slice(1)}` })) };
    const stack = active.get(id.key) ?? [];
    stack.push(item);
    active.set(id.key, stack);
    return item;
  },
  returned(id, value) {
    const item = active.get(id.key)?.at(-1);
    if (item) {
      const captured = snapshot(value);
      item.expected = captured.value;
      item.portable &&= captured.portable;
      item.reasons.push(...captured.reasons.map((r) => ({ ...r, path: `expected${r.path.slice(1)}` })));
      item.outcome = 'return';
    }
  },
  thrown(id, error) {
    const item = active.get(id.key)?.at(-1);
    if (!item) return;
    try { item.thrown = { name: String(error?.name ?? 'Error'), message: String(error?.message ?? error) }; }
    catch { item.thrown = { name: 'Error', message: 'unreadable thrown value' }; }
    item.outcome = 'throw';
  },
  finish(id, args) {
    const stack = active.get(id.key);
    const item = stack?.pop();
    if (!item) return;
    const captured = snapshot(args);
    item.input_after = captured.value;
    item.portable &&= captured.portable;
    item.reasons.push(...captured.reasons.map((r) => ({ ...r, path: `input_after${r.path.slice(1)}` })));
    records.push(item);
    if (!stack.length) active.delete(id.key);
    const dest = process.env.CODE_CORPUS_CAPTURE;
    if (dest) { try { fs.appendFileSync(dest, `${JSON.stringify(item)}\n`); } catch { /* Capture I/O must not alter source behavior. */ } }
  },
};
globalThis.__codeCorpusCapture = runtime;
module.exports = { ...runtime, snapshot };
