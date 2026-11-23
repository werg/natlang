// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): structured facts read out of each item, judged field
// by field against the values its meaning allows. Several readings may be acceptable for a field; list them all.
// oracle: { items: { <id>: { <field>: [<acceptable value>, ...] } | null },   null: the item yields no record
//           shape: 'by_item' (value { <id>: { <field>: value } }) | 'rows' (value [{ <id_key>: <id>, <field>: value }]),
//           id_key?: string, tolerance?: number (absolute, for numbers; default 0),
//           extra_fields?: 'forbid' | 'allow' (default 'allow': fields the oracle does not name are not judged) }
// Strings compare after trimming, collapsing spaces and lowercasing; a numeric string equals its number; null is a value.
// score = right fields / (oracle fields + stray or repeated records + forbidden extra fields).
const norm = value => {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed !== '' && Number.isFinite(Number(trimmed)) ? Number(trimmed) : trimmed.replace(/\s+/g, ' ').toLowerCase();
  }
  return value === undefined ? null : value;
};
const sameValue = (a, b, tolerance) => {
  const x = norm(a), y = norm(b);
  if (typeof x === 'number' && typeof y === 'number') return Math.abs(x - y) <= tolerance + 1e-9;
  if (x && typeof x === 'object' || y && typeof y === 'object') return JSON.stringify(canon(x)) === JSON.stringify(canon(y));
  return x === y;
};
function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canon(value[key])]));
  return norm(value);
}
const key = id => typeof id === 'string' ? id.trim().toLowerCase() : typeof id === 'number' ? String(id) : null;

export function verifyCandidate({ value, oracle }) {
  const tolerance = oracle.tolerance ?? 0;
  const given = new Map();
  let strays = 0;
  const issues = [];
  if (oracle.shape === 'rows') {
    if (!Array.isArray(value)) return { ok: false, issues: ['not a list of records'], score: 0 };
    for (const row of value) {
      const id = row && typeof row === 'object' ? key(row[oracle.id_key]) : null;
      if (id === null || given.has(id)) { strays += 1; issues.push(`stray or repeated record: ${JSON.stringify(row)?.slice(0, 80)}`); continue; }
      given.set(id, row);
    }
  } else if (oracle.shape === 'by_item') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, issues: ['not a record of records'], score: 0 };
    for (const [id, record] of Object.entries(value)) given.set(key(id), record);
  } else throw new Error('records verifier: unknown shape ' + oracle.shape);
  const items = new Map(Object.entries(oracle.items).map(([id, fields]) => [key(id), fields]));
  let right = 0, total = 0;
  for (const [id, fields] of items) {
    const record = given.get(id);
    if (fields === null) {
      total += 1;
      if (record === undefined || record === null) right += 1; else issues.push(`${id}: should yield no record`);
      continue;
    }
    for (const [field, accepted] of Object.entries(fields)) {
      total += 1;
      if (!record || typeof record !== 'object') { issues.push(`${id}: missing`); continue; }
      const have = record[field];
      if (accepted.some(option => sameValue(have, option, tolerance))) right += 1;
      else issues.push(`${id}.${field}: ${JSON.stringify(have)}`);
    }
    if (record && typeof record === 'object' && oracle.extra_fields === 'forbid') {
      const extra = Object.keys(record).filter(field => field !== oracle.id_key && !(field in fields));
      if (extra.length) { strays += extra.length; issues.push(`${id}: unexpected fields ${extra.join(', ')}`); }
    }
  }
  for (const id of given.keys()) if (!items.has(id)) { strays += 1; issues.push(`not an item: ${id}`); }
  const score = total + strays ? right / (total + strays) : 1;
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
