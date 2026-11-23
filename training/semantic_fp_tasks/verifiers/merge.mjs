// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): the outcome of a three-way merge of records (std
// semanticMerge or a program's own). Where both edits can stand, the merged record keeps both: each judged field equals
// one acceptable value or contains a piece of each side's edit and none of the excluded text. Where they cannot, the
// field is listed as unresolved (and holds what the program keeps there, judged like any field).
// oracle: { shape: 'by_item' (value { <id>: { value: <record>, unresolved: [<path>] } }) | 'single' ({ value, unresolved }),
//           items: { <id>: { fields: { <dotted path>: Rule }, unresolved: [<dotted path>, ...],
//                            unresolved_optional?: [<dotted path>, ...] } } }      ('single': one item, any id)
//   Rule: { equals: [<acceptable value>, ...] }
//       | { includes: [[<alternative text>, ...], ...], excludes?: [<text>, ...] }   a string field containing at least
//         one alternative of every group (case and spacing ignored) and none of excludes
//       | { any: [Rule, ...] }   one of several readings (e.g. the conflict combined, or kept unresolved at base; list
//         the path in unresolved_optional too)
// A path in unresolved is a dotted string or a list of keys. The listed paths must be exactly the required ones plus
// any of the optional ones. Strings compare after trimming, collapsing spaces and lowercasing; a numeric string equals
// its number. score = right checks / (fields + 1 for the unresolved list, per item + stray items).
const norm = value => {
  if (typeof value === 'string') {
    const trimmed = value.trim();
    return trimmed !== '' && Number.isFinite(Number(trimmed)) ? Number(trimmed) : trimmed.replace(/\s+/g, ' ').toLowerCase();
  }
  return value === undefined ? null : value;
};
function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canon(value[key])]));
  return norm(value);
}
const same = (a, b) => JSON.stringify(canon(a)) === JSON.stringify(canon(b));
const flat = text => String(text).replace(/\s+/g, ' ').toLowerCase();
const at = (record, path) => path.split('.').reduce((node, key) => node && typeof node === 'object' ? node[key] : undefined, record);
const pathOf = path => Array.isArray(path) ? path.map(String).join('.') : typeof path === 'string' ? path.trim() : null;

function ruleHolds(have, rule) {
  if (rule.any) return rule.any.some(option => ruleHolds(have, option));
  if (rule.equals) return rule.equals.some(option => same(have, option));
  if (typeof have !== 'string') return false;
  const text = flat(have);
  return (rule.includes ?? []).every(group => group.some(piece => text.includes(flat(piece)))) &&
    !(rule.excludes ?? []).some(piece => text.includes(flat(piece)));
}

export function verifyCandidate({ value, oracle }) {
  const issues = [];
  let given;
  const entries = Object.entries(oracle.items);
  if (oracle.shape === 'single') {
    if (entries.length !== 1) throw new Error('merge verifier: shape single needs exactly one item');
    given = new Map([[entries[0][0], value]]);
  } else if (oracle.shape === 'by_item') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, issues: ['not a record of merges'], score: 0 };
    given = new Map(Object.entries(value));
  } else throw new Error('merge verifier: unknown shape ' + oracle.shape);
  let right = 0, total = 0;
  for (const [id, expected] of entries) {
    const merge = given.get(id);
    const fields = Object.entries(expected.fields ?? {});
    total += fields.length + 1;
    if (!merge || typeof merge !== 'object' || !merge.value || typeof merge.value !== 'object') { issues.push(`${id}: no merged record`); continue; }
    for (const [path, rule] of fields) {
      const have = at(merge.value, path);
      if (ruleHolds(have, rule)) right += 1; else issues.push(`${id}.${path}: ${JSON.stringify(have)?.slice(0, 120)}`);
    }
    const listed = Array.isArray(merge.unresolved) ? merge.unresolved.map(pathOf) : null;
    const required = new Set(expected.unresolved ?? []), optional = new Set(expected.unresolved_optional ?? []);
    if (!listed || listed.includes(null)) { issues.push(`${id}: unresolved is not a list of paths`); continue; }
    const missing = [...required].filter(path => !listed.includes(path));
    const extra = listed.filter(path => !required.has(path) && !optional.has(path));
    if (!missing.length && !extra.length && new Set(listed).size === listed.length) right += 1;
    else issues.push(`${id}: unresolved ${JSON.stringify(listed)}${missing.length ? `, missing ${missing.join(', ')}` : ''}${extra.length ? `, not a conflict ${extra.join(', ')}` : ''}`);
  }
  let strays = 0;
  for (const id of given.keys()) if (!oracle.items[id]) { strays += 1; issues.push(`not an item: ${id}`); }
  const score = total + strays ? right / (total + strays) : 1;
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
