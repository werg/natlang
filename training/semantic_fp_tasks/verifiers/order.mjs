// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): an ordering of a collection, judged by the precedence
// its items' meaning implies. Any order that keeps every precedence passes; many do.
// oracle: { items: [<id>, ...], before: [[<earlier id>, <later id>], ...], first?: <id>, last?: <id>,
//           excluded?: [<id>, ...] (items that must not appear: their notes take them out of the plan),
//           key?: string (value is a list of records and key names their id field; otherwise a list of ids) }
// Ids compare after trimming, collapsing spaces and lowercasing. score = satisfied checks / checks, where the checks
// are: every included item appears exactly once, every excluded item is absent, every precedence pair, first and last.
const norm = value => typeof value === 'string' ? value.trim().replace(/\s+/g, ' ').toLowerCase()
  : typeof value === 'number' && Number.isFinite(value) ? String(value) : null;

export function verifyCandidate({ value, oracle }) {
  if (!Array.isArray(value)) return { ok: false, issues: ['not a list'], score: 0 };
  const ids = value.map(entry => norm(oracle.key ? entry?.[oracle.key] : entry));
  const excluded = new Set((oracle.excluded ?? []).map(norm));
  const included = oracle.items.map(norm).filter(id => !excluded.has(id));
  const known = new Set([...included, ...excluded]);
  const position = new Map();
  const issues = [];
  ids.forEach((id, index) => {
    if (id === null || !known.has(id)) issues.push(`not an item: ${id}`);
    else if (position.has(id)) issues.push(`listed twice: ${id}`);
    else position.set(id, index);
  });
  let passed = 0, checks = 0;
  for (const id of included) { checks += 1; if (position.has(id)) passed += 1; else issues.push(`missing: ${id}`); }
  for (const id of excluded) { checks += 1; if (!position.has(id)) passed += 1; else issues.push(`should be left out: ${id}`); }
  for (const [a, b] of oracle.before) {
    const x = norm(a), y = norm(b);
    if (excluded.has(x) || excluded.has(y)) continue;
    checks += 1;
    if (position.has(x) && position.has(y) && position.get(x) < position.get(y)) passed += 1;
    else issues.push(`${x} must come before ${y}`);
  }
  for (const [end, at] of [['first', 0], ['last', ids.length - 1]]) if (oracle[end] !== undefined) {
    checks += 1;
    if (ids[at] === norm(oracle[end])) passed += 1; else issues.push(`${norm(oracle[end])} must be ${end}`);
  }
  const strays = ids.length - position.size;
  const score = passed / (checks + strays);
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
