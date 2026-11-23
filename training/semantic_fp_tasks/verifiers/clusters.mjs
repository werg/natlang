// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): a grouping of items into clusters (duplicates, the
// same underlying issue, one conversation thread), judged up to the clusters' names. Only who is together matters.
// oracle: { clusters: [[<id>, ...], ...] (every item in exactly one cluster; singletons included),
//           loose?: [[<id>, <id>], ...] (pairs that may be together or apart: their sameness is defensible either way),
//           shape: 'lists' (value [[<id>, ...], ...]) | 'by_item' (value { <id>: <cluster name> })
//                  | 'by_cluster' (value { <cluster name>: [<id>, ...] }) }
// Ids compare after trimming, collapsing spaces and lowercasing. Judged on pairs: every pair of items is together in
// the answer exactly when it is together in the oracle (loose pairs excepted). score = right pairs / (pairs + strays).
const norm = value => typeof value === 'string' ? value.trim().replace(/\s+/g, ' ').toLowerCase()
  : typeof value === 'number' && Number.isFinite(value) ? String(value) : null;
const pairKey = (a, b) => a < b ? a + '\u0000' + b : b + '\u0000' + a;

export function verifyCandidate({ value, oracle }) {
  const issues = [];
  let groups = [];
  if (oracle.shape === 'lists') {
    if (!Array.isArray(value) || !value.every(Array.isArray)) return { ok: false, issues: ['not a list of lists'], score: 0 };
    groups = value;
  } else if (oracle.shape === 'by_cluster') {
    if (!value || typeof value !== 'object' || Array.isArray(value) || !Object.values(value).every(Array.isArray))
      return { ok: false, issues: ['not a record of id lists'], score: 0 };
    groups = Object.values(value);
  } else if (oracle.shape === 'by_item') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, issues: ['not a record of cluster names'], score: 0 };
    const byName = new Map();
    for (const [id, name] of Object.entries(value)) {
      const label = norm(name) ?? JSON.stringify(name);
      byName.set(label, [...(byName.get(label) ?? []), id]);
    }
    groups = [...byName.values()];
  } else throw new Error('clusters verifier: unknown shape ' + oracle.shape);
  const items = oracle.clusters.flat().map(norm);
  const truth = new Map();
  oracle.clusters.forEach((cluster, index) => cluster.forEach(id => truth.set(norm(id), index)));
  const at = new Map();
  let strays = 0;
  groups.forEach((group, index) => group.forEach(id => {
    const n = norm(id);
    if (n === null || !truth.has(n)) { strays += 1; issues.push(`not an item: ${n}`); }
    else if (at.has(n)) { strays += 1; issues.push(`placed twice: ${n}`); }
    else at.set(n, index);
  }));
  for (const id of items) if (!at.has(id)) issues.push(`not placed: ${id}`);
  const loose = new Set((oracle.loose ?? []).map(([a, b]) => pairKey(norm(a), norm(b))));
  let right = 0, pairs = 0;
  for (let i = 0; i < items.length; i += 1) for (let j = i + 1; j < items.length; j += 1) {
    const a = items[i], b = items[j];
    if (loose.has(pairKey(a, b))) continue;
    pairs += 1;
    const together = truth.get(a) === truth.get(b);
    const answer = at.has(a) && at.has(b) && at.get(a) === at.get(b);
    if (together === answer) right += 1;
    else issues.push(together ? `${a} and ${b} belong together` : `${a} and ${b} do not belong together`);
  }
  const score = pairs + strays ? right / (pairs + strays) : 1;
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
