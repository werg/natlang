// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): a chosen subset of a collection, judged by what each
// item's meaning requires. Items whose inclusion is defensible either way go in `optional`.
// oracle: { required: [<id>, ...], forbidden: [<id>, ...], optional?: [<id>, ...],
//           min?: number, max?: number (bounds on the number chosen), key?: string (value is a list of records and key
//           names their id field; otherwise a list of ids), ordered?: boolean (the list must keep `required`'s order) }
// Every id is one of required, forbidden or optional. Ids compare after trimming, collapsing spaces and lowercasing.
// score = (required chosen + forbidden left out) / (required + forbidden + strays and repeats).
const norm = value => typeof value === 'string' ? value.trim().replace(/\s+/g, ' ').toLowerCase()
  : typeof value === 'number' && Number.isFinite(value) ? String(value) : null;

export function verifyCandidate({ value, oracle }) {
  if (!Array.isArray(value)) return { ok: false, issues: ['not a list'], score: 0 };
  const ids = value.map(entry => norm(oracle.key ? entry?.[oracle.key] : entry));
  const required = oracle.required.map(norm), forbidden = new Set(oracle.forbidden.map(norm));
  const known = new Set([...required, ...forbidden, ...(oracle.optional ?? []).map(norm)]);
  const chosen = new Set(), issues = [];
  let strays = 0;
  for (const id of ids) {
    if (id === null || !known.has(id)) { strays += 1; issues.push(`not an item: ${id}`); }
    else if (chosen.has(id)) { strays += 1; issues.push(`chosen twice: ${id}`); }
    else chosen.add(id);
  }
  let right = 0;
  for (const id of required) if (chosen.has(id)) right += 1; else issues.push(`missing: ${id}`);
  for (const id of forbidden) if (!chosen.has(id)) right += 1; else issues.push(`should be left out: ${id}`);
  if (oracle.min !== undefined && chosen.size < oracle.min) issues.push(`chose ${chosen.size}, at least ${oracle.min} needed`);
  if (oracle.max !== undefined && chosen.size > oracle.max) issues.push(`chose ${chosen.size}, at most ${oracle.max} allowed`);
  if (oracle.ordered) {
    const order = ids.filter(id => required.includes(id));
    if (order.some((id, index) => id !== required.filter(r => order.includes(r))[index])) issues.push('required items out of order');
  }
  const total = required.length + forbidden.size + strays;
  const score = total ? right / total : 1;
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
