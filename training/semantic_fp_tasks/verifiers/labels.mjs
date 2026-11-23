// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): a labelling of a collection, judged item by item
// against the labels its meaning allows. The model never sees this file or the oracle data.
// oracle: { items: { <id>: [<acceptable label>, ...] }, labels: [<label>, ...],
//           shape: 'by_label' (value { <label>: [<id>, ...] }) | 'by_item' (value { <id>: <label> })
//                  | 'rows' (value [{ <id_key>: <id>, <label_key>: <label> }, ...]),
//           id_key?: string, label_key?: string }
// Labels and ids compare after trimming, collapsing spaces and lowercasing; order never matters. Every item must be
// labelled exactly once. score = items labelled acceptably / (items + ids that are not items or repeat an item).
const norm = value => typeof value === 'string' ? value.trim().replace(/\s+/g, ' ').toLowerCase()
  : typeof value === 'number' && Number.isFinite(value) ? String(value) : null;

export function verifyCandidate({ value, oracle }) {
  const issues = [];
  const items = new Map(Object.entries(oracle.items).map(([id, labels]) => [norm(id), new Set(labels.map(norm))]));
  const labels = new Set(oracle.labels.map(norm));
  const pairs = [];
  if (oracle.shape === 'by_label') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, issues: ['not a record of label lists'], score: 0 };
    for (const [label, ids] of Object.entries(value)) {
      if (!Array.isArray(ids)) { issues.push(`label ${label} does not hold a list`); continue; }
      for (const id of ids) pairs.push([norm(id), norm(label)]);
    }
  } else if (oracle.shape === 'by_item') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, issues: ['not a record of item labels'], score: 0 };
    for (const [id, label] of Object.entries(value)) pairs.push([norm(id), norm(label)]);
  } else if (oracle.shape === 'rows') {
    if (!Array.isArray(value)) return { ok: false, issues: ['not a list of rows'], score: 0 };
    for (const row of value) pairs.push(row && typeof row === 'object' ? [norm(row[oracle.id_key]), norm(row[oracle.label_key])] : [null, null]);
  } else throw new Error('labels verifier: unknown shape ' + oracle.shape);
  const seen = new Set();
  let right = 0, stray = 0;
  for (const [id, label] of pairs) {
    if (id === null || !items.has(id)) { stray += 1; issues.push(`not an item: ${id}`); continue; }
    if (seen.has(id)) { stray += 1; issues.push(`labelled twice: ${id}`); continue; }
    seen.add(id);
    if (label === null || !labels.has(label)) issues.push(`${id}: ${label} is not a label`);
    else if (items.get(id).has(label)) right += 1;
    else issues.push(`${id}: ${label}`);
  }
  for (const id of items.keys()) if (!seen.has(id)) issues.push(`unlabelled: ${id}`);
  const score = right / (items.size + stray);
  return { ok: issues.length === 0 && right === items.size, issues, score: issues.length ? Math.min(score, 0.999) : score };
}
