// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): an assignment of items to slots (people to sessions,
// orders to vans, guests to tables), judged by the constraints the items' notes state. Any assignment that satisfies
// them all passes; usually many do.
// oracle: { items: [<id>, ...], slots: [<slot>, ...], none?: <label for "not assigned"> (default "none"),
//           allowed: { <id>: [<slot or none label>, ...] }   (what that item's own note permits),
//           together?: [[<id>, <id>], ...], apart?: [[<id>, <id>], ...], capacity?: { <slot>: <max items> },
//           shape: 'by_item' (value { <id>: <slot> }) | 'by_slot' (value { <slot>: [<id>, ...] }) }
// Ids and slots compare after trimming, collapsing spaces and lowercasing. Items missing from a by_slot value are
// "none". score = satisfied checks / checks: one per item (assigned once, to a slot its note allows), one per pair,
// one per capacity.
const norm = value => typeof value === 'string' ? value.trim().replace(/\s+/g, ' ').toLowerCase()
  : typeof value === 'number' && Number.isFinite(value) ? String(value) : null;

export function verifyCandidate({ value, oracle }) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, issues: ['not a record'], score: 0 };
  const none = norm(oracle.none ?? 'none');
  const slots = new Set(oracle.slots.map(norm));
  const items = oracle.items.map(norm);
  const known = new Set(items);
  const at = new Map(), issues = [];
  let strays = 0;
  const place = (id, slot) => {
    if (id === null || !known.has(id)) { strays += 1; issues.push(`not an item: ${id}`); return; }
    if (at.has(id)) { strays += 1; issues.push(`assigned twice: ${id}`); return; }
    at.set(id, slot === null || slot === undefined ? none : slot);
  };
  if (oracle.shape === 'by_item') for (const [id, slot] of Object.entries(value)) place(norm(id), norm(slot));
  else if (oracle.shape === 'by_slot') {
    for (const [slot, ids] of Object.entries(value)) {
      if (!Array.isArray(ids)) { issues.push(`slot ${slot} does not hold a list`); continue; }
      for (const id of ids) place(norm(id), norm(slot));
    }
    for (const id of items) if (!at.has(id)) at.set(id, none);
  } else throw new Error('assignment verifier: unknown shape ' + oracle.shape);
  let passed = 0, checks = 0;
  const allowed = new Map(Object.entries(oracle.allowed).map(([id, list]) => [norm(id), new Set(list.map(norm))]));
  for (const id of items) {
    checks += 1;
    const slot = at.get(id);
    if (slot === undefined) issues.push(`unassigned: ${id}`);
    else if (slot !== none && !slots.has(slot)) issues.push(`${id}: ${slot} is not a slot`);
    else if (!(allowed.get(id) ?? new Set()).has(slot)) issues.push(`${id} cannot go to ${slot}`);
    else passed += 1;
  }
  for (const [kind, pairs] of [['together', oracle.together ?? []], ['apart', oracle.apart ?? []]]) for (const [a, b] of pairs) {
    checks += 1;
    const x = at.get(norm(a)), y = at.get(norm(b));
    const ok = kind === 'together' ? x !== undefined && x === y && x !== none
      : x === undefined || y === undefined || x === none || y === none || x !== y;
    if (ok) passed += 1; else issues.push(`${norm(a)} and ${norm(b)} must be ${kind}`);
  }
  for (const [slot, max] of Object.entries(oracle.capacity ?? {})) {
    checks += 1;
    const used = [...at.values()].filter(s => s === norm(slot)).length;
    if (used <= max) passed += 1; else issues.push(`${norm(slot)} holds ${used}, at most ${max}`);
  }
  const score = passed / (checks + strays);
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
