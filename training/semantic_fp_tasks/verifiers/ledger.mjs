// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): a list of operations, executed on a copy of the
// starting records, judged by the state it leaves. The requests' meaning fixes the goal state, not the operations:
// any sequence that reaches it passes (several usually do).
// oracle: { initial: { <id>: { <field>: value, ... } },
//           goal: the complete final state, { <id>: { <field>: value, ... } | null (removed) } for every record,
//           fields: [<field>, ...] (the fields an update may change), creatable?: boolean }
// value: [{ op: 'update', id, set: { <field>: value } } | { op: 'remove', id } | { op: 'add', id, record: {...} }, ...]
// Strings compare after trimming, collapsing spaces and lowercasing; a numeric string equals its number. An operation
// on an unknown id, an unknown field, or an add without creatable fails as a check of its own.
// score = (records equal to their goal) / (records the goal or the operations change + failed operations).
const norm = value => typeof value === 'string' ? (value.trim() !== '' && Number.isFinite(Number(value.trim()))
  ? Number(value.trim()) : value.trim().replace(/\s+/g, ' ').toLowerCase()) : value;
const same = (a, b) => JSON.stringify(canon(a)) === JSON.stringify(canon(b));
function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') return Object.fromEntries(Object.keys(value).sort().map(key => [key, canon(value[key])]));
  return norm(value);
}

export function verifyCandidate({ value, oracle }) {
  if (!Array.isArray(value)) return { ok: false, issues: ['not a list of operations'], score: 0 };
  const state = new Map(Object.entries(structuredClone(oracle.initial)).map(([id, record]) => [norm(String(id)), record]));
  const fields = new Set(oracle.fields);
  const issues = [];
  let bad = 0;
  for (const [index, entry] of value.entries()) {
    const id = norm(String(entry?.id ?? ''));
    const fail = text => { bad += 1; issues.push(`operation ${index}: ${text}`); };
    if (entry?.op === 'update') {
      if (!state.has(id) || state.get(id) === null) { fail(`no record ${id}`); continue; }
      if (!entry.set || typeof entry.set !== 'object' || Array.isArray(entry.set)) { fail('update needs set'); continue; }
      const unknown = Object.keys(entry.set).filter(field => !fields.has(field));
      if (unknown.length) { fail(`fields ${unknown.join(', ')} cannot change`); continue; }
      state.set(id, { ...state.get(id), ...entry.set });
    } else if (entry?.op === 'remove') {
      if (!state.has(id) || state.get(id) === null) { fail(`no record ${id}`); continue; }
      state.set(id, null);
    } else if (entry?.op === 'add') {
      if (!oracle.creatable) { fail('records cannot be added'); continue; }
      if (state.has(id) && state.get(id) !== null) { fail(`record ${id} exists`); continue; }
      if (!entry.record || typeof entry.record !== 'object') { fail('add needs record'); continue; }
      state.set(id, entry.record);
    } else fail(`unknown op ${JSON.stringify(entry?.op)}`);
  }
  const goal = new Map(Object.entries(oracle.goal).map(([id, record]) => [norm(String(id)), record]));
  const start = new Map(Object.entries(oracle.initial).map(([id, record]) => [norm(String(id)), record]));
  const equal = (a, b) => (a ?? null) === null ? (b ?? null) === null : (b ?? null) !== null && same(a, b);
  // Only records the goal changes or the operations changed count: a record that already matched and was left alone
  // earns nothing (as task_rollouts.effect_fraction counts file effects), so doing nothing scores 0.
  let right = 0, counted = 0;
  for (const id of new Set([...goal.keys(), ...state.keys()])) {
    const expected = goal.has(id) ? goal.get(id) : null;
    const have = state.has(id) ? state.get(id) : null;
    const before = start.has(id) ? start.get(id) : null;
    if (equal(expected, before) && equal(have, before)) continue;
    counted += 1;
    if (equal(have, expected)) right += 1;
    else issues.push(`${id}: ${JSON.stringify(have)} should be ${JSON.stringify(expected)}`);
  }
  const score = counted + bad ? right / (counted + bad) : 1;
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(score, 0.999) : 1 };
}
