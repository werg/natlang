// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts): numbers computed from what the items mean (an amount
// owed after reading which charges stand, a count of real requests), judged against every acceptable result.
// oracle: { value: <number> | { <name>: <number>, ... },   the result; a record is judged name by name
//           accept?: [<other acceptable value of the same shape>, ...] (other defensible readings, each whole),
//           tolerance?: number (absolute; default 0.005) }
// A numeric string equals its number. score = right names / names (a plain number: 0 or 1); extra names count as wrong.
const num = value => typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' &&
  Number.isFinite(Number(value.trim().replace(/,/g, ''))) ? Number(value.trim().replace(/,/g, '')) : NaN;

function judge(value, target, tolerance) {
  if (typeof target === 'number') {
    const ok = Math.abs(num(value) - target) <= tolerance;
    return { ok, right: ok ? 1 : 0, total: 1, issues: ok ? [] : [`${JSON.stringify(value)} should be ${target}`] };
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return { ok: false, right: 0, total: Object.keys(target).length, issues: ['not a record of numbers'] };
  const issues = [];
  let right = 0;
  for (const [name, expected] of Object.entries(target)) {
    if (Math.abs(num(value[name]) - expected) <= tolerance) right += 1;
    else issues.push(`${name}: ${JSON.stringify(value[name])}`);
  }
  const extra = Object.keys(value).filter(name => !(name in target));
  if (extra.length) issues.push(`unexpected: ${extra.join(', ')}`);
  return { ok: issues.length === 0, right, total: Object.keys(target).length + extra.length, issues };
}

export function verifyCandidate({ value, oracle }) {
  const tolerance = oracle.tolerance ?? 0.005;
  const verdicts = [oracle.value, ...(oracle.accept ?? [])].map(target => judge(value, target, tolerance));
  const best = verdicts.find(verdict => verdict.ok) ??
    verdicts.reduce((a, b) => (b.right / b.total > a.right / a.total ? b : a));
  const score = best.total ? best.right / best.total : 0;
  return { ok: best.ok, issues: best.issues, score: best.ok ? 1 : Math.min(score, 0.999) };
}
