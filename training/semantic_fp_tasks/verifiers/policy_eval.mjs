// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts) for higher-order policy synthesis (`compileRule`,
// `.nl` returning a typed judgment function, or rule-evaluation outputs over hidden cases).
//
// Accepts EITHER:
// 1. A map of hidden-case results `{ [caseId]: verdict }` produced by running the synthesized rule inside the task, OR
// 2. An array of `{ id, result }` entries, OR
// 3. A `{ results: { [caseId]: verdict }, metadata?: ... }` object.
// Enforces both exact accuracy on hidden cases AND metamorphic consistency across paraphrase pairs (`twins: [[idA, idB]]`).
//
// oracle: {
//   cases: { <id>: { expected: unknown, weight?: number, critical?: boolean } },
//   twins?: [[string, string]],   // pairs of case IDs that must receive the SAME verdict (anti-cue metamorphic check)
//   min_accuracy?: number,        // default 1.0; every case still needs a verdict
// }
function canon(value) {
  if (typeof value === 'string') return value.trim().toLowerCase();
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map(k => [k, canon(value[k])]));
  }
  return value;
}
const same = (a, b) => JSON.stringify(canon(a)) === JSON.stringify(canon(b));

function resultsMap(value) {
  if (!value || typeof value !== 'object') return null;
  if (Array.isArray(value)) {
    const map = new Map();
    for (const item of value) {
      if (!item || typeof item !== 'object' || typeof item.id !== 'string') return null;
      if (map.has(item.id)) return null;
      map.set(item.id, 'result' in item ? item.result : item.value);
    }
    return map;
  }
  const source = value.results && typeof value.results === 'object' && !Array.isArray(value.results) ? value.results : value;
  return new Map(Object.entries(source));
}

export function verifyCandidate({ value, oracle }) {
  const issues = [];
  const given = resultsMap(value);
  if (!given) return { ok: false, issues: ['expected a map or array of policy verdicts by case id'], score: 0 };
  const oracleCases = oracle.cases ?? {};
  const cases = Object.entries(oracleCases);
  if (!cases.length) throw new Error('policy_eval verifier: oracle.cases must be nonempty');

  let weightPassed = 0, weightTotal = 0;
  let malformed = false;
  for (const [id, spec] of cases) {
    const w = spec.weight ?? 1;
    weightTotal += w;
    if (!given.has(id)) { malformed = true; issues.push(`${id}: missing verdict`); continue; }
    const got = given.get(id);
    if (same(got, spec.expected)) weightPassed += w;
    else issues.push(`${id}: expected ${JSON.stringify(spec.expected)}, got ${JSON.stringify(got)}${spec.critical ? ' (critical)' : ''}`);
  }
  for (const [a, b] of oracle.twins ?? []) {
    if (!given.has(a) || !given.has(b) || !same(given.get(a), given.get(b))) {
      issues.push(`metamorphic twin mismatch: ${a} (${JSON.stringify(given.get(a))}) != ${b} (${JSON.stringify(given.get(b))})`);
    }
  }
  for (const id of given.keys()) {
    if (!Object.hasOwn(oracleCases, id)) {
      malformed = true;
      issues.push(`unexpected case id: ${id}`);
    }
  }

  const accuracy = weightTotal ? weightPassed / weightTotal : 0;
  const minAcc = oracle.min_accuracy ?? 1.0;
  const criticalFailed = cases.some(([id, spec]) => spec.critical && (!given.has(id) || !same(given.get(id), spec.expected)));
  const twinFailed = (oracle.twins ?? []).some(([a, b]) => !given.has(a) || !given.has(b) || !same(given.get(a), given.get(b)));
  // Every case needs a verdict: skipping the unsure ones would otherwise pass min_accuracy on the rest.
  const ok = !criticalFailed && !twinFailed && accuracy >= minAcc && !malformed;
  return { ok, issues: ok ? [] : issues, score: ok ? 1 : Math.min(accuracy * (twinFailed ? 0.5 : 1), 0.999) };
}
