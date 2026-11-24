// Trusted verifier (ts-host/src/teacher/trusted-exact-verifier.ts) for WS5 O3 proper scoring of calibrated
// `readout: decision` distributions and escalation decisions.
//
// oracle: {
//   items: { <id>: { label: unknown, escalate_expected?: boolean } },
//   shape?: 'by_item' | 'single',
//   max_brier?: number,           // maximum allowed mean Brier score (default 0.25)
//   max_log_loss?: number,        // maximum allowed bounded mean log-loss (default 1.0)
//   min_accuracy?: number,        // minimum argmax accuracy in [0, 1] (default 0.75)
//   risk_floors?: { <label>: number }, // if P(label) >= threshold, item must be escalated (when escalated field checked)
// }
// Candidate value per item: { value, probabilities: [{ value, probability }] | Record<string, number>, escalated?: boolean }
const EPS = 1e-6;

function canon(value) {
  if (Array.isArray(value)) return value.map(canon);
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.keys(value).sort().map(k => [k, canon(value[k])]));
  }
  return value;
}

const keyOf = value => typeof value === 'string' ? value : JSON.stringify(canon(value));

function distributionOf(entry) {
  if (!entry || typeof entry !== 'object') return null;
  const map = new Map();
  if (Array.isArray(entry.probabilities)) {
    for (const item of entry.probabilities) {
      if (!item || typeof item !== 'object' || typeof item.probability !== 'number' || !Number.isFinite(item.probability)) return null;
      const k = keyOf(item.value);
      if (map.has(k)) return null;
      map.set(k, item.probability);
    }
  } else if (entry.probabilities && typeof entry.probabilities === 'object') {
    for (const [k, p] of Object.entries(entry.probabilities)) {
      if (typeof p !== 'number' || !Number.isFinite(p)) return null;
      map.set(k, p);
    }
  } else return null;
  if (!map.size) return null;
  const sum = [...map.values()].reduce((a, b) => a + b, 0);
  if (!(sum > 0.98 && sum < 1.02) || [...map.values()].some(p => p < -1e-6 || p > 1 + 1e-6)) return null;
  return map;
}

export function verifyCandidate({ value, oracle }) {
  const issues = [];
  const items = oracle.items ?? {};
  const entries = Object.entries(items);
  if (!entries.length) throw new Error('proper_score verifier: oracle.items must be nonempty');
  let given;
  if (oracle.shape === 'single') {
    if (entries.length !== 1) throw new Error('proper_score verifier: shape single needs 1 item');
    given = new Map([[entries[0][0], value]]);
  } else {
    if (!value || typeof value !== 'object' || Array.isArray(value))
      return { ok: false, issues: ['expected a record keyed by item id'], score: 0 };
    given = new Map(Object.entries(value));
  }

  let brierSum = 0, logLossSum = 0, correct = 0, valid = 0, escalationViolations = 0;
  for (const [id, spec] of entries) {
    const entry = given.get(id);
    const dist = distributionOf(entry);
    if (!dist) { issues.push(`${id}: missing valid probability distribution`); continue; }
    valid++;
    const truthKey = keyOf(spec.label);
    const pTruth = Math.min(1, Math.max(0, dist.get(truthKey) ?? 0));
    let brier = 0;
    const keys = new Set([...dist.keys(), truthKey]);
    for (const k of keys) {
      const p = dist.get(k) ?? 0;
      const y = k === truthKey ? 1 : 0;
      brier += (p - y) * (p - y);
    }
    brierSum += brier;
    logLossSum += -Math.log(Math.max(EPS, pTruth));
    const maxProb = Math.max(...dist.values());
    const valKey = keyOf(entry.value);
    const isArgmax = dist.has(valKey) && dist.get(valKey) >= maxProb - 1e-6;
    if (!isArgmax) {
      issues.push(`${id}: value ${valKey} does not match argmax of probabilities`);
    } else if (valKey === truthKey) {
      correct++;
    }
    if (oracle.risk_floors) {
      for (const [riskLabel, floor] of Object.entries(oracle.risk_floors)) {
        if ((dist.get(riskLabel) ?? 0) >= floor && entry.escalated !== true) {
          escalationViolations++;
          issues.push(`${id}: P(${riskLabel})=${(dist.get(riskLabel) ?? 0).toFixed(2)} >= ${floor} required escalation`);
        }
      }
    }
    if (spec.escalate_expected !== undefined && entry.escalated !== spec.escalate_expected) {
      escalationViolations++;
      issues.push(`${id}: expected escalated=${spec.escalate_expected}, got ${entry.escalated}`);
    }
  }
  for (const id of given.keys()) if (!Object.hasOwn(items, id)) issues.push(`unexpected item ${id}`);
  if (valid === 0) return { ok: false, issues, score: 0 };

  const meanBrier = brierSum / entries.length + (entries.length - valid) * 2 / entries.length;
  const maxItemLogLoss = -Math.log(EPS);
  const meanLogLoss = (logLossSum + (entries.length - valid) * maxItemLogLoss) / entries.length;
  const accuracy = correct / entries.length;
  const maxBrier = oracle.max_brier ?? 0.25;
  const maxLogLoss = oracle.max_log_loss ?? 1.0;
  const minAccuracy = oracle.min_accuracy ?? 0.75;
  if (meanBrier > maxBrier) issues.push(`mean Brier ${meanBrier.toFixed(3)} > ${maxBrier}`);
  if (meanLogLoss > maxLogLoss) issues.push(`mean log-loss ${meanLogLoss.toFixed(3)} > ${maxLogLoss}`);
  if (accuracy < minAccuracy) issues.push(`accuracy ${accuracy.toFixed(3)} < ${minAccuracy}`);

  const quality = Math.max(0, Math.min(1, 1 - meanBrier / 2)) * (escalationViolations ? 0.5 : 1);
  return { ok: issues.length === 0, issues, score: issues.length ? Math.min(quality, 0.999) : 1 };
}
