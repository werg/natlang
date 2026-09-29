/** Adapted from Ax b780a14a3cb94d5ac572db04038399aef655c76c. Copyright Ax contributors. Apache-2.0. Modified for native keys and deterministic randomness; see vendor/ax-gepa/CHANGES.md. */
// Shared Pareto / multi-objective helpers for GEPA optimizers

export function dominatesVector(
  a: Readonly<Record<string, number>>,
  b: Readonly<Record<string, number>>
): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let atLeastAsGood = true;
  let strictlyBetter = false;
  for (const k of keys) {
    const va = a[k] ?? 0;
    const vb = b[k] ?? 0;
    if (va < vb) {
      atLeastAsGood = false;
      break;
    }
    if (va > vb) strictlyBetter = true;
  }
  return atLeastAsGood && strictlyBetter;
}

export function dominatesVectorEps(
  a: Readonly<Record<string, number>>,
  b: Readonly<Record<string, number>>,
  eps = 0
): boolean {
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  let atLeastAsGood = true;
  let strictlyBetter = false;
  for (const k of keys) {
    const va = a[k] ?? 0;
    const vb = b[k] ?? 0;
    if (va + eps < vb) {
      atLeastAsGood = false;
      break;
    }
    if (va > vb + eps) strictlyBetter = true;
  }
  return atLeastAsGood && strictlyBetter;
}

export function buildParetoFront(
  items: ReadonlyArray<{
    idx: number;
    scores: Readonly<Record<string, number>>;
  }>,
  eps = 0
): Array<{
  idx: number;
  scores: Readonly<Record<string, number>>;
  dominated: number;
}> {
  const front: Array<{
    idx: number;
    scores: Readonly<Record<string, number>>;
    dominated: number;
  }> = [];
  for (let i = 0; i < items.length; i++) {
    let dominatedCount = 0;
    let isDominated = false;
    for (let j = 0; j < items.length; j++) {
      if (i === j) continue;
      if (dominatesVectorEps(items[j]!.scores, items[i]!.scores, eps)) {
        isDominated = true;
        break;
      }
      if (dominatesVectorEps(items[i]!.scores, items[j]!.scores, eps))
        dominatedCount++;
    }
    if (!isDominated)
      front.push({
        idx: items[i]!.idx,
        scores: items[i]!.scores,
        dominated: dominatedCount,
      });
  }
  return front;
}

export function computeCrowdingDistances(
  front: ReadonlyArray<{
    idx: number;
    scores: Readonly<Record<string, number>>;
  }>
): Map<number, number> {
  const dist = new Map<number, number>();
  if (front.length === 0) return dist;
  const keys = new Set<string>();
  for (const f of front) for (const k of Object.keys(f.scores)) keys.add(k);
  for (const f of front) dist.set(f.idx, 0);
  for (const key of keys) {
    const sorted = [...front].sort(
      (a, b) => (a.scores[key] ?? 0) - (b.scores[key] ?? 0)
    );
    const min = sorted[0] ? (sorted[0]!.scores[key] ?? 0) : 0;
    const max = sorted[sorted.length - 1]
      ? (sorted[sorted.length - 1]!.scores[key] ?? 0)
      : 0;
    const range = Math.max(max - min, 1e-9);
    if (sorted.length > 0) dist.set(sorted[0]!.idx, Number.POSITIVE_INFINITY);
    if (sorted.length > 1)
      dist.set(sorted[sorted.length - 1]!.idx, Number.POSITIVE_INFINITY);
    for (let i = 1; i < sorted.length - 1; i++) {
      const prev = sorted[i - 1]!.scores[key] ?? 0;
      const next = sorted[i + 1]!.scores[key] ?? 0;
      const inc = (next - prev) / range;
      dist.set(sorted[i]!.idx, (dist.get(sorted[i]!.idx) ?? 0) + inc);
    }
  }
  return dist;
}

export function hypervolume2D(
  front: ReadonlyArray<Readonly<Record<string, number>>>
): number | undefined {
  if (front.length === 0) return undefined;
  const keys = Object.keys(front[0] ?? {});
  if (keys.length !== 2) return undefined;
  const [k1, k2] = keys;
  const sorted = [...front].sort((a, b) => (b[k1!] ?? 0) - (a[k1!] ?? 0));
  let hv = 0;
  let prevY = 0;
  for (const p of sorted) {
    const x = p[k1!] ?? 0;
    const y = p[k2!] ?? 0;
    const dy = Math.max(y - prevY, 0);
    hv += x * dy;
    prevY = Math.max(prevY, y);
  }
  return hv;
}
