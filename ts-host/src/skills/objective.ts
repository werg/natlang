/** Host-only, independently computed quality for small optimization skill episodes. */
export const OBJECTIVE_KINDS = ['knapsack', 'bin-packing', 'weighted-tardiness', 'graph-coloring', 'tsp'] as const;
export type ObjectiveKind = typeof OBJECTIVE_KINDS[number];
/** Kinds whose objective is maximized; all others are minimized. */
const MAXIMIZE = new Set<ObjectiveKind>(['knapsack']);
/** TSPLIB EUC_2D convention: integer-rounded Euclidean edge lengths keep tour sums exact. */
const distance = (a: any, b: any) => Math.round(Math.hypot(a.x - b.x, a.y - b.y));
export type ObjectiveBound = { kind: 'objective-bound'; worst: number; best: number };
export type SkillObjectiveScore = { quality: number; gates: Record<string, boolean>; objective?: number };

const invalid = (reason: string): SkillObjectiveScore => ({ quality: 0, gates: { feasible: false, [reason]: false } });
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n);
function checkBounds(expected: unknown): expected is ObjectiveBound {
  const x = expected as Partial<ObjectiveBound> | null;
  return !!x && x.kind === 'objective-bound' && finite(x.worst) && finite(x.best);
}
function instanceValue(instance: unknown): any {
  if (typeof instance !== 'string') return instance;
  try { return JSON.parse(instance); } catch { throw Error('instance is not valid JSON'); }
}
function normalize(kind: ObjectiveKind, value: number, bounds: ObjectiveBound): number {
  const raw = bounds.best === bounds.worst ? (value === bounds.best ? 1 : NaN)
    : MAXIMIZE.has(kind) ? (value - bounds.worst) / (bounds.best - bounds.worst)
      : (bounds.worst - value) / (bounds.worst - bounds.best);
  return Math.max(0, Math.min(1, raw));
}
function score(kind: ObjectiveKind, value: number, expected: unknown): SkillObjectiveScore {
  if (!checkBounds(expected)) return invalid('valid_bounds');
  const [low, high] = [Math.min(expected.best, expected.worst), Math.max(expected.best, expected.worst)];
  if (value < low || value > high) return invalid('within_reference_bounds');
  const quality = normalize(kind, value, expected);
  if (!finite(quality)) return invalid('finite_quality');
  return { quality, gates: { feasible: true, within_reference_bounds: true }, objective: value };
}

/** Compute small exact objective bounds. Instances are deliberately bounded to keep host work predictable. */
export function exactObjectiveBounds(kind: ObjectiveKind, instance: unknown): ObjectiveBound {
  const x = instanceValue(instance);
  if (kind === 'knapsack') {
    if (!finite(x?.capacity) || x.capacity < 0 || !Array.isArray(x.items) || x.items.length > 20) throw Error('invalid or oversized knapsack instance');
    if (x.items.some((i: any) => !i || typeof i.id !== 'string' || !finite(i.weight) || i.weight < 0 || !finite(i.value) || i.value < 0) || new Set(x.items.map((i: any) => i.id)).size !== x.items.length)
      throw Error('invalid knapsack item');
    let best = 0;
    for (let mask = 0; mask < 2 ** x.items.length; mask++) {
      let weight = 0, value = 0;
      for (let i = 0; i < x.items.length; i++) if (mask & (2 ** i)) { const item = x.items[i]; weight += item.weight; value += item.value; }
      if (weight <= x.capacity) best = Math.max(best, value);
    }
    const worst = 0;
    return { kind: 'objective-bound', worst, best };
  }
  if (kind === 'bin-packing') {
    if (!finite(x?.capacity) || x.capacity <= 0 || !Array.isArray(x.items) || x.items.length > 14) throw Error('invalid or oversized bin-packing instance');
    if (x.items.some((i: any) => !i || typeof i.id !== 'string') || new Set(x.items.map((i: any) => i.id)).size !== x.items.length)
      throw Error('invalid bin-packing item identity');
    const sizes: number[] = x.items.map((i: any) => i.size).sort((a: number, b: number) => b - a);
    if (sizes.some((n: number) => !finite(n) || n <= 0 || n > x.capacity)) throw Error('item cannot fit in any bin');
    if (!sizes.length) return { kind: 'objective-bound', worst: 0, best: 0 };
    let best = sizes.length;
    const bins: number[] = [];
    const place = (i: number) => {
      if (i === sizes.length) { best = Math.min(best, bins.length); return; }
      if (bins.length >= best) return;
      const seen = new Set<number>();
      for (let b = 0; b < bins.length; b++) {
        const load = bins[b]!, size = sizes[i]!;
        if (load + size <= x.capacity && !seen.has(load)) {
          seen.add(load); bins[b] = load + size; place(i + 1); bins[b] = load;
        }
      }
      bins.push(sizes[i]!); place(i + 1); bins.pop();
    };
    place(0);
    return { kind: 'objective-bound', worst: sizes.length, best };
  }
  if (kind === 'weighted-tardiness') {
    if (!Array.isArray(x?.jobs) || x.jobs.length < 1 || x.jobs.length > 16) throw Error('invalid or oversized scheduling instance');
    const jobs = x.jobs;
    if (jobs.some((j: any) => !j || typeof j.id !== 'string' || !finite(j.processing) || j.processing < 0 || !finite(j.due) || !finite(j.weight) || j.weight < 0) || new Set(jobs.map((j: any) => j.id)).size !== jobs.length)
      throw Error('invalid scheduling job');
    const states = 2 ** jobs.length, min = new Float64Array(states).fill(Infinity), max = new Float64Array(states).fill(-Infinity);
    min[0] = 0; max[0] = 0;
    for (let mask = 0; mask < states; mask++) {
      let totalTime = 0; for (let i = 0; i < jobs.length; i++) if (mask & (2 ** i)) totalTime += jobs[i].processing;
      for (let i = 0; i < jobs.length; i++) if (!(mask & (2 ** i))) {
        const next = mask + 2 ** i, job = jobs[i], add = job.weight * Math.max(0, totalTime + job.processing - job.due);
        min[next] = Math.min(min[next]!, min[mask]! + add); max[next] = Math.max(max[next]!, max[mask]! + add);
      }
    }
    return { kind: 'objective-bound', worst: max[states - 1]!, best: min[states - 1]! };
  }
  if (kind === 'graph-coloring') {
    if (!Array.isArray(x?.nodes) || x.nodes.length < 1 || x.nodes.length > 12 || !Array.isArray(x.edges)) throw Error('invalid or oversized coloring instance');
    const ids: string[] = x.nodes;
    if (ids.some(id => typeof id !== 'string') || new Set(ids).size !== ids.length) throw Error('invalid coloring node');
    const index = new Map(ids.map((id, i) => [id, i]));
    const adjacent = ids.map(() => new Set<number>());
    for (const edge of x.edges) {
      if (!Array.isArray(edge) || edge.length !== 2 || !index.has(edge[0]) || !index.has(edge[1]) || edge[0] === edge[1]) throw Error('invalid coloring edge');
      adjacent[index.get(edge[0])!]!.add(index.get(edge[1])!); adjacent[index.get(edge[1])!]!.add(index.get(edge[0])!);
    }
    // Chromatic number by backtracking over increasing color counts.
    const colors = new Array<number>(ids.length).fill(-1);
    const fits = (k: number, i = 0): boolean => {
      if (i === ids.length) return true;
      for (let c = 0; c < k; c++) if (![...adjacent[i]!].some(j => colors[j] === c)) {
        colors[i] = c; if (fits(k, i + 1)) return true; colors[i] = -1;
      }
      return false;
    };
    let best = 1; while (!fits(best)) { colors.fill(-1); best++; }
    return { kind: 'objective-bound', worst: ids.length, best };
  }
  if (kind === 'tsp') {
    if (!Array.isArray(x?.cities) || x.cities.length < 2 || x.cities.length > 10) throw Error('invalid or oversized tsp instance');
    const cities = x.cities;
    if (cities.some((c: any) => !c || typeof c.id !== 'string' || !finite(c.x) || !finite(c.y)) || new Set(cities.map((c: any) => c.id)).size !== cities.length)
      throw Error('invalid tsp city');
    // Held-Karp for both the shortest and the longest closed tour starting at city 0.
    const n = cities.length, full = 2 ** n, at = (mask: number, j: number) => mask * n + j;
    const lo = new Float64Array(full * n).fill(Infinity), hi = new Float64Array(full * n).fill(-Infinity);
    lo[at(1, 0)] = 0; hi[at(1, 0)] = 0;
    for (let mask = 1; mask < full; mask += 2) for (let j = 0; j < n; j++) {
      if (!(mask & (2 ** j)) || !Number.isFinite(lo[at(mask, j)]!)) continue;
      for (let k = 1; k < n; k++) if (!(mask & (2 ** k))) {
        const next = at(mask + 2 ** k, k), d = distance(cities[j], cities[k]);
        lo[next] = Math.min(lo[next]!, lo[at(mask, j)]! + d); hi[next] = Math.max(hi[next]!, hi[at(mask, j)]! + d);
      }
    }
    let best = Infinity, worst = -Infinity;
    for (let j = 1; j < n; j++) {
      const d = distance(cities[j], cities[0]);
      best = Math.min(best, lo[at(full - 1, j)]! + d); worst = Math.max(worst, hi[at(full - 1, j)]! + d);
    }
    return { kind: 'objective-bound', worst, best };
  }
  throw Error(`unknown objective kind ${kind}`);
}

/** Recompute feasibility and objective from the public instance and returned solution only. */
export function scoreSkillObjective(kind: ObjectiveKind, instance: unknown, value: unknown, expected: unknown): SkillObjectiveScore {
  let x: any;
  try { x = instanceValue(instance); } catch { return invalid('instance_json'); }
  let solution: any = value;
  if (typeof solution === 'string') { try { solution = JSON.parse(solution); } catch { return invalid('solution_json'); } }
  if (!x || !solution || typeof solution !== 'object') return invalid('solution_shape');
  if (!checkBounds(expected)) return invalid('valid_bounds');
  let exact: ObjectiveBound;
  try { exact = exactObjectiveBounds(kind, x); } catch { return invalid('reference_instance'); }
  if (expected.best !== exact.best || expected.worst !== exact.worst) return invalid('reference_bounds_match');
  let objective: number;
  if (kind === 'knapsack') {
    if (!finite(x.capacity) || !Array.isArray(x.items) || !Array.isArray(solution.selectedIds)) return invalid('solution_shape');
    const ids = x.items.map((i: any) => i.id), selected = solution.selectedIds;
    if (new Set(ids).size !== ids.length || selected.some((id: unknown) => !ids.includes(id)) || new Set(selected).size !== selected.length) return invalid('item_identity');
    const items = selected.map((id: unknown) => x.items.find((i: any) => i.id === id));
    const weight = items.reduce((sum: number, i: any) => sum + i.weight, 0);
    if (items.some((i: any) => !finite(i.weight) || !finite(i.value) || i.weight < 0) || weight > x.capacity) return invalid('capacity');
    objective = items.reduce((sum: number, i: any) => sum + i.value, 0);
  } else if (kind === 'bin-packing') {
    if (!finite(x.capacity) || !Array.isArray(x.items) || !Array.isArray(solution.bins)) return invalid('solution_shape');
    const ids = x.items.map((i: any) => i.id), placed = solution.bins.flatMap((b: any) => b?.itemIds ?? []);
    if (new Set(ids).size !== ids.length || placed.length !== ids.length || new Set(placed).size !== placed.length || placed.some((id: unknown) => !ids.includes(id))) return invalid('item_identity');
    for (const b of solution.bins) {
      if (!Array.isArray(b?.itemIds) || !b.itemIds.length) return invalid('nonempty_bins');
      const load = b.itemIds.reduce((sum: number, id: unknown) => sum + x.items.find((i: any) => i.id === id).size, 0);
      if (load > x.capacity) return invalid('bin_capacity');
    }
    objective = solution.bins.length;
  } else if (kind === 'weighted-tardiness') {
    if (!Array.isArray(x.jobs) || !Array.isArray(solution.order)) return invalid('solution_shape');
    const ids = x.jobs.map((j: any) => j.id), order = solution.order;
    if (new Set(ids).size !== ids.length || order.length !== ids.length || new Set(order).size !== order.length || order.some((id: unknown) => !ids.includes(id))) return invalid('job_identity');
    let time = 0; objective = 0;
    for (const id of order) { const job = x.jobs.find((j: any) => j.id === id); if (![job.processing, job.due, job.weight].every(finite) || job.processing < 0 || job.weight < 0) return invalid('job_fields'); time += job.processing; objective += job.weight * Math.max(0, time - job.due); }
  } else if (kind === 'graph-coloring') {
    if (!Array.isArray(x.nodes) || !Array.isArray(x.edges) || !solution.colors || typeof solution.colors !== 'object' || Array.isArray(solution.colors)) return invalid('solution_shape');
    const assigned = solution.colors as Record<string, unknown>;
    if (Object.keys(assigned).length !== x.nodes.length || x.nodes.some((id: string) => !Number.isSafeInteger(assigned[id]) || (assigned[id] as number) < 0)) return invalid('node_identity');
    if (x.edges.some((edge: string[]) => assigned[edge[0]!] === assigned[edge[1]!])) return invalid('proper_coloring');
    objective = new Set(Object.values(assigned)).size;
  } else if (kind === 'tsp') {
    if (!Array.isArray(x.cities) || !Array.isArray(solution.tour)) return invalid('solution_shape');
    const ids = x.cities.map((c: any) => c.id), tour = solution.tour;
    if (tour.length !== ids.length || new Set(tour).size !== tour.length || tour.some((id: unknown) => !ids.includes(id))) return invalid('city_identity');
    const city = (id: unknown) => x.cities.find((c: any) => c.id === id);
    objective = 0;
    for (let i = 0; i < tour.length; i++) objective += distance(city(tour[i]), city(tour[(i + 1) % tour.length]));
  } else return invalid('objective_kind');
  return score(kind, objective, expected);
}
