/** Small exact host scorers for additional optimization episode families. */
import type { ObjectiveBound, SkillObjectiveScore } from './objective.js';

export const EXTENDED_OBJECTIVE_KINDS = ['weighted-set-cover', 'budgeted-max-coverage', 'facility-location',
  'bottleneck-assignment', 'matrix-chain'] as const;
export type ExtendedObjectiveKind = typeof EXTENDED_OBJECTIVE_KINDS[number];

const MAXIMIZE = new Set<ExtendedObjectiveKind>(['budgeted-max-coverage']);
const invalid = (reason: string): SkillObjectiveScore => ({ quality: 0, gates: { feasible: false, [reason]: false } });
const safe = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;
const object = (value: unknown): value is Record<string, any> => !!value && typeof value === 'object' && !Array.isArray(value);

function jsonValue(value: unknown): any {
  if (typeof value !== 'string') return value;
  return JSON.parse(value);
}
function uniqueIds(values: unknown[], label: string): asserts values is string[] {
  if (values.some(value => typeof value !== 'string' || !value) || new Set(values).size !== values.length)
    throw Error(`invalid ${label} identity`);
}
function safeSum(...values: number[]): number {
  const total = values.reduce((sum, value) => sum + value, 0);
  if (!Number.isSafeInteger(total)) throw Error('objective exceeds safe integer range');
  return total;
}
function bound(best: number, worst: number): ObjectiveBound {
  return { kind: 'objective-bound', best, worst };
}
function checkBound(value: unknown): value is ObjectiveBound {
  return object(value) && value.kind === 'objective-bound' && Number.isSafeInteger(value.best) && Number.isSafeInteger(value.worst);
}

type SetCover = { universe: string[]; sets: { id: string; cost: number; elements: string[] }[] };
type MaxCoverage = { budget: number; universe: { id: string; weight: number }[]; items: { id: string; cost: number; elements: string[] }[] };
type Facility = { id: string; openCost: number; serviceCosts: Record<string, number> };
type FacilityInstance = { clients: string[]; facilities: Facility[] };
type Assignment = { agents: string[]; tasks: string[]; costs: number[][] };
type MatrixChain = { dimensions: number[] };

function checkSetCover(x: any): SetCover {
  if (!object(x) || !Array.isArray(x.universe) || x.universe.length < 1 || x.universe.length > 15 ||
      !Array.isArray(x.sets) || x.sets.length < 1 || x.sets.length > 18) throw Error('invalid or oversized set-cover instance');
  uniqueIds(x.universe, 'universe element');
  uniqueIds(x.sets.map((item: any) => item?.id), 'set');
  const universe = new Set(x.universe);
  for (const item of x.sets) {
    if (!safe(item.cost) || !Array.isArray(item.elements) || item.elements.some((id: unknown) => !universe.has(id as string)) ||
        new Set(item.elements).size !== item.elements.length) throw Error('invalid set-cover member');
  }
  const covered = new Set(x.sets.flatMap((item: any) => item.elements));
  if (x.universe.some((id: string) => !covered.has(id))) throw Error('set-cover universe is not coverable');
  return x as SetCover;
}
function checkMaxCoverage(x: any): MaxCoverage {
  if (!object(x) || !safe(x.budget) || !Array.isArray(x.universe) || x.universe.length < 1 || x.universe.length > 15 ||
      !Array.isArray(x.items) || x.items.length < 1 || x.items.length > 18) throw Error('invalid or oversized max-coverage instance');
  uniqueIds(x.universe.map((item: any) => item?.id), 'universe element');
  uniqueIds(x.items.map((item: any) => item?.id), 'coverage item');
  const universe = new Set(x.universe.map((item: any) => item.id));
  for (const item of x.universe) if (!safe(item.weight)) throw Error('invalid coverage weight');
  for (const item of x.items) if (!safe(item.cost) || !Array.isArray(item.elements) ||
      item.elements.some((id: unknown) => !universe.has(id as string)) || new Set(item.elements).size !== item.elements.length)
    throw Error('invalid coverage item');
  return x as MaxCoverage;
}
function checkFacility(x: any): FacilityInstance {
  if (!object(x) || !Array.isArray(x.clients) || x.clients.length < 1 || x.clients.length > 7 ||
      !Array.isArray(x.facilities) || x.facilities.length < 1 || x.facilities.length > 6)
    throw Error('invalid or oversized facility-location instance');
  uniqueIds(x.clients, 'client'); uniqueIds(x.facilities.map((item: any) => item?.id), 'facility');
  for (const facility of x.facilities) {
    if (!safe(facility.openCost) || !object(facility.serviceCosts) ||
        x.clients.some((client: string) => !safe(facility.serviceCosts[client])) ||
        Object.keys(facility.serviceCosts).some(client => !x.clients.includes(client))) throw Error('invalid facility costs');
  }
  return x as FacilityInstance;
}
function checkAssignment(x: any): Assignment {
  if (!object(x) || !Array.isArray(x.agents) || x.agents.length < 1 || x.agents.length > 8 ||
      !Array.isArray(x.tasks) || x.tasks.length !== x.agents.length || !Array.isArray(x.costs) ||
      x.costs.length !== x.agents.length) throw Error('invalid or oversized assignment instance');
  uniqueIds(x.agents, 'agent'); uniqueIds(x.tasks, 'task');
  if (x.costs.some((row: unknown) => !Array.isArray(row) || row.length !== x.tasks.length || row.some((n: unknown) => !safe(n))))
    throw Error('invalid assignment costs');
  return x as Assignment;
}
function checkMatrixChain(x: any): MatrixChain {
  if (!object(x) || !Array.isArray(x.dimensions) || x.dimensions.length < 3 || x.dimensions.length > 11 ||
      x.dimensions.some((n: unknown) => !Number.isSafeInteger(n) || (n as number) <= 0 || (n as number) > 10_000))
    throw Error('invalid or oversized matrix-chain instance');
  const n = x.dimensions.length - 1;
  const dp = Array.from({ length: n }, () => Array<number>(n).fill(0));
  for (let width = 2; width <= n; width++) for (let i = 0; i + width <= n; i++) {
    const j = i + width - 1;
    dp[i]![j] = Math.min(...Array.from({ length: width - 1 }, (_, offset) => {
      const k = i + offset;
      return safeSum(dp[i]![k]!, dp[k + 1]![j]!, x.dimensions[i]! * x.dimensions[k + 1]! * x.dimensions[j + 1]!);
    }));
  }
  if (!Number.isSafeInteger(dp[0]![n - 1]!)) throw Error('matrix-chain bound exceeds safe integer range');
  return x as MatrixChain;
}

function enumerateAssignments(facilities: Facility[], clients: string[], visit: (assignment: number[]) => void): void {
  const assignment: number[] = [];
  const walk = (client: number) => {
    if (client === clients.length) { visit(assignment); return; }
    for (let facility = 0; facility < facilities.length; facility++) {
      assignment.push(facility); walk(client + 1); assignment.pop();
    }
  };
  walk(0);
}
function facilityCost(x: FacilityInstance, assignment: number[]): number {
  const used = new Set(assignment);
  const opening = [...used].map(index => x.facilities[index]!.openCost);
  const service = assignment.map((index, clientIndex) => x.facilities[index]!.serviceCosts[x.clients[clientIndex]!]!);
  return safeSum(...opening, ...service);
}

export function exactExtendedObjectiveBounds(kind: ExtendedObjectiveKind, instance: unknown): ObjectiveBound {
  const x = jsonValue(instance);
  if (kind === 'weighted-set-cover') {
    const data = checkSetCover(x), n = data.sets.length;
    const universeIds = new Map(data.universe.map((id, index) => [id, 2 ** index]));
    const target = 2 ** data.universe.length - 1;
    let best = Infinity, worst = -Infinity;
    for (let mask = 0; mask < 2 ** n; mask++) {
      let cover = 0, cost = 0;
      for (let i = 0; i < n; i++) if (mask & (2 ** i)) {
        const set = data.sets[i]!; cost = safeSum(cost, set.cost);
        for (const id of set.elements) cover |= universeIds.get(id)!;
      }
      if (cover === target) { best = Math.min(best, cost); worst = Math.max(worst, cost); }
    }
    return bound(best, worst);
  }
  if (kind === 'budgeted-max-coverage') {
    const data = checkMaxCoverage(x), n = data.items.length;
    let best = 0;
    for (let mask = 0; mask < 2 ** n; mask++) {
      let cost = 0; const covered = new Set<string>();
      for (let i = 0; i < n; i++) if (mask & (2 ** i)) {
        const item = data.items[i]!; cost = safeSum(cost, item.cost);
        for (const id of item.elements) covered.add(id);
      }
      if (cost <= data.budget) best = Math.max(best, safeSum(...data.universe.filter(row => covered.has(row.id)).map(row => row.weight)));
    }
    return bound(best, 0);
  }
  if (kind === 'facility-location') {
    const data = checkFacility(x);
    let best = Infinity, worst = -Infinity;
    enumerateAssignments(data.facilities, data.clients, assignment => {
      const cost = facilityCost(data, assignment);
      best = Math.min(best, cost); worst = Math.max(worst, cost);
    });
    return bound(best, worst);
  }
  if (kind === 'bottleneck-assignment') {
    const data = checkAssignment(x), n = data.agents.length, used = new Array<boolean>(n).fill(false);
    let best = Infinity, worst = -Infinity;
    const walk = (agent: number, maxCost: number) => {
      if (agent === n) { best = Math.min(best, maxCost); worst = Math.max(worst, maxCost); return; }
      for (let task = 0; task < n; task++) if (!used[task]) {
        used[task] = true; walk(agent + 1, Math.max(maxCost, data.costs[agent]![task]!)); used[task] = false;
      }
    };
    walk(0, 0); return bound(best, worst);
  }
  if (kind === 'matrix-chain') {
    const data = checkMatrixChain(x), dims = data.dimensions, n = dims.length - 1;
    const lo = Array.from({ length: n }, () => Array<number>(n).fill(0));
    const hi = Array.from({ length: n }, () => Array<number>(n).fill(0));
    for (let width = 2; width <= n; width++) for (let i = 0; i + width <= n; i++) {
      const j = i + width - 1;
      const choices = Array.from({ length: width - 1 }, (_, offset) => {
        const k = i + offset, multiply = dims[i]! * dims[k + 1]! * dims[j + 1]!;
        return [safeSum(lo[i]![k]!, lo[k + 1]![j]!, multiply), safeSum(hi[i]![k]!, hi[k + 1]![j]!, multiply)];
      });
      lo[i]![j] = Math.min(...choices.map(pair => pair[0]!));
      hi[i]![j] = Math.max(...choices.map(pair => pair[1]!));
    }
    return bound(lo[0]![n - 1]!, hi[0]![n - 1]!);
  }
  throw Error(`unknown extended objective kind ${kind}`);
}

function scoreValue(kind: ExtendedObjectiveKind, objective: number, expected: unknown): SkillObjectiveScore {
  if (!checkBound(expected) || expected.best < 0 || expected.worst < 0) return invalid('valid_bounds');
  const min = Math.min(expected.best, expected.worst), max = Math.max(expected.best, expected.worst);
  if (objective < min || objective > max) return invalid('within_reference_bounds');
  const quality = expected.best === expected.worst ? 1 : MAXIMIZE.has(kind)
    ? (objective - expected.worst) / (expected.best - expected.worst)
    : (expected.worst - objective) / (expected.worst - expected.best);
  if (!Number.isFinite(quality)) return invalid('finite_quality');
  return { quality: Math.max(0, Math.min(1, quality)), gates: { feasible: true, within_reference_bounds: true }, objective };
}

export function scoreExtendedObjective(kind: ExtendedObjectiveKind, instance: unknown, value: unknown,
    expected: unknown): SkillObjectiveScore {
  try {
    const x = jsonValue(instance), solution = jsonValue(value);
    if (!object(solution) || !checkBound(expected)) return invalid('solution_shape');
    const exact = exactExtendedObjectiveBounds(kind, x);
    if (expected.best !== exact.best || expected.worst !== exact.worst) return invalid('reference_bounds_match');
    let objective: number;
    if (kind === 'weighted-set-cover') {
      const data = checkSetCover(x), ids = data.sets.map(row => row.id), selected = solution.selectedIds;
      if (!Array.isArray(selected) || selected.some((id: unknown) => typeof id !== 'string' || !ids.includes(id)) || new Set(selected).size !== selected.length)
        return invalid('set_identity');
      const covered = new Set<string>(); let cost = 0;
      for (const id of selected) { const set = data.sets.find(row => row.id === id)!; cost = safeSum(cost, set.cost); set.elements.forEach(item => covered.add(item)); }
      if (data.universe.some(id => !covered.has(id))) return invalid('complete_cover');
      objective = cost;
    } else if (kind === 'budgeted-max-coverage') {
      const data = checkMaxCoverage(x), ids = data.items.map(row => row.id), selected = solution.selectedIds;
      if (!Array.isArray(selected) || selected.some((id: unknown) => typeof id !== 'string' || !ids.includes(id)) || new Set(selected).size !== selected.length)
        return invalid('item_identity');
      const covered = new Set<string>(); let cost = 0;
      for (const id of selected) { const item = data.items.find(row => row.id === id)!; cost = safeSum(cost, item.cost); item.elements.forEach(element => covered.add(element)); }
      if (cost > data.budget) return invalid('budget');
      objective = safeSum(...data.universe.filter(row => covered.has(row.id)).map(row => row.weight));
    } else if (kind === 'facility-location') {
      const data = checkFacility(x), assigned = solution.assignments;
      if (!object(assigned) || Object.keys(assigned).length !== data.clients.length || data.clients.some(id => typeof assigned[id] !== 'string') ||
          Object.keys(assigned).some(id => !data.clients.includes(id))) return invalid('client_assignment');
      const indices = data.clients.map(client => data.facilities.findIndex(facility => facility.id === assigned[client]));
      if (indices.some(index => index < 0)) return invalid('facility_identity');
      objective = facilityCost(data, indices);
    } else if (kind === 'bottleneck-assignment') {
      const data = checkAssignment(x), assigned = solution.assignments;
      if (!object(assigned) || Object.keys(assigned).length !== data.agents.length || data.agents.some(id => typeof assigned[id] !== 'string') ||
          Object.keys(assigned).some(id => !data.agents.includes(id))) return invalid('agent_assignment');
      const taskIndices = data.agents.map(agent => data.tasks.indexOf(assigned[agent]));
      if (taskIndices.some(index => index < 0) || new Set(taskIndices).size !== data.tasks.length) return invalid('task_assignment');
      objective = Math.max(...taskIndices.map((task, agent) => data.costs[agent]![task]!));
    } else {
      const data = checkMatrixChain(x), dims = data.dimensions, n = dims.length - 1;
      let leaf = 0;
      const evalTree = (node: any): { start: number; end: number; rows: number; cols: number; cost: number } => {
        if (!object(node)) throw Error('invalid tree node');
        if (Object.keys(node).length === 1 && typeof node.matrix === 'string' && node.matrix === `A${leaf + 1}`) {
          const index = leaf++;
          return { start: index, end: index, rows: dims[index]!, cols: dims[index + 1]!, cost: 0 };
        }
        if (Object.keys(node).length !== 2 || !('left' in node) || !('right' in node)) throw Error('invalid binary split');
        const left = evalTree(node.left), right = evalTree(node.right);
        if (left.end + 1 !== right.start || left.cols !== right.rows) throw Error('noncontiguous or incompatible matrix chain');
        return { start: left.start, end: right.end, rows: left.rows, cols: right.cols,
          cost: safeSum(left.cost, right.cost, left.rows * left.cols * right.cols) };
      };
      const result = evalTree(solution.tree);
      if (leaf !== n || result.start !== 0 || result.end !== n - 1) return invalid('matrix_leaf_coverage');
      objective = result.cost;
    }
    return scoreValue(kind, objective, expected);
  } catch {
    return invalid('invalid_instance_or_solution');
  }
}
