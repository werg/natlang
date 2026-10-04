/** Exact small Boolean-DNF minimization with correctness-gated cost scoring. */
import type { ObjectiveBound, SkillObjectiveScore } from './objective.js';

export const EFFICIENCY_OBJECTIVE_KINDS = ['boolean-dnf'] as const;
export type EfficiencyObjectiveKind = typeof EFFICIENCY_OBJECTIVE_KINDS[number];
const invalid = (reason: string): SkillObjectiveScore => ({ quality: 0, gates: { feasible: false, [reason]: false } });
const safeCost = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;
const record = (value: unknown): value is Record<string, any> => !!value && typeof value === 'object' && !Array.isArray(value);

type DnfInstance = { variables: string[]; truthTable: boolean[]; literalCost: number; termCost: number };
type Implicant = { pattern: string; coverage: number; cost: number };

function parse(value: unknown): any {
  if (typeof value !== 'string') return value;
  return JSON.parse(value);
}
function checkedInstance(value: unknown): DnfInstance {
  const x = parse(value);
  if (!record(x) || !Array.isArray(x.variables) || x.variables.length < 1 || x.variables.length > 4 ||
      x.variables.some((v: unknown) => typeof v !== 'string' || !v) || new Set(x.variables).size !== x.variables.length ||
      !Array.isArray(x.truthTable) || x.truthTable.length !== 2 ** x.variables.length ||
      x.truthTable.some((v: unknown) => typeof v !== 'boolean') ||
      !safeCost(x.literalCost) || !safeCost(x.termCost)) throw Error('invalid or oversized Boolean DNF instance');
  return x as DnfInstance;
}
function implicants(data: DnfInstance): Implicant[] {
  const n = data.variables.length, count = 2 ** n, result: Implicant[] = [];
  const totalPatterns = 3 ** n;
  for (let code = 0; code < totalPatterns; code++) {
    let rest = code, pattern = '';
    for (let i = 0; i < n; i++) { pattern = ['0', '1', '-'][rest % 3]! + pattern; rest = Math.floor(rest / 3); }
    let coverage = 0, valid = true;
    for (let row = 0; row < count; row++) {
      const bits = row.toString(2).padStart(n, '0');
      if ([...pattern].every((symbol, i) => symbol === '-' || symbol === bits[i])) {
        if (!data.truthTable[row]) { valid = false; break; }
        coverage |= 2 ** row;
      }
    }
    if (valid && coverage) {
      const literals = [...pattern].filter(symbol => symbol !== '-').length;
      const cost = data.termCost + literals * data.literalCost;
      if (!Number.isSafeInteger(cost)) throw Error('DNF term cost exceeds safe integer range');
      result.push({ pattern, coverage, cost });
    }
  }
  return result;
}
function safeTotal(values: number[]): number {
  const value = values.reduce((sum, item) => sum + item, 0);
  if (!Number.isSafeInteger(value)) throw Error('DNF objective exceeds safe integer range');
  return value;
}

export function exactEfficiencyObjectiveBounds(kind: EfficiencyObjectiveKind, instance: unknown): ObjectiveBound {
  if (kind !== 'boolean-dnf') throw Error(`unknown efficiency objective kind ${kind}`);
  const data = checkedInstance(instance), terms = implicants(data), rows = data.truthTable.length;
  let target = 0;
  data.truthTable.forEach((enabled, row) => { if (enabled) target |= 2 ** row; });
  if (target === 0) return { kind: 'objective-bound', best: 0, worst: 0 };
  // 0/1 set-cover DP over at most 16 truth-table rows and 81 valid implicants.
  const min = new Float64Array(2 ** rows).fill(Infinity);
  min[0] = 0;
  for (const term of terms) {
    for (let cover = min.length - 1; cover >= 0; cover--) {
      if (!Number.isFinite(min[cover]!)) continue;
      const next = cover | term.coverage, cost = min[cover]! + term.cost;
      if (!Number.isSafeInteger(cost)) throw Error('DNF objective exceeds safe integer range');
      min[next] = Math.min(min[next]!, cost);
    }
  }
  const worst = safeTotal(terms.map(term => term.cost));
  return { kind: 'objective-bound', best: min[target]!, worst };
}

export function scoreEfficiencyObjective(kind: EfficiencyObjectiveKind, instance: unknown, value: unknown,
    expected: unknown): SkillObjectiveScore {
  try {
    if (kind !== 'boolean-dnf') return invalid('objective_kind');
    const data = checkedInstance(instance), solution = parse(value);
    if (!record(solution) || !Array.isArray(solution.terms) || !record(expected) || expected.kind !== 'objective-bound' ||
        !safeCost(expected.best) || !safeCost(expected.worst)) return invalid('solution_shape');
    const exact = exactEfficiencyObjectiveBounds(kind, data);
    if (expected.best !== exact.best || expected.worst !== exact.worst) return invalid('reference_bounds_match');
    const validTerms = new Map(implicants(data).map(term => [term.pattern, term]));
    if (solution.terms.some((term: unknown) => typeof term !== 'string' || !validTerms.has(term)) ||
        new Set(solution.terms).size !== solution.terms.length) return invalid('valid_implicants');
    let coverage = 0;
    const cost: number[] = [];
    for (const pattern of solution.terms as string[]) {
      const term = validTerms.get(pattern)!;
      coverage |= term.coverage; cost.push(term.cost);
    }
    let target = 0;
    data.truthTable.forEach((enabled, row) => { if (enabled) target |= 2 ** row; });
    if (coverage !== target) return invalid('truth_table_equivalence');
    const objective = safeTotal(cost);
    const quality = expected.best === expected.worst ? 1 : (expected.worst - objective) / (expected.worst - expected.best);
    if (!Number.isFinite(quality) || objective < expected.best || objective > expected.worst) return invalid('objective_bounds');
    return { quality: Math.max(0, Math.min(1, quality)), gates: { feasible: true, correct: true, within_reference_bounds: true }, objective };
  } catch {
    return invalid('invalid_instance_or_solution');
  }
}
