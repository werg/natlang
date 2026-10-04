/** Bounded, host-only finite-CSP solver and partial-assignment reward. Model scores are never trusted. */
import { hexDigest } from '../native/hash.js';

export type CspValue = string | number | boolean;
export type CspVariable = { id: string; domain: CspValue[]; label?: string };
export type CspConstraint =
  | { kind: 'allDifferent'; vars: string[] }
  | { kind: 'sum'; vars: string[]; op: 'eq' | 'le' | 'ge'; value: number }
  | { kind: 'order'; before: string; after: string; op?: 'lt' | 'le' }
  | { kind: 'allowedTuples'; vars: string[]; tuples: CspValue[][] }
  | { kind: 'equal'; variable: string; value: CspValue }
  | { kind: 'lineRuns'; vars: string[]; runs: number[] };
export type FiniteCsp = { schema: 'natlang.finite-csp/1'; template: string; narrative: string;
  variables: CspVariable[]; constraints: CspConstraint[] };
export type CspProgressBound = { schema: 'natlang.skill-csp/1'; kind: 'csp-progress-bound';
  instance_sha256: string; variable_count: number; unique_solution: true };
export type CspSolutionReport = { solutions: Record<string, CspValue>[]; status: 'unique' | 'multiple' | 'satisfiable' | 'unsat' | 'limit'; nodes: number };
export type CspScore = { quality: number; gates: Record<string, boolean>; assigned: number; total: number;
  extendable: boolean; complete: boolean; solver_nodes?: number };

export const CSP_SOLVER_DEFAULTS = Object.freeze({ maxVariables: 28, maxDomain: 8, maxConstraints: 96,
  maxTuplesPerConstraint: 512, maxNodes: 250_000 });

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => [k, canonical(v)]));
  return value;
}
export function finiteCspSha256(instance: unknown): string {
  return hexDigest(JSON.stringify(canonical(instance)));
}
function fail(message: string): never { throw new Error(`invalid finite CSP: ${message}`); }
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}
function valueKey(value: CspValue): string { return JSON.stringify(value); }
function number(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value); }
function parseInstance(input: unknown): FiniteCsp {
  let raw = input;
  if (typeof raw === 'string') { try { raw = JSON.parse(raw); } catch { fail('instance is not valid JSON'); } }
  if (!isRecord(raw) || raw.schema !== 'natlang.finite-csp/1' || typeof raw.template !== 'string' ||
      typeof raw.narrative !== 'string' || !Array.isArray(raw.variables) || !Array.isArray(raw.constraints)) fail('shape or schema');
  const csp = raw as unknown as FiniteCsp;
  if (csp.variables.length < 1 || csp.variables.length > CSP_SOLVER_DEFAULTS.maxVariables) fail('variable count cap');
  if (csp.constraints.length > CSP_SOLVER_DEFAULTS.maxConstraints) fail('constraint count cap');
  const vars = new Map<string, CspVariable>();
  for (const variable of csp.variables) {
    if (!isRecord(variable) || typeof variable.id !== 'string' || !/^[A-Za-z][A-Za-z0-9_.-]{0,63}$/.test(variable.id) ||
        vars.has(variable.id) || !Array.isArray(variable.domain) || variable.domain.length < 1 ||
        variable.domain.length > CSP_SOLVER_DEFAULTS.maxDomain) fail('variable identity/domain');
    const domain = variable.domain as unknown[];
    if (domain.some(v => !(typeof v === 'string' || typeof v === 'boolean' || number(v))) ||
        new Set(domain.map(v => valueKey(v as CspValue))).size !== domain.length) fail(`invalid domain for ${variable.id}`);
    vars.set(variable.id, variable as CspVariable);
  }
  const needsVars = (ids: unknown, min = 1) => {
    if (!Array.isArray(ids) || ids.length < min || ids.some(id => typeof id !== 'string' || !vars.has(id)) ||
        new Set(ids).size !== ids.length) fail('constraint variable references');
  };
  for (const c of csp.constraints) {
    if (!isRecord(c) || typeof c.kind !== 'string') fail('constraint shape');
    if (c.kind === 'allDifferent') needsVars(c.vars, 2);
    else if (c.kind === 'sum') {
      needsVars(c.vars); if (!['eq', 'le', 'ge'].includes(c.op as string) || !number(c.value) ||
          (c.vars as string[]).some(id => vars.get(id)!.domain.some(v => !number(v)))) fail('sum constraint');
    } else if (c.kind === 'order') {
      needsVars([c.before, c.after], 2); if (!['lt', 'le', undefined].includes(c.op as string | undefined) ||
          [c.before, c.after].some(id => vars.get(id as string)!.domain.some(v => !number(v)))) fail('order constraint');
    } else if (c.kind === 'allowedTuples') {
      needsVars(c.vars, 2); if (!Array.isArray(c.tuples) || c.tuples.length < 1 || c.tuples.length > CSP_SOLVER_DEFAULTS.maxTuplesPerConstraint ||
          c.tuples.some(tuple => !Array.isArray(tuple) || tuple.length !== (c.vars as string[]).length || tuple.some((v, i) =>
            !(vars.get((c.vars as string[])[i]!)!.domain.some(d => valueKey(d) === valueKey(v as CspValue)))))) fail('allowed tuple table');
    } else if (c.kind === 'equal') {
      needsVars([c.variable]); if (!vars.get(c.variable as string)!.domain.some(v => valueKey(v) === valueKey(c.value as CspValue))) fail('equality clue');
    } else if (c.kind === 'lineRuns') {
      needsVars(c.vars, 1);
      if (!Array.isArray(c.runs) || c.runs.some(n => !Number.isSafeInteger(n) || (n as number) <= 0) ||
          (c.vars as string[]).some(id => vars.get(id)!.domain.some(v => v !== 0 && v !== 1))) fail('line-run clue');
      const total = (c.runs as number[]).reduce((a, b) => a + b, 0) + (c.runs as number[]).length - 1;
      if (total > (c.vars as string[]).length) fail('line runs cannot fit');
    } else fail('unsupported constraint kind');
  }
  return csp;
}

function tupleKey(tuple: readonly CspValue[]): string { return JSON.stringify(tuple); }
function lineCanMatchPartial(ids: readonly string[], runs: readonly number[], known: Record<string, CspValue>): boolean {
  const failed = new Set<string>();
  const visit = (index: number, runIndex: number, active: number): boolean => {
    const key = `${index}/${runIndex}/${active}`;
    if (failed.has(key)) return false;
    if (index === ids.length) {
      const completedRunIndex = active > 0 && runs[runIndex] === active ? runIndex + 1 : active === 0 ? runIndex : -1;
      return completedRunIndex === runs.length;
    }
    const value = known[ids[index]!];
    const choices = value === undefined ? [0, 1] : [value];
    for (const choice of choices) {
      if (choice === 1) {
        if (runIndex < runs.length && active < runs[runIndex]! && visit(index + 1, runIndex, active + 1)) return true;
      } else if (active === 0) {
        if (visit(index + 1, runIndex, 0)) return true;
      } else if (runs[runIndex] === active && visit(index + 1, runIndex + 1, 0)) return true;
    }
    failed.add(key);
    return false;
  };
  return visit(0, 0, 0);
}

/** Validate a proposed partial or complete assignment against domains and every currently decidable constraint. */
export function checkCspAssignment(instance: unknown, assignment: unknown): { valid: boolean; assigned: number; total: number; reason?: string } {
  let csp: FiniteCsp;
  try { csp = parseInstance(instance); } catch (error) { return { valid: false, assigned: 0, total: 0, reason: (error as Error).message }; }
  if (!isRecord(assignment)) return { valid: false, assigned: 0, total: csp.variables.length, reason: 'assignment must be an object' };
  const variables = new Map(csp.variables.map(v => [v.id, v]));
  const entries = Object.entries(assignment);
  if (entries.some(([id, value]) => !variables.has(id) || !variables.get(id)!.domain.some(d => valueKey(d) === valueKey(value as CspValue))))
    return { valid: false, assigned: entries.length, total: csp.variables.length, reason: 'unknown variable or out-of-domain value' };
  const known = assignment as Record<string, CspValue>;
  for (const c of csp.constraints) {
    const vals = (ids: string[]) => ids.map(id => known[id]);
    if (c.kind === 'allDifferent') {
      const chosen = vals(c.vars).filter(v => v !== undefined);
      if (new Set(chosen.map(valueKey)).size !== chosen.length) return { valid: false, assigned: entries.length, total: csp.variables.length, reason: 'allDifferent violated' };
    } else if (c.kind === 'sum') {
      const ids = c.vars, selected = ids.filter(id => known[id] !== undefined);
      const sum = selected.reduce((s, id) => s + (known[id] as number), 0), rest = ids.filter(id => known[id] === undefined);
      const low = sum + rest.reduce((s, id) => s + Math.min(...variables.get(id)!.domain as number[]), 0);
      const high = sum + rest.reduce((s, id) => s + Math.max(...variables.get(id)!.domain as number[]), 0);
      if (c.op === 'eq' ? c.value < low || c.value > high : c.op === 'le' ? low > c.value : high < c.value)
        return { valid: false, assigned: entries.length, total: csp.variables.length, reason: 'sum cannot be satisfied' };
    } else if (c.kind === 'order') {
      const before = known[c.before], after = known[c.after];
      const bdom = before === undefined ? variables.get(c.before)!.domain as number[] : [before as number];
      const adom = after === undefined ? variables.get(c.after)!.domain as number[] : [after as number];
      const op = c.op ?? 'lt';
      if (!bdom.some(a => adom.some(b => op === 'lt' ? a < b : a <= b))) return { valid: false, assigned: entries.length, total: csp.variables.length, reason: 'order cannot be satisfied' };
    } else if (c.kind === 'allowedTuples') {
      const ids = c.vars;
      if (!c.tuples.some(tuple => ids.every((id, i) => known[id] === undefined || valueKey(known[id]!) === valueKey(tuple[i]!))))
        return { valid: false, assigned: entries.length, total: csp.variables.length, reason: 'allowed tuple has no compatible row' };
    } else if (c.kind === 'equal') {
      if (known[c.variable] !== undefined ? valueKey(known[c.variable]!) !== valueKey(c.value) : !variables.get(c.variable)!.domain.some(v => valueKey(v) === valueKey(c.value)))
        return { valid: false, assigned: entries.length, total: csp.variables.length, reason: 'fixed clue violated' };
    } else if (c.kind === 'lineRuns') {
      if (!lineCanMatchPartial(c.vars, c.runs, known))
        return { valid: false, assigned: entries.length, total: csp.variables.length, reason: 'line clue cannot be satisfied' };
    }
  }
  return { valid: true, assigned: entries.length, total: csp.variables.length };
}

/** Exact, deterministic bounded enumeration. A `limit` result is never treated as unique. */
export function solveFiniteCsp(input: unknown, options: { maxNodes?: number; maxSolutions?: number } = {}): CspSolutionReport {
  const csp = parseInstance(input), maxNodes = options.maxNodes ?? CSP_SOLVER_DEFAULTS.maxNodes;
  const maxSolutions = options.maxSolutions ?? 2;
  if (!Number.isSafeInteger(maxNodes) || maxNodes < 1 || !Number.isSafeInteger(maxSolutions) || maxSolutions < 1 || maxSolutions > 16)
    fail('solver caps');
  const solutions: Record<string, CspValue>[] = [], assignment: Record<string, CspValue> = {};
  let nodes = 0, hitLimit = false;
  const visit = () => {
    if (solutions.length >= maxSolutions) return;
    if (++nodes > maxNodes) { hitLimit = true; return; }
    const check = checkCspAssignment(csp, assignment);
    if (!check.valid) return;
    if (Object.keys(assignment).length === csp.variables.length) { solutions.push({ ...assignment }); return; }
    const remaining = csp.variables.filter(v => assignment[v.id] === undefined);
    const degree = (v: CspVariable) => csp.constraints.filter(c => {
      const ids = c.kind === 'order' ? [c.before, c.after] : c.kind === 'equal' ? [c.variable] : c.vars;
      return ids.includes(v.id);
    }).length;
    remaining.sort((a, b) => a.domain.length - b.domain.length || degree(b) - degree(a) || a.id.localeCompare(b.id));
    const variable = remaining[0]!;
    for (const value of variable.domain) {
      assignment[variable.id] = value;
      visit();
      delete assignment[variable.id];
      if (solutions.length >= maxSolutions || hitLimit) return;
    }
  };
  visit();
  const status = hitLimit ? 'limit' : solutions.length >= 2 ? 'multiple'
    : solutions.length === 0 ? 'unsat' : maxSolutions === 1 ? 'satisfiable' : 'unique';
  return { solutions, status, nodes: Math.min(nodes, maxNodes) };
}

export function cspProgressBound(instance: unknown, options: { maxNodes?: number } = {}): CspProgressBound {
  const csp = parseInstance(instance), result = solveFiniteCsp(csp, { ...options, maxSolutions: 2 });
  if (result.status !== 'unique') throw new Error(`CSP must be uniquely solvable before scoring (status=${result.status}, nodes=${result.nodes})`);
  return { schema: 'natlang.skill-csp/1', kind: 'csp-progress-bound', instance_sha256: finiteCspSha256(csp),
    variable_count: csp.variables.length, unique_solution: true };
}

/** Reward progress only if the partial assignment remains extendable; contradictions and dead ends score zero. */
export function scoreCspProgress(instance: unknown, proposed: unknown, expected: unknown,
  options: { maxNodes?: number } = {}): CspScore {
  let csp: FiniteCsp;
  try { csp = parseInstance(instance); } catch { return { quality: 0, gates: { instance_valid: false }, assigned: 0, total: 0, extendable: false, complete: false }; }
  const bound = expected as Partial<CspProgressBound> | null;
  if (!bound || bound.schema !== 'natlang.skill-csp/1' || bound.kind !== 'csp-progress-bound' || bound.unique_solution !== true ||
      bound.instance_sha256 !== finiteCspSha256(csp) || bound.variable_count !== csp.variables.length)
    return { quality: 0, gates: { reference_binding: false }, assigned: 0, total: csp.variables.length, extendable: false, complete: false };
  let assignment = proposed;
  if(typeof assignment === 'string'){try{assignment=JSON.parse(assignment);}catch{return {quality:0,gates:{assignment_valid:false},assigned:0,total:csp.variables.length,extendable:false,complete:false};}}
  proposed=assignment;
  if (isRecord(proposed) && 'assignment' in proposed) assignment = proposed.assignment;
  const checked = checkCspAssignment(csp, assignment);
  if (!checked.valid) return { quality: 0, gates: { instance_valid: true, assignment_valid: false, consistent: false },
    assigned: checked.assigned, total: csp.variables.length, extendable: false, complete: false };
  const enumeration = solveFiniteCsp(csp, { maxNodes: options.maxNodes, maxSolutions: 2 });
  // Search for one completion while respecting the submitted choices.
  const constrained: FiniteCsp = { ...csp, constraints: [...csp.constraints,
    ...Object.entries(assignment as Record<string, CspValue>).map(([variable, value]) => ({ kind: 'equal' as const, variable, value }))] };
  const completion = solveFiniteCsp(constrained, { maxNodes: options.maxNodes, maxSolutions: 1 });
  const extendable = completion.solutions.length > 0 && completion.status !== 'limit';
  if (enumeration.status !== 'unique') return { quality: 0, gates: { instance_valid: true, reference_binding: false,
      unique_reference: false }, assigned: checked.assigned, total: csp.variables.length, extendable: false, complete: false,
    solver_nodes: enumeration.nodes + completion.nodes };
  const complete = checked.assigned === csp.variables.length;
  const quality = extendable ? checked.assigned / csp.variables.length : 0;
  return { quality, gates: { instance_valid: true, assignment_valid: true, consistent: true,
      extendable, progress: checked.assigned > 0, complete, unique_reference: enumeration.status === 'unique' },
    assigned: checked.assigned, total: csp.variables.length, extendable, complete,
    solver_nodes: enumeration.nodes + completion.nodes };
}
