/**
 * Fusion plans and their crisp parts (plans/FUSED_PIPELINES.md): the plan type, the crisp planner (the default), and
 * the verifier that checks any plan, including a natural-language planner's, against the facts.
 *
 * The verifier is the safety property: a fused edge has exactly one reader, its consumer, and both sides share one model
 * and one dialect. A plan the verifier rejects is not used for the rejected edges.
 */
import type { EdgeFacts, FusionFacts } from './facts.js';
import type { FusionEdgeSpec } from '../runtime/fusion.js';

export const FUSION_PLAN_SCHEMA = 'natlang.fusion-plan/1';

/** Supporting recorded runs a plan needs before it accepts a reader set that was observed rather than proven. */
export const DEFAULT_OBSERVED_MIN_RUNS = 20;

export type PlanOptions = {
  /** Supporting runs needed for an edge whose readers are observed (default `DEFAULT_OBSERVED_MIN_RUNS`). */
  observedMinRuns?: number;
};

export type EdgeDecision = { edge: string; decision: 'fuse' | 'keep-text'; reason: string };
/** The observed evidence behind an edge of a plan: how many runs, the minimum the verifier required, and the store revision. */
export type PlanEvidence = { edge: string; readers: 'observed'; runs: number; contradicted: number; minRuns: number; revision: string };
export type FusionPlan = { schema: typeof FUSION_PLAN_SCHEMA; planner: 'crisp' | 'nl' | 'crisp-fallback'; edges: EdgeDecision[];
  /** Edges whose reader set was observed in recorded runs; every other edge's readers were proven from the source. */
  evidence?: PlanEvidence[] };

/** The plan with the observed evidence of the facts' edges recorded in it. */
export function withEvidence(facts: FusionFacts, plan: FusionPlan, options: PlanOptions = {}): FusionPlan {
  const evidence: PlanEvidence[] = facts.edges.filter(edge => edge.observed).map(edge => ({ edge: edge.id, readers: 'observed' as const,
    runs: edge.observed!.runs, contradicted: edge.observed!.contradicted, minRuns: options.observedMinRuns ?? DEFAULT_OBSERVED_MIN_RUNS,
    revision: edge.observed!.revision }));
  const { evidence: _drop, ...rest } = plan;
  return evidence.length ? { ...rest, evidence } : rest;
}
export type PlanProblem = { edge: string; problem: string };

const normalize = (type: string): string => type.replace(/\s+/g, ' ').trim();

/** Why an edge cannot be fused, by the exact conditions; `undefined` when every condition holds. */
export function unfusableBecause(edge: EdgeFacts, options: PlanOptions = {}): string | undefined {
  if (edge.alreadySoft) return 'the producer or the consumer already uses a Neuralese type';
  const outside = edge.readers.filter(reader => reader.kind !== 'consumer');
  if (outside.length) {
    const first = outside[0]!;
    return `${first.kind === 'other-call' ? 'another call reads' : first.kind === 'crisp-code' ? 'crisp code reads' :
      first.kind === 'service' ? 'a service reads' : first.kind === 'host-return' ? 'the host reads' :
      first.kind === 'trace-ui' ? 'a log or display shows' : first.certain ? 'the orchestrating model reads' : 'an unknown reader may read'} the value` +
      ` (${first.detail})${outside.length > 1 ? ` and ${outside.length - 1} more reader${outside.length > 2 ? 's' : ''}` : ''}`;
  }
  const consumers = edge.readers.filter(reader => reader.kind === 'consumer');
  if (consumers.length !== 1) return `the value has ${consumers.length} consumers, not one`;
  if (!consumers[0]!.certain) return 'the consumer is only implied by the prose';
  if (edge.observed) {
    const minimum = options.observedMinRuns ?? DEFAULT_OBSERVED_MIN_RUNS;
    if (edge.observed.runs < minimum)
      return `the readers are only observed (${edge.observed.runs} recorded run${edge.observed.runs === 1 ? '' : 's'}, at least ${minimum} needed)`;
  }
  if (edge.producer.model !== edge.consumer.model)
    return `the two sides run on different models (${edge.producer.model ?? 'default'} and ${edge.consumer.model ?? 'default'})`;
  if (!edge.dialect) return 'no shared dialect';
  if (!edge.consumer.param) return 'the consumer parameter that receives the value is unknown';
  if (edge.consumer.paramType && normalize(edge.consumer.paramType) !== normalize(edge.type))
    return `the consumer takes ${edge.consumer.paramType}, not the producer's ${edge.type}`;
  return undefined;
}

/** The crisp planner: the exact conditions, plus the one judgment it can make by rule (a finite result stays text). */
export function crispPlan(facts: FusionFacts, options: PlanOptions = {}): FusionPlan {
  return withEvidence(facts, { schema: FUSION_PLAN_SCHEMA, planner: 'crisp', edges: facts.edges.map(edge => {
    const because = unfusableBecause(edge, options);
    if (because) return { edge: edge.id, decision: 'keep-text' as const, reason: because };
    if (edge.finite) return { edge: edge.id, decision: 'keep-text' as const,
      reason: `the value is ${edge.type}: its text is a token or two, nothing to save` };
    return { edge: edge.id, decision: 'fuse' as const, reason: edge.observed ?
      `one consumer reads it in ${edge.observed.runs} recorded runs and no run shows another reader; both sides share the model and dialect` :
      'one consumer reads it, both sides share the model and dialect' };
  }) }, options);
}

/** Check a plan against the facts. Every problem names the edge and what is wrong. */
export function verifyPlan(facts: FusionFacts, plan: unknown, options: PlanOptions = {}): { ok: boolean; problems: PlanProblem[]; plan?: FusionPlan } {
  const problems: PlanProblem[] = [];
  const known = new Map(facts.edges.map(edge => [edge.id, edge]));
  const decided = new Map<string, EdgeDecision>();
  const entries = plan && typeof plan === 'object' && Array.isArray((plan as FusionPlan).edges) ? (plan as FusionPlan).edges : undefined;
  if (!entries) return { ok: false, problems: [{ edge: '*', problem: 'the plan has no edges list' }] };
  for (const entry of entries) {
    if (!entry || typeof entry.edge !== 'string' || (entry.decision !== 'fuse' && entry.decision !== 'keep-text') || typeof entry.reason !== 'string') {
      problems.push({ edge: String((entry as { edge?: unknown })?.edge ?? '*'), problem: 'an entry needs edge, decision ("fuse" or "keep-text") and reason' });
      continue;
    }
    const edge = known.get(entry.edge);
    if (!edge) { problems.push({ edge: entry.edge, problem: 'no such edge in the facts' }); continue; }
    if (decided.has(entry.edge)) { problems.push({ edge: entry.edge, problem: 'decided twice' }); continue; }
    decided.set(entry.edge, entry);
    if (entry.decision === 'fuse') {
      const because = unfusableBecause(edge, options);
      if (because) problems.push({ edge: entry.edge, problem: `fused, but ${because}` });
    }
  }
  for (const id of known.keys()) if (!decided.has(id) && !problems.some(item => item.edge === id))
    problems.push({ edge: id, problem: 'the plan does not decide this edge' });
  return { ok: !problems.length, problems, ...(!problems.length ? { plan: withEvidence(facts, plan as FusionPlan, options) } : {}) };
}

/**
 * The edges the runtime can engage: planned `fuse` edges whose orchestrator is a natural-language function. A fused
 * hand-off inside crisp TypeScript is planned and reported, but the runtime only sees calls made by natural-language
 * parents, so it keeps those as text.
 */
export function fusedEdges(facts: FusionFacts, plan: FusionPlan): FusionEdgeSpec[] {
  const fuse = new Set(plan.edges.filter(entry => entry.decision === 'fuse').map(entry => entry.edge));
  return facts.edges.filter(edge => fuse.has(edge.id) && edge.scopeKind === 'nl' && edge.consumer.param).map(edge => ({
    id: edge.id, scope: edge.scope, producer: { source: edge.producer.source },
    consumer: { source: edge.consumer.source, param: edge.consumer.param! }, type: edge.type }));
}

/** The plan with every problem edge set to text, and the reasons recorded. */
export function repairedPlan(facts: FusionFacts, plan: FusionPlan | undefined, problems: readonly PlanProblem[], planner: FusionPlan['planner'],
    options: PlanOptions = {}): FusionPlan {
  const bad = new Map(problems.map(item => [item.edge, item.problem]));
  const given = new Map((plan?.edges ?? []).filter(entry => entry && typeof entry.edge === 'string').map(entry => [entry.edge, entry]));
  return withEvidence(facts, { schema: FUSION_PLAN_SCHEMA, planner, edges: facts.edges.map(edge => {
    const problem = bad.get(edge.id), entry = given.get(edge.id);
    if (problem || !entry || (entry.decision !== 'fuse' && entry.decision !== 'keep-text'))
      return { edge: edge.id, decision: 'keep-text' as const, reason: `kept as text: ${problem ?? 'the planner did not decide this edge'}` };
    return entry;
  }) }, options);
}
