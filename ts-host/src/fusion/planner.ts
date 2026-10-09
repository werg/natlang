/**
 * Selecting and running the fusion planner (plans/FUSED_PIPELINES.md). `fusion.planner` is `crisp` (default: the exact
 * rules in plan.ts), `nl` (the natlang program applications/fusion-planner decides, a crisp verifier checks) or
 * `shadow` (both run, crisp is served, disagreements are traced). Selection is `pluggable()`.
 */
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { FusionFacts, EdgeFacts } from './facts.js';
import { FUSION_PLAN_SCHEMA, crispPlan, repairedPlan, verifyPlan, type EdgeDecision, type FusionPlan, type PlanProblem } from './plan.js';
import { pluggable, type PluggableMode } from '../runtime/pluggable.js';
import { loadNatlang } from '../runtime/node-files.js';
import type { NatlangRuntime } from '../runtime/runtime.js';

export type FusionPlannerMode = PluggableMode;

/** One run of the natural-language planner over the facts' edges; `problems` are the verifier's objections to an earlier answer. */
export type NlPlanner = (edges: EdgeFacts[], problems: PlanProblem[]) => Promise<EdgeDecision[]>;

/** The planner program in this checkout (`NATLANG_FUSION_PLANNER` overrides). */
export function fusionPlannerDirectory(): string | undefined {
  const candidates = [process.env.NATLANG_FUSION_PLANNER, fileURLToPath(new URL('../../../applications/fusion-planner', import.meta.url))];
  return candidates.find(item => item && existsSync(join(item, 'planner.nl')));
}

/** The natural-language planner as a function; it must be called inside `runtime.run`. */
export function nlPlannerFrom(directory = fusionPlannerDirectory()): NlPlanner {
  if (!directory) throw new Error('the fusion planner program was not found; set NATLANG_FUSION_PLANNER to its directory');
  const planner = loadNatlang(join(directory, 'planner.nl'), directory) as unknown as
    (edges: EdgeFacts[], problems?: PlanProblem[]) => Promise<EdgeDecision[]>;
  return (edges, problems) => planner(edges, problems.length ? problems : undefined);
}

/** Natural-language plan, verified; one retry with the verifier's problems; problem edges fall back to text. */
export async function nlPlan(facts: FusionFacts, nl: NlPlanner): Promise<FusionPlan> {
  if (!facts.edges.length) return { schema: FUSION_PLAN_SCHEMA, planner: 'nl', edges: [] };
  const attempt = async (problems: PlanProblem[]): Promise<FusionPlan | Error> => {
    try { return { schema: FUSION_PLAN_SCHEMA, planner: 'nl', edges: await nl(facts.edges, problems) }; }
    catch (error) { return error instanceof Error ? error : new Error(String(error)); }
  };
  const first = await attempt([]);
  if (first instanceof Error) {
    const problems = facts.edges.map(edge => ({ edge: edge.id, problem: `the planner failed: ${first.message.split('\n')[0]}` }));
    return repairedPlan(facts, undefined, problems, 'crisp-fallback');
  }
  const checked = verifyPlan(facts, first);
  if (checked.ok) return first;
  const second = await attempt(checked.problems);
  if (second instanceof Error) return repairedPlan(facts, first, checked.problems, 'nl');
  const again = verifyPlan(facts, second);
  return again.ok ? second : repairedPlan(facts, second, again.problems, 'nl');
}

const sameDecisions = (a: FusionPlan, b: FusionPlan): boolean =>
  a.edges.length === b.edges.length && a.edges.every((entry, index) => entry.edge === b.edges[index]!.edge && entry.decision === b.edges[index]!.decision);

/**
 * The plan of the selected planner. `nl` is needed for `nl` and `shadow` (build it with `nlPlannerFrom` and call this
 * inside `runtime.run`); `crisp` needs nothing.
 */
export async function planFusion(facts: FusionFacts, options: { planner?: FusionPlannerMode; nl?: NlPlanner } = {}): Promise<FusionPlan> {
  const mode = options.planner ?? 'crisp';
  if (mode !== 'crisp' && !options.nl) throw new Error(`fusion.planner "${mode}" needs the natural-language planner and a model`);
  return pluggable<[], FusionPlan>({ crisp: () => crispPlan(facts), nl: () => nlPlan(facts, options.nl!) }, mode,
    { name: 'fusion-plan', serve: 'crisp', same: sameDecisions })();
}

/** Run `planFusion` with a runtime for the natural-language planner. */
export async function planFusionWith(runtime: NatlangRuntime, facts: FusionFacts, planner: FusionPlannerMode, directory?: string): Promise<FusionPlan> {
  if (planner === 'crisp') return crispPlan(facts);
  return runtime.run(() => planFusion(facts, { planner, nl: nlPlannerFrom(directory) }));
}
