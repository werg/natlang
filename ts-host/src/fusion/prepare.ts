/** Wiring fusion into a launched runtime and into `natlang check` (plans/FUSED_PIPELINES.md). Node only. */
import { fusionFacts, type FusionFacts } from './facts.js';
import { crispPlan, fusedEdges, type FusionPlan } from './plan.js';
import { planFusionWith } from './planner.js';
import { fusionReport, type FusionReport } from './report.js';
import { readFusionCertificate, type FusionSettings } from './settings.js';
import { nodeSourceFiles } from '../runtime/node-files.js';
import type { NatlangRuntime } from '../runtime/runtime.js';
import type { FusionOptions } from '../runtime/fusion.js';

/** Facts, the selected planner's plan and the report for the project at `root`. `runtime` is needed for planners that use a model. */
export async function analyzeFusion(root: string, settings: FusionSettings, runtime?: NatlangRuntime):
    Promise<{ facts: FusionFacts; plan: FusionPlan; report: FusionReport }> {
  const facts = fusionFacts(root, nodeSourceFiles(root));
  const plan = settings.planner === 'crisp' || !runtime ? crispPlan(facts) : await planFusionWith(runtime, facts, settings.planner);
  return { facts, plan, report: fusionReport(facts, plan, settings) };
}

/** The runtime option for a launched program, and nothing when fusion is off. */
export async function prepareFusion(runtime: NatlangRuntime, root: string, settings: FusionSettings): Promise<FusionOptions | undefined> {
  if (settings.mode === 'off') return undefined;
  const { facts, plan } = await analyzeFusion(root, settings, runtime);
  return { mode: settings.mode, edges: fusedEdges(facts, plan), certificate: readFusionCertificate(root, settings), weights: settings.weights };
}
