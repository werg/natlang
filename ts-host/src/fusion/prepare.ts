/** Wiring fusion into a launched runtime and into `natlang check` (plans/FUSED_PIPELINES.md). Node only. */
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fusionFacts, type FusionFacts } from './facts.js';
import { observedFromStore } from './observed.js';
import { machineStoreRoot, openCallStore } from '../calls/store.js';
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
  const facts = fusionFacts(root, nodeSourceFiles(root), { observed: observedSource(root, settings) });
  const options = { observedMinRuns: settings.observed?.minRuns };
  const plan = settings.planner === 'crisp' || !runtime ? crispPlan(facts, options) : await planFusionWith(runtime, facts, settings.planner, undefined, options);
  return { facts, plan, report: fusionReport(facts, plan, settings) };
}

/** Recorded runs for observed readers: only when the settings ask for them and the store exists. */
function observedSource(root: string, settings: FusionSettings): ReturnType<typeof observedFromStore> | undefined {
  if (!settings.observed) return undefined;
  const storeRoot = settings.observed.store ? resolve(root, settings.observed.store) : machineStoreRoot();
  if (!storeRoot || !existsSync(join(storeRoot, 'calls.sqlite'))) return undefined;
  return observedFromStore(openCallStore(storeRoot), settings.observed.minRuns);
}

/** The runtime option for a launched program, and nothing when fusion is off. */
export async function prepareFusion(runtime: NatlangRuntime, root: string, settings: FusionSettings): Promise<FusionOptions | undefined> {
  if (settings.mode === 'off') return undefined;
  const { facts, plan } = await analyzeFusion(root, settings, runtime);
  return { mode: settings.mode, edges: fusedEdges(facts, plan), certificate: readFusionCertificate(root, settings), weights: settings.weights };
}
