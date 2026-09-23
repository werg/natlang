import type { Case, Plan, Candidate, Trial, Metric, Analysis, Report } from "../types.js";
export default function assemble(plan: Plan, metrics: Metric[], trials: Trial[], analysis: Analysis): Report {
return { plan: plan, metrics: metrics, trials: trials,
         interpretation: analysis.interpretation,
         unknowns: analysis.unknowns, followups: analysis.followups };
}
