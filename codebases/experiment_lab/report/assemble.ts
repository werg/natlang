/*---
args:
  plan: Plan
  metrics: Metric[]
  trials: Trial[]
  analysis: Analysis
returns: Report
---*/
return { plan: args.plan, metrics: args.metrics, trials: args.trials,
         interpretation: args.analysis.interpretation,
         unknowns: args.analysis.unknowns, followups: args.analysis.followups };
