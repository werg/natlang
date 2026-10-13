---
description: Materialize the three source-bound protocol meanings for one trial.
args:
  intake: Intake
  lineage: Lineage
  trial: TrialData
  request: ProtocolRequest
uses: [populations, timing, metrics]
returns: PlanState
---
1. For targetProtocolId, leftImplementationId and rightImplementationId, gather selected definition
   links by role from lineage. An owner without a required role remains non-executable and unresolved.
2. For each owner call populations(ownerId, trial, lineage, intake.question), timing(ownerId, trial,
   lineage, intake.question) and metrics(ownerId, lineage, intake.question). Consume distinct typed views.
3. Build an ExecutablePlan only when inclusion IDs, every included unit anchor, window bounds and
   numerator outcome are resolved. Do not coerce a missing anchor to0 or choose an outcome by label.
4. Preserve role-specific witnesses and local decisions in BoundPlan. Population, timing and metric
   semantic meanings remain inspectable alongside the exact numeric plan.
5. Call measurementLab.evaluate(trial, executablePlan) for each executable owner. The service applies
   supplied IDs, times and arithmetic only; it does not interpret eligible or successful itself.
6. Return trial, lineage, target/left/right BoundPlans and exact receipts. A missing executable plan
   yields no fabricated numerical receipt, even if another owner's percent is available.
7. Preserve selectedReadingIds, numerator and denominator. Two equal percentages with different
   selected readings or numerator propositions must remain distinguishable in subsequent comparison.
8. This department is reused for baseline, counterfactual probes and edited definitions. Reusing
   the same NL materialization prevents a repair from being tested by a simpler disconnected formula.
