---
description: Rebind trial facts and predict each hypothesis through the existing semantic departments.
args:
  context: InvestigationContext
  seed: ProbeSeed
  hypotheses: Hypothesis[]
uses: [plans]
returns: ProbePrediction
---
1. Call plans(context.intake, context.baseline.lineage, seed.data, context.request). Consume the
   three reconstructed plans and exact receipts; selected definitions stay fixed while trial facts change.
2. For each possible or supported hypothesis compare its affected role's materialized choices with
   baseline. Identify changed included units, selected readings, anchor offsets or numerator proposition.
3. State the expectedChangedReadingIds from those actual plan/receipt differences. Keep absence
   and polarity distinct: removing a row and counting its opposite outcome are different predictions.
4. A hypothesis can predict no headline difference while still predicting a different row set. Record
   that distinction explicitly instead of reducing every prediction to percent equality.
5. Classify predicted relation different, same or unresolved under the specific mechanism. Preserve
   missing decisive inputs as unresolved; a null denominator does not mean0percent.
6. Return seedId, the actual PlanState, targetedHypothesisIds and predictedMechanisms with reasons
   and exact witnesses. The returned state is the experiment's materialized computation, not a guess.
7. The same function is reused during repair replay. A repaired hypothesis should no longer produce
   the formerly distinguishing semantic discrepancy under the same raw intervention.
8. Exact arithmetic is delegated to measurementLab through plans. Probe significance and mechanism
   prediction remain NL decisions over the returned data and source clauses.
