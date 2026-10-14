---
description: Compare one actual observation with a theory-conditioned prediction.
args:
  world: JointWorld
  observation: Observation
  theories: ExpressionTheory[]
  inheritanceRules: SourceClause[]
returns: ObservationAssessment
---
1. Select the world's actual expression theory and accession assignment. Call
   predictObservedTrait(world, observation, theory, inheritanceRules) and consume its WorldPrediction.
2. Compare the observation's reported answer using its source-visible answer grammar and readout
   resolution. A coarse report corresponds to the sum of the compatible predicted answer masses.
3. If the observation validly measures this quantity, assign its exact likelihood and compatibility:
   zero excludes the world; a positive value retains it. Keep the actual prediction in the assessment.
4. If scope says it measures a different quantity, treat it as uninformative for expression-world
   elimination with likelihood one, explaining that distinction using the scope's witnesses.
5. If scope or prediction is unresolved, return compatible null and likelihood null. Missing evidence
   must remain a posterior uncertainty rather than an impossible observation.
6. Return the source observation/world IDs, actual prediction, likelihood and exact supporting
   witnesses. The posterior updater owns mass arithmetic and consumes this decision.
