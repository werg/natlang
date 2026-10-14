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
3. For any valid measurement, use the actual prediction for its reported biological quantity to
   assign likelihood and compatibility. Zero excludes the joint world; a positive value retains it.
   A coat or marker observation can constrain cytoplasm or genotype even when it is not adult expression.
4. Assign neutral likelihood one only when source scope establishes that this record supplies no
   information about the represented joint worlds, such as an explicitly unusable sensitivity regime.
   Explain that neutrality with the actual scope witnesses and retain its unusable prediction.
5. If measurement scope, referent or applicable prediction is unresolved, return compatible null
   and likelihood null. Missing interpretation stays a posterior uncertainty, not an impossible world.
6. Return the source observation/world IDs, actual prediction, likelihood and exact supporting
   witnesses. The posterior updater owns mass arithmetic and consumes this decision.
