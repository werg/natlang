---
description: Predict every supplied experiment across the actual surviving joint worlds.
args:
  source: BankSource
  posterior: Posterior
returns: PredictionIndex
---
1. Visit the actual experiment plans from source.stocks. Find each linked assay and retain even a
   rejected or invalid proposal so the certificate can explain the source alternative.
2. Call interpretCross(experiment, source.pedigree, assay). Consume the oriented parent/subject
   interpretation and biological compartment rather than deriving direction from a generated label.
3. For each posterior world, call deriveOffspring(world, cross, source.pedigree.inheritanceRules).
   Then call readOffspring(world, experiment, assay, cross, offspring, actualTheory).
4. Keep each actual WorldPrediction with its original world/experiment IDs and witnesses. A failure
   or missing semantic contribution remains unresolved; it cannot become an empty valid partition.
5. Call partitionAnswers(experiment, cross, predictions, posterior). Consume its observable branches,
   goal-class memberships, usable status and reason into the returned PredictionIndex.
6. Preserve all unresolved predictions. Planning receives typed predictions and partitions, rather
   than raw bank documents or a case-specific answer table.
