---
description: Form observational equivalence branches from actual world predictions.
args:
  experiment: ExperimentPlan
  cross: CrossReading
  assay: Assay
  predictions: WorldPrediction[]
  posterior: Posterior
returns: ExperimentPartition
---
1. Inventory the positive-posterior JointWorld IDs and join exactly one actual prediction for this
   experiment to each. Check original world/item identity before interpreting answers. An absent
   prediction is missing information; an explicit zero likelihood is a supplied quantitative claim.
2. Read cross.interpretable and each prediction.usable with their reasons. If a possible world lacks
   an interpretable, usable answer distribution, preserve that blocker. Set usable=false for established
   invalidity and usable=null for unresolved meaning or missing evidence.
   Keep its actual prediction for audit rather than inventing an observable category for the failure.
3. Interpret answer identity from the actual assay.readout and assay.answerGrammar supplied here.
   Use its compartment, stage and regime to preserve the observation scope. Identify which differences the instrument actually reports. A uniformity answer can merge
   different colors, and equal observed colors can merge different genotypes or mechanisms when the
   source readout says so. A distinction unavailable to the assay cannot become an observation branch.
4. Visit the actual predictions, then their actual AnswerMass entries. Build the finite collection of
   observable answer classes from those entries. For each proposed merge, use the readout meaning and
   grammar to decide observational equivalence; preserve distinguishable categories separately.
5. For every resulting answer class, visit every posterior world. Sum only that world's supplied
   likelihoods for observations belonging to this class. Retain one worldLikelihood row per original
   world, including explicit zero. Unresolved equivalence stays a validity blocker, not a forced merge.
6. Compute the class marginal probability by exact sum of posterior weight times the world-specific
   likelihood. This arithmetic consumes the semantic grouping already decided above; it does not
   classify offspring, infer compartment identity or create a case-specific mechanism table.
7. Derive goalClasses from worlds with positive conditional mass in this branch. Keep distinct worlds
   of one mechanism when their genotypes differ. A one-mechanism branch satisfies the stated mechanism
   goal while preserving those other uncertainties for the later audit.
8. Check each usable world's answer-class likelihoods sum to one and the posterior-weighted reachable
   branches sum to one. Keep zero-probability predictions as audit evidence; executable planning uses
   only positive-probability branches. A malformed distribution is an explicit invalid partition.
9. Return the actual experiment/cross, original predictions, constructed branches, usable status and
   reason. These marginal probabilities describe this supplied posterior. At a later belief node the
   planner reweights the same world-specific likelihoods against that node's actual conditional weights.
