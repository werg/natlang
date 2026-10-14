---
description: Form observational equivalence branches from actual world predictions.
args:
  experiment: ExperimentPlan
  cross: CrossReading
  predictions: WorldPrediction[]
  posterior: Posterior
returns: ExperimentPartition
---
1. Check that every posterior world has exactly one actual prediction for this experiment. Retain
   unresolved or unusable predictions instead of treating missing answers as zero likelihood.
2. Compare answers at the supplied readout resolution and visible serialization grammar. Merge
   equivalent observations into one branch; distinct genotypes alone do not create separate observations.
3. For each branch, retain every world's actual likelihood of that answer. Its marginal probability
   is the exact sum of posterior probability times that likelihood.
4. Record goal classes from the theories of worlds with positive branch likelihood. A branch can
   identify one mechanism while preserving several genotype assignments within that mechanism.
5. Exclude zero-probability branches from reachable planning but keep their supporting predictions
   available for audit. Check total observable probability one for every usable world.
6. Return ExperimentPartition with the actual cross, predictions, branches and validity reason.
   Current branch probabilities are for the root posterior; later belief nodes reweight them locally.
