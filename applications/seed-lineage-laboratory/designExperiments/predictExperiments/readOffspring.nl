---
description: Map actual biological classes through a theory and the assay resolution.
args:
  world: JointWorld
  experiment: ExperimentPlan
  assay: Assay
  cross: CrossReading
  offspring: OffspringDistribution
  theory: ExpressionTheory
returns: WorldPrediction
---
1. Bind only the actual returned cross/classes, theory, assay and experiment. Declare a local
   ReadoutMeaning with answers, usable, reason and quote support, then directly await a scoped literal.
2. The literal first checks regime sensitivity and compartment. It then interprets the theory for
   each supplied biological class, preserving maternal tissue ownership where that controls the readout.
3. It finally maps those class outcomes into the assay's actual resolution: a uniformity reading
   can merge all-bright and all-dark trays, whereas a fraction reading keeps them distinguishable.
4. Return probabilities of observable answers under the source protocol. For a replica-complete tray,
   its exact fraction is one deterministic answer, not a lottery over individual seed phenotypes.
5. Consume the actual literal result, bind original source witnesses and check nonempty exact quotes.
   Use exact rational arithmetic only to combine already interpreted class masses or reduce fractions.
6. Return WorldPrediction with original IDs. An invalid warm regime remains unusable and cannot
   supply a dark answer or a spurious distinction between surviving mechanisms.
