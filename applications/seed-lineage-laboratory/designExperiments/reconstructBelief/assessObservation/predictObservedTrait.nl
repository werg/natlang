---
description: Interpret one theory against the observed accession and biological scope.
args:
  world: JointWorld
  observation: Observation
  theory: ExpressionTheory
  inheritanceRules: SourceClause[]
returns: WorldPrediction
---
1. Join observation.accessionId to the world's complete assignment. Preserve its genotype and
   cytoplasm independently; a maternal label is not a second nuclear allele.
2. Read observation.scope and the theory's actual compartment/stage. Decide whether this observation
   concerns the theory's quantity, an independent maternal tissue, or an unresolved measurement.
3. For an applicable deterministic trait, apply the source theory to the returned assignment and
   derive the observable answer under its regime. Carry the exact theory and scope witnesses.
4. For a separately defined coat or marker quantity, interpret its own source inheritance/tissue
   convention. Do not borrow an adult expression rule merely because both reports use color words.
5. Return a WorldPrediction with explicit answer probabilities and usable status. A sensitivity loss
   is not a dark phenotype; preserve it as unusable or unresolved according to the actual protocol.
6. Use only source-visible answer labels. Numeric probabilities may use exact arithmetic after the
   semantic phenotype and readout classes have been selected.
