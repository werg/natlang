---
description: Derive source-permitted offspring classes from one joint world and oriented cross.
args:
  world: JointWorld
  cross: CrossReading
  inheritanceRules: SourceClause[]
returns: OffspringDistribution
---
1. If the cross is unresolved, return an unresolved distribution with its actual witnesses. For a
   subject-only assay, return the subject's actual assignment as one class of probability one.
2. Join the actual mother/father IDs to their world assignments. Use established fixed accession
   assignments carried into the world; absent parent data stays unresolved.
3. Interpret the source gamete convention for each genotype. Derive its possible transmitted alleles
   and their exact probabilities; retain maternal and paternal roles through the cross.
4. Combine the two finite gamete collections into source-permitted offspring nuclear genotypes.
   Assign each class the mother's cytoplasm according to the separately witnessed transmission rule.
5. Merge classes with identical genotype and cytoplasm by exact probability addition. The nursery's
   replica-complete convention determines whether these probabilities are exact tray proportions.
6. Return the actual classes with transmission/parent witnesses. A sample-based regime would require
   its own observation model; source-complete trays do not introduce sampling uncertainty.
