---
description: reconcileCollection for Historical Record Linkage Atelier.
args: { stageOutputs: StageOutputs; expectedCoverage: string[] }
returns: CollectionFinding
---
Reconcile all stage outputs against frozen coverage. Find cross-stage identity collisions, incompatible dispositions, unsupported totals, and missing partitions. Return the smallest affected components without rewriting sound artifacts.
