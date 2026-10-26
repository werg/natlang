---
description: reconcileCollection for Cross Cultural Training Adaptation Studio.
args: { stageOutputs: StageOutputs; expectedCoverage: string[] }
returns: CollectionFinding
---
Reconcile all stage outputs against frozen coverage. Find cross-stage identity collisions, incompatible dispositions, unsupported totals, and missing partitions. Return the smallest affected components without rewriting sound artifacts.
