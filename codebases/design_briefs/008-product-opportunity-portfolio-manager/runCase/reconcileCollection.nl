---
args: { results: StageOutputs, snapshot: Snapshot }
returns: CollectionFinding
---
1. Check duplicate identities, incompatible outcomes, shared-resource conflicts, and missing candidate coverage. Return only the minimal conflict components.
2. Check global consistency after local decisions: duplicated identities, incompatible dispositions, shared-resource conflicts, unsupported totals, and incomplete partitions. Return minimal conflict components and exact affected IDs, not a wholesale rewritten collection.
3. Return an explicit artifact; preserve unknowns and source coverage. This procedure proposes no production write.
