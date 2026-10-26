---
args: { snapshot: Snapshot, currentEvidence: EvidencePage[] }
returns: InvalidationProposal
---
1. Separate new facts, correction, access change, and editorial change. Invalidate only materially dependent outputs and preserve unaffected artifacts.
2. Distinguish new facts, corrections, permissions changes, and editorial changes. Traverse evidence and field-dependency edges, then judge semantic materiality. Preserve unaffected artifacts and label unresolved impact explicitly; host services perform the exact graph traversal.
3. Return an explicit artifact; preserve unknowns and source coverage. This procedure proposes no production write.
