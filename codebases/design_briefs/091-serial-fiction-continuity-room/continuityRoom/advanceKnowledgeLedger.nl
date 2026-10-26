---
description: Reconstruct what each character knows at the proposed scene boundary.
args: { edition:SerialEdition; proposal:SerialScene }
returns: KnowledgeLedgerAnalysis
---
Bind trace = `deriveSceneExposureEvents(edition, proposal)`. Return the complete KnowledgeLedgerAnalysis with trace.events at the proposal boundary, retaining unaffected prior entries with exact edition and scene provenance. Remove only trace.invalidatedEventIds from the old downstream ledger so later scenes are re-evaluated after a changed exposure. Preserve unresolvedExposureIds as explicit unknown-character-knowledge findings.
