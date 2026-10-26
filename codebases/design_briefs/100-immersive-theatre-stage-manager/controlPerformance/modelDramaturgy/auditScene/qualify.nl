---
description: Finite independent gate for a theatre scene's prerequisites and identity-scoped knowledge.
args: { evidence: EvidenceSlice; draft: DraftTheatreScene }
returns: "pass" | "fail" | "needs-review"
readout: decision
---
If the source slice is incomplete, a required beat/knowledge field is absent, or an audience/character path is not evidenced, return needs-review. Return pass only when current approved evidence supports the scene, all prerequisites are revealed before use, and each character and audience group knows only what its authorized path conveyed. Return fail when current approved evidence contradicts the required beat or a hard reveal-order or consent boundary is violated. Unknown status never passes. Return only this verdict.
