---
description: assessDelta for Volunteer Deployment Coordinator.
args: { committedSnapshot: string[]; sourceDelta: EvidenceItem[]; stageOutputs: StageOutputs }
returns: InvalidationProposal
---
Compare a committed snapshot with the supplied source delta. Classify new facts, corrections, permission changes, and editorial changes. Identify candidate dependent artifacts and which material change warrants recomputation; preserve unaffected outputs and mark uncertain impacts for review. The host performs exact dependency traversal and authorization checks.
