---
description: prepareReview for Volunteer Deployment Coordinator.
args: { stageOutputs: StageOutputs; finding: CollectionFinding; questions: string[] }
returns: ReviewPacket
---
Build a reviewer packet with qualified artifacts, unresolved blockers, evidence coverage, and required approvals. Keep proposals distinct from committed effects.
