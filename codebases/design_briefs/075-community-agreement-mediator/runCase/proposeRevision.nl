---
description: proposeRevision for Community Agreement Mediator.
args: { failure: StageAudit; artifact: StageDraft }
returns: RevisionProposal
---
Propose the smallest bounded correction linked to a failed gate. State the expected effect and regression risk; do not weaken the gate or discard the last qualified checkpoint.
