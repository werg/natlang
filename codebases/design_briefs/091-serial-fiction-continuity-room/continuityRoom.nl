---
description: Test a proposed serial-fiction scene against the active approved edition and resume its affected continuity review.
args: { sourceRecord: StorySeries; sourceEvidence: EvidenceItem[]; edition:SerialEdition; change:StoryChange; evidence:EvidenceItem[]; checkpoint:ContinuityCheckpoint }
returns: ContinuityDossier
---
Read `seriesId`, `manuscriptIds`, `canonVersion`, `editorId` and `creativeBrief` from the top level of the StorySeries record. Treat its subject and source refs as the active continuity issue; preserve the approved canon version and manuscript identity in every repair branch. Carry every source fact and evidence ref into the analysis branch that depends on it; when any transformed proposal conflicts with this anchor, hold that branch for review instead of accepting the placeholder.

Bind story = `bindStorySeriesContext(sourceRecord, sourceEvidence, edition, change)`. Use story.edition, story.change and story.evidence for all following calls. The seriesId and canonVersion anchor the edition identity; manuscriptIds bound the review scope; editorId is the accountable editorial authority; creativeBrief sets the continuity obligations. If story.evidenceGaps is nonempty, preserve those gaps as blockers and do not qualify a repair.

Require the exact active approved `story.edition`; if its revision differs from `checkpoint.revision` without recorded invalidations, stop with blocked status. Bind canon = `recoverStoryCanon(story.edition, story.evidence)` to keep world facts, approved plans, character beliefs, and unreliable narration as different assertion kinds. Drafts from other editions may inform conflicts but cannot silently replace active canon.

Bind knowledge = `advanceKnowledgeLedger(story.edition, story.change.proposal)`. Knowledge changes only at an approved scene event that exposes the secret; a belief or unreliable narration does not establish world truth. Bind findings = `assessSerialChange(story.edition, story.change.proposal, canon, knowledge, story.edition.obligations)` to test chronology, character-specific knowledge at scene entry, motivations, world rules, and open story promises. A deliberate contradiction is acceptable only with an explicit approved reveal rationale.

If findings conflict, bind repairs = `mapContinuityRepairs(story.edition, story.change.proposal, findings)`. Preserve distinct options to revise this scene, seed earlier material, or submit a reviewed canon amendment; include every downstream scene invalidated by each choice and do not choose for the author.

When source evidence is incomplete, the edition is not approved, or checkpoint identity/revision is stale, do not verify or apply any chosen repair. Return the complete blocked dossier from `assembleContinuityDossier(story, canon, knowledge, findings, repairs, checkpoint)`; it must retain exact gaps, the active edition, prior checkpoint lineage and every dependent scene.

Only when the active edition and checkpoint match and source evidence is complete may an author-selected repair with matching editorial approval call `verifySerialEdition(story.edition, story.change, findings, repairs)` to recheck affected descendants and preserve the prior revision. Return its complete verified dossier. If no such exact choice and approval exists, return the complete review packet from `assembleContinuityDossier(story, canon, knowledge, findings, repairs, checkpoint)`.

When findings contain no conflicts, return `assembleContinuityDossier(story, canon, knowledge, findings, [], checkpoint)` as a complete dossier. This preserves the active edition and current checkpoint until author intent or approval authorizes a revision. The dossier records every finding, the exact proposal, affected descendants, open canon questions and the next checkpoint; it does not promote the proposal to approved material.
