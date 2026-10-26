---
description: Recheck an approved story revision and freeze its continuity checkpoint.
args: { edition:SerialEdition; change:StoryChange; findings:ContinuityFinding[]; repairs:StoryRevision[] }
returns: ContinuityDossier
---
Require the exact chosen repair and approval for every canon amendment. Re-evaluate the changed scene and all downstream scenes whose facts, knowledge or obligations depend on it. Keep the preceding edition immutable. Bind the result to the new edition revision, show an editorial diff, and leave unresolved quality judgments for the author rather than disguising them as mechanical continuity checks.
