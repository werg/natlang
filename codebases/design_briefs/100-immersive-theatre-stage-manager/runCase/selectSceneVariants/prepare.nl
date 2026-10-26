---
description: Prepare one selectSceneVariants evidence slice.
args: { goal: string; records: TheatreConstraint[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Select or generate approved-style variants that fit audience choices and current resources. Preserve the dramatic beat, character knowledge, and consent boundaries; avoid replacing a missing prop with an unsafe physical action or unsupported audience involvement. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
