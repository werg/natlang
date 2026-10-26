---
description: Prepare one recoverCanon evidence slice.
args: { goal: string; records: StorySeries[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Extract events, relationships, world rules, and character knowledge from approved manuscripts and notes. Distinguish established canon from plans, unreliable narration, and character beliefs; preserve conflicting drafts without silently promoting one to truth. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
