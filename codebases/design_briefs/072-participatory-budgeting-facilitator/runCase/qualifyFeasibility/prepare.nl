---
description: Prepare one qualifyFeasibility evidence slice.
args: { goal: string; records: ProposalRelationship[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare proposals with supplied eligibility, resource, and delivery rules, using exact estimates and qualified assessments where necessary. Separate infeasibility from missing information and provide a clear path for revision rather than silently excluding difficult proposals. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
