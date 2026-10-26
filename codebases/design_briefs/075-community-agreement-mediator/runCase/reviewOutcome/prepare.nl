---
description: Prepare one reviewOutcome evidence slice.
args: { goal: string; records: AgreementTrial[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Compare actual use and participant feedback with the trial criteria, preserving divergent experiences. Propose continuation, revision, or termination, and record only the terms participants explicitly accept rather than treating silence as agreement. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
