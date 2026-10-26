---
description: Prepare one formNarratives evidence slice.
args: { goal: string; records: CuratorialEvidence[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Generate distinct interpretive arguments that connect artifacts meaningfully rather than merely grouping them by topic. Include alternative perspectives and uncertainties, identify missing evidence, and avoid inventing historical continuity to make a more dramatic story. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
