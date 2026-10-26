---
description: Prepare one assembleFulfillment evidence slice.
args: { goal: string; records: RecordDisclosureFinding[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Build a complete or explicitly partial packet with a manifest and explanation of outstanding items. Apply approved redactions through trusted rendering services, verify extracted content and metadata, and bind approval to exact artifact hashes. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
