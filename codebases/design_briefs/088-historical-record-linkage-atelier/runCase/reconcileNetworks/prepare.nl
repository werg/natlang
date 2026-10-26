---
description: Prepare one reconcileNetworks evidence slice.
args: { goal: string; records: HistoricalIdentityFinding[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Check proposed identity and relationship links globally for impossible dates, incompatible simultaneous identities, and circular assumptions. Prevent one uncertain link from becoming evidence that confirms itself through a derived family or institutional network. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
