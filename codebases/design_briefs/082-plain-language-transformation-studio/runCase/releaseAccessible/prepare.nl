---
description: Prepare one releaseAccessible evidence slice.
args: { goal: string; records: ComprehensionProbe[]; evidence: EvidenceItem[] }
returns: EvidenceSlice
---
Select a bounded evidence slice for the exact subject and task. Include supporting and contradictory versions, authority, and freshness. Record searched scope and missing decisive facts. Apply this stage's relevance rule: Reconcile reviewer feedback, maintain side-by-side source mapping, and publish the accessible edition with its scope and version. Revalidate affected meaning units after source changes and preserve access to authoritative wording where required. Return source item IDs and evidence, not conclusions or a corpus-wide paraphrase.
