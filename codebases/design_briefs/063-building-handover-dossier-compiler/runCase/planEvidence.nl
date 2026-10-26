---
args: { input: CaseInput, work: WorkGraph }
returns: EvidencePlan
---
For every work node, identify decisive fields and evidence needed. Keep searched universe explicit; retrieval absence does not prove nonexistence.
Return the typed result, preserving missing facts as missing and citing evidence IDs where a decision depends on evidence.
