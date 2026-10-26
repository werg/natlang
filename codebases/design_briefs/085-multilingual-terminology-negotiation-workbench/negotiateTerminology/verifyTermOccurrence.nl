---
description: Exact counterfactual gate for contextual term occurrence identity and language use.
args: { evidence: EvidenceSlice; draft: DraftTermOccurrence }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when term, language, context, candidate concept and source reference are preserved from evidence. Fail when context is removed in a way that changes meaning or a regional form is relabeled inconsistent without scope evidence. Return needs-review when sense, region or authority is ambiguous. Return only the finite verdict.
