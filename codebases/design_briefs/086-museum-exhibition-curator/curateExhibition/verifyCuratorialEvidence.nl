---
description: Exact counterfactual gate for provenance-supported artifact interpretations and display restrictions.
args: { evidence: EvidenceSlice; draft: DraftCuratorialEvidence }
returns: 'pass' | 'fail' | 'needs-review'
readout: decision
---
Pass only when artifact identity, supported interpretation, provenance limit and display constraint are evidenced. Fail when a withdrawn loan or restricted item is treated as available for display or publication. Return needs-review when provenance, community permission or conservation status is missing. Return only the finite verdict.
