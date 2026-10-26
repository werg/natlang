---
args: { request: AmendmentReviewRequest, editions: EvidencePage[] }
returns: AmendmentGuardrail
---
Resolve obsolete section numbers against the authenticated executed edition and its clause map. If multiple clause mappings remain plausible, stop target attribution and ask for the signed schedule or cross-reference; never choose by document position alone.
