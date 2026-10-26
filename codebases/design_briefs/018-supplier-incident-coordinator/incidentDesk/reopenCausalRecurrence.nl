---
args: { request: SupplierIncidentRequest, evidence: EvidencePage[] }
returns: IncidentGuardrail
---
Compare post-closure failures with the prior closed incident's evidenced mechanism, product revision, lot and release criteria. If materially similar evidence contradicts closure, reopen only affected causal verification and preserve the original closure record before any new closure review.
