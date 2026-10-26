---
args: { request: SupplierIncidentRequest, evidence: EvidencePage[] }
returns: IncidentGuardrail
---
Compare the required supplier response deadline with an authoritative receipt of response. Escalate by the configured path when missed; retain the unanswered claim as unverified and never accept it by silence.
