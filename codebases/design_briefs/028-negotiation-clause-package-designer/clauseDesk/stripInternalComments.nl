---
args: { request: NegotiationPackageRequest, comments: EvidencePage[] }
returns: NegotiationGuardrail
---
Identify internal fallback notes and comments in the exact export version, strip them from the external artifact, and inspect the rendered output. Preserve internal negotiation history in a separate protected record; stripping comments does not delete the audit trail.
