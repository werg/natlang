---
args: { request: SupplierIncidentRequest, evidence: EvidencePage[] }
returns: IncidentGuardrail
---
Bind a corrected notice to the original message, recipient set, delivery receipt and corrected fact. Preserve the initial notification and delivery history; issue a reviewed amendment instead of silently replacing the prior notice.
