---
args: { request: ProcurementReviewRequest, approvals: EvidencePage[] }
returns: ProcurementGuardrail
---
Revalidate agreement effective date, price validity and purchase authorization at order release. If an applicable term expired while approval waited, withhold release and refresh only affected pricing, scope and approval dependencies.
