---
args: { request: ProcurementReviewRequest, versions: EvidencePage[] }
returns: ProcurementGuardrail
---
Bind each approved line to the catalog item identifier, description revision and timestamp observed during review. Preserve old and new descriptions; changed wording requires new conformance analysis where the changed property matters.
