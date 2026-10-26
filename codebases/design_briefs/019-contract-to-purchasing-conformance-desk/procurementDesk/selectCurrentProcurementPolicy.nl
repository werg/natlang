---
args: { request: ProcurementReviewRequest, policies: PolicyCandidate[] }
returns: PolicySelection
---
Select purchasing and exception rules only when their effective date, entity, territory, category, and commitment scope cover the requested purchase. Retain competing interpretations as unresolved; do not derive affiliate coverage from a parent company's portal or relationship.
