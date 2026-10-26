---
args: { request: SupplierIncidentRequest, exposure: StageResultExposureFinding, policy: PolicySelection, evidence: EvidencePage[] }
returns: StageResultContainmentPlan
---
Choose only policy-authorized actions for confirmed exposure or explicitly precautionary scope. Keep unaffected orders intact, state lot-specific release conditions, and request capacity evidence before promising alternate supply. Missing trace or unclear authority stays unresolved; do not treat a supplier's narrative as release evidence.
