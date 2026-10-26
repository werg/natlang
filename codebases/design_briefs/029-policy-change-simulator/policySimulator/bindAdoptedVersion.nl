---
args: { request: PolicyScenarioRequest, receipts: EvidencePage[] }
returns: PolicyGuardrail
---
If a policy is adopted, pin the tested version and exact replay receipts to the adoption record. Any later edit forks a new candidate and requires fresh case replay and review; adoption does not certify subsequent revisions.
