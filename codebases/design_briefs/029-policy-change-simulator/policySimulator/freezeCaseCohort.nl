---
args: { request: PolicyScenarioRequest, cases: PolicyCaseFacts[], dedupEvidence: EvidencePage[] }
returns: CohortReview
---
Freeze exact case identifiers, source versions, protected attributes, and required facts. Mark cases missing a field required by either edition indeterminate rather than imputing it. Reconcile deduplication changes as cohort edits with preserved included/excluded lineage; never conflate a changed denominator with a policy outcome.
