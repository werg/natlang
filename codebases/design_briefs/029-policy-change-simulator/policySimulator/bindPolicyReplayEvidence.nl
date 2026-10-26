---
args: { request: PolicyScenarioRequest, cohort: CohortReview, receipts: PolicyReplayPair[] }
returns: PolicyReplayBinding
---
For each frozen cohort case, bind one baseline and one candidate execution receipt to exact policy hashes, identical case and input digest, same engine/runtime configuration, and distinct execution IDs. Return only complete verified pairs; partial or stale pairs remain blocked with a concrete repair request. This procedure validates artifacts and does not run a simulator.
