---
args: { prior: EligibilitySnapshot, changes: EvidencePage[], lineage: StudyLineage, observations: ObservationLedger }
returns: ObservationLedger
---
When eligibility rules change mid-review, version the new criteria and re-screen affected canonical studies against the new rules. Return an updated observation ledger with old and new screening provenance. Preserve every prior included/excluded decision; do not apply new criteria retroactively without recording the review version.
