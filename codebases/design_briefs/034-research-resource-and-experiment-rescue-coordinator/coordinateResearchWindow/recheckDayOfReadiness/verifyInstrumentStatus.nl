---
args: { assignments: AssignmentSearch, snapshot: ResearchOperationsSnapshot }
returns: ReadinessEvidenceReview[]
---
For each assigned instrument inspect the most recent calibration and maintenance evidence, requiring it to remain valid through the proposed end time. Missing, expired, contradictory, or stale evidence blocks readiness. Keep the existing booking and identify only the instrument-specific evidence needed to unblock it.
