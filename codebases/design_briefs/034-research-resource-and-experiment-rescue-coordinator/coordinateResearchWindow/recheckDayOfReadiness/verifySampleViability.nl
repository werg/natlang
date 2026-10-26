---
args: { assignments: AssignmentSearch, mandates: ProtocolMandate[], snapshot: ResearchOperationsSnapshot }
returns: ReadinessEvidenceReview[]
---
Compare the proposed interval against the newest sample-viability evidence and approved protocol window. If the usable window shrank, retain the calendar reservation as historical/currently-booked but mark its slot blocked and require a fresh exact scheduling run.
