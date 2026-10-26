---
args: { request: EvidenceMapRequest, reviews: EvidenceReview[] }
returns: ReviewFrame
---
Bind a single review question, corpus, owner, eligibility policy ID and cutoff date. Keep the review question and the approved evidence universe stable; later changes create a new review version.
