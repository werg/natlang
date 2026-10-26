---
args: { check: EstimandCheck, output: ApprovedAnalysisOutput, accounts: EvidencePage[] }
returns: FeedbackReview
---
If the check's provenance, assignment, approval or measurement-window findings block the output, return a withheld review with those exact blockers and no fitted mechanism. Otherwise, for each proposed mechanism and each relevant account call classifyAccountExperience(check, output, mechanism, account). Build the returned mechanisms from actual child classifications, preserving supportive accounts, dissatisfied successes, satisfied failures, rival explanation and discriminating observation. Keep account evidence links in each mechanism; never treat the account sample as causal proof.
