---
args: { check: EstimandCheck, output: ApprovedAnalysisOutput, mechanism: string, account: EvidencePage }
returns: AccountEvidenceReview
---
Bind link = qualifyAccountLink(check, output, account). Classify the account as supportive, dissatisfied-success, satisfied-failure, contrary, or unlinked only from its actual outcome, experience, and mechanism evidence. An unlinked anecdote cannot be fitted to this analysis. Return evidence IDs and any mismatch for the parent mechanism review.
