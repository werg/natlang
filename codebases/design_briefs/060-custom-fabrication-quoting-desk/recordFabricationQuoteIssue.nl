---
args: { book: QuoteDecisionBook, issue: FabricationQuoteIssue }
returns: QuoteIssueReview
---
1. Find an existing quote record with issue.quoteId. If it already has the same quoteRevision and term digest recorded as issued, return the unchanged book as "replayed". If the identity exists with different terms, hold the issue and request a new quote identity; never overwrite an issued or accepted quote.
2. Bind readiness = verifyQuoteIssueReadiness(issue). If readiness.issuable is false, return the unchanged book as "held" with its exact reasons as nextTasks. This decision requires supplied current evidence; a clean-looking proposal alone is insufficient.
3. Append a new QuoteDecisionRecord for issue.quoteId with issue.quoteRevision, issuedTermDigest equal to issue.termDigest, status "issued", acceptedTermDigest null, customerResponseId null, responseEvidence equal to readiness.approvalRefs plus readiness.evidenceRefs, and no pending changes. The supplied termDigest binds the exact rendered commercial terms; do not recompute or alter it.
4. Increment the book revision once and preserve all existing quote records, response events, and consumed response IDs. Return the updated book as "issued". Issuing terms creates no customer acceptance, shop reservation, purchase order, or production start.
