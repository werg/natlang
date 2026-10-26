---
args: { book: QuoteDecisionBook, issue: FabricationQuoteIssue }
returns: QuoteIssueReview
---
1. Find an existing quote record with issue.quoteId. If it already has the same quoteRevision and term digest recorded as issued, bind result = { book, outcome: "replayed", reasons: [], nextTasks: [] }; return unchanged. If identity exists with different terms, bind result = { book, outcome: "held", reasons: ["quote identity already binds different terms"], nextTasks: ["issue a new quote identity"] }; return unchanged. Never overwrite an issued or accepted quote.
2. Bind readiness = verifyQuoteIssueReadiness(issue). If readiness.issuable is false, bind result = { book, outcome: "held", reasons: readiness.reasons, nextTasks: readiness.reasons }; return unchanged. This decision requires supplied current evidence; a clean-looking proposal alone is insufficient.
3. Append a new QuoteDecisionRecord for issue.quoteId with issue.quoteRevision, issuedTermDigest equal to issue.termDigest, status "issued", acceptedTermDigest null, customerResponseId null, responseEvidence equal to readiness.approvalRefs plus readiness.evidenceRefs, and no pending changes. The supplied termDigest binds the exact rendered commercial terms; do not recompute or alter it.
4. Increment revision once and preserve existing quote records and response events. Bind result = { book: updatedBook, outcome: "issued", reasons: [], nextTasks: [] }; return result. Issuing terms creates no customer acceptance, shop reservation, purchase order, or production start.
