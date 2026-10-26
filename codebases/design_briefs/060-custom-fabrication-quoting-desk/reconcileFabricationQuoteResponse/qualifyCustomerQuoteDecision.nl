---
args: { response: CustomerQuoteResponse, issuedRecord: QuoteDecisionRecord, current: QuoteCurrentFacts }
returns: QuoteResponseReadiness
---
1. A decision is recordable only when current.responseAuthority is "verified", response.respondingPartyId is among current.authorizedRespondingPartyIds, and response.authorityEvidence and response.evidenceRef are present.
2. Bind the decision to the exact issued quote: response.quoteId and current.quoteId must equal issuedRecord.quoteId; response.quoteRevision and current.currentRevision must equal issuedRecord.quoteRevision; response.presentedTermDigest and current.currentTermDigest must both equal issuedRecord.issuedTermDigest. Any mismatch makes the response held evidence, not a decision about current terms.
3. For an exact, authorized "decline" or "request-change", decisionMayBeRecorded is true. These responses do not prove agreement to any terms.
4. For "accept", acceptanceMayBeRecorded is true only when decisionMayBeRecorded is true, current.validity is "current", current.feasibility is "ready", current.feasibilityEvidence is present, current.allRequiredApprovalsPresent is true, and current.approvalRefs are present. An expired, blocked, unknown, provisional, or unapproved quote stays unaccepted.
5. Return each unmet condition by name. Do not infer freshness, feasibility, or authority from the wording of the response.
