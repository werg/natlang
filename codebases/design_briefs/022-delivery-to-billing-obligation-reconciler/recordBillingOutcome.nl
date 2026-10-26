---
args: {"book": "ObligationBook", "outcome": "InvoiceOutcome"}
returns: "BillingOutcomeReview"
---
Find the existing packet and operation before changing any capacity. Call verifyInvoiceOutcome(book, outcome). When that check returns a discrepancy, leave consumption unchanged and open a reconciliation item. Otherwise attach the issued invoice ID to its exact phase amounts once. For a voided outcome require the authoritative invoice ledger's confirmed void receipt and version before releasing only that operation's consumed amounts. An unknown outcome remains unknown; a repeated identical receipt is a no-op. Return the new book with historical earned entitlements untouched.
