---
args: {"book": "ObligationBook", "outcome": "InvoiceOutcome"}
returns: "string[]"
---
Compare packet, operation, customer obligation, phase, amount, currency and ledger version against the approved consumption entries. A void command or email is not a confirmed ledger void. Reject an invoice receipt reused for another packet and any amount exceeding the approved phase capacity. Return mismatches or missing authority as precise reconciliation questions; never compensate by reducing a different milestone.
