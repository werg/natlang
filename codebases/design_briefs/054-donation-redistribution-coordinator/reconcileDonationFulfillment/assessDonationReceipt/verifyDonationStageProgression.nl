---
args: { book: DonationFulfillmentBook, receipt: DonationFulfillmentReceipt }
returns: DonationConsentAndStockReview
---
1. Require donorAvailability "confirmed" and stockReconciliation "conserved" for a new reservation. A mismatch quarantines the affected quantity; an unknown result keeps it unavailable.
2. Permit collected only from reserved; delivered only from collected with current recipient consent and consentRef; accepted or declined only from delivered with actual recipient response evidence. Do not infer acceptance from delivery completion.
3. A declined item stays allocated and unavailable until an actual returned receipt confirms the same offer, recipient, bundle and quantity with conserved stock. Then set reallocationRequired; the old allocation is not silently reused.
4. For unknown stage or consent, keep the allocation unknown and block further transfer. Report missing consent or stock receipts precisely.
