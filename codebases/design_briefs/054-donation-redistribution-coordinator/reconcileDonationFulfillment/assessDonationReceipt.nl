---
args: { book: DonationFulfillmentBook, plan: DonationRoundProposal, receipt: DonationFulfillmentReceipt }
returns: DonationFulfillmentAssessment
---
1. Bind allocation = matchDonationAllocation(plan, receipt) and progression = verifyDonationStageProgression(book, receipt).
2. If safety holds this offer or stock reconciliation is mismatch/unknown, set state to quarantined or unknown and keep the quantity unavailable. A recall or handling restriction cannot be cleared by a delivery receipt.
3. Accept only the stage progression the two checks establish. A decline does not release quantity; only an actual return with conserved stock does. After a confirmed return, require a new fairness allocation before assigning that stock again.
4. Return acceptedTransition, the current exact state, reallocationRequired, and the specific donor, consent, inventory, or recipient follow-up. Do not infer receipt, consent, safe condition, or remaining quantity from the proposal.
