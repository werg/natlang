---
args: { book: DonationFulfillmentBook, plan: DonationRoundProposal, receipt: DonationFulfillmentReceipt }
returns: DonationFulfillmentReview
---
1. For a consumed receiptId compare every field to the stored receipt. Exact replay locates its allocation state and returns { book, outcome: "replayed", offerStatus: stored allocation state, reallocationRequired: stored return-state flag, followUp: stored outstanding stock/consent tasks }. Conflicting or unbound identity returns { book, outcome: "held", offerStatus: "unknown", reallocationRequired: false, followUp: ["reconcile fulfillment receipt identity and physical stock"] }; neither branch changes the book.
2. Bind assessment = assessDonationReceipt(book, plan, receipt). A mismatch in round, bundle, recipient, offer, eligibility, consent, or stock conservation cannot advance the allocation.
3. Record actual stage outcomes only. Keep failed or unknown delivery unavailable, retain declines until returned inventory is verified, and quarantine recall or handling holds. Update only the matching offer/bundle/recipient allocation state.
4. After a verified return, set reallocationRequired and rerun the supplied fairness policy before assigning that stock. Never select a new recipient by convenience or infer an unrecorded quantity balance.
5. Append each new receipt once, increment revision once, preserve prior delivery and consent history, and bind followUp to assessment.reasons plus stock, consent, recipient, or fairness tasks. Bind review = { book: updatedBook, outcome: assessment.acceptedTransition ? "recorded" : "held", offerStatus: assessment.state, reallocationRequired: assessment.reallocationRequired, followUp }; return review.
