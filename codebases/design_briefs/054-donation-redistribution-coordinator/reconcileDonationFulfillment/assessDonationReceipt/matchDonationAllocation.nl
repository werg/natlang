---
args: { plan: DonationRoundProposal, receipt: DonationFulfillmentReceipt }
returns: DonationAllocationMatch
---
Find the exact offer, feasible bundle, recipient need, and allocation linking receipt.offerId, bundleId, and recipientId. The offer must be eligible under the plan's safety assessment and not held or unresolved. Confirm that the bundle satisfies the recipient's purpose and storage constraints and that the allocation names this bundle. Return every missing or mismatched identity; a planned bundle alone is not a stock reservation.
