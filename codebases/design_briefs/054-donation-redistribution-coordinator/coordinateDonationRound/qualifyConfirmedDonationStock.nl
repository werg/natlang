---
args: { round: DonationRoundRequest, safety: DonationSafetyAssessment }
returns: ConfirmedDonationStock[]
---
For every offered lot verify donor confirmation, exact quantity, condition, collection window, and permitted-use restrictions. Exclude every offerId and lotRef held by safety before returning allocatable stock. Keep tentative quantities in a separate pending list. If collection-day quantity changes, preserve prior reservations and identify only bundles whose feasibility changed.
Bind changed = reconcileCollectionDayQuantity(round.priorAllocations, round.collectionDayUpdates). Recalculate affected reservations under the same approved allocation rule and keep every newly unmet need visible.
