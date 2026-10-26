---
args: { round: DonationRoundRequest }
returns: DonationRoundProposal
---
1. Bind safety = quarantineRestrictedOffers(round.offeredLots, round). Inspect every lot against supplied recall and handling restrictions before any inventory can enter a bundle.
2. Bind offers = qualifyConfirmedDonationStock(round, safety). Keep tentative promises outside confirmed stock and mark quarantined quantities unavailable.
3. Bind needs = confirmRecipientConstraints(round). Use only recipient-stated essentials, timing, consent, and storage limits. Do not infer need or deservingness from sensitive circumstances.
4. Bind bundles = assemblePurposeFitBundles(offers, needs, round). Check complete components and permitted substitutions together; split large bundles to fit storage limits.
5. Bind allocation = applyFairAllocationRules(bundles, round). Use supplied fairness policy and exact confirmed inventory/transport capacity; preserve unmet needs and explain tradeoffs.
6. Bind delivery = proposeConsentedDelivery(allocation, round). Keep donor details separate from recipient identity and share only consented fields.
7. Return { roundId: round.roundId, offers, needs, bundles, allocation, delivery, safety, commitAllowed: false }.
