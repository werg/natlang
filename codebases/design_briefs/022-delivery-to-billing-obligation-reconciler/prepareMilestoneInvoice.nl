---
args: {"docket": "BillingDocket", "book": "ObligationBook | null"}
returns: "MilestonePacket"
---
1. Bind prior to the supplied obligation book or an empty book at revision zero. Bind clocks = buildContractClocks(docket, prior.clocks). Preserve every earned slice's original effective rate schedule.
2. Bind deliveries = mapDeliveryClaims(docket.agreementSources, clocks). Keep project labels as aliases of contract milestone identities, never as replacement identities. Drafts shared for feedback remain distinct from contractual handovers.
3. For each milestone and phase call adjudicateAcceptance(clock, deliveries, docket.authorityRecords, docket.agreementSources). Collect each returned decision in acceptance; an unauthorized acknowledgment can remain useful evidence while a confirmation request stays open.
4. Bind slices = computeUnconsumedEntitlement(clocks, acceptance, prior, docket.agreementSources). Allocate only the accepted phase or authorized partial fraction. Preserve unresolved triggers, caps, conversion bases and timing prerequisites as holds.
5. Bind packet = bindCustomerPacket(docket, slices, acceptance, prior). Recheck late amendments and changed acceptance against the exact proposed packet before release. Return a new book revision carrying clocks, acceptance, earned slices and unchanged actual consumption. A packet is ready only when every proposed amount is supported and its exact approval bindings are current; it does not create an invoice.
