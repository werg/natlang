---
args: { prior: FairDonationAllocation[], updates: DonationOfferUpdate[] }
returns: ChangedReservationSet
---
Join each update to its offer ID and confirmed measured quantity. Recompute only reservations that draw from that offer. Preserve the prior allocation version and report newly unmet bundles rather than hiding shortages by reducing requested need.
