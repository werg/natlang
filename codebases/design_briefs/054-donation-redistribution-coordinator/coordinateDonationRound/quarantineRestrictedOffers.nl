---
args: { offers: DonationOffer[], round: DonationRoundRequest }
returns: DonationSafetyAssessment
---
Compare every offered lot with the supplied recall and handling restrictions. Quarantine only affected quantities and preserve lot/offer identity. Return eligible offer IDs separately from holds, and require each hold to name its release condition. Missing lot evidence cannot be treated as a clean safety result.
