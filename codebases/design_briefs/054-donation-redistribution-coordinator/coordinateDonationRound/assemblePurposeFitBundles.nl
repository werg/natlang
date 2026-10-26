---
args: { offers: ConfirmedDonationStock[], needs: RecipientNeedProfile[], round: DonationRoundRequest }
returns: PurposeFitDonationBundle[]
---
Combine items only when the complete bundle can serve the recipient's confirmed purpose. Compare condition and practical fit, list missing components, and propose substitutions only from stated acceptable alternatives. If a recipient cannot store the full bundle, consider separated delivery or a smaller feasible bundle.
