---
args: { allocation: FairDonationAllocation, round: DonationRoundRequest }
returns: DonationDeliveryPlan[]
---
Verify donor availability and recipient consent immediately before dispatch planning. Include only the minimum recipient information needed for the selected collection or delivery. Record delivery attempt, recipient acceptance, and decline as separate states; declined items return to allocation under the same fairness rule.
Bind disclosure = minimizeRecipientDisclosure(round.recipientConsents, round.deliveryMode). Exclude any recipient detail not explicitly consented for this delivery.
