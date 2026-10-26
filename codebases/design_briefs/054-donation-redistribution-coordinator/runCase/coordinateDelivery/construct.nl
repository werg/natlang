---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftDonationDelivery
---
Construct a DonationDelivery draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Arrange consented collection or delivery, verify availability before dispatch, and track actual recipient acceptance. Reallocate declined or unavailable items through the same policy rather than informal side agreements, and preserve privacy in donor communications.
