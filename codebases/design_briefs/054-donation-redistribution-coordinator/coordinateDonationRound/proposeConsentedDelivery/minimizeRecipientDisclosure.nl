---
args: { consents: RecipientConsent[], mode: string }
returns: DeliveryDisclosure
---
For the selected handoff mode, list only the information required to complete that handoff. Compare every proposed donor-facing field with the recipient's explicit consent. Remove unconsented identity, address, and circumstance details; if consent is absent, provide a coordinator-mediated handoff instead.
