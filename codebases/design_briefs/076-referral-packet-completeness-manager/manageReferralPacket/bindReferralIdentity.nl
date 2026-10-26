---
description: Bind every referral document to the patient using approved identity evidence, never name similarity.
args: { dossier: ReferralDossier; change: PacketChange }
returns: IdentityBinding
---
Require the approved identity evidence specified by the receiving service. Match patient identifiers and provenance, not names alone. Mark a mismatch when a document belongs to another chart. If identity evidence is absent or ambiguous, block document matching and retain the exact missing verification requirement.
