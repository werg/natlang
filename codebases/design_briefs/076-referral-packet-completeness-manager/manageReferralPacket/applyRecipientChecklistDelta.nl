---
description: Apply only the receiving service's current authorized checklist version to pending packet requirements.
args: { dossier: ReferralDossier; change: PacketChange; identity: IdentityBinding }
returns: ChecklistDelta
---
Verify the checklist source, effective date and receiving service. Compare its requirement IDs with the prior version. Identify additions, changed validity/signature rules and unchanged requirements. Revalidate pending items only when affected; do not infer that a checklist update removes a requirement without explicit source authority.
