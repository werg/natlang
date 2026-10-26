---
description: Classify each attached document against its exact checklist requirement and referral episode.
args: { dossier: ReferralDossier; change: PacketChange; identity: IdentityBinding; checklist: ChecklistDelta }
returns: DocumentFinding[]
---
Match one document to one requirement only when document identity, required content, signature, validity period and episode all fit. Preserve a present but unsigned record as unsigned, not complete. Keep unrelated history out of the packet and report it as wrong-episode. Record the checklist and document source IDs for every classification.
