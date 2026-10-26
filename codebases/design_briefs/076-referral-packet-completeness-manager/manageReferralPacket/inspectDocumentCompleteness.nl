---
description: Match patient-bound documents to recipient requirements and episode scope, preserving precise gaps.
args: { dossier: ReferralDossier; change: PacketChange; identity: IdentityBinding; checklist: ChecklistDelta }
returns: DocumentFinding[]
---
Call `classifyDocumentRequirements(dossier, change, identity, checklist)`. For each required item, confirm patient, episode, required signature, date validity and source. Distinguish unsigned from missing and wrong-episode from wrong-patient. An identity mismatch blocks use of the document even if its content appears to fit.
