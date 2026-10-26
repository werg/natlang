---
description: Advance packet transfer through authorization, dispatch and recipient acknowledgment without ambiguous resend.
args: { dossier: ReferralDossier; change: PacketChange; identity: IdentityBinding; documents: DocumentFinding[]; requests: ReferralRequest[] }
returns: TransferLedger
---
Call `readRecipientStatus(dossier, change)` when the dispatch receipt is delayed, absent or ambiguous. Do not resend sensitive records while delivery is unknown. Require current authorization, verified patient identity, closed checklist gaps and episode-limited disclosure before dispatch. Mark acknowledged only from an exact recipient receipt bound to this packet version.
