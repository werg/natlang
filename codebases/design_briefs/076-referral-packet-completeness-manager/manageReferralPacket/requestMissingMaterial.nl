---
description: Request only the document, signature or verification needed to close a named referral gap.
args: { dossier: ReferralDossier; documents: DocumentFinding[] }
returns: ReferralRequest[]
---
For each gap, name the exact recipient checklist requirement and responsible source owner. Request only the missing item or signature needed for this episode. Preserve clinician priority as authoritative context but never request a new independent triage decision.
