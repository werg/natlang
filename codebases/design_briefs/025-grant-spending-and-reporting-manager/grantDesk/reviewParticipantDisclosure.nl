---
args: { request: GrantCycleRequest, disclosures: EvidencePage[] }
returns: ParticipantDisclosureReview
---
For each participant disclosure request, bind the participant identifier, consent, allowed purpose, minimization and requester authority. Return only participants whose complete permission chain matches the current funder and report. Put every other identifier in withheldParticipantIds. Include permitted aggregate evidence references separately; do not treat a missing consent record as permission.
