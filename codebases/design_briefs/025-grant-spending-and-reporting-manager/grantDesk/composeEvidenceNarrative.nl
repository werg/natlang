---
args: { request: GrantCycleRequest; conditions: GrantConditionSet; allocations: GrantAllocationResult; activities: GrantActivityRegister; evidence: EvidencePage[]; disclosure: ParticipantDisclosureReview }
returns: GrantNarrativeDraft
---
Draft each report claim from accepted activity evidence and the grant's actual reporting trigger. Cross-reference its supported quantity and period to the corresponding allocation schedule. State deviation, partial delivery and missing impact evidence where observed; do not convert aspirations into accomplishments. Include participant identifiers only for IDs in disclosure.allowedParticipantIds. For withheldParticipantIds, use an allowed aggregate reference only when it independently answers the reporting need; otherwise block the claim. Return an evidence-bound narrative draft with unsupported claims blocked.
