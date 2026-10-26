---
args: { request: GrantCycleRequest; conditions: GrantConditionSet; allocations: GrantAllocationResult; activities: GrantActivityRegister; evidence: EvidencePage[] }
returns: GrantNarrativeDraft
---
Draft each report claim from accepted activity evidence and the grant's actual reporting trigger. Cross-reference its supported quantity and period to the corresponding allocation schedule. State deviation, partial delivery and missing impact evidence where observed; do not convert aspirations into accomplishments. Exclude personal identifiers unless the separate disclosure review records consent, reporting purpose and funder authority. Return an evidence-bound narrative draft with unsupported claims blocked.