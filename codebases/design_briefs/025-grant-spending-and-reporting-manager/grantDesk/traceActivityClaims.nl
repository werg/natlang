---
args: { request: GrantCycleRequest; conditions: GrantConditionSet; allocations: GrantAllocationResult; evidence: EvidencePage[] }
returns: GrantActivityRegister
---
Match each milestone to its own definition and acceptance evidence. Record delivered activity separately from outcome and long-term impact; a participation count supports a count only. Require allocation and allowed-use linkage before attributing funded activity. Preserve partial completion with remaining requirement and evidence gaps instead of rounding it to complete. Return accepted activity records, partial milestones, unsupported claim topics and source refs.