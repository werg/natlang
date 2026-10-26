---
args: { request: GrantCycleRequest, evidence: EvidencePage[] }
returns: PeriodAllocationResult
---
For every source expense in the supplied period evidence, bind its original amount, currency and scale, service interval, and the approved basis reference. Derive one slice per reporting period only when the basis and dates support it; every slice carries that same original amount and currency metadata alongside its own amount. Preserve unresolved expense IDs when evidence or basis is incomplete. Return these sourced slices and the basis references; do not claim that their arithmetic conserves until the exact checker has examined them.
