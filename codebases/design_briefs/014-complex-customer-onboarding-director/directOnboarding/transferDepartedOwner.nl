---
args: { request: CustomerLaunchRequest, receipts: EvidencePage[], outcomes: OnboardingOutcome[] }
returns: OnboardingGuardrail
---
For each accepted outcome whose owner is departing, resolve an authorized successor before constructing dependent tasks. Require a current transfer receipt naming that successor and a separate credential revocation event. Preserve the former assignee in history; without both controls keep dependent work unassigned and blocked.
