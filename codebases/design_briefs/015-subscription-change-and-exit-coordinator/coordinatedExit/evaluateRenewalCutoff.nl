---
args: { request: SubscriptionExitRequest, billing: EvidencePage[] }
returns: ExitGuardrail
---
For a request near renewal, compare the authoritative request-received instant with the renewal instant and the governing timezone/cutoff rule. A customer message timestamp or UI display alone is insufficient when the billing service receipt conflicts. Keep effective date unresolved if the cutoff cannot be established.
