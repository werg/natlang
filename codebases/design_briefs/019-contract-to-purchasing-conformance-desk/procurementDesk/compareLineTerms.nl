---
args: { intent: StageResultPurchaseIntent, agreement: StageResultAgreementScope, schedules: EvidencePage[], catalog: EvidencePage[] }
returns: StageResultConformanceFinding
---
Compare each purchase line independently against current contract inclusion, exclusion, quantity minimum, bundle condition, intended-use limit, and price schedule. Keep commercial price compliance separate from functional scope. A catalog description supports only the exact catalog revision and timestamp supplied.
