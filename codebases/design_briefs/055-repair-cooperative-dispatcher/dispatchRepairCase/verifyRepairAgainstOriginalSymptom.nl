---
args: { work: RepairSequence, symptom: SymptomSeparation, repair: CooperativeRepairRequest }
returns: VerifiedRepairOutcome
---
Bind comparison = comparePostWorkChecksToOriginal(symptom, repair.outcomeChecks, repair.customerEvidence). Bind recurrence = identifyTemporaryMasking(comparison, work). Verify against the original observations under equivalent conditions. If a check is absent or a temporary mask can explain improvement, keep verification open and add a recurrence watch; do not claim a full repair from an unobserved result.
