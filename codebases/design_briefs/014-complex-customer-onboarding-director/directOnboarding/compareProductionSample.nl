---
args: { request: CustomerLaunchRequest, comparisons: EvidencePage[] }
returns: OnboardingGuardrail
---
Compare sample-data and production observations by schema version, identifier cardinality, null behavior, access classification, and representative account classes. A mismatch requires a privacy-approved limited validation with reconciled identifiers before bulk import; sample success cannot close this gate.
