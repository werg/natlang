---
args: { prior: ObservatoryCheckpoint | null, request: CustomerObservatoryRequest, dossier: CustomerSignalDossier, consent: ConsentDerivativePlan, invalidations: ObservatoryInvalidation }
returns: ConsentEffectReview
---
Match each consent withdrawal to attributable quote identities and derivative artifacts that consumed those quotes. Keep removal and recomputation as different pending effects with distinct source references. Preserve non-attributable incident IDs and aggregate evidence only when their lineage does not reveal the withdrawn identity. When a matching external receipt is absent, retain the effect as pending; a proposal does not establish deletion.
