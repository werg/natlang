---
args: { metric: string, window: string, schemas: EvidencePage[], missingness: EvidencePage[] }
returns: { treatmentExposure: string[]; controlExposure: string[]; eligibleDenominators: string[]; missingOutcomeRecords: string[]; instrumentationChanges: string[]; evidenceIds: string[] }
---
Locate only records matching this metric and window. Bind treatment and control exposure separately, retain each supplied eligible denominator, and distinguish missing outcome records from instrumentation downtime. Do not calculate rates from unlike populations or infer an exposure count from raw event volume. Return the exact matching source identities and unresolved denominator gaps.
