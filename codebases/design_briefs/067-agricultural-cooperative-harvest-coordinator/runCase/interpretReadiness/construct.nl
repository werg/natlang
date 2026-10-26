---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHarvestReadiness
---
Construct a HarvestReadiness draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Read grower notes and approved assessment records to identify likely readiness, quality, and uncertainty. Distinguish tentative expectations from confirmed availability and preserve variety or lot differences that affect harvest timing and customer suitability.
