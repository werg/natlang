---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftHarvestRevision
---
Construct a HarvestRevision draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Process changed field observations, resource failures, or customer responses as scoped deltas. Reallocate only affected lots, obtain consent for material substitutions, and reconcile actual harvested and delivered quantities before declaring commitments fulfilled.
