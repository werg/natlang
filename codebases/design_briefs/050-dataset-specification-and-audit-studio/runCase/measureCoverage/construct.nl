---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftCoverageFinding
---
Construct a CoverageFinding draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compute exact slice coverage and overlap using the approved population definitions. Interpret whether apparent balance masks duplicated sources or missing mechanisms, and identify the most consequential acquisition gaps rather than optimizing superficial counts.
