---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftScopeDifference
---
Construct a ScopeDifference draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Compare the instruction with the agreed baseline and approved prior changes. Identify additions, omissions, substitutions, or clarifications, preserve exclusions, and avoid treating a contractor’s preferred method as contractual scope unless the baseline supports it.
