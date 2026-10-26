---
args: { request: SubstituteEvaluationRequest, policies: PolicyCandidate[] }
returns: PolicySelection
---
Select only policies whose authority, effective period, intended product use, and tenant scope cover this request. Preserve unresolved precedence and do not choose a policy because it is most convenient for the substitution.
