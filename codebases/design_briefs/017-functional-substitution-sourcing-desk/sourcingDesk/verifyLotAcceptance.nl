---
args: { request: SubstituteEvaluationRequest, rules: EvidencePage[] }
returns: QualificationGuardrail
---
Bind sample result to sampled lot, method, acceptance threshold and production-lot variability rule. If a passing sample cannot represent production lots under the approved rule, require lot-level testing before broad substitution.
