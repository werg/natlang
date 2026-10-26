---
description: Check voice rules across ordinary, error and sensitive-support contexts.
args:{system:BrandSystemState;directions:BrandDirection[];constraints:{supportedClaimIds:string[];blockedClaims:string[];requiredDisclosures:string[];rightsQuestions:string[];}}
returns:{usableRules:string[];contradictions:string[];missingContexts:string[];}
---
Return `resolveRuleConflicts(system, directions, constraints)`. Use its usable rule IDs when adapting each channel. Keep every conflicting rule pair and missing context in the returned arrays so the final review packet can preserve why a direction or support interaction remains blocked; do not collapse distinct channel exceptions into a generic voice warning.
