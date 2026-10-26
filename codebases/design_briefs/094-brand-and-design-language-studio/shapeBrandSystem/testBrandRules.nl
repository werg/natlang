---
description: Check voice rules across ordinary, error and sensitive-support contexts.
args:{system:BrandSystemState;directions:BrandDirection[];constraints:{supportedClaimIds:string[];blockedClaims:string[];requiredDisclosures:string[];rightsQuestions:string[];}}
returns:{usableRules:string[];contradictions:string[];missingContexts:string[];}
---
Check that each rule can guide an asset and that examples follow it. Require distinct support/error behavior where promotional voice would mislead, shame or obscure instructions. Flag rules that contradict positioning, prohibited claims or mandatory disclosures. Keep stylistic preferences out of the factual defect list.
