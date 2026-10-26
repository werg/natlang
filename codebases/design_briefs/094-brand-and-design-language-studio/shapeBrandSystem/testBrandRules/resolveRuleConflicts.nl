---
description: Resolve rule applicability across brand directions, channels, support contexts and explicit exceptions.
args:{system:BrandSystemState;directions:BrandDirection[];constraints:{supportedClaimIds:string[];blockedClaims:string[];requiredDisclosures:string[];rightsQuestions:string[]}}
returns:{usableRules:string[];contradictions:string[];missingContexts:string[]}
---
For each VoiceRule bind its context to the direction and asset use it governs. Compare allowed and disallowed examples with supportedClaimIds, blockedClaims and requiredDisclosures. When one rule conflicts with another or a direction, report both rule IDs and the affected contexts; do not choose the more persuasive wording as precedence. Treat an exception as applicable only when the system's exact approvedScope names that rule and context and approvedByClient is true. A general system approval does not waive a blocked product claim or missing disclosure. Return usable rule IDs, unresolved conflicts and contexts lacking a usable rule.
