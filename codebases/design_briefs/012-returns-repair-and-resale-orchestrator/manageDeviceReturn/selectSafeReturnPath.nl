---
description: Select only a policy-authorized, consent-compatible device return route.
args:{scope:GoalContract;policy:PolicySelection;condition:StageResultConditionAssessment;reports:EvidencePage[];customerPreference:string}
returns:StageResultReturnDisposition
---
Bind terms = `matchReturnEntitlement(scope, policy, reports)`. Compare condition certainty with purchase-date rights and approved handling restrictions. Respect stated customer preference only within authorized paths. If a diagnosis is unresolved, preserve specialist inspection as a route. Do not promise repair or replacement before required checks and customer consent.
