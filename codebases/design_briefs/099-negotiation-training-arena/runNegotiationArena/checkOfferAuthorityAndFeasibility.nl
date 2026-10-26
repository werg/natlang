---
description: Check exact offer artifact against scenario revision and each role mandate.
args:{scenario:ArenaScenario;roleId:string;offer:ParsedArenaOffer;artifacts:FeasibilityArtifact[]}
returns:{valid:boolean;questions:string[];authorityFailures:string[];feasibility:FeasibilityArtifact|null;}
---
Resolve exactly one `ArenaRole` for roleId; missing or duplicate roles make authority unresolved. Bind mandate = `checkIssueLevelMandates(scenario, role, offer)` and use its term-level authority failures, unknown issue IDs, bounds violations and condition questions. Match exactly one feasibility artifact to scenario revision and offer.turnId; duplicates or stale artifacts remain unresolved. Return valid only when the role mandate has no failures, every term is within bounds, no condition question remains, and the exact artifact confirms feasible, authorityValid and withinIssueBounds. Preserve every precise question and authority failure. Never infer feasibility from private preferences, and keep utility bands hidden from learner-facing output.
