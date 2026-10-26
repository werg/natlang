---
description: Check exact offer artifact against scenario revision and each role mandate.
args:{scenario:ArenaScenario;roleId:string;offer:ParsedArenaOffer;artifacts:FeasibilityArtifact[]}
returns:{valid:boolean;questions:string[];authorityFailures:string[];feasibility:FeasibilityArtifact|null;}
---
Require one exact verifier artifact bound to current scenario revision and offer identifier. Check role authority for each issue term and condition. If no matching artifact exists, keep feasibility unknown; never infer a mutually beneficial trade from incomplete values. Hide private utility bands unless the debrief release scope authorizes them.
