---
description: Replay discriminating scenarios against version-matched supplied simulator results.
args:{scope:GoalContract;specification:StageResultRequirement;schemas:EvidencePage[];simulatorResults:SimulatorReplay[]}
returns:StageResultAcceptanceScenario
---
Bind scenarios = `deriveRequirementScenarios(scope, specification.accepted)`. For every mandatory requirement create ordinary, boundary and adverse cases that distinguish success from failure. Bind each scenario to its schema version and exact host-produced simulator result. If result or schema is missing, leave it unknown; do not emulate the simulator. A counterexample reopens only its named requirement and dependent agreement. Preserve unrelated accepted scenarios.
