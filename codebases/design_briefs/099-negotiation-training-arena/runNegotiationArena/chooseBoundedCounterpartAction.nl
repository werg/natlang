---
description: Select a counterpart response from permitted actions using only that role's information.
args:{scenario:ArenaScenario;roleId:string;offer:ParsedArenaOffer;review:{valid:boolean;questions:string[];authorityFailures:string[];feasibility:FeasibilityArtifact|null;};turns:ArenaTurn[]}
returns:CounterpartChoice
---
Resolve exactly one current role from scenario.roles and bind visibility = `filterRoleVisibleFacts(scenario, role, turns, offer)`. When a consequential question remains, use a clarification action rather than accepting inferred terms. Otherwise choose a legal response consistent with the role's private priorities and reservation conditions, while revealing only visibility.permittedFactIds. A policy conflict or prohibitedActionId prevents that action and is preserved in conductCheck. Respect role conduct limits; do not use deception, coercion or hidden score leakage as a training tactic. Record selected action and role-view basis privately, keeping visibility.privateFactIds out of learner-facing language.
