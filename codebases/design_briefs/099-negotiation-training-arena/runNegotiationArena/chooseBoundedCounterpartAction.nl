---
description: Select a counterpart response from permitted actions using only that role's information.
args:{scenario:ArenaScenario;roleId:string;offer:ParsedArenaOffer;review:{valid:boolean;questions:string[];authorityFailures:string[];feasibility:FeasibilityArtifact|null;};turns:ArenaTurn[]}
returns:CounterpartChoice
---
When a consequential question remains, use a clarification action rather than accepting inferred terms. Otherwise choose a legal response consistent with this role's private priorities and reservation conditions. Reveal only role-authorized facts. Respect conduct limits; do not use deception, coercion or hidden score leakage as a training tactic. Record the selected action and its role-view basis privately.
