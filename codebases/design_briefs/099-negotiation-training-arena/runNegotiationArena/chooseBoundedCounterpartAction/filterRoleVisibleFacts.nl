---
description: Filter counterpart knowledge to the role's explicit disclosure rights and the current exercise policy.
args:{scenario:ArenaScenario;role:ArenaRole;turns:ArenaTurn[];offer:ParsedArenaOffer}
returns:{permittedFactIds:string[];privateFactIds:string[];prohibitedActionIds:string[];policyConflicts:string[]}
---
Use only facts listed in this role's disclosableFacts. Treat privatePriorities, reservationConditions and any utility-derived inference as internal decision inputs, never as learner-visible facts. Check the exact action permitted by scenario.allowedTactics against prohibitedTactics and the role's conductLimits. A fact mentioned by another role in history does not grant this role permission to reveal it. Return the exact permitted facts and withheld private facts to the action chooser; if policy or role scope conflicts, report it and permit no revealing action.
