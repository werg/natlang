---
description: Lower one rule's governed prerequisites into true, false and unknown semantic branches.
args:
  rule: RuleMeaning
  schema: StateSchema
returns: BranchPlan
---
1. Join prerequisiteMeanings with their declared state keys and actor/time owners. Identify which
   independent facts establish readiness, custody, queue status and hazard absence for this rule.
2. Apply only the rule's returned exceptionBindings. Remove a waived queue prerequisite from the
   active guard when its exception scope holds, preserving the independent retained duty predicates.
3. State guardMeaning as a source-grounded conjunction/disjunction with event interval and actor scope.
   An absence of affirmative support for a required fact differs from an affirmative contradicting fact.
4. State trueEffect as the authorized state or permission contribution. Keep falseEffect as the
   specific unmet source condition, and unknownEffect as the missing premise with preserved alternatives.
5. A waived guard fact has no ordinary completion effects: emergency priority does not mark a queue
   booking complete, and receiving a key does not mark its holder safety-trained.
6. Return ruleId, guardMeaning, trueEffect, falseEffect, unknownEffect, requiredStateKeys and witnesses.
   The generated guard literal and scenario judge consume these exact branches rather than inventing them.
