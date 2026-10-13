---
description: Build governed duties and permissions while preserving precisely scoped waived prerequisites.
args:
  intake: Intake
  selection: EditionSelection
returns: RequirementSet
---
1. Inventory substantive selected clauses by source action and actor role. Keep their event-time
   intervals and authority witnesses, including historical clauses needed by earlier scenarios.
2. For each clause call interpretRule(clause, intake.goal, selection). Consume its RuleMeaning,
   including actual actor/action/prerequisite/effect distinctions rather than a sentence-level summary.
3. Partition duties and permissions by the returned semantics. A permission allows an action if its
   prerequisites hold; a duty can remain owed even when the action it would enable is not yet permitted.
4. For each selected scoped exception call resolveException(exceptionClause, ruleMeanings, selection).
   Consume exactly which prerequisites it waives and which independent duties it retains. Attach that
   binding to the affected rules; do not delete every prerequisite on a neighboring action.
5. Preserve explicit forbidden consequences from the governing clauses: retroactive permission,
   shared custody inferred from appearance and acknowledgment inferred from receiving an object are
   separate prohibitions, each relevant to a different state or temporal join.
6. If a prerequisite is ambiguous, retain its missing meaning in unresolved and its state-key need
   in the rule. Ambiguity is not a reason to silently lower the rule to an unconditional permission.
7. Return selection, rules, permissions, forbiddenConsequences and unresolved. Model inference reads
   the actual predicate dependencies; scenario replay consumes the returned exception bindings.
