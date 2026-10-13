---
description: Repair one demonstrated generated-program divergence on a private folder without changing source policy.
kind: directory-reducer
args:
  state: RevisionState
uses: [writing, scenarios, audit]
returns: RevisionState
---
1. Select one original remainingScenarioId and remove it from a copied frontier. Find its current
   verdict and original input history. If prior repairs already satisfy it, record that actual verdict
   and settle the ID without inventing a new semantic defect.
2. For a violated or unresolved verdict call explainCounterexample(verdict, program, model). Consume
   its earliestEventId, affectedRuleIds, affectedPaths and source-backed repairObligation.
3. Preserve an unresolved source authority or scenario fact as unresolved. A generated-policy repair
   cannot invent the missing source amendment or manufacture the scenario's absent safety acknowledgment.
4. Revise only the demonstrated modules/typed handoffs on this private folder candidate. Reuse their
   existing semantic responsibilities; a mistake does not justify replacing all modules with one final-answer stage.
5. Re-run writing for changed plans, then scenarios for every original history, then audit the complete
   generated program. A repaired emergency case that breaks normal custody transfer is not an accepted candidate.
6. Keep the revised program/suite only if the actual returned source/interface/scenario obligations
   hold, otherwise retain the previous valid parts and append the specific remaining defect finding.
7. Append the actual RevisionTarget and settle this scenario ID. Return copied remainingScenarioIds,
   targets, settledScenarioIds, program, suite, findings, model and intake. A failed repair terminates
   with an explicit target/finding rather than endlessly retrying an unchanged candidate.
8. The caller consumes the returned program and suite, including unresolved failures. Exact file
   serialization and private Folder commit are mechanisms; candidate semantic validity stays in NL.
