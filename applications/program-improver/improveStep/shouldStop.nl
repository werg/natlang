---
args:
  facts: StopFacts | CounterexampleStopFacts
returns: StopDecision
---
Decide whether the improvement search ends after this experiment; in the counterexample-guided search an experiment is one round. Read facts.search when it is present: `counterexample` means the counterexample-guided search, and the steps under "Counterexample search" apply. Without it, the steps under "Source search" apply.

Source search. facts describes the experiment that just finished (disposition, reason), the selected quality, the objective, the experiment count and facts.recent (the last experiments with the cases they failed).

1. When facts.disposition is `fixture-error`, stop with facts.reason.
2. When facts.objective is `quality` and facts.selectedQuality is 1, stop with the reason "Declared objective satisfied."
3. When facts.disposition is `no-hypothesis` or `no-opportunity`, stop with the reason "No further evidenced change."
4. When facts.iteration + 1 reaches facts.maxExperiments, stop with the reason "Declared experiments completed."
5. When the last three entries of facts.recent were rejected and name the same failing cases, stop with the reason "The last three experiments failed the same cases."
6. Otherwise continue, with the reason "Continue with a distinct hypothesis."

Counterexample search. facts describes the round that just finished: facts.admitted examples were admitted by the independent oracle, facts.remainingChecks is the oracle allowance left, and facts.repair describes the repair on the new suite (it is absent when nothing was admitted).

1. When facts.admitted is 0, stop with the reason "No counterexample was admitted; the source is retained."
2. When facts.repair.trainingQuality is 1, stop with the reason "The repaired training suite passes."
3. When facts.remainingChecks is 0, stop with the reason "The independent oracle allowance is exhausted."
4. When facts.repair.eligible is false, stop with the reason "The repair did not produce a completed eligible result."
5. Otherwise continue, with the reason "Continue with further counterexamples."
