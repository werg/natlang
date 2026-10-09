---
args:
  facts: StopFacts
returns: StopDecision
---
Decide whether the improvement search ends after this experiment. facts describes the experiment that just finished (disposition, reason), the selected quality, the objective, the experiment count and facts.recent (the last experiments with the cases they failed).

1. When facts.disposition is `fixture-error`, stop with facts.reason.
2. When facts.objective is `quality` and facts.selectedQuality is 1, stop with the reason "Declared objective satisfied."
3. When facts.disposition is `no-hypothesis` or `no-opportunity`, stop with the reason "No further evidenced change."
4. When facts.iteration + 1 reaches facts.maxExperiments, stop with the reason "Declared experiments completed."
5. When the last three entries of facts.recent were rejected and name the same failing cases, stop with the reason "The last three experiments failed the same cases."
6. Otherwise continue, with the reason "Continue with a distinct hypothesis."
