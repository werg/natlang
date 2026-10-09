---
args:
  facts: OpportunityFacts
returns: Opportunity
---
Say what the next experiment on the program should look at. facts.rows lists the training rows (caseId, passed, modelCalls, failureKind) and facts.objective is `quality`, `source-size` or `model-calls`.

1. When a row has failureKind `fixture`, return kind `fixture` and a reason that says the independent fixture failed before the program ran.
2. When a row has passed false, return kind `quality` and a reason that names the failed case ids.
3. When facts.objective is `model-calls` and a passing row used more than one model request, return kind `efficiency` and a reason that names those case ids and says the cost objective is still open.
4. When facts.objective is `source-size`, return kind `source-size` and a reason that says a behavior-preserving simplification can be measured.
5. Otherwise return kind `none` and the reason "No observed failure or cost opportunity supports this objective."
