---
args: { factory: RegressionFactoryRequest }
returns: RegressionPlanProposal
---
1. Bind behavior = recoverObservableContracts(factory). Split required outcome from implementation hints, and preserve unresolved owner decisions rather than turning guesses into assertions.
2. Bind partitions = designDecisivePartitions(behavior, factory). Add invalid inputs, boundary cases, interactions, and the specific failure behavior for each requirement.
3. Bind tests = authorIndependentOracles(partitions, factory). Each assertion must cite a requirement or approved interpretation and derive its expected behavior from that contract, never by copying current implementation output.
4. Bind sandbox = inspectExecutionBoundary(factory). Block before launch when production credentials or an unapproved environment are present. Keep fixture privacy checks separate from the product oracle.
5. Bind failures = diagnoseTestFailures(factory.priorRuns, tests, sandbox). Separate product failures from fixture, harness, load-flakiness, and unavailable-dependency findings. A flaky failure keeps its timing trace; retry policy stays bounded.
6. Bind qualification = challengeSuiteWithSeededDefects(tests, partitions, behavior, factory). A deleted assertion or undetected required mutation blocks qualification. Changed requirements invalidate only linked assertions and seed a targeted rebuild.
7. Return { projectId: factory.projectId, behavior, partitions, tests, sandbox, failures, qualification, revisionPlan: invalidateChangedAssertions(factory, behavior, tests), commitAllowed: false }.
