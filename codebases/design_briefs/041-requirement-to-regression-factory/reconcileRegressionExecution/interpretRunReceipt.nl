---
args: { plan: RegressionPlanProposal, receipt: RegressionExecutionReceipt }
returns: RegressionExecutionResult
---
1. Bind receipt.testId to exactly one test in plan.tests and require suiteVersion and sourceRevision to match the compiled test contract. Unknown or stale tests are unresolved and cover no requirement.
2. If the run environment is unavailable or outside the plan's approved sandbox, classify environment-unavailable; preserve the trace but do not grade product behavior.
3. For a passed run, require an independent oracle and actual trace references before classifying pass. A successful process exit with no oracle observation is unresolved.
4. For a failed run, use the trace, fixture provenance, and independent assertion to distinguish product-defect, oracle-defect, fixture-defect, harness-failure, environment-unavailable, or unresolved. Never copy implementation output into an expected result.
5. When repeatedRunIds include valid executions whose outcomes disagree under the same declared scenario, classify flaky and keep the full run history. Report requirement coverage only for a valid execution against the matching suite revision.
