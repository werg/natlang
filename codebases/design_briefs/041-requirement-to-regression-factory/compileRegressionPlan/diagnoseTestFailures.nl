---
args: { runs: TestRunEvidence[], tests: RegressionTestSpec[], sandbox: SandboxEligibility }
returns: TestFailureClassification[]
---
Bind trace = align each failing assertion with its requirement, harness trace, and fixture health. If behavior violates the independent contract, classify product defect. If setup, environment, credentials, or dependency availability failed first, classify execution invalidity and exclude it from coverage. If load changes timing, preserve the measured window and bounded retry count instead of retrying until green. Bind flake = preserveLoadFailureWindow(runs, trace); never widen the retry budget until a failure disappears.
