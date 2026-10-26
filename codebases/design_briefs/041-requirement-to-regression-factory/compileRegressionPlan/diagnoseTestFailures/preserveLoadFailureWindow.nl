---
args: { runs: TestRunEvidence[], trace: TestFailureClassification[] }
returns: FlakyRunEvidence
---
Compare failing and passing run intervals, load conditions, and retry count. Preserve the original failure trace and classify instability explicitly. A later pass does not erase a load failure or count it as product coverage.
