---
args: { tests: RegressionTestSpec[], partitions: RegressionPartition[], contracts: BehavioralContract[], factory: RegressionFactoryRequest }
returns: SuiteQualification
---
For each required seeded defect choose the contract partition whose observable behavior it violates. Confirm that at least one independent assertion detects it and that the test does not rely on the mutation's implementation text. A missing detection or deleted assertion leaves the suite unqualified. Retain regression cases for repaired defects; remove redundancy only when the full decisive partition remains covered.
