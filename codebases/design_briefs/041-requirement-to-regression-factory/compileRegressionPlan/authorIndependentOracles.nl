---
args: { partitions: RegressionPartition[], factory: RegressionFactoryRequest }
returns: RegressionTestSpec[]
---
For each decisive partition construct deterministic setup, exact action, observable assertion, and cleanup under the approved harness. Derive expected output from the contract and approved interpretation. If an oracle depends on current implementation output, mark it circular and reject the test until an independent owner supplies expected behavior.
