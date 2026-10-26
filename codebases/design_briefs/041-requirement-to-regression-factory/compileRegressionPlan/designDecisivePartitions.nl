---
args: { contracts: BehavioralContract[], factory: RegressionFactoryRequest }
returns: RegressionPartition[]
---
For each contract split the input and state space where the expected outcome changes. Include normal behavior, each boundary, invalid preconditions, interaction with changed requirements, and the forbidden outcome. Identify uncovered combinations by requirement; do not inflate counts with equivalent surface variants.
