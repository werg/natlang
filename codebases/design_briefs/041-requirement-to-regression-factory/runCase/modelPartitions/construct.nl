---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftTestPartition
---
Construct a TestPartition draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Partition input and state space by behaviorally meaningful distinctions, including invalid inputs and interacting requirements. Identify boundary cases and previously untested combinations rather than generating many superficial variants of the same happy path.
