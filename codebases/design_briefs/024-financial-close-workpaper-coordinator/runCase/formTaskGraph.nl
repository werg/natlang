---
args: { goal: GoalContract, snapshot: Snapshot }
returns: WorkGraphProposal
---
1. Name each stage, its prerequisite, typed input, and distinct output. Block a dependent node whose required input or capability is absent.
2. Decompose the goal into named stage invocations, inputs, prerequisite edges, result contracts, and blocked questions. Each node must consume an available capability and contribute a distinct artifact or state decision. Detect missing prerequisites before proposing execution.
3. Return an explicit artifact; preserve unknowns and source coverage. This procedure proposes no production write.
