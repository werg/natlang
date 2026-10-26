---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: DocumentedPromise[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for probeBehavior. Form bounded sandbox probes using approved capabilities and representative inputs. Record environment and release identity, compare observed outcomes with the documented promise, and avoid treating one failed environment as proof of a product defect. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
