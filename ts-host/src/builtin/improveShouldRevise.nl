---
description: Decide whether a program function's instructions should be revised, given what its runs show (natlang improve DIR).
args:
  target: FunctionUnderReview
  evidence: RunEvidence
  diagnosis: string
returns: boolean
readout: decision
types:
  FunctionUnderReview: "{ path: string, declaration: string, instructions: string }"
  RunEvidence: "{ calls: number, failures: { inputs: unknown, outcome: string, reason: string }[], escalations: { inputs: unknown, local: unknown, fallback: unknown, agreed: boolean, reason: string }[], repairs: { inputs: unknown, problems: string[], outcome: string }[], feedback: { inputs: unknown, output: unknown, verdict: string, expected?: unknown }[], wrong: { inputs: unknown, expected: unknown, received?: unknown, error?: string }[] }"
---
target is one natural-language function of a program, evidence is what its recorded calls show, and diagnosis is a reading of that evidence.

Answer true when the diagnosis names a cause in target's instructions that rewording or completing them can fix, and the evidence supports it with more than one troubled call or with a tester's or example's expected result. Answer false when the diagnosis finds nothing to fix, when the trouble lies outside the instructions (a service that failed, an input that was missing), or when the evidence is a single call the instructions already cover.
